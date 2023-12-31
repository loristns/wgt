import {Command} from './commands';
import {Tensor} from './tensor';
import {DeviceTensor} from './deviceTensor';
import {Op} from './op';

/**
 * The WGT class is the main class of the WGT library.
 *
 * It represents a computation graph that can be executed on the GPU.
 */
export class WGT {
  static device: GPUDevice;

  /**
   * Set up the WebGPU device.
   */
  static async initializeGpu() {
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();

    if (!device) {
      throw new Error('WebGPU not supported');
    }

    WGT.device = device;
  }

  inputs: DeviceTensor[];
  outputs: DeviceTensor[];
  private _commands: Command[];

  /**
   * Build a computation graph.
   *
   * @param inputs List of input DeviceTensor
   * @param outputs List of output DeviceTensor
   */
  constructor(inputs: DeviceTensor[], outputs: DeviceTensor[]) {
    this.inputs = inputs;
    this.outputs = outputs;

    // Mark all output DeviceTensor as readable.
    this.outputs.forEach(output => {
      output.markAsReadable();
    });

    // Get all needed commands.
    const commands = new Set<Command>(); // Use a Set to avoid duplicates.

    this.outputs.forEach(output => {
      output.sourceCommands.forEach(command => {
        commands.add(command);
      });
    });

    this._commands = Array.from(commands);
  }

  /**
   * Run the computation graph.
   * @param inputs List of input Tensor, in the same order as the inputs
   *              of the computation graph.
   * @returns List of output Tensor, in the same order as the outputs
   *          of the computation graph.
   */
  async run(inputs: Tensor[]): Promise<Tensor[]> {
    // Write the input tensors to the GPU.
    inputs.forEach((input, i) => {
      this.inputs[i].write(input);
    });

    const encoder = WGT.device.createCommandEncoder();

    this._commands.forEach(command => {
      command.execute(encoder);
    });

    // Submit the commands.
    const commandBuffer = encoder.finish();
    WGT.device.queue.submit([commandBuffer]);

    // Read the output tensors.
    return await Promise.all(
      this.outputs.map(outputBuffer => outputBuffer.read())
    );
  }

  /**
   * Destroy the computation graph recursively.
   */
  destroy() {
    this.outputs.forEach(output => output.destroy(true));
  }

  getRecipe() {
    return this._commands
      .filter(command => command instanceof Op)
      .map(command => (command as Op).label);
  }

  getDotTree() {
    const nodes = new Set<string>();
    const edges = new Set<string>();

    this._commands
      .filter(command => command instanceof Op)
      .forEach(command => {
        const op = command as Op;

        nodes.add(
          `"${op.outputs[0].uuid}" [label="${
            op.label
          }" style=filled fillcolor=${
            this.outputs.includes(op.outputs[0]) ? 'green' : 'white'
          }]`
        );

        op.inputs
          //.filter(input => input.sourceOp != null)
          .forEach(input => {
            if (input.sourceOp == null) {
              nodes.add(
                `"${input.uuid}" [label="input (${
                  input.shape
                })" shape=box style=filled fillcolor=${
                  this.inputs.includes(input) ? 'red' : 'gray'
                }]`
              );
              edges.add(`"${input.uuid}" -> "${op.outputs[0].uuid}"`);
              return;
            }

            edges.add(
              `"${input.sourceOp.outputs[0].uuid}" -> "${op.outputs[0].uuid}"`
            );
          });
      });

    return `digraph {
      ${Array.from(nodes).join('\n      ')}
      ${Array.from(edges).join('\n      ')}
    }`;
  }
}
