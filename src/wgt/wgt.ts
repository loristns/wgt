import {Command} from './commands';
import {Tensor} from './tensor';
import {DeviceTensor} from './deviceTensor';

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
    let commands = this.outputs
      .flatMap(output => output.sourceCommands)
      .filter((command): command is Command => command != null);

    // Filter out duplicate commands.
    commands = commands.filter((command, index) => {
      const firstCommandIndex = commands.findIndex(
        otherCommand => otherCommand === command
      );
      return index === firstCommandIndex;
    });

    this._commands = commands;
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
}
