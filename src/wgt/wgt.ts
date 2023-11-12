import {Command} from './commands';
import {Op} from './op';
import {Tensor} from './tensor';
import {TensorBuffer} from './tensorBuffer';

/**
 * The WGT class is the main class of the WGT library.
 *
 * It is used to create and execute operations on the GPU.
 */
export class WGT {
  static device: GPUDevice;

  static async initializeGpu() {
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();

    if (!device) {
      throw new Error('WebGPU not supported');
    }

    WGT.device = device;
  }

  inputBuffers: TensorBuffer[];
  outputBuffers: TensorBuffer[];
  private _commands: Command[];

  private get _outputOps(): Op[] {
    return this.outputBuffers
      .map(outputBuffer => outputBuffer.parentOp)
      .filter((op): op is Op => op != null);
  }

  constructor(inputs: TensorBuffer[], outputs: TensorBuffer[]) {
    this.inputBuffers = inputs;
    this.outputBuffers = outputs;

    // Mark all output buffers as readable.
    this.outputBuffers.forEach(outputBuffer => {
      outputBuffer.markAsReadable();
    });

    // Get all needed commands.
    let commands = [
      ...this._outputOps.flatMap(op => op.commands),
      ...this.outputBuffers.map(outputBuffer => outputBuffer.readCommand),
    ];

    // Filter out duplicate commands.
    commands = commands.filter((command, index) => {
      const firstCommandIndex = commands.findIndex(
        otherCommand => otherCommand === command
      );
      return index === firstCommandIndex;
    });

    this._commands = commands;
  }

  async run(inputs: Tensor[]): Promise<Tensor[]> {
    // Write the input tensors to the input buffers.
    inputs.forEach((input, i) => {
      this.inputBuffers[i].write(input);
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
      this.outputBuffers.map(outputBuffer => outputBuffer.read())
    );
  }

  destroy() {
    this._outputOps.forEach(outputOp => outputOp.destroySubTree());
  }
}
