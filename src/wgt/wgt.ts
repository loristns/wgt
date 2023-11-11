import {Op} from './ops';
import {Tensor} from './tensor';

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

  static async run(outputOps: Op[]): Promise<Tensor[]> {
    // Mark all output ops as readable.
    outputOps.forEach(op => {
      op.buffer.markAsReadable();
    });

    // Encode the operations.
    let commands = [
      ...outputOps.flatMap(op => op.getCommands()),
      ...outputOps.map(op => op.buffer.getReadCommand()),
    ];

    // Filter out duplicate commands.
    commands = commands.filter((command, index) => {
      const firstCommandIndex = commands.findIndex(
        otherCommand => otherCommand === command
      );
      return index === firstCommandIndex;
    });

    const encoder = WGT.device.createCommandEncoder();

    commands.forEach(command => {
      command.execute(encoder);
    });

    // Submit the commands.
    const commandBuffer = encoder.finish();
    WGT.device.queue.submit([commandBuffer]);

    // Read the output tensors.
    return await Promise.all(outputOps.map(op => op.buffer.read()));
  }
}
