import {Tensor} from './tensor';
import {Op, OpType} from './ops';

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

  static async run(outputOps: Op[]) {
    let commands = outputOps.flatMap(op => op.getCommands());

    // Filter out duplicate commands, eg:
    //  input1 --> op1 --> op2 --> output1
    //  input2 -|      |-> op3 --> output2
    //  Command would be [input1, input2, op1, op2, input1, input2, op1, op3]
    //  After filtering: [input1, input2, op1, op2, op3]
    // TODO: make this code more readable
    commands = commands.filter((command, index) => {
      // Find the index of the first command that is identical to the current command
      const firstCommandIndex = commands.findIndex(
        otherCommand =>
          // If both commands are compute commands and have the same pipeline and params
          ((command.type === OpType.COMPUTE || command.type == null) &&
            (otherCommand.type === OpType.COMPUTE ||
              otherCommand.type == null) &&
            command.pipeline === otherCommand.pipeline &&
            command.params.every(
              (param, i) => param === otherCommand.params[i]
            )) ||
          // If both commands are copy commands and have the same src and dst buffers
          (command.type === OpType.COPY &&
            otherCommand.type === OpType.COPY &&
            command.src === otherCommand.src &&
            command.dst === otherCommand.dst)
      );
      // Keep only the first command and remove duplicates
      return index === firstCommandIndex;
    });

    const encoder = WGT.device.createCommandEncoder();

    commands.forEach(command => {
      // Compute command
      if (command.type === OpType.COMPUTE || command.type == null) {
        const bindGroup = WGT.device.createBindGroup({
          layout: command.pipeline.getBindGroupLayout(0),
          entries: command.params.map((param, i) => ({
            binding: i,
            resource: {
              buffer: param,
            },
          })),
        });

        const pass = encoder.beginComputePass();

        pass.setPipeline(command.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(...command.workgroups);
        pass.end();

        // Copy command
      } else if (command.type === OpType.COPY) {
        encoder.copyBufferToBuffer(
          command.src,
          0,
          command.dst,
          0,
          command.src.size
        );

        // Unknown command type
      } else {
        throw new Error(`Unknown command type: ${command.type}`);
      }
    });

    // Copy all output variables to the CPU
    outputOps.forEach(op => {
      encoder.copyBufferToBuffer(
        op.buffer,
        0,
        op.readableBuffer,
        0,
        op.shape.size
      );
    });

    const commandBuffer = encoder.finish();
    WGT.device.queue.submit([commandBuffer]);

    return Promise.all(
      outputOps.map(async op => {
        await op.readableBuffer.mapAsync(GPUMapMode.READ);

        const arrayBuffer = new ArrayBuffer(op.shape.size);
        new Uint8Array(arrayBuffer).set(
          new Uint8Array(op.readableBuffer.getMappedRange())
        );

        op.readableBuffer.unmap();

        return new Tensor(arrayBuffer);
      })
    );
  }

  static destroy(ops: Op[]) {
    ops.forEach(op => {
      op.destroy();
      op.dependencies.forEach(dep => dep.destroy());
    });
  }
}
