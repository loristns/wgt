import {Tensor} from './tensor';
import {Op} from './ops';

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
    commands = commands.filter((command, index) => {
      const firstCommandIndex = commands.findIndex(
        otherCommand =>
          command.pipeline === otherCommand.pipeline &&
          command.params.every((param, i) => param === otherCommand.params[i])
      );
      return index === firstCommandIndex;
    });

    const encoder = WGT.device.createCommandEncoder();

    commands.forEach(command => {
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
