import {TensorBuffer} from './tensorBuffer';
import {WGT} from './wgt';

export abstract class Command {
  abstract execute(encoder: GPUCommandEncoder): void;
}

export class ComputeCommand extends Command {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  workgroups: [number, number?, number?];

  constructor(params: {
    args: TensorBuffer[];
    workgroups: [number, number?, number?];
    code: string;
    entryPoint?: string;
  }) {
    super();

    const {args, workgroups, code, entryPoint = 'main'} = params;

    this.pipeline = WGT.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: WGT.device.createShaderModule({
          code,
        }),
        entryPoint,
      },
    });

    this.bindGroup = WGT.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: args.map((arg, i) => ({
        binding: i,
        resource: {
          buffer: arg.buffer,
        },
      })),
    });

    this.workgroups = workgroups;
  }

  execute(encoder: GPUCommandEncoder) {
    const pass = encoder.beginComputePass();

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(...this.workgroups);

    pass.end();
  }
}

export class RawCopyCommand extends Command {
  source: GPUBuffer;
  destination: GPUBuffer;

  constructor(source: GPUBuffer, destination: GPUBuffer) {
    super();

    this.source = source;
    this.destination = destination;
  }

  execute(encoder: GPUCommandEncoder) {
    encoder.copyBufferToBuffer(
      this.source,
      0,
      this.destination,
      0,
      this.source.size
    );
  }
}
