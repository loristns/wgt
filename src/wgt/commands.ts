import {WGT} from './wgt';

/**
 * A Command is a GPU operation that can be executed.
 */
export abstract class Command {
  abstract execute(encoder: GPUCommandEncoder): void;
}

/**
 * A ComputeCommand runs a compute shader on the GPU.
 */
export class ComputeCommand extends Command {
  private _pipeline: GPUComputePipeline;
  private _bindGroup: GPUBindGroup;
  private _workgroups: [number, number?, number?];

  constructor(params: {
    buffers: GPUBuffer[];
    workgroups: [number, number?, number?];
    code: string;
    entryPoint?: string;
  }) {
    super();

    const {buffers, workgroups, code, entryPoint = 'main'} = params;

    this._pipeline = WGT.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: WGT.device.createShaderModule({
          code,
        }),
        entryPoint,
      },
    });

    this._bindGroup = WGT.device.createBindGroup({
      layout: this._pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, i) => ({
        binding: i,
        resource: {buffer},
      })),
    });

    this._workgroups = workgroups;
  }

  execute(encoder: GPUCommandEncoder) {
    const pass = encoder.beginComputePass();

    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, this._bindGroup);
    pass.dispatchWorkgroups(...this._workgroups);

    pass.end();
  }
}

/**
 * A CopyCommand copies data from one GPU buffer to another.
 */
export class CopyCommand extends Command {
  private _source: GPUBuffer;
  private _destination: GPUBuffer;

  constructor(source: GPUBuffer, destination: GPUBuffer) {
    super();

    this._source = source;
    this._destination = destination;
  }

  execute(encoder: GPUCommandEncoder) {
    encoder.copyBufferToBuffer(
      this._source,
      0,
      this._destination,
      0,
      this._source.size
    );
  }
}
