import {WGT} from '../index';
import {TensorShape} from '../tensor';

export enum OpType {
  COMPUTE,
  COPY,
}

/**
 * A simplified description of a compute pass on the GPU.
 */
export type OpCommand =
  | {
      type?: OpType.COMPUTE; // This is the default type, so it can be omitted.
      pipeline: GPUComputePipeline;
      params: GPUBuffer[];
      workgroups: [number, number?, number?];
    }
  | {
      type: OpType.COPY;
      src: GPUBuffer;
      dst: GPUBuffer;
    };

/**
 * Base class for all operations.
 *
 * All operations have an associated output shape, buffer (that stores the output tensor) and dependencies.
 *
 * The output buffer is created when the operation is created, so it size must be known at
 * construction time.
 *
 * Dependencies are operations that must be executed before this operation, and are used to
 * build the command array.
 */
export abstract class Op {
  readonly dependencies: Op[];
  readonly shape: TensorShape;

  readonly buffer: GPUBuffer;
  private _readableBuffer?: GPUBuffer;

  /**
   * Returns a buffer that can be read from the CPU.
   */
  get readableBuffer() {
    if (this._readableBuffer == null) {
      this._readableBuffer = WGT.device.createBuffer({
        size: this.shape.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }

    return this._readableBuffer;
  }

  constructor(shape: TensorShape, dependencies: Op[], buffer?: GPUBuffer) {
    this.shape = shape;
    this.dependencies = dependencies;

    // If a buffer is provided, use it. Otherwise, create a new buffer.
    this.buffer =
      buffer ??
      WGT.device.createBuffer({
        size: this.shape.size,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
  }

  getCommands(): OpCommand[] {
    // By default, an operation has no commands.
    // It simply returns the commands of its dependencies.
    return [...this.dependencies.flatMap(d => d.getCommands())];
  }

  destroy() {
    this.buffer.destroy();
    this._readableBuffer?.destroy();
  }
}
