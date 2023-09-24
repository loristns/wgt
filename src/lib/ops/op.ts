import {WGT} from '../index';
import {TensorShape} from '../tensor';

/**
 * A simplified description of a compute pass on the GPU.
 */
export type OpCommand = {
  pipeline: GPUComputePipeline;
  params: GPUBuffer[];
  workgroups: [number, number?, number?];
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

  constructor(shape: TensorShape, dependencies: Op[]) {
    this.shape = shape;
    this.dependencies = dependencies;

    this.buffer = WGT.device.createBuffer({
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

  clean() {
    this.buffer.destroy();
    this._readableBuffer?.destroy();
  }
}
