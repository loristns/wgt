import {WGT} from './wgt';
import {Shape, Tensor} from './tensor';
import {RawCopyCommand} from './commands';
import {Op} from './op';

/**
 * A TensorBuffer is a GPU buffer that may contain a tensor.
 */
export class TensorBuffer {
  readonly shape: Shape;
  readonly parentOp?: Op;

  private _buffer: GPUBuffer;
  private _readableBuffer?: GPUBuffer;

  constructor(shape: Shape, parentOp?: Op) {
    this.shape = shape;
    this.parentOp = parentOp;

    this._buffer = WGT.device.createBuffer({
      size: shape.size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
  }

  static fromTensor(tensor: Tensor): TensorBuffer {
    const tensorBuffer = new TensorBuffer(tensor.shape);
    tensorBuffer.write(tensor);

    return tensorBuffer;
  }

  get buffer(): GPUBuffer {
    return this._buffer;
  }

  get isReadable(): boolean {
    return this._readableBuffer != null;
  }

  /**
   * Setup read functionality for the tensor buffer.
   */
  markAsReadable() {
    if (!this.isReadable) {
      this._readableBuffer = WGT.device.createBuffer({
        size: this.shape.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
  }

  get readCommand() {
    if (!this.isReadable) {
      throw new Error(
        'Cannot add read command for tensor buffer that is not marked as readable.'
      );
    }

    return new RawCopyCommand(this._buffer, this._readableBuffer!);
  }

  /**
   * Read the tensor buffer.
   */
  async read(): Promise<Tensor> {
    if (!this.isReadable) {
      throw new Error(
        'Cannot read tensor buffer that is not marked as readable.'
      );
    }

    await this._readableBuffer!.mapAsync(GPUMapMode.READ);
    const arrayBuffer = new ArrayBuffer(this.shape.size);

    new Uint8Array(arrayBuffer).set(
      new Uint8Array(this._readableBuffer!.getMappedRange())
    );

    this._readableBuffer!.unmap();

    return new Tensor(arrayBuffer);
  }

  /**
   * Update the tensor buffer with the given tensor.
   */
  write(tensor: Tensor) {
    // Assert that the tensor shape is the same as the buffer shape.
    if (!this.shape.equals(tensor.shape)) {
      throw new Error(
        `Cannot write tensor of shape ${tensor.shape} to tensor buffer of shape ${this.shape}.`
      );
    }

    WGT.device.queue.writeBuffer(this._buffer, 0, tensor.arrayBuffer);
  }

  destroy() {
    this._buffer.destroy();
    this._readableBuffer?.destroy();
  }
}
