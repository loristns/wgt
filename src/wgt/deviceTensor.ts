import {WGT} from './wgt';
import {Shape, ShapeLike, Tensor} from './tensor';
import {Command, CopyCommand} from './commands';
import {Op} from './op';

/**
 * A DeviceTensor holds references to GPU memory storing a Tensor
 * at some time in the computation graph.
 *
 * The DeviceTensor can originate from a Tensor on the CPU or from
 * the output of an Op. It can be readable from the CPU or not.
 */
export class DeviceTensor {
  readonly shape: Shape;

  private _buffer: GPUBuffer;
  private _readableBuffer?: GPUBuffer;

  get buffer(): GPUBuffer {
    return this._buffer;
  }

  get isReadable(): boolean {
    return this._readableBuffer != null;
  }

  sourceOp?: Op;

  /**
   * List of all DeviceTensor that are needed to compute this DeviceTensor.
   */
  private get _dependencies(): DeviceTensor[] {
    const sourceTensors = new Set<DeviceTensor>();
    const queue: DeviceTensor[] = [this];

    while (queue.length > 0) {
      const deviceTensor = queue.shift()!;

      if (sourceTensors.has(deviceTensor)) {
        continue;
      }
      sourceTensors.add(deviceTensor);
      queue.push(...(deviceTensor.sourceOp?.inputs ?? []));
    }

    return Array.from(sourceTensors).reverse();
  }

  /**
   * List of all commands needed to compute (and read) this DeviceTensor.
   */
  get sourceCommands(): Command[] {
    const commands = new Set<Command>();

    this._dependencies.forEach(deviceTensor => {
      if (deviceTensor.sourceOp != null) {
        commands.add(deviceTensor.sourceOp);
      }

      if (deviceTensor.isReadable) {
        commands.add(
          new CopyCommand(deviceTensor._buffer, deviceTensor._readableBuffer!)
        );
      }
    });

    return Array.from(commands);
  }

  constructor(shape: ShapeLike) {
    this.shape = Shape.from(shape);

    this._buffer = WGT.device.createBuffer({
      size: this.shape.size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
  }

  static fromTensor(tensor: Tensor): DeviceTensor {
    const deviceTensor = new DeviceTensor(tensor.shape);
    deviceTensor.write(tensor);

    return deviceTensor;
  }

  markAsReadable() {
    if (!this.isReadable) {
      this._readableBuffer = WGT.device.createBuffer({
        size: this.shape.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
  }

  /**
   * Read a tensor from the DeviceTensor.
   */
  async read(): Promise<Tensor> {
    if (!this.isReadable) {
      throw new Error(
        'Cannot read DeviceTensor that is not marked as readable.'
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
   * Update the DeviceTensor with the given tensor.
   */
  write(tensor: Tensor) {
    // Assert that the tensor shape is the same as the buffer shape.
    if (!this.shape.equals(tensor.shape)) {
      throw new Error(
        `Cannot write tensor of shape ${tensor.shape} to DeviceTensor of shape ${this.shape}.`
      );
    }

    WGT.device.queue.writeBuffer(this._buffer, 0, tensor.arrayBuffer);
  }

  destroy(destroyDependencies = false) {
    this._buffer.destroy();
    this._readableBuffer?.destroy();

    if (destroyDependencies) {
      this._dependencies.forEach(sourceTensor => {
        sourceTensor.destroy();
      });
    }
  }
}
