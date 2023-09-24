import {WGT} from '../index';
import {Tensor, TensorShape} from '../tensor';
import {Op} from './op';

/**
 * The input operation takes a tensor and writes it to the GPU.
 */
export class Input extends Op {
  constructor(shape: TensorShape);
  constructor(rows: number, cols: number);
  constructor(shapeOrRows: TensorShape | number, cols?: number) {
    const shape =
      typeof shapeOrRows === 'number'
        ? new TensorShape(shapeOrRows, cols!)
        : shapeOrRows;

    // The input operation has no dependencies so we pass an empty array.
    super(shape, []);
  }

  write(tensor: Tensor) {
    WGT.device.queue.writeBuffer(this.buffer, 0, tensor.arrayBuffer);
  }
}
