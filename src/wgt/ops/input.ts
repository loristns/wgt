import {Shape, ShapeLike} from '../tensor';
import {TensorBuffer} from '../tensorBuffer';

export function input(shapeLike: ShapeLike): TensorBuffer {
  const shape = Shape.from(shapeLike);
  return new TensorBuffer(shape);
}
