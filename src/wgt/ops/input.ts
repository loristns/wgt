import {Shape, ShapeLike} from '../tensor';
import {DeviceTensor} from '../deviceTensor';

export function input(shapeLike: ShapeLike): DeviceTensor {
  const shape = Shape.from(shapeLike);
  return new DeviceTensor(shape);
}
