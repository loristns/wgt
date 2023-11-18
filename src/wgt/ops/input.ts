import {ShapeLike} from '../tensor';
import {DeviceTensor} from '../deviceTensor';

export function input(shape: ShapeLike): DeviceTensor {
  return new DeviceTensor(shape);
}
