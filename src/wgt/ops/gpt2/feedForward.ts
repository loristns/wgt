import {DeviceTensor} from '../../deviceTensor';

import {Parameters} from '../parameters';
import {linear, LinearParameters} from '../linear';
import {gelu} from '../gelu';

export interface FeedForwardParameters extends Parameters {
  linear1: LinearParameters;
  linear2: LinearParameters;
}

/**
 * A feed-forward operation (with a GELU activation function)
 */
export function feedForward(
  input: DeviceTensor,
  params: FeedForwardParameters
): DeviceTensor {
  const linear1 = linear(input, params.linear1);
  const gelu1 = gelu(linear1);
  const linear2 = linear(gelu1, params.linear2);

  return linear2;
}
