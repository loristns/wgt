import {gemm} from './gemm';
import {Tensor} from '../tensor';
import {DeviceTensor} from '../deviceTensor';
import {parameters, RootParameters} from './parameters';

export interface LinearParameters extends RootParameters {
  weights: Tensor;
  bias: Tensor;
}

export function linear(
  input: DeviceTensor,
  params: LinearParameters
): DeviceTensor {
  const {weights, bias} = parameters(params);

  const output = gemm(input, weights, bias);

  return output;
}
