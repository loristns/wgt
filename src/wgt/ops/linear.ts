import {gemm} from './gemm';
import {Tensor} from '../tensor';
import {TensorBuffer} from '../tensorBuffer';
import {parameters, RootParameters} from './parameters';

export interface LinearParameters extends RootParameters {
  weights: Tensor;
  bias: Tensor;
}

export function linear(
  input: TensorBuffer,
  params: LinearParameters
): TensorBuffer {
  const {weights, bias} = parameters(params);

  const output = gemm(input, weights, bias);

  return output;
}
