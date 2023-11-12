import {Tensor} from '../tensor';
import {TensorBuffer} from '../tensorBuffer';

export type Parameters<Self = unknown> = Record<string, Self | Tensor>;
export type RootParameters = Parameters<never>;

type RootParameterBuffers<P extends RootParameters> = Record<
  keyof P,
  TensorBuffer
>;

export function parameters<P extends RootParameters>(
  parameters: P
): RootParameterBuffers<P> {
  const parameterBuffers = {} as RootParameterBuffers<P>;

  Object.entries(parameters).forEach(([name, param]) => {
    parameterBuffers[name as keyof P] = TensorBuffer.fromTensor(param);
  });

  return parameterBuffers;
}
