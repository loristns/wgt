import {Tensor} from '../tensor';
import {DeviceTensor} from '../deviceTensor';
import {Op} from '../op';
import {parameters, RootParameters} from './parameters';

export interface LayerNormParameters extends RootParameters {
  scale: Tensor;
  bias: Tensor;
}

export function layerNorm(
  input: DeviceTensor,
  params: LayerNormParameters
): DeviceTensor {
  const shape = input.shape;
  const {scale, bias} = parameters(params);

  const output = new DeviceTensor(shape);
  output.sourceOp = new Op({
    label: 'layerNorm',
    inputs: [input, scale, bias],
    outputs: [output],
    workgroups: [
      Math.ceil(shape.rows / 16),
      Math.ceil(shape.cols / 16),
      shape.batches,
    ],
    code: /* wgsl */ `
      ${Tensor.WGSL}
        
      // Input
      @group(0) @binding(0) var<storage, read> input: Tensor;
      @group(0) @binding(1) var<storage, read> scale: Tensor;
      @group(0) @binding(2) var<storage, read> bias: Tensor;

      // Output
      @group(0) @binding(3) var<storage, read_write> result: Tensor;

      @compute @workgroup_size(16, 16, 1) fn main(
        @builtin(global_invocation_id) id: vec3<u32>,
      ) {
        result.shape = input.shape;

        let batch = id.z;
        let row = id.x;
        let col = id.y;

        let value = input.tensor[tensor_idx(input.shape, batch, row, col)];
        let scaleValue = scale.tensor[tensor_idx(scale.shape, batch, row, col)];
        let biasValue = bias.tensor[tensor_idx(bias.shape, batch, row, col)];

        var mean = 0.0;
        for (var i = 0u; i < input.shape.cols; i = i + 1u) {
          mean += input.tensor[tensor_idx(input.shape, batch, row, i)];
        }
        mean /= f32(input.shape.cols);

        var variance = 0.0;
        for (var i = 0u; i < input.shape.cols; i = i + 1u) {
          variance += pow(input.tensor[tensor_idx(input.shape, batch, row, i)] - mean, 2.0);
        }
        variance /= f32(input.shape.cols);

        result.tensor[tensor_idx(result.shape, batch, row, col)] = \
          ((value - mean) / sqrt(variance + 0.00001)) * scaleValue + biasValue;
      }
    `,
  });

  return output;
}
