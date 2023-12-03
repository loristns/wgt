import {Shape, Tensor} from '../tensor';
import {DeviceTensor} from '../deviceTensor';
import {Op} from '../op';
import {parameters, Parameters} from './parameters';

import {linear, LinearParameters} from './linear';
import {softmax} from './softmax';
import {transpose} from './transpose';
import {gemm} from './gemm';
import {merge, MergeMethod} from './merge';

export interface SelfAttentionParameters extends Parameters {
  query: LinearParameters;
  key: LinearParameters;
  value: LinearParameters;
  projection: LinearParameters;
}

/**
 * A self-attention operation (with masking as in a Transformer decoder)
 */
export function selfAttention(
  input: DeviceTensor,
  params: SelfAttentionParameters
): DeviceTensor {
  // 1. Linear projection of input to query, key, and value.
  const query = linear(input, params.query);
  const key = linear(input, params.key);
  const value = linear(input, params.value);

  // 2. Compute attention matrix.
  const keyT = transpose(key);
  const attention = gemm(query, keyT);

  // Scale attention matrix
  const {scale} = parameters({
    scale: Tensor.fromArray([Math.sqrt(query.shape.cols)]),
  });
  const attentionScaled = merge(attention, scale, MergeMethod.Div);

  // Mask out the upper triangle of the attention matrix (decoder)
  const attentionMasked = new DeviceTensor(attentionScaled.shape);
  attentionMasked.sourceOp = new Op({
    label: 'attentionMask',
    inputs: [attentionScaled],
    outputs: [attentionMasked],
    workgroups: [
      Math.ceil(attentionScaled.shape.rows / 16),
      Math.ceil(attentionScaled.shape.cols / 16),
      attentionScaled.shape.batches,
    ],
    code: /* wgsl */ `
      ${Tensor.WGSL}

      // Input
      @group(0) @binding(0) var<storage, read> input: Tensor;

      // Output
      @group(0) @binding(1) var<storage, read_write> result: Tensor;
  
      @compute @workgroup_size(16, 16, 1) fn main(
        @builtin(global_invocation_id) id: vec3<u32>
      ) {
        result.shape = input.shape;

        let batch = id.z;
        let row = id.x;
        let col = id.y;

        var value = input.tensor[tensor_idx(input.shape, batch, row, col)];

        // Use min to fix (strange issue?)
        // TODO: Investigate further
        if (min(row, result.shape.rows - 1u) < min(col, result.shape.cols - 1u)) {
          value = -0x1p+127f;
        }

        result.tensor[tensor_idx(result.shape, batch, row, col)] = value;
      }
    `,
  });

  // 3. Apply softmax to attention matrix to get attention weights per token
  // And project the value matrix.
  const attentionValue = gemm(softmax(attentionMasked), value);

  // 4. Flatten
  const attentionValueFlattenedShape = new Shape({
    rows: attentionValue.shape.rows,
    cols: attentionValue.shape.cols * attentionValue.shape.batches,
    batches: 1,
  });

  const attentionValueFlattened = new DeviceTensor(
    attentionValueFlattenedShape
  );
  attentionValueFlattened.sourceOp = new Op({
    label: 'attentionFlatten',
    inputs: [attentionValue],
    outputs: [attentionValueFlattened],
    workgroups: [
      Math.ceil(attentionValueFlattenedShape.rows / 16),
      Math.ceil(attentionValueFlattenedShape.cols / 16),
      attentionValueFlattenedShape.batches,
    ],
    code: /* wgsl */ `
      ${Tensor.WGSL}

      // Input
      @group(0) @binding(0) var<storage, read> input: Tensor;

      // Output
      @group(0) @binding(1) var<storage, read_write> result: Tensor;

      @compute @workgroup_size(16, 16, 1) fn main(
        @builtin(global_invocation_id) id: vec3<u32>,
      ) {
        let batch = id.z;
        let row = id.x;
        let col = id.y;

        let result_shape = Shape(
          1,
          input.shape.rows,
          input.shape.cols * input.shape.batches
        );

        if (batch == 0u && row == 0u && col == 0u) {
          result.shape = result_shape;
        }

        let input_batch = u32(col / input.shape.cols);
        let input_col = col % input.shape.cols;
        let input_row = row;

        result.tensor[tensor_idx(result_shape, batch, row, col)] = \
          input.tensor[tensor_idx(input.shape, input_batch, input_row, input_col)];
      }
      `,
  });

  // 5. Final linear projection
  const output = linear(attentionValueFlattened, params.projection);

  return output;
}
