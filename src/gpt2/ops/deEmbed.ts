import {Shape, Tensor} from '../../wgt/tensor';
import {DeviceTensor} from '../../wgt/deviceTensor';
import {Op} from '../../wgt/op';

import {parameters} from '../../wgt/ops/parameters';
import {gemm} from '../../wgt/ops/gemm';

import {EmbedParameters} from './embed';
import {transpose} from '../../wgt/ops/transpose';

export function deEmbed(
  input: DeviceTensor,
  params: EmbedParameters
): DeviceTensor {
  const {chunk1, chunk2} = parameters(params);

  // chunks are [1, N_VOCAB // 2, EMBED_SIZE]
  // input is [1, LEN, EMBED_SIZE]
  // output is [1, LEN, N_VOCAB]
  const chunk1Scores = gemm(input, transpose(chunk1));
  const chunk2Scores = gemm(input, transpose(chunk2));

  // concat cols
  const shape = new Shape({
    batches: chunk1Scores.shape.batches,
    rows: chunk1Scores.shape.rows,
    cols: chunk1Scores.shape.cols + chunk2Scores.shape.cols,
  });

  const output = new DeviceTensor(shape);
  output.sourceOp = new Op({
    label: 'deEmbed',
    inputs: [chunk1Scores, chunk2Scores],
    outputs: [output],
    workgroups: [
      Math.ceil(shape.rows / 16),
      Math.ceil(shape.cols / 16),
      shape.batches,
    ],
    code: /* wgsl */ `
      ${Tensor.WGSL}

      // Input
      @group(0) @binding(0) var<storage, read> chunk1: Tensor;
      @group(0) @binding(1) var<storage, read> chunk2: Tensor;

      // Output
      @group(0) @binding(2) var<storage, read_write> result: Tensor;

      @compute @workgroup_size(16, 16, 1) fn main(
        @builtin(global_invocation_id) id: vec3<u32>,
      ) {
        let batch = id.z;
        let row = id.x;
        let col = id.y;

        let result_shape = Shape(chunk1.shape.batches, chunk1.shape.rows, chunk1.shape.cols + chunk2.shape.cols);

        if (batch == 0 && row == 0 && col == 0) {
          result.shape = result_shape;
        }

        if (col < chunk1.shape.cols) {
          result.tensor[tensor_idx(result_shape, batch, row, col)] = \
            chunk1.tensor[tensor_idx(chunk1.shape, batch, row, col)];
        } else {
          result.tensor[tensor_idx(result_shape, batch, row, col)] = \
            chunk2.tensor[tensor_idx(chunk2.shape, batch, row, col - chunk1.shape.cols)];
        }
      }
    `,
  });

  return output;
}
