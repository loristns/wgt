import {Shape, Tensor} from '../../wgt/tensor';
import {DeviceTensor} from '../../wgt/deviceTensor';
import {Op} from '../../wgt/op';
import {parameters, RootParameters} from '../../wgt/ops/parameters';

export interface EmbedParameters extends RootParameters {
  chunk1: Tensor;
  chunk2: Tensor;
}

export function embed(
  input: DeviceTensor,
  params: EmbedParameters
): DeviceTensor {
  const {chunk1, chunk2} = parameters(params);

  const shape = new Shape({
    batches: input.shape.batches,
    rows: input.shape.cols,
    cols: chunk1.shape.cols,
  });

  const output = new DeviceTensor(shape);
  output.sourceOp = new Op({
    label: 'embed',
    inputs: [input, chunk1, chunk2],
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
      @group(0) @binding(1) var<storage, read> chunk1: Tensor;
      @group(0) @binding(2) var<storage, read> chunk2: Tensor;

      // Output
      @group(0) @binding(3) var<storage, read_write> result: Tensor;

      @compute @workgroup_size(16, 16, 1) fn main(
        @builtin(global_invocation_id) id: vec3<u32>,
      ) {
        let batch = id.z;
        let row = id.x;
        let col = id.y;

        let result_shape = Shape(input.shape.batches, input.shape.cols, chunk1.shape.cols);

        if (batch == 0 && row == 0 && col == 0) {
          result.shape = result_shape;
        }

        let key = u32(input.tensor[tensor_idx(input.shape, batch, 0, row)]);

        if (key < chunk1.shape.rows) {
          result.tensor[tensor_idx(result_shape, batch, row, col)] = \
            chunk1.tensor[tensor_idx(chunk1.shape, 0, key, col)];
        } else {
          result.tensor[tensor_idx(result_shape, batch, row, col)] = \
            chunk2.tensor[tensor_idx(chunk2.shape, 0, key - chunk1.shape.rows, col)];
        }
      }
    `,
  });

  return output;
}
