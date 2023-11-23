import {Shape, Tensor} from '../tensor';
import {DeviceTensor} from '../deviceTensor';
import {Op} from '../op';

/**
 * A transpose operation (swaps rows and columns).
 */
export function transpose(input: DeviceTensor): DeviceTensor {
  const shape = new Shape({
    rows: input.shape.cols,
    cols: input.shape.rows,
    batches: input.shape.batches,
  });

  const output = new DeviceTensor(shape);
  output.sourceOp = new Op({
    label: 'transpose',
    inputs: [input],
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

      // Output
      @group(0) @binding(1) var<storage, read_write> result: Tensor;

      @compute @workgroup_size(16, 16, 1) fn main(
        @builtin(global_invocation_id) id: vec3<u32>,
      ) {
        result.shape = Shape(input.shape.batches, input.shape.cols, input.shape.rows);

        let batch = id.z;
        let row = id.x;
        let col = id.y;

        result.tensor[tensor_idx(result.shape, batch, row, col)] = \
          input.tensor[tensor_idx(input.shape, batch, col, row)];
      }
    `,
  });

  return output;
}
