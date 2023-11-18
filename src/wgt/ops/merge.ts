import {Tensor} from '../tensor';
import {DeviceTensor} from '../deviceTensor';
import {Op} from '../op';

export enum MergeMethod {
  Add,
  Sub,
  Mul,
  Div,
  Min,
  Max,
}

/**
 * A merge operation between two tensors.
 */
export function merge(
  a: DeviceTensor,
  b: DeviceTensor,
  method: MergeMethod
): DeviceTensor {
  const shape = a.shape;

  const output = new DeviceTensor(shape);
  output.sourceOp = new Op({
    inputs: [a, b],
    outputs: [output],
    workgroups: [
      Math.ceil(shape.rows / 16),
      Math.ceil(shape.cols / 16),
      shape.batches,
    ],
    code: /* wgsl */ `
      ${Tensor.WGSL}
        
      // Input
      @group(0) @binding(0) var<storage, read> a: Tensor;
      @group(0) @binding(1) var<storage, read> b: Tensor;

      // Output
      @group(0) @binding(2) var<storage, read_write> result: Tensor;

      @compute @workgroup_size(16, 16, 1) fn main(
        @builtin(global_invocation_id) id: vec3<u32>,
      ) {
        result.shape = a.shape;

        let batch = id.z;
        let row = id.x;
        let col = id.y;

        let a_value = a.tensor[tensor_idx(a.shape, batch, row, col)];
        let b_value = b.tensor[tensor_idx(b.shape, batch, row, col)];

        ${
          method === MergeMethod.Add
            ? 'result.tensor[tensor_idx(result.shape, batch, row, col)] = a_value + b_value;'
            : method === MergeMethod.Sub
            ? 'result.tensor[tensor_idx(result.shape, batch, row, col)] = a_value - b_value;'
            : method === MergeMethod.Mul
            ? 'result.tensor[tensor_idx(result.shape, batch, row, col)] = a_value * b_value;'
            : method === MergeMethod.Div
            ? 'result.tensor[tensor_idx(result.shape, batch, row, col)] = a_value / b_value;'
            : method === MergeMethod.Min
            ? 'result.tensor[tensor_idx(result.shape, batch, row, col)] = min(a_value, b_value);'
            : method === MergeMethod.Max
            ? 'result.tensor[tensor_idx(result.shape, batch, row, col)] = max(a_value, b_value);'
            : ''
        }
      }
    `,
  });

  return output;
}
