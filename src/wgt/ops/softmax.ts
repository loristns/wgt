import {Tensor} from '../tensor';
import {DeviceTensor} from '../deviceTensor';
import {Op} from '../op';

/**
 * A softmax operation for each row of a tensor.
 *
 * The softmax function is defined as:
 * ```
 * softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 * ```
 * This implementation is probably suboptimal, as it does not parallelize the sum and max operations.
 * TODO: Implement a faster kernel.
 */
export function softmax(input: DeviceTensor): DeviceTensor {
  const shape = input.shape;

  const output = new DeviceTensor(shape);
  output.sourceOp = new Op({
    label: 'softmax',
    inputs: [input],
    outputs: [output],
    workgroups: [Math.ceil(shape.rows / 256), shape.batches],
    code: /* wgsl */ `
      ${Tensor.WGSL}
        
      // Input
      @group(0) @binding(0) var<storage, read> input: Tensor;
      
      // Output
      @group(0) @binding(1) var<storage, read_write> result: Tensor;
      
      @compute @workgroup_size(256, 1) fn main(
        @builtin(global_invocation_id) id: vec3<u32>,
      ) {
        result.shape = input.shape;
        
        let batch = id.y;
        let row = id.x;
    
        // Get the max value in the row (across all columns)
          var rowMax: f32 = 0.0;
        for (var col = 0u; col < input.shape.cols; col += 1u) {
          rowMax = max(rowMax, input.tensor[tensor_idx(input.shape, batch, row, col)]);
        }

        // Compute the sum of all unnormalized softmax values in the row
        var sum: f32 = 0.0;

        for (var col = 0u; col < input.shape.cols; col += 1u) {
          var unnormalizedSoftmax: f32 = exp(
            input.tensor[tensor_idx(input.shape, batch, row, col)] - rowMax
          );

          sum += unnormalizedSoftmax;
          result.tensor[tensor_idx(result.shape, batch, row, col)] = unnormalizedSoftmax;
        }

        // Normalize the softmax values
        for (var col = 0u; col < input.shape.cols; col += 1u) {
          result.tensor[tensor_idx(result.shape, batch, row, col)] /= sum;
        }
      }
    `,
  });

  return output;
}
