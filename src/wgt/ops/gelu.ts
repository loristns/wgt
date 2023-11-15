import {Tensor} from '../tensor';
import {DeviceTensor} from '../deviceTensor';
import {Op} from '../op';

/**
 * A GEneralized Linear Unit (GELU) activation over the input.
 */
export function gelu(x: DeviceTensor): DeviceTensor {
  const output = new DeviceTensor(x.shape);
  output.sourceOp = new Op({
    inputs: [x],
    outputs: [output],
    workgroups: [
      Math.ceil(x.shape.rows / 16),
      Math.ceil(x.shape.cols / 16),
      x.shape.batches,
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
        result.shape = input.shape;
        
        let batch = id.z;
        let row = id.x;
        let col = id.y;

        let value = input.tensor[tensor_idx(input.shape, batch, row, col)];
      
        result.tensor[tensor_idx(result.shape, batch, row, col)] = \
          0.5 * value * (1 + tanh(0.797884 * (value + 0.044715 * pow(value, 3.0))));
      }
    `,
  });

  return output;
}
