import {WGT} from '../index';
import {WGSL_TENSOR_TYPE} from '../tensor';
import {Op, OpCommand} from './op';

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
export class Softmax extends Op {
  pipeline: GPUComputePipeline;

  constructor(input: Op) {
    const shape = input.shape;
    super(shape, [input]);

    this.pipeline = WGT.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: WGT.device.createShaderModule({
          code: /* wgsl */ `
            ${WGSL_TENSOR_TYPE}
      
            // Input
            @group(0) @binding(0) var<storage, read> input: Tensor;
            
            // Output
            @group(0) @binding(1) var<storage, read_write> result: Tensor;
           
            @compute @workgroup_size(256) fn main(
              @builtin(global_invocation_id) id: vec3<u32>,
            ) {
              result.rows = input.rows;
              result.cols = input.cols;
              
              let row = id.x;
              let rowOffset = row * input.cols;
         
              // Get the max value in the row (across all columns)
               var rowMax: f32 = 0.0;
              for (var col = 0u; col < input.cols; col += 1u) {
                rowMax = max(rowMax, input.matrix[rowOffset + col]);
              }

              // Compute the sum of all unnormalized softmax values in the row
              var sum: f32 = 0.0;

              for (var col = 0u; col < input.cols; col += 1u) {
                var unnormalizedSoftmax: f32 = exp(
                  input.matrix[rowOffset + col] - rowMax
                );

                sum += unnormalizedSoftmax;
                result.matrix[rowOffset + col] = unnormalizedSoftmax;
              }

              // Normalize the softmax values
              for (var col = 0u; col < input.cols; col += 1u) {
                result.matrix[rowOffset + col] /= sum;
              }
            }
          `,
        }),
        entryPoint: 'main',
      },
    });
  }

  getCommands(): OpCommand[] {
    return [
      ...super.getCommands(),
      {
        pipeline: this.pipeline,
        params: [this.dependencies[0].buffer, this.buffer],
        workgroups: [Math.ceil(this.shape.rows / 256)],
      },
    ];
  }
}
