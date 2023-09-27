import {WGT} from '../index';
import {WGSL_TENSOR} from '../tensor';
import {Op, OpCommand} from './op';

/**
 * A GEneralized Linear Unit (GELU) operation over each element of a tensor.
 */
export class Gelu extends Op {
  pipeline: GPUComputePipeline;

  constructor(input: Op) {
    const shape = input.shape;
    super(shape, [input]);

    this.pipeline = WGT.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: WGT.device.createShaderModule({
          code: /* wgsl */ `
            ${WGSL_TENSOR}
      
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

              let value = input.matrix[tensor_idx(input.shape, batch, row, col)];
            
              result.matrix[tensor_idx(result.shape, batch, row, col)] = \
                0.5 * value * (1 + tanh(0.797884 * (value + 0.044715 * pow(value, 3.0))));
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
        workgroups: [
          Math.ceil(this.shape.rows / 16),
          Math.ceil(this.shape.cols / 16),
          this.shape.batches, // We put batches last because its value is likely to be small (and the z dimension is generally smaller)
        ],
      },
    ];
  }
}
