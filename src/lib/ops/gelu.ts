import {WGT} from '../index';
import {WGSL_TENSOR_TYPE} from '../tensor';
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
            ${WGSL_TENSOR_TYPE}
      
            // Input
            @group(0) @binding(0) var<storage, read> input: Tensor;
            
            // Output
            @group(0) @binding(1) var<storage, read_write> result: Tensor;
           
            @compute @workgroup_size(64) fn main(
              @builtin(global_invocation_id) id: vec3<u32>,
            ) {
              result.rows = input.rows;
              result.cols = input.cols;
              
              let row = id.x;
              let col = id.y;
              let value = input.matrix[row * input.cols + col];
            
              result.matrix[row * result.cols + col] = \
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
        workgroups: [this.shape.rows, this.shape.cols],
      },
    ];
  }
}
