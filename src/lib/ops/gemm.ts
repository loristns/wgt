import {WGT} from '../index';
import {TensorShape, WGSL_TENSOR_TYPE} from '../tensor';
import {Op, OpCommand} from './op';

/**
 * A General Matrix Multiply (GEMM) operation : A @ B (+ C)
 */
export class Gemm extends Op {
  pipeline: GPUComputePipeline;

  constructor(a: Op, b: Op, c?: Op) {
    const shape = new TensorShape(a.shape.rows, b.shape.cols);

    if (c != null) {
      super(shape, [a, b, c]);
    } else {
      super(shape, [a, b]);
    }

    this.pipeline = WGT.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: WGT.device.createShaderModule({
          code: /* wgsl */ `
            ${WGSL_TENSOR_TYPE}
      
            // Inputs
            @group(0) @binding(0) var<storage, read> a: Tensor;
            @group(0) @binding(1) var<storage, read> b: Tensor;

            ${
              c != null
                ? '@group(0) @binding(2) var<storage, read> c: Tensor;'
                : ''
            }
            
            // Output
            @group(0) @binding(${
              c != null ? '3' : '2'
            }) var<storage, read_write> result: Tensor;
            
            @compute @workgroup_size(16, 16) fn main(
              @builtin(global_invocation_id) id: vec3<u32>
            ) {
              result.rows = a.rows;
              result.cols = b.cols;
              
              let row = id.x;
              let col = id.y;              
         
              var value: f32 = 0.0;
            
              for (var i = 0u; i < a.cols; i += 1u) {
                value += a.matrix[row * a.cols + i] * b.matrix[i * b.cols + col];
              }

              ${c != null ? 'value += c.matrix[row * c.cols + col];' : ''}
            
              result.matrix[row * result.cols + col] = value;
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
        params: [...this.dependencies.map(dep => dep.buffer), this.buffer],
        workgroups: [
          Math.ceil(this.shape.rows / 16),
          Math.ceil(this.shape.cols / 16),
        ],
      },
    ];
  }
}
