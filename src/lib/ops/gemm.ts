import {WGT} from '../index';
import {TensorShape, WGSL_TENSOR} from '../tensor';
import {Op, OpCommand} from './op';

/**
 * A General Matrix Multiply (GEMM) operation : A @ B (+ C)
 */
export class Gemm extends Op {
  pipeline: GPUComputePipeline;

  constructor(a: Op, b: Op, c?: Op) {
    const shape = new TensorShape(
      Math.max(a.shape.batches, b.shape.batches),
      a.shape.rows,
      b.shape.cols
    );

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
            ${WGSL_TENSOR}
      
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
            
            @compute @workgroup_size(16, 16, 1) fn main(
              @builtin(global_invocation_id) id: vec3<u32>
            ) {
              result.shape = TensorShape();
              if (a.shape.batches > b.shape.batches) {
                result.shape.batches = a.shape.batches;
              } else {
                result.shape.batches = b.shape.batches;
              }
              result.shape.rows = a.shape.rows;
              result.shape.cols = b.shape.cols;

              let batch = id.z;
              let row = id.x;
              let col = id.y;
         
              var value: f32 = 0.0;
            
              for (var i = 0u; i < a.shape.cols; i += 1u) {
                value += \
                  a.matrix[tensor_idx(a.shape, batch, row, i)] \
                * b.matrix[tensor_idx(b.shape, batch, i, col)];
              }

              ${
                c != null
                  ? 'value += c.matrix[tensor_idx(c.shape, batch, row, col)];'
                  : ''
              }
            
              result.matrix[tensor_idx(result.shape, batch, row, col)] = value;
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
          this.shape.batches,
        ],
      },
    ];
  }
}
