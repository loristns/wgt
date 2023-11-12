import {ComputeCommand} from '../commands';
import {Shape, Tensor} from '../tensor';
import {TensorBuffer} from '../tensorBuffer';
import {Op} from '../op';

/**
 * A General Matrix Multiply (GEMM) operation : A @ B (+ C)
 */
export function gemm(
  a: TensorBuffer,
  b: TensorBuffer,
  c?: TensorBuffer
): TensorBuffer {
  const op = new Gemm(a, b, c);
  return op.buffer;
}

class Gemm extends Op {
  constructor(a: TensorBuffer, b: TensorBuffer, c?: TensorBuffer) {
    const shape = new Shape({
      batches: Math.max(a.shape.batches, b.shape.batches),
      rows: a.shape.rows,
      cols: b.shape.cols,
    });

    if (c != null) {
      super({shape, dependencies: [a, b, c]});
    } else {
      super({shape, dependencies: [a, b]});
    }

    this.opCommands = [
      new ComputeCommand({
        args: [a, b, ...(c != null ? [c] : []), this.buffer],
        workgroups: [
          Math.ceil(shape.rows / 16),
          Math.ceil(shape.cols / 16),
          shape.batches,
        ],
        code: /* wgsl */ `
          ${Tensor.WGSL}
    
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
            let batch = id.z;
            let row = id.x;
            let col = id.y;

            let shape = Shape(
              max(a.shape.batches, b.shape.batches),
              a.shape.rows,
              b.shape.cols
            );
        
            // Set the shape of the output tensor (only once)
            if (id.x == 0u && id.y == 0u && id.z == 0u) {
              result.shape = shape;
            }

            var value: f32 = 0.0;
          
            for (var i = 0u; i < a.shape.cols; i += 1u) {
              value += \
                a.tensor[tensor_idx(a.shape, batch, row, i)] \
              * b.tensor[tensor_idx(b.shape, batch, i, col)];
            }

            ${
              c != null
                ? 'value += c.tensor[tensor_idx(c.shape, batch, row, col)];'
                : ''
            }
          
            result.tensor[tensor_idx(shape, batch, row, col)] = value;
          }
      `,
      }),
    ];
  }
}
