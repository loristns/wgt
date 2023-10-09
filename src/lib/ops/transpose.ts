import {WGT} from '../index';
import {TensorShape, WGSL_TENSOR} from '../tensor';
import {Op, OpCommand} from './op';

/**
 * A matrix transpose operation (invert rows and columns)
 */
export class Transpose extends Op {
  pipeline: GPUComputePipeline;

  constructor(input: Op) {
    const shape = new TensorShape(
      input.shape.batches,
      input.shape.cols,
      input.shape.rows
    );
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
              result.shape = TensorShape(input.shape.batches, input.shape.cols, input.shape.rows);

              let batch = id.z;
              let row = id.x;
              let col = id.y;

              result.matrix[tensor_idx(result.shape, batch, row, col)] = \
                input.matrix[tensor_idx(input.shape, batch, col, row)];
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
          Math.ceil(this.dependencies[0].shape.rows / 16),
          Math.ceil(this.dependencies[0].shape.cols / 16),
          this.dependencies[0].shape.batches,
        ],
      },
    ];
  }
}
