import {WGT} from '../index';
import {WGSL_TENSOR} from '../tensor';
import {Op, OpCommand} from './op';

/**
 * A layer normalization operation
 */
export class LayerNorm extends Op {
  pipeline: GPUComputePipeline;

  constructor(input: Op, scale: Op, offset: Op) {
    const shape = input.shape;
    super(shape, [input, scale, offset]);

    this.pipeline = WGT.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: WGT.device.createShaderModule({
          code: /* wgsl */ `
            ${WGSL_TENSOR}

            // Input
            @group(0) @binding(0) var<storage, read> input: Tensor;
            @group(0) @binding(1) var<storage, read> scale: Tensor;
            @group(0) @binding(2) var<storage, read> offset: Tensor;

            // Output
            @group(0) @binding(3) var<storage, read_write> result: Tensor;

            @compute @workgroup_size(16, 16, 1) fn main(
              @builtin(global_invocation_id) id: vec3<u32>,
            ) {
              result.shape = input.shape;

              let batch = id.z;
              let row = id.x;
              let col = id.y;

              let value = input.matrix[tensor_idx(input.shape, batch, row, col)];
              let scaleValue = scale.matrix[tensor_idx(scale.shape, batch, row, col)];
              let offsetValue = offset.matrix[tensor_idx(offset.shape, batch, row, col)];

              var mean = 0.0;
              for (var i = 0u; i < input.shape.cols; i = i + 1u) {
                mean += input.matrix[tensor_idx(input.shape, batch, row, i)];
              }
              mean /= f32(input.shape.cols);

              var variance = 0.0;
              for (var i = 0u; i < input.shape.cols; i = i + 1u) {
                variance += pow(input.matrix[tensor_idx(input.shape, batch, row, i)] - mean, 2.0);
              }
              variance /= f32(input.shape.cols);

              result.matrix[tensor_idx(result.shape, batch, row, col)] = \
                ((value - mean) / sqrt(variance + 0.00001)) * scaleValue + offsetValue;
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
        params: [
          this.dependencies[0].buffer,
          this.dependencies[1].buffer,
          this.dependencies[2].buffer,
          this.buffer,
        ],
        workgroups: [
          Math.ceil(this.shape.rows / 16),
          Math.ceil(this.shape.cols / 16),
          this.shape.batches,
        ],
      },
    ];
  }
}
