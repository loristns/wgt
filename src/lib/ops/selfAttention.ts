import {WGT} from '../index';
import {TensorShape, WGSL_TENSOR} from '../tensor';
import {Gemm} from './gemm';
import {Op, OpCommand} from './op';
import {Softmax} from './softmax';
import {Transpose} from './transpose';

class SelfAttentionMask extends Op {
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

              if (row < col) {
                result.matrix[tensor_idx(result.shape, batch, row, col)] = -0x1p+127f;
              } else {
                result.matrix[tensor_idx(result.shape, batch, row, col)] = \
                  input.matrix[tensor_idx(input.shape, batch, row, col)];
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
        workgroups: [
          Math.ceil(this.shape.rows / 16),
          Math.ceil(this.shape.cols / 16),
          this.shape.batches,
        ],
      },
    ];
  }
}

class SelfAttentionFlatten extends Op {
  pipeline: GPUComputePipeline;

  constructor(input: Op) {
    const shape = new TensorShape(
      1,
      input.shape.rows,
      input.shape.cols * input.shape.batches
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
              var result_shape = TensorShape();
              result_shape.batches = 1;
              result_shape.rows = input.shape.rows;
              result_shape.cols = input.shape.cols * input.shape.batches;

              if (id.x == 0 && id.y == 0 && id.z == 0) {
                result.shape = result_shape;
              }

              let batch = id.z;
              let row = id.x;
              let col = id.y;

              result.matrix[tensor_idx(result_shape, 0, row, input.shape.batches * batch + col)] = \
                input.matrix[tensor_idx(input.shape, batch, row, col)];
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

/**
 * A causal self-attention operation
 */
export class SelfAttention extends Op {
  /**
   * Constructs a causal self-attention operation.
   *
   * @param input [1, length, dim]
   * @param queryWeights [nHeads, dim, attDim]
   * @param queryBias [nHeads, 1, attDim]
   * @param keyWeights [nHeads, dim, attDim]
   * @param keyBias [nHeads, 1, attDim]
   * @param valueWeights [nHeads, dim, attDim]
   * @param valueBias [nHeads, 1, attDim]
   * @param projWeights [1, attDim, outDim]
   * @param projBias [1, 1, outDim]
   */
  constructor(
    input: Op,
    queryWeights: Op,
    queryBias: Op,
    keyWeights: Op,
    keyBias: Op,
    valueWeights: Op,
    valueBias: Op,
    projWeights: Op,
    projBias: Op
  ) {
    const shape = new TensorShape(1, input.shape.rows, projWeights.shape.cols);

    // [1, length, dim] -> [nHeads, length, attDim]
    const query = new Gemm(input, queryWeights, queryBias);
    const key = new Gemm(input, keyWeights, keyBias);
    const value = new Gemm(input, valueWeights, valueBias);

    // [nHeads, length, attDim] -> [nHeads, length, length]
    let attention = new Gemm(query, new Transpose(key));
    attention = new SelfAttentionMask(attention);
    attention = new Softmax(attention);

    // [nHeads, length, length] -> [nHeads, length, attDim]
    let output = new Gemm(attention, value);

    // [nHeads, length, attDim] -> [1, length, attDim * nHeads]
    output = new SelfAttentionFlatten(output);

    // [1, length, attDim * nHeads] -> [1, length, outDim]
    output = new Gemm(output, projWeights, projBias);

    super(shape, [output], output.buffer);
  }
}
