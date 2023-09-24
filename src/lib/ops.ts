import {WGT} from './index';
import {Tensor, TensorShape, WGSL_TENSOR_TYPE} from './tensor';

/**
 * A simplified description of a compute pass on the GPU.
 */
export type OpCommand = {
  pipeline: GPUComputePipeline;
  params: GPUBuffer[];
  workgroups: [number, number?, number?];
};

/**
 * Base class for all operations.
 *
 * All operations have an associated output shape, buffer (that stores the output tensor) and dependencies.
 *
 * The output buffer is created when the operation is created, so it size must be known at
 * construction time.
 *
 * Dependencies are operations that must be executed before this operation, and are used to
 * build the command array.
 */
export abstract class Op {
  readonly dependencies: Op[];
  readonly shape: TensorShape;

  readonly buffer: GPUBuffer;
  private _readableBuffer?: GPUBuffer;

  /**
   * Returns a buffer that can be read from the CPU.
   */
  get readableBuffer() {
    if (this._readableBuffer == null) {
      this._readableBuffer = WGT.device.createBuffer({
        size: this.shape.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }

    return this._readableBuffer;
  }

  constructor(shape: TensorShape, dependencies: Op[]) {
    this.shape = shape;
    this.dependencies = dependencies;

    this.buffer = WGT.device.createBuffer({
      size: this.shape.size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
  }

  getCommands(): OpCommand[] {
    // By default, an operation has no commands.
    // It simply returns the commands of its dependencies.
    return [...this.dependencies.flatMap(d => d.getCommands())];
  }

  clean() {
    this.buffer.destroy();
    this._readableBuffer?.destroy();
  }
}

/**
 * The input operation takes a tensor and writes it to the GPU.
 */
export class Input extends Op {
  constructor(shape: TensorShape);
  constructor(rows: number, cols: number);
  constructor(shapeOrRows: TensorShape | number, cols?: number) {
    const shape =
      typeof shapeOrRows === 'number'
        ? new TensorShape(shapeOrRows, cols!)
        : shapeOrRows;

    // The input operation has no dependencies so we pass an empty array.
    super(shape, []);
  }

  write(tensor: Tensor) {
    WGT.device.queue.writeBuffer(this.buffer, 0, tensor.arrayBuffer);
  }
}

/**
 * A matrix multiplication operation between two tensors.
 */
export class Matmul extends Op {
  pipeline: GPUComputePipeline;

  constructor(a: Op, b: Op) {
    const shape = new TensorShape(a.shape.rows, b.shape.cols);
    super(shape, [a, b]);

    this.pipeline = WGT.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: WGT.device.createShaderModule({
          code: /* wgsl */ `
            ${WGSL_TENSOR_TYPE}
      
            // Inputs
            @group(0) @binding(0) var<storage, read> a: Tensor;
            @group(0) @binding(1) var<storage, read> b: Tensor;
            
            // Output
            @group(0) @binding(2) var<storage, read_write> result: Tensor;
            
            @compute @workgroup_size(64) fn main(
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
        params: [
          this.dependencies[0].buffer,
          this.dependencies[1].buffer,
          this.buffer,
        ],
        workgroups: [this.shape.rows, this.shape.cols],
      },
    ];
  }
}

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
           
            @compute @workgroup_size(64) fn main(
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
        workgroups: [this.shape.rows, 1, 1],
      },
    ];
  }
}
