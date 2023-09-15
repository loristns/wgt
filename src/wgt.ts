class GPUDeviceSingleton {
  private static device: GPUDevice;
  private static limits: GPUSupportedLimits;

  private constructor() {}

  static async initialize() {
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();

    if (!device) {
      throw new Error('WebGPU not supported');
    }

    GPUDeviceSingleton.device = device;
    GPUDeviceSingleton.limits = device.limits;
  }

  static getDevice() {
    if (!GPUDeviceSingleton.device) {
      throw new Error('GPU Device not initialized');
    }

    return GPUDeviceSingleton.device;
  }

  static getLimits() {
    if (!GPUDeviceSingleton.limits) {
      throw new Error('GPU Device not initialized');
    }

    return GPUDeviceSingleton.limits;
  }
}

type TensorShape = {
  batches: number;
  rows: number;
  cols: number;
};

/**
 * 3D tensor type.
 */
class Tensor {
  static WGSL_TYPE = /* wgsl */ `
    struct Tensor {
      batches: u32,
      rows: u32,
      cols: u32,
      matrix: array<f32>,
    }
  `;

  arrayBuffer: ArrayBuffer;

  rawData: Uint32Array;
  rawMatrix: Float32Array;
  rawShape: Uint32Array;

  constructor(arrayBuffer: ArrayBuffer) {
    this.arrayBuffer = arrayBuffer;

    this.rawData = new Uint32Array(this.arrayBuffer);
    this.rawMatrix = new Float32Array(this.arrayBuffer, 3 * 4);
    this.rawShape = new Uint32Array(this.arrayBuffer, 0, 3);
  }

  get shape(): TensorShape {
    return {
      batches: this.rawShape[0],
      rows: this.rawShape[1],
      cols: this.rawShape[2],
    };
  }

  get data(): number[][][] {
    const {batches, rows, cols} = this.shape;

    const data: number[][][] = [];

    for (let i = 0; i < batches; i++) {
      data.push([]);

      for (let j = 0; j < rows; j++) {
        data[i].push([]);

        for (let k = 0; k < cols; k++) {
          data[i][j].push(this.rawMatrix[i * rows * cols + j * cols + k]);
        }
      }
    }

    return data;
  }

  static fromArray(array: number[][][]): Tensor {
    const [batches, rows, cols] = [
      array.length,
      array[0].length,
      array[0][0].length,
    ];

    const arrayBuffer = new ArrayBuffer(batches * rows * cols * 4 + 3 * 4);
    const rawData = new Uint32Array(arrayBuffer);
    const rawMatrix = new Float32Array(arrayBuffer, 3 * 4);

    rawData.set([batches, rows, cols], 0);
    rawMatrix.set(array.flat(2), 0);

    return new Tensor(rawData);
  }
}

/**
 * A simplified version of GPUBufferUsage.
 */
enum VariableMode {
  GPU,
  READABLE,
}

/**
 * A tiny wrapper around GPUBuffer.
 */
class Variable {
  buffer: GPUBuffer;

  constructor(byteLength: number, mode: VariableMode = VariableMode.GPU) {
    this.buffer = GPUDeviceSingleton.getDevice().createBuffer({
      size: byteLength,
      usage:
        mode === VariableMode.GPU
          ? GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST
          : GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  write(data: ArrayBuffer) {
    GPUDeviceSingleton.getDevice().queue.writeBuffer(this.buffer, 0, data);
  }

  async read() {
    await this.buffer.mapAsync(GPUMapMode.READ);
    return this.buffer.getMappedRange();
  }

  dispose() {
    this.buffer.destroy();
  }
}

/**
 * Type of the command to be executed on the GPU.
 */
enum OpCommandType {
  EXECUTE_OP,
  COPY_VARIABLE,
}

type OpCommand =
  | {
      type: OpCommandType.EXECUTE_OP;
      op: Op;
      variables: Variable[];
      workgroups: [number, number?, number?];
    }
  | {
      type: OpCommandType.COPY_VARIABLE;
      src: Variable;
      dst: Variable;
    };

interface Op {
  pipeline: GPUComputePipeline;

  createOutputVariables(mode?: VariableMode): Variable[];
  createCommand(...args: unknown[]): OpCommand;
}

class MatmulOp implements Op {
  static WGSL_CODE = /* wgsl */ `
    ${Tensor.WGSL_TYPE}

    // Inputs
    @group(0) @binding(0) var<storage, read> a: Tensor;
    @group(0) @binding(1) var<storage, read> b: Tensor;
    
    // Output
    @group(0) @binding(2) var<storage, read_write> result: Tensor;
    
    @compute @workgroup_size(64) fn main(
      @builtin(global_invocation_id) id: vec3<u32>
    ) {
      result.batches = a.batches;
      result.rows = a.rows;
      result.cols = b.cols;

      let batch = id.x;
      let row = id.y;
      let col = id.z;
   
      var value: f32 = 0.0;
    
      for (var i = 0u; i < a.cols; i = i + 1u) {
        value = value \
          + a.matrix[batch * a.rows * a.cols + row * a.cols + i] \
          * b.matrix[batch * b.rows * b.cols + i * b.cols + col];
      }
    
      result.matrix[batch * result.rows * result.cols + row * result.cols + col] = value;
    }
  `;

  pipeline: GPUComputePipeline;

  aShape: TensorShape;
  bShape: TensorShape;

  get outputShape(): TensorShape {
    return {
      batches: this.aShape.batches,
      rows: this.aShape.rows,
      cols: this.bShape.cols,
    };
  }

  constructor(aShape: TensorShape, bShape: TensorShape) {
    const device = GPUDeviceSingleton.getDevice();

    const module = device.createShaderModule({
      code: MatmulOp.WGSL_CODE,
    });

    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'main',
      },
    });

    this.aShape = aShape;
    this.bShape = bShape;
  }

  createOutputVariables(mode: VariableMode = VariableMode.GPU): Variable[] {
    return [
      new Variable(
        3 * 4 + this.aShape.batches * this.aShape.rows * this.bShape.cols * 4,
        mode
      ),
    ];
  }

  createCommand(
    input1: Variable,
    input2: Variable,
    output: Variable
  ): OpCommand {
    return {
      type: OpCommandType.EXECUTE_OP,
      op: this,
      variables: [input1, input2, output],
      workgroups: [this.aShape.batches, this.aShape.rows, this.bShape.cols],
    };
  }
}

function runCommands(commands: OpCommand[]) {
  const device = GPUDeviceSingleton.getDevice();
  const encoder = device.createCommandEncoder();

  for (const command of commands) {
    switch (command.type) {
      case OpCommandType.EXECUTE_OP: {
        const {op, variables, workgroups} = command;

        const bindGroup = device.createBindGroup({
          layout: op.pipeline.getBindGroupLayout(0),
          entries: variables.map((variable, i) => ({
            binding: i,
            resource: {
              buffer: variable.buffer,
            },
          })),
        });

        const pass = encoder.beginComputePass();
        pass.setPipeline(op.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(...workgroups);
        pass.end();

        break;
      }

      case OpCommandType.COPY_VARIABLE: {
        const {src, dst} = command;

        encoder.copyBufferToBuffer(
          src.buffer,
          0,
          dst.buffer,
          0,
          src.buffer.size
        );

        break;
      }
    }
  }

  const commandsBuffer = encoder.finish();

  device.queue.submit([commandsBuffer]);
}

export async function main() {
  await GPUDeviceSingleton.initialize();

  // const a = Tensor.fromArray([
  //   [
  //     [1, 2, 3],
  //     [4, 5, 6],
  //   ],
  //   [
  //     [1, 2, 3],
  //     [4, 5, 6],
  //   ],
  // ]);

  // const b = Tensor.fromArray([
  //   [
  //     [7, 8],
  //     [9, 10],
  //     [11, 12],
  //   ],
  //   [
  //     [7, 8],
  //     [9, 10],
  //     [11, 12],
  //   ],
  // ]);

  const a = Tensor.fromArray([Array(1024).fill(Array(1024).fill(1))]);

  console.log(a.data);

  const b = Tensor.fromArray([Array(1024).fill(Array(1024).fill(1))]);

  const matmul = new MatmulOp(a.shape, b.shape);

  const aVariable = new Variable(a.arrayBuffer.byteLength);
  aVariable.write(a.arrayBuffer);

  const bVariable = new Variable(b.arrayBuffer.byteLength);
  bVariable.write(b.arrayBuffer);

  const [resultVariable] = matmul.createOutputVariables();

  const matmul2 = new MatmulOp(matmul.outputShape, matmul.outputShape);
  const [resultVariable2] = matmul2.createOutputVariables();

  const [readResultVariable] = matmul2.createOutputVariables(
    VariableMode.READABLE
  );

  console.time('matmul_2d');

  runCommands([
    matmul.createCommand(aVariable, bVariable, resultVariable),
    matmul2.createCommand(resultVariable, resultVariable, resultVariable2),
    {
      type: OpCommandType.COPY_VARIABLE,
      src: resultVariable2,
      dst: readResultVariable,
    },
  ]);

  const result = new Tensor(await readResultVariable.read());

  console.log(result.data);
  console.log(result.shape);
  console.log(result.rawData);

  aVariable.dispose();
  bVariable.dispose();
  resultVariable.dispose();
  readResultVariable.dispose();

  console.timeEnd('matmul_2d');
}
