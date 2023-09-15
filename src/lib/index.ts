export class GPUDeviceSingleton {
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

export type TensorShape = {
  batches: number;
  rows: number;
  cols: number;
};

/**
 * 3D tensor type.
 */
export class Tensor {
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
export enum VariableMode {
  GPU,
  READABLE,
}

/**
 * A tiny wrapper around GPUBuffer.
 */
export class Variable {
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
export enum OpCommandType {
  EXECUTE_OP,
  COPY_VARIABLE,
}

export type OpCommand =
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

export interface Op {
  pipeline: GPUComputePipeline;

  createOutputVariables(mode?: VariableMode): Variable[];
  createCommand(...args: unknown[]): OpCommand;
}

export function runCommands(commands: OpCommand[]) {
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
