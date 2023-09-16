import {
  GPUDeviceSingleton,
  Op,
  OpCommand,
  OpCommandType,
  Tensor,
  TensorShape,
  Variable,
  VariableMode,
} from '../index';

export class AddOp implements Op {
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
      result.cols = a.cols;

      let batch = id.x;
      let row = id.y;
      let col = id.z;
   
      result.matrix[batch * result.rows * result.cols + row * result.cols + col] = \
          a.matrix[batch * a.rows * a.cols + row * a.cols + col] \
        + b.matrix[batch * b.rows * b.cols + row * b.cols + col];
    }
  `;

  pipeline: GPUComputePipeline;

  shape: TensorShape;

  get outputShape(): TensorShape {
    return this.shape;
  }

  constructor(shape: TensorShape) {
    const device = GPUDeviceSingleton.getDevice();

    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: AddOp.WGSL_CODE,
        }),
        entryPoint: 'main',
      },
    });

    this.shape = shape;
  }

  createOutputVariables(mode: VariableMode = VariableMode.GPU): Variable[] {
    return [
      new Variable(
        3 * 4 + this.shape.batches * this.shape.rows * this.shape.cols * 4,
        mode
      ),
    ];
  }

  getCommands(
    input1: Variable,
    input2: Variable,
    output: Variable
  ): OpCommand[] {
    return [
      {
        type: OpCommandType.EXECUTE_OP,
        op: this,
        variables: [input1, input2, output],
        workgroups: [this.shape.batches, this.shape.rows, this.shape.cols],
      },
    ];
  }
}
