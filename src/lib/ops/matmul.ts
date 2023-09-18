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

export class MatmulOp implements Op {
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

  pipeline: Record<string, GPUComputePipeline>;

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

    this.pipeline = {
      main: device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: device.createShaderModule({
            code: MatmulOp.WGSL_CODE,
          }),
          entryPoint: 'main',
        },
      }),
    };

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

  getCommands(
    input1: Variable,
    input2: Variable,
    output: Variable
  ): OpCommand[] {
    return [
      {
        type: OpCommandType.EXECUTE_OP,
        op: this,
        pipeline: 'main',
        variables: [input1, input2, output],
        workgroups: [this.aShape.batches, this.aShape.rows, this.bShape.cols],
      },
    ];
  }
}
