import {ComputeCommand} from './commands';
import {DeviceTensor} from './deviceTensor';

/**
 * An Op is a ComputeCommand that takes DeviceTensor as inputs and outputs.
 *
 * It handles references to the inputs and outputs DeviceTensor to build the
 * computation graph.
 */
export class Op extends ComputeCommand {
  inputs: DeviceTensor[];
  outputs: DeviceTensor[];

  constructor(params: {
    inputs: DeviceTensor[];
    outputs: DeviceTensor[];
    workgroups: [number, number?, number?];
    code: string;
    entryPoint?: string;
  }) {
    const {inputs, outputs, workgroups, code, entryPoint = 'main'} = params;
    super({
      buffers: [...inputs, ...outputs].map(arg => arg.buffer),
      workgroups,
      code,
      entryPoint,
    });

    this.inputs = inputs;
    this.outputs = outputs;
  }
}
