import {Command} from '../commands';
import {Shape, Tensor} from '../tensor';
import {TensorBuffer} from '../tensorBuffer';

type _OpParameters<Self> = Record<string, Self | Tensor>;
type OpRootParameters = _OpParameters<never>;
type OpRootParameterBuffers = Record<keyof OpRootParameters, TensorBuffer>;
export interface OpParameters extends _OpParameters<OpParameters> {}

export class Op {
  readonly buffer: TensorBuffer;

  readonly dependencies: Op[];
  readonly parameterBuffers: OpRootParameterBuffers;

  constructor(params: {
    shape: Shape;
    dependencies: (Op | TensorBuffer)[];
    parameters?: OpRootParameters;
  }) {
    const {shape, dependencies, parameters} = params;

    this.buffer = new TensorBuffer(shape, this);
    this.dependencies = dependencies
      .map(dependency =>
        dependency instanceof Op ? dependency : dependency.parentOp
      )
      .filter((dependency): dependency is Op => dependency != null);

    // Create the parameter buffers from the parameters.
    this.parameterBuffers = {} as OpRootParameterBuffers;
    Object.entries(parameters ?? {}).forEach(([name, param]) => {
      this.parameterBuffers[name] = TensorBuffer.fromTensor(param);
    });
  }

  getCommands(): Command[] {
    // By default, an operation has no commands.
    // It simply returns the commands of its dependencies.
    return [
      ...this.dependencies.flatMap(dependency => dependency.getCommands()),
    ];
  }

  destroy() {
    this.buffer.destroy();

    Object.values(this.parameterBuffers).forEach(parameterBuffer => {
      parameterBuffer.destroy();
    });
  }
}
