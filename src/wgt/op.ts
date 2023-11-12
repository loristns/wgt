import {Command} from './commands';
import {Shape} from './tensor';
import {TensorBuffer} from './tensorBuffer';

export abstract class Op {
  readonly buffer: TensorBuffer;
  readonly dependencies: TensorBuffer[];

  protected opCommands: Command[] = [];

  constructor(params: {shape: Shape; dependencies: TensorBuffer[]}) {
    const {shape, dependencies} = params;

    this.buffer = new TensorBuffer(shape, this);
    this.dependencies = dependencies;
  }

  get commands(): Command[] {
    return [
      // The commands of an operation are the commands of the operations it depends on...
      ...this.dependencies
        .map(dependency => dependency.parentOp)
        .filter((opDependency): opDependency is Op => opDependency != null)
        .flatMap(opDependency => opDependency.commands),

      // ...followed by its own commands.
      ...this.opCommands,
    ];
  }

  destroySubTree() {
    this.buffer.destroy();

    this.dependencies.forEach(dependency => {
      if (dependency.parentOp != null) {
        dependency.parentOp.destroySubTree();
      }
    });
  }
}
