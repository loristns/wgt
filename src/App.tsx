import {
  GPUDeviceSingleton,
  OpCommandType,
  runCommands,
  Tensor,
  Variable,
  VariableMode,
} from './lib';
import {MatmulOp} from './lib/ops/matmul';
import {AddOp} from './lib/ops/add';

async function run() {
  await GPUDeviceSingleton.initialize();

  const a = Tensor.fromArray([
    [
      [1, 2, 3],
      [4, 5, 6],
    ],
    [
      [1, 2, 3],
      [4, 5, 6],
    ],
  ]);

  const b = Tensor.fromArray([
    [
      [7, 8],
      [9, 10],
      [11, 12],
    ],
    [
      [7, 8],
      [9, 10],
      [11, 12],
    ],
  ]);

  const matmul = new MatmulOp(a.shape, b.shape);

  const aVariable = new Variable(a.arrayBuffer.byteLength);
  aVariable.write(a.arrayBuffer);

  const bVariable = new Variable(b.arrayBuffer.byteLength);
  bVariable.write(b.arrayBuffer);

  const [resultVariable] = matmul.createOutputVariables();

  const matmul2 = new MatmulOp(matmul.outputShape, matmul.outputShape);
  const [resultVariable2] = matmul2.createOutputVariables();

  const add = new AddOp(matmul2.outputShape);
  const [resultVariable3] = add.createOutputVariables();

  const [readResultVariable] = add.createOutputVariables(VariableMode.READABLE);

  console.time('matmul_2d');

  await runCommands([
    ...matmul.getCommands(aVariable, bVariable, resultVariable),
    ...matmul2.getCommands(resultVariable, resultVariable, resultVariable2),
    ...add.getCommands(resultVariable, resultVariable2, resultVariable3),
    {
      type: OpCommandType.COPY_VARIABLE,
      src: resultVariable3,
      dst: resultVariable2,
    },
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

function App() {
  return (
    <>
      <p>Please open the console.</p>
      <button onClick={() => run()}>Run</button>
    </>
  );
}

export default App;
