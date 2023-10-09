import {WGT} from './lib';
import {Tensor} from './lib/tensor';
import {Input, SelfAttention} from './lib/ops';

async function run() {
  await WGT.initializeGpu();

  // Compute graph
  const input = new Input(1, 32, 1024);

  const queryWeights = new Input(10, 1024, 512);
  const queryBias = new Input(10, 1, 512);

  const keyWeights = new Input(10, 1024, 512);
  const keyBias = new Input(10, 1, 512);

  const valueWeights = new Input(10, 1024, 512);
  const valueBias = new Input(10, 1, 512);

  const projWeights = new Input(1, 10 * 512, 1024);
  const projBias = new Input(1, 1, 1024);

  const attention = new SelfAttention(
    input,
    queryWeights,
    queryBias,
    keyWeights,
    keyBias,
    valueWeights,
    valueBias,
    projWeights,
    projBias
  );

  // Input data
  const inputTensor = Tensor.fromShape(input.shape, 1);

  const qw = Tensor.fromShape(queryWeights.shape, 'random');
  const qb = Tensor.fromShape(queryBias.shape, 'random');

  const kw = Tensor.fromShape(keyWeights.shape, 'random');
  const kb = Tensor.fromShape(keyBias.shape, 'random');

  const vw = Tensor.fromShape(valueWeights.shape, 'random');
  const vb = Tensor.fromShape(valueBias.shape, 'random');

  const pw = Tensor.fromShape(projWeights.shape, 'random');
  const pb = Tensor.fromShape(projBias.shape, 'random');

  input.write(inputTensor);
  queryWeights.write(qw);
  queryBias.write(qb);
  keyWeights.write(kw);
  keyBias.write(kb);
  valueWeights.write(vw);
  valueBias.write(vb);
  projWeights.write(pw);
  projBias.write(pb);

  console.time('attention');
  const [output] = await WGT.run([attention]);
  console.timeEnd('attention');

  console.log(output.data);
  console.log(output.shape);
  console.log(output.rawData);

  // Cleanup
  WGT.destroy([attention]);
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
