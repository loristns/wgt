import {WGT} from './wgt/wgt';
import {Tensor} from './wgt/tensor';
import {input} from './wgt/ops/input';
import {linear, LinearParameters} from './wgt/ops/linear';

async function run() {
  await WGT.initializeGpu();

  const params: LinearParameters = {
    weights: Tensor.random({batches: 1, rows: 10, cols: 200}),
    bias: Tensor.zeros({batches: 1, rows: 1, cols: 200}),
  };

  const input1 = input({batches: 1, rows: 1, cols: 10});
  const linear1 = linear(input1, params);
  const graph = new WGT([input1], [linear1]);

  const a = Tensor.random({batches: 1, rows: 1, cols: 10});
  console.log(await graph.run([a]));

  graph.destroy();
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
