import {WGT} from './wgt/wgt';
import {Tensor} from './wgt/tensor';
import {input} from './wgt/ops/input';
import {layerNorm, LayerNormParameters} from './wgt/ops/layerNorm';
import {linear, LinearParameters} from './wgt/ops/linear';
import {gelu} from './wgt/ops/gelu';
import {softmax} from './wgt/ops/softmax';
import {transpose} from './wgt/ops/transpose';

async function run() {
  await WGT.initializeGpu();

  const linear1Params: LinearParameters = {
    weights: Tensor.random({batches: 1, rows: 10, cols: 200}),
    bias: Tensor.zeros({batches: 1, rows: 1, cols: 200}),
  };

  const layerNorm1Params: LayerNormParameters = {
    scale: Tensor.ones({batches: 1, rows: 1, cols: 200}),
    bias: Tensor.zeros({batches: 1, rows: 1, cols: 200}),
  };

  const input1 = input({batches: 1, rows: 1, cols: 10});
  const linear1 = linear(input1, linear1Params);
  const layerNorm1 = transpose(layerNorm(linear1, layerNorm1Params));

  const input2 = input({batches: 1, rows: 1, cols: 5});
  const gelu1 = gelu(input2);
  const softmax1 = softmax(gelu1);

  const graph = new WGT(
    [input1, input2],
    [linear1, layerNorm1, gelu1, softmax1]
  );

  const a = Tensor.random({batches: 1, rows: 1, cols: 10});
  const b = Tensor.random({batches: 1, rows: 1, cols: 5});
  console.log(await graph.run([a, b]));

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
