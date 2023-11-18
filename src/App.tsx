import {WGT} from './wgt/wgt';
import {Tensor} from './wgt/tensor';
import {input} from './wgt/ops/input';
import {LinearParameters} from './wgt/ops/linear';
import {selfAttention, SelfAttentionParameters} from './wgt/ops/selfAttention';
import {merge, MergeMethod} from './wgt/ops/merge';

async function run() {
  await WGT.initializeGpu();

  const linear1Params: LinearParameters = {
    weights: Tensor.random({batches: 120, rows: 10, cols: 1024}),
    bias: Tensor.zeros({batches: 120, rows: 1, cols: 1024}),
  };

  const linear2Params: LinearParameters = {
    weights: Tensor.random({batches: 1, rows: 1024, cols: 12}),
    bias: Tensor.zeros({batches: 1, rows: 1, cols: 12}),
  };

  const attentionParams: SelfAttentionParameters = {
    query: linear1Params,
    key: linear1Params,
    value: linear1Params,
    projection: linear2Params,
  };

  const input1 = input({batches: 1, rows: 8, cols: 10});
  const att = selfAttention(input1, attentionParams);

  const merged = merge(att, att, MergeMethod.Div);

  const graph = new WGT([input1], [att, merged]);

  const a = Tensor.random({batches: 1, rows: 8, cols: 10});

  console.time('run');
  console.log(await graph.run([a]));
  console.timeEnd('run');

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
