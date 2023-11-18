import {WGT} from './wgt/wgt';
import {Tensor} from './wgt/tensor';
import {input} from './wgt/ops/input';
import {gpt2Block, Gpt2BlockParameters} from './wgt/ops/gpt2/gpt2Block';

async function run() {
  await WGT.initializeGpu();

  const INPUT_LENGTH = 8;
  const HIDDEN_SIZE = 1024;
  const ATTENTION_HEADS = 2;

  const blockParams: Gpt2BlockParameters = {
    layerNorm1: {
      scale: Tensor.random({batches: 1, rows: 1, cols: HIDDEN_SIZE}),
      bias: Tensor.zeros({batches: 1, rows: 1, cols: HIDDEN_SIZE}),
    },
    selfAttention: {
      query: {
        weights: Tensor.random({
          batches: ATTENTION_HEADS,
          rows: HIDDEN_SIZE,
          cols: HIDDEN_SIZE,
        }),
        bias: Tensor.zeros({
          batches: ATTENTION_HEADS,
          rows: 1,
          cols: HIDDEN_SIZE,
        }),
      },
      key: {
        weights: Tensor.random({
          batches: ATTENTION_HEADS,
          rows: HIDDEN_SIZE,
          cols: HIDDEN_SIZE,
        }),
        bias: Tensor.zeros({
          batches: ATTENTION_HEADS,
          rows: 1,
          cols: HIDDEN_SIZE,
        }),
      },
      value: {
        weights: Tensor.random({
          batches: ATTENTION_HEADS,
          rows: HIDDEN_SIZE,
          cols: HIDDEN_SIZE,
        }),
        bias: Tensor.zeros({
          batches: ATTENTION_HEADS,
          rows: 1,
          cols: HIDDEN_SIZE,
        }),
      },
      projection: {
        weights: Tensor.random({
          batches: 1,
          rows: ATTENTION_HEADS * HIDDEN_SIZE,
          cols: HIDDEN_SIZE,
        }),
        bias: Tensor.zeros({batches: 1, rows: 1, cols: HIDDEN_SIZE}),
      },
    },
    layerNorm2: {
      scale: Tensor.random({batches: 1, rows: 1, cols: HIDDEN_SIZE}),
      bias: Tensor.zeros({batches: 1, rows: 1, cols: HIDDEN_SIZE}),
    },
    feedForward: {
      linear1: {
        weights: Tensor.random({
          batches: 1,
          rows: HIDDEN_SIZE,
          cols: 4 * HIDDEN_SIZE,
        }),
        bias: Tensor.zeros({batches: 1, rows: 1, cols: 4 * HIDDEN_SIZE}),
      },
      linear2: {
        weights: Tensor.random({
          batches: 1,
          rows: 4 * HIDDEN_SIZE,
          cols: HIDDEN_SIZE,
        }),
        bias: Tensor.zeros({batches: 1, rows: 1, cols: HIDDEN_SIZE}),
      },
    },
  };

  const input1 = input({batches: 1, rows: INPUT_LENGTH, cols: HIDDEN_SIZE});

  const block1 = gpt2Block(input1, blockParams);

  const graph = new WGT([input1], [input1, block1]);

  const a = Tensor.random({batches: 1, rows: INPUT_LENGTH, cols: HIDDEN_SIZE});

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
