import {WGT} from './wgt/wgt';
import {Tensor} from './wgt/tensor';
import {input} from './wgt/ops/input';
import {gpt2Block, Gpt2BlockParameters} from './wgt/ops/gpt2/gpt2Block';
import {TensorUtils} from './wgt/tensorUtils';

async function run() {
  await WGT.initializeGpu();

  const VOCAB_SIZE = 10000;
  const INPUT_LENGTH = 8;
  const HIDDEN_SIZE = 1024;
  const ATTENTION_HEADS = 16;

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
  const block2 = gpt2Block(block1, blockParams);
  const block3 = gpt2Block(block2, blockParams);
  const block4 = gpt2Block(block3, blockParams);
  const block5 = gpt2Block(block4, blockParams);
  const block6 = gpt2Block(block5, blockParams);
  const block7 = gpt2Block(block6, blockParams);
  const block8 = gpt2Block(block7, blockParams);
  const block9 = gpt2Block(block8, blockParams);
  const block10 = gpt2Block(block9, blockParams);
  const block11 = gpt2Block(block10, blockParams);
  const block12 = gpt2Block(block11, blockParams);

  console.time('compile');
  const graph = new WGT([input1], [block12]);
  console.timeEnd('compile');

  console.log(graph.getRecipe());
  console.log(graph.getDotTree());

  const embeddings = Tensor.random({
    batches: 1,
    rows: VOCAB_SIZE,
    cols: HIDDEN_SIZE,
  });
  const i = TensorUtils.getEmbeddings([0, 0, 2, 3, 4, 5, 6, 7], embeddings);
  console.log(i);

  console.time('run');
  console.log(await graph.run([i]));
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
