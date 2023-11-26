import {useEffect, useState} from 'react';

import {WGT} from './wgt/wgt';
import {Tensor} from './wgt/tensor';
import {input} from './wgt/ops/input';

import {gpt2, Gpt2Parameters} from './gpt2/ops/gpt2';
import {Gpt2BlockParameters} from './gpt2/ops/gpt2Block';

import vocab from './gpt2/vocab.json';
import {Tokenizer} from './gpt2/tokenizer';

async function init() {
  await WGT.initializeGpu();

  const tokenizer = new Tokenizer(vocab);

  const blocks: Gpt2BlockParameters[] = [];
  for (let blockId = 0; blockId < 12; blockId++) {
    blocks.push({
      layerNorm1: {
        scale: await Tensor.fromURL(
          `/gpt2_weights/block${blockId}.ln1.scale.bin`
        ),
        bias: await Tensor.fromURL(
          `/gpt2_weights/block${blockId}.ln1.bias.bin`
        ),
      },
      selfAttention: {
        query: {
          weights: await Tensor.fromURL(
            `/gpt2_weights/block${blockId}.attention.query.weights.bin`
          ),
          bias: await Tensor.fromURL(
            `/gpt2_weights/block${blockId}.attention.query.bias.bin`
          ),
        },
        key: {
          weights: await Tensor.fromURL(
            `/gpt2_weights/block${blockId}.attention.key.weights.bin`
          ),
          bias: await Tensor.fromURL(
            `/gpt2_weights/block${blockId}.attention.key.bias.bin`
          ),
        },
        value: {
          weights: await Tensor.fromURL(
            `/gpt2_weights/block${blockId}.attention.value.weights.bin`
          ),
          bias: await Tensor.fromURL(
            `/gpt2_weights/block${blockId}.attention.value.bias.bin`
          ),
        },
        projection: {
          weights: await Tensor.fromURL(
            `/gpt2_weights/block${blockId}.attention.proj.weights.bin`
          ),
          bias: await Tensor.fromURL(
            `/gpt2_weights/block${blockId}.attention.proj.bias.bin`
          ),
        },
      },
      layerNorm2: {
        scale: await Tensor.fromURL(
          `/gpt2_weights/block${blockId}.ln2.scale.bin`
        ),
        bias: await Tensor.fromURL(
          `/gpt2_weights/block${blockId}.ln2.bias.bin`
        ),
      },
      feedForward: {
        linear1: {
          weights: await Tensor.fromURL(
            `/gpt2_weights/block${blockId}.ff.linear1.weights.bin`
          ),
          bias: await Tensor.fromURL(
            `/gpt2_weights/block${blockId}.ff.linear1.bias.bin`
          ),
        },
        linear2: {
          weights: await Tensor.fromURL(
            `/gpt2_weights/block${blockId}.ff.linear2.weights.bin`
          ),
          bias: await Tensor.fromURL(
            `/gpt2_weights/block${blockId}.ff.linear2.bias.bin`
          ),
        },
      },
    });
  }

  const params: Gpt2Parameters = {
    tokenEmbeddings: {
      chunk1: await Tensor.fromURL('/gpt2_weights/embeddings.chunk1.bin'),
      chunk2: await Tensor.fromURL('/gpt2_weights/embeddings.chunk2.bin'),
    },
    positionEmbeddings: {
      chunk1: await Tensor.fromURL(
        '/gpt2_weights/position_embeddings.chunk1.bin'
      ),
      chunk2: await Tensor.fromURL(
        '/gpt2_weights/position_embeddings.chunk2.bin'
      ),
    },
    layerNorm: {
      scale: await Tensor.fromURL('/gpt2_weights/ln_final.scale.bin'),
      bias: await Tensor.fromURL('/gpt2_weights/ln_final.bias.bin'),
    },
    blocks,
  };

  console.log('params', params);

  return {tokenizer, params};
}

async function run(tokens: number[], params: Gpt2Parameters) {
  const input1 = input({
    batches: 1,
    rows: 1,
    cols: tokens.length,
  });
  const gpt = gpt2(input1, params);

  console.time('compile');
  const graph = new WGT([input1], [gpt]);
  console.timeEnd('compile');

  console.time('run');

  const [pred] = await graph.run([Tensor.fromArray(tokens)]);

  const nextTokens = pred.array[0].map(row => {
    const max = Math.max(...row);
    return row.indexOf(max);
  });
  const nextToken = nextTokens[nextTokens.length - 1];

  console.timeEnd('run');

  console.log('next token:', nextTokens);
  console.log(pred);

  // console.log(graph.getDotTree());

  graph.destroy();

  return nextToken;
}

function App() {
  const [loaded, setLoaded] = useState(false);
  const [params, setParams] = useState<Gpt2Parameters | null>(null);
  const [tokenizer, setTokenizer] = useState<Tokenizer | null>(null);

  const [text, setText] = useState('');
  const [predTokens, setPredTokens] = useState<number[]>([]);

  useEffect(() => {
    init().then(({params, tokenizer}) => {
      setLoaded(true);
      setParams(params);
      setTokenizer(tokenizer);
    });
  }, []);

  return (
    <>
      <h1>WGT</h1>

      <p>
        <strong>Status: </strong> {loaded ? 'Loaded' : 'Loading...'}
      </p>

      <hr />
      <h2>Tokenizer</h2>

      <textarea
        placeholder="Type something..."
        value={text}
        cols={80}
        rows={10}
        onChange={e => {
          setText(e.target.value);
          setPredTokens(tokenizer?.encode(e.target.value) ?? []);
        }}
      />

      <p>
        <strong>Tokenized: </strong>
        {tokenizer?.encode(text).join(', ')}
      </p>

      <hr />
      <h2>Model</h2>

      <textarea
        placeholder="The model will predict the next token..."
        value={tokenizer?.decode(predTokens)}
        cols={80}
        rows={10}
        readOnly
      />

      <p>
        <strong>Tokenized: </strong>
        {predTokens.join(', ')}
      </p>

      <br />

      <button
        disabled={!loaded}
        onClick={async () =>
          setPredTokens([...predTokens, await run(predTokens, params!)])
        }
      >
        Run
      </button>
    </>
  );
}

export default App;
