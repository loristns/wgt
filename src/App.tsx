import {WGT} from './lib';
import {Tensor} from './lib/tensor';
import {Gelu, Gemm, Input, LayerNorm, Softmax} from './lib/ops';

async function run() {
  await WGT.initializeGpu();

  // Compute graph
  const input1 = new Input(1, 1024, 1024);
  const input2 = new Input(1, 1024, 1024);
  const input3 = new Input(1, 1, 1024);
  const matmul = new Gemm(input1, input2, input3);

  const softmax = new Softmax(matmul);
  const gelu = new Gelu(matmul);
  const layerNorm = new LayerNorm(matmul, matmul, gelu);

  // Input data
  const a = Tensor.fromShape(input1.shape, 'random');
  const b = Tensor.fromShape(input2.shape, 1);
  const c = Tensor.fromShape(input3.shape, 'random');

  // Run
  input1.write(a);
  input2.write(b);
  input3.write(c);

  console.time('matmul_2d');
  const [output1, output2, output3, output4] = await WGT.run([
    matmul,
    softmax,
    gelu,
    layerNorm,
  ]);
  console.timeEnd('matmul_2d');

  console.log(output1.data);
  console.log(output2.data);
  console.log(output3.data);
  console.log(output4.data);

  // Cleanup
  WGT.destroy([matmul, softmax, gelu, layerNorm]);
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
