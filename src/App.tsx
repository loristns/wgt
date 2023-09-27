import {WGT} from './lib';
import {Tensor} from './lib/tensor';
import {Gelu, Gemm, Input, Softmax} from './lib/ops';

async function run() {
  await WGT.initializeGpu();

  // Compute graph
  const input1 = new Input(1, 1024, 1024);
  const input2 = new Input(1, 1024, 1024);
  const matmul = new Gemm(input1, input1, input2);

  const softmax = new Softmax(matmul);
  const gelu = new Gelu(matmul);

  // Input data
  const a = Tensor.fromArray(
    Array(1024)
      .fill(1)
      .map(() =>
        Array(1024)
          .fill(1)
          .map(() => Math.random() * 0.001)
      )
  );

  const b = Tensor.fromArray(
    Array(1024)
      .fill(1)
      .map(() =>
        Array(1024)
          .fill(1)
          .map(() => Math.random() * 10)
      )
  );

  // Run
  input1.write(a);
  input2.write(b);

  console.time('matmul_2d');
  const [output1, output2, output3] = await WGT.run([matmul, softmax, gelu]);
  console.timeEnd('matmul_2d');

  console.log(output1.data);
  console.log(output2.data);
  console.log(output3.data);

  // Cleanup
  WGT.destroy([matmul, softmax, gelu]);
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
