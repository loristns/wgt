import {WGT} from './lib';
import {Tensor} from './lib/tensor';
import {Input, Matmul, Softmax} from './lib/ops';

async function run() {
  await WGT.initializeGpu();

  // Compute graph
  // (input1 @ input2) @ (input1 @ input2)
  const input1 = new Input(1024, 1024);
  const input2 = new Input(1024, 1024);
  const matmul = new Matmul(input1, input2);
  const matmul2 = new Matmul(matmul, matmul);

  const softmax = new Softmax(matmul);

  // Input data
  const a = Tensor.fromArray(
    Array(1024)
      .fill(1)
      .map(() =>
        Array(1024)
          .fill(1)
          .map(() => Math.random())
      )
  );

  const b = Tensor.fromArray(
    Array(1024)
      .fill(1)
      .map(() =>
        Array(1024)
          .fill(1)
          .map(() => Math.random())
      )
  );

  // Run
  console.time('matmul_2d');

  input1.write(a);
  input2.write(b);

  const [output1, output2] = await WGT.run([matmul, softmax]);

  console.timeEnd('matmul_2d');
  console.log(output1.data);
  console.log(output2.data);

  // Cleanup
  input1.clean();
  input2.clean();
  matmul.clean();
  matmul2.clean();
  softmax.clean();
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
