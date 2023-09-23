import {WGT} from './lib';
import {Tensor} from './lib/tensor';
import {Input, Matmul} from './lib/ops';

async function run() {
  await WGT.initializeGpu();

  // Compute graph
  // (input1 @ input2) @ (input1 @ input2)
  const input1 = new Input(1, 2, 3);
  const input2 = new Input(1, 3, 2);
  const matmul = new Matmul(input1, input2);
  const matmul2 = new Matmul(matmul, matmul);

  // Input data
  const a = Tensor.fromArray([
    [
      [1, 2, 3],
      [4, 5, 6],
    ],
  ]);

  const b = Tensor.fromArray([
    [
      [7, 8],
      [9, 10],
      [11, 12],
    ],
  ]);

  // Run
  console.time('matmul_2d');

  input1.write(a);
  input2.write(b);

  const [output1, output2] = await WGT.run([matmul, matmul2]);

  console.timeEnd('matmul_2d');
  console.log(output1.data);
  console.log(output2.data);

  // Cleanup
  input1.clean();
  input2.clean();
  matmul.clean();
  matmul2.clean();
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
