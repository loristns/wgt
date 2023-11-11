import {WGT} from './wgt/wgt';
import {input} from './wgt/tensorBuffer';
import {Shape, Tensor} from './wgt/tensor';
import {Gemm} from './wgt/ops/gemm';

async function run() {
  await WGT.initializeGpu();

  const a = Tensor.random(new Shape({batches: 1, rows: 500, cols: 5000}));
  const b = Tensor.random(new Shape({batches: 10, rows: 5000, cols: 500}));

  const input1 = input(1, 500, 5000);
  const input2 = input(10, 5000, 500);

  const outputOp = new Gemm(input1, input2);

  input1.write(a);
  input2.write(b);
  console.log(await WGT.run([outputOp]));
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
