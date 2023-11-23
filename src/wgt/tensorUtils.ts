import {Shape, Tensor} from './tensor';

export class TensorUtils {
  static getEmbeddings(input: number[], weights: Tensor): Tensor {
    const shape = new Shape({
      batches: 1,
      rows: input.length,
      cols: weights.shape.cols,
    });

    const arrayBuffer = new ArrayBuffer(shape.size);
    const rawShape = new Uint32Array(arrayBuffer, 0, 3);
    const rawTensor = new Float32Array(arrayBuffer, 3 * 4);

    rawShape.set([shape.batches, shape.rows, shape.cols]);

    const weightsRawTensor = new Float32Array(weights.arrayBuffer, 3 * 4);

    input.forEach((idx, pos) => {
      const weightRow = weightsRawTensor.slice(
        idx * shape.cols,
        (idx + 1) * shape.cols
      );

      rawTensor.set(weightRow, pos * shape.cols);
    });

    return new Tensor(arrayBuffer);
  }
}
