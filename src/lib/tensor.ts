/**
 * Represents a 3D tensor shape.
 * Uses explicit fields instead of an array for readability.
 */
export class TensorShape {
  readonly batches: number;
  readonly rows: number;
  readonly cols: number;

  constructor(batches: number, rows: number, cols: number) {
    this.batches = batches;
    this.rows = rows;
    this.cols = cols;
  }

  /**
   * Returns the size of the tensor in bytes.
   */
  get size(): number {
    return this.batches * this.rows * this.cols * 4 + 3 * 4;
  }
}

/**
 * A 3D tensor type with an internal binary representation :
 *  - First 3 32-bit unsigned integers represent the shape of the tensor (batches, rows, cols).
 *  - The rest of the buffer is a 32-bit float array representing the tensor.
 */
export class Tensor {
  arrayBuffer: ArrayBuffer;

  rawData: Uint32Array;
  rawShape: Uint32Array;
  rawMatrix: Float32Array;

  /**
   * Constructs a tensor from its raw binary representation.
   * @param arrayBuffer An array buffer containing the raw binary representation of the tensor.
   */
  constructor(arrayBuffer: ArrayBuffer) {
    this.arrayBuffer = arrayBuffer;

    this.rawData = new Uint32Array(this.arrayBuffer);
    this.rawShape = new Uint32Array(this.arrayBuffer, 0, 3);
    this.rawMatrix = new Float32Array(this.arrayBuffer, 3 * 4);
  }

  get shape(): TensorShape {
    return new TensorShape(
      this.rawShape[0],
      this.rawShape[1],
      this.rawShape[2]
    );
  }

  get data(): number[][][] {
    const {batches, rows, cols} = this.shape;

    const data: number[][][] = [];

    for (let i = 0; i < batches; i++) {
      data.push([]);

      for (let j = 0; j < rows; j++) {
        data[i].push([]);

        for (let k = 0; k < cols; k++) {
          data[i][j].push(this.rawMatrix[i * rows * cols + j * cols + k]);
        }
      }
    }

    return data;
  }

  /**
   * Constructs a tensor from a number, a vector, a matrix or a 3D tensor.
   */
  static fromArray(
    array: number | number[] | number[][] | number[][][]
  ): Tensor {
    if (typeof array === 'number') {
      return Tensor.fromArray([[[array]]]);
    }
    if (typeof array[0] === 'number') {
      return Tensor.fromArray([[array as number[]]]);
    }
    if (typeof array[0][0] === 'number') {
      return Tensor.fromArray([array as number[][]]);
    }

    const [batches, rows, cols] = [
      array.length,
      array[0].length,
      array[0][0].length,
    ];

    const arrayBuffer = new ArrayBuffer(batches * rows * cols * 4 + 3 * 4);
    const rawData = new Uint32Array(arrayBuffer);
    const rawMatrix = new Float32Array(arrayBuffer, 3 * 4);

    rawData.set([batches, rows, cols], 0);
    rawMatrix.set(array.flat(2), 0);

    return new Tensor(rawData);
  }

  static fromShape(shape: TensorShape, value: number | 'random' = 0): Tensor {
    const arrayBuffer = new ArrayBuffer(shape.size);
    const rawData = new Uint32Array(arrayBuffer);
    const rawMatrix = new Float32Array(arrayBuffer, 3 * 4);

    rawData.set([shape.batches, shape.rows, shape.cols], 0);

    for (let i = 0; i < shape.batches * shape.rows * shape.cols; i++) {
      rawMatrix[i] = value === 'random' ? Math.random() : value;
    }

    return new Tensor(rawData);
  }
}

/**
 * The WGSL representation of a tensor.
 */
export const WGSL_TENSOR = /* wgsl */ `
  struct TensorShape {
    batches: u32,
    rows: u32,
    cols: u32,
  }

  struct Tensor {
    shape: TensorShape,
    matrix: array<f32>,
  }

  // Utility functions to access a tensor's matrix.
  fn tensor_idx(shape: TensorShape, batch: u32, row: u32, col: u32) -> u32 {
    // The use of min() allow to make all kernels works for spread tensors.
    // For example, a [1, 10, 3] tensor can be accessed as a [10, 10, 3] tensor (the first batch is spread)

    // TODO: Check that the spread tensor is valid (i.e. that the spread dimension is 1).
    // Unchecked case is for example a [2, 10, 3] tensor accessed as a [10, 10, 3] tensor, which is invalid.
    return min(batch, shape.batches - 1) * shape.rows * shape.cols \
      + min(row, shape.rows - 1) * shape.cols \
      + min(col, shape.cols - 1);
  }
`;
