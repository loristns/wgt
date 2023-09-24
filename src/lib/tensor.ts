/**
 * Represents a 2D tensor shape.
 * Uses explicit fields instead of an array for readability.
 */
export class TensorShape {
  readonly rows: number;
  readonly cols: number;

  constructor(rows: number, cols: number) {
    this.rows = rows;
    this.cols = cols;
  }

  /**
   * Returns the size of the tensor in bytes.
   */
  get size(): number {
    return this.rows * this.cols * 4 + 2 * 4;
  }
}

/**
 * The WGSL struct representation of a tensor.
 */
export const WGSL_TENSOR_TYPE = /* wgsl */ `
  struct Tensor {
    rows: u32,
    cols: u32,
    matrix: array<f32>,
  }
`;

/**
 * A 2D tensor type with an internal binary representation :
 *  - First 2 32-bit unsigned integers represent the shape of the tensor (rows, cols).
 *  - The rest of the buffer is a 32-bit float array representing the matrix.
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
    this.rawShape = new Uint32Array(this.arrayBuffer, 0, 2);
    this.rawMatrix = new Float32Array(this.arrayBuffer, 2 * 4);
  }

  get shape(): TensorShape {
    return new TensorShape(this.rawShape[0], this.rawShape[1]);
  }

  get data(): number[][] {
    const {rows, cols} = this.shape;

    const data: number[][] = [];

    for (let i = 0; i < rows; i++) {
      data.push([]);

      for (let j = 0; j < cols; j++) {
        data[i].push(this.rawMatrix[i * cols + j]);
      }
    }

    return data;
  }

  /**
   * Constructs a tensor from a 2D array (array of arrays).
   */
  static fromArray(array: number[][]): Tensor {
    const [rows, cols] = [array.length, array[0].length];

    const arrayBuffer = new ArrayBuffer(rows * cols * 4 + 2 * 4);
    const rawData = new Uint32Array(arrayBuffer);
    const rawMatrix = new Float32Array(arrayBuffer, 2 * 4);

    rawData.set([rows, cols], 0);
    rawMatrix.set(array.flat(), 0);

    return new Tensor(rawData);
  }
}
