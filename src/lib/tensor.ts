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
 * The WGSL struct representation of a tensor.
 */
export const WGSL_TENSOR_TYPE = /* wgsl */ `
  struct Tensor {
    batches: u32,
    rows: u32,
    cols: u32,
    matrix: array<f32>,
  }
`;

/**
 * A 3D tensor type with an internal binary representation :
 *  - First 3 32-bit unsigned integers represent the shape of the tensor (batches, rows, cols).
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
   * Constructs a tensor from a 3D array (array of arrays of arrays).
   */
  static fromArray(array: number[][][]): Tensor {
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
}
