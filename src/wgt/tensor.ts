interface ShapeParams {
  batches: number;
  rows: number;
  cols: number;
}

export type ShapeLike = Shape | ShapeParams;

export class Shape {
  static WGSL = /* wgsl */ `
    struct Shape {
      batches: u32,
      rows: u32,
      cols: u32,
    }
  `;

  readonly batches: number;
  readonly rows: number;
  readonly cols: number;

  constructor(params: ShapeParams) {
    const {batches, rows, cols} = params;

    this.batches = batches;
    this.rows = rows;
    this.cols = cols;
  }

  static from(shapeLike: ShapeLike): Shape {
    if (shapeLike instanceof Shape) {
      return shapeLike;
    }

    return new Shape(shapeLike);
  }

  /**
   * Returns the predicted size of the tensor in bytes.
   */
  get size(): number {
    return this.batches * this.rows * this.cols * 4 + 3 * 4;
  }

  equals(shape: Shape): boolean {
    return (
      this.batches === shape.batches &&
      this.rows === shape.rows &&
      this.cols === shape.cols
    );
  }

  toString(): string {
    return `${this.batches}x${this.rows}x${this.cols}`;
  }
}

/**
 * A tensor is a multi-dimensional array of numbers.
 */
export class Tensor {
  static WGSL = /* wgsl */ `
    ${Shape.WGSL}

    struct Tensor {
      shape: Shape,
      tensor: array<f32>,
    }

    // Utility functions to access a tensor's matrix.
    fn tensor_idx(shape: Shape, batch: u32, row: u32, col: u32) -> u32 {
      // The use of min() allow to make all kernels works for spread tensors.
      // For example, a [1, 10, 3] tensor can be accessed as a [10, 10, 3] tensor (the first batch is spread)

      // TODO: Check that the spread tensor is valid (i.e. that the spread dimension is 1).
      // Unchecked case is for example a [2, 10, 3] tensor accessed as a [10, 10, 3] tensor, which is invalid.
      return min(batch, shape.batches - 1) * shape.rows * shape.cols \
        + min(row, shape.rows - 1) * shape.cols \
        + min(col, shape.cols - 1);
    }
  `;

  arrayBuffer: ArrayBuffer;

  constructor(arrayBuffer: ArrayBuffer) {
    this.arrayBuffer = arrayBuffer;
  }

  get shape(): Shape {
    const rawShape = new Uint32Array(this.arrayBuffer, 0, 3);

    return new Shape({
      batches: rawShape[0],
      rows: rawShape[1],
      cols: rawShape[2],
    });
  }

  get array(): number[][][] {
    const rawTensor = new Float32Array(this.arrayBuffer, 3 * 4);

    const {batches, rows, cols} = this.shape;
    const array: number[][][] = [];

    for (let i = 0; i < batches; i++) {
      array.push([]);

      for (let j = 0; j < rows; j++) {
        array[i].push([]);

        for (let k = 0; k < cols; k++) {
          array[i][j].push(rawTensor[i * rows * cols + j * cols + k]);
        }
      }
    }

    return array;
  }

  static fromArray(
    array: number | number[] | number[][] | number[][][]
  ): Tensor {
    // Support for 0D, 1D, 2D, and 3D arrays.
    if (typeof array === 'number') {
      return Tensor.fromArray([[[array]]]);
    }
    if (typeof array[0] === 'number') {
      return Tensor.fromArray([[array as number[]]]);
    }
    if (typeof array[0][0] === 'number') {
      return Tensor.fromArray([array as number[][]]);
    }

    const shape = new Shape({
      batches: array.length,
      rows: array[0].length,
      cols: array[0][0].length,
    });

    const arrayBuffer = new ArrayBuffer(shape.size);
    const rawShape = new Uint32Array(arrayBuffer, 0, 3);
    const rawTensor = new Float32Array(arrayBuffer, 3 * 4);

    rawShape.set([shape.batches, shape.rows, shape.cols]);
    rawTensor.set(array.flat(2));

    return new Tensor(arrayBuffer);
  }

  static fromURL(url: string): Promise<Tensor> {
    return fetch(url)
      .then(response => response.blob())
      .then(blob => blob.arrayBuffer())
      .then(arrayBuffer => new Tensor(arrayBuffer));
  }

  static random(shapeLike: ShapeLike, avg = 0, std = 0.1): Tensor {
    const shape = Shape.from(shapeLike);

    const arrayBuffer = new ArrayBuffer(shape.size);
    const rawShape = new Uint32Array(arrayBuffer, 0, 3);
    const rawTensor = new Float32Array(arrayBuffer, 3 * 4);

    rawShape.set([shape.batches, shape.rows, shape.cols]);

    for (let i = 0; i < rawTensor.length; i++) {
      rawTensor[i] = Math.random() * 2 * std - std + avg;
    }

    return new Tensor(arrayBuffer);
  }

  static zeros(shapeLike: ShapeLike): Tensor {
    const shape = Shape.from(shapeLike);

    const arrayBuffer = new ArrayBuffer(shape.size);
    const rawShape = new Uint32Array(arrayBuffer, 0, 3);
    const rawTensor = new Float32Array(arrayBuffer, 3 * 4);

    rawShape.set([shape.batches, shape.rows, shape.cols]);
    rawTensor.fill(0);

    return new Tensor(arrayBuffer);
  }

  static ones(shapeLike: ShapeLike): Tensor {
    const shape = Shape.from(shapeLike);

    const arrayBuffer = new ArrayBuffer(shape.size);
    const rawShape = new Uint32Array(arrayBuffer, 0, 3);
    const rawTensor = new Float32Array(arrayBuffer, 3 * 4);

    rawShape.set([shape.batches, shape.rows, shape.cols]);
    rawTensor.fill(1);

    return new Tensor(arrayBuffer);
  }
}
