import type { EncodedData } from './types';

/**
 * General Numeric Encoder
 *
 * Parses any numeric CSV/JSON data:
 *   - If vector: reshape to nearest-square 2D and pad with zeros
 *   - If matrix: normalize to [0,1] and scale to 256x256
 * Returns as ImageData.
 */

function parseNumeric(text: string): { matrix: number[][]; flat: number[] } {
  const trimmed = text.trim();

  // Try JSON first
  try {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) {
      // Check if it's a 2D array (matrix)
      if (Array.isArray(parsed[0])) {
        const matrix = parsed.map((row: any[]) => row.map(Number));
        const flat = matrix.flat();
        return { matrix, flat };
      }
      // 1D array (vector)
      const flat = parsed.map(Number).filter((n: number) => !isNaN(n));
      return { matrix: [], flat };
    }
  } catch {
    // Not JSON
  }

  // CSV parsing
  const lines = trimmed.split(/\r?\n/).filter((l) => l.trim() && !l.startsWith('#'));
  const rows: number[][] = [];

  for (const line of lines) {
    const parts = line.split(/[,\t\s]+/).map(Number).filter((n) => !isNaN(n));
    if (parts.length > 0) {
      rows.push(parts);
    }
  }

  if (rows.length === 0) {
    return { matrix: [], flat: [] };
  }

  // If all rows have same length > 1, treat as matrix
  const isMatrix = rows.length > 1 && rows.every((r) => r.length === rows[0].length) && rows[0].length > 1;

  if (isMatrix) {
    return { matrix: rows, flat: rows.flat() };
  }

  // Otherwise flatten to vector
  return { matrix: [], flat: rows.flat() };
}

function nearestSquare(n: number): number {
  return Math.ceil(Math.sqrt(n));
}

function bilinearSample(
  source: Float32Array,
  srcW: number,
  srcH: number,
  u: number,
  v: number
): number {
  const x = u * (srcW - 1);
  const y = v * (srcH - 1);

  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, srcW - 1);
  const y1 = Math.min(y0 + 1, srcH - 1);

  const fx = x - x0;
  const fy = y - y0;

  const v00 = source[y0 * srcW + x0];
  const v10 = source[y0 * srcW + x1];
  const v01 = source[y1 * srcW + x0];
  const v11 = source[y1 * srcW + x1];

  return v00 * (1 - fx) * (1 - fy) +
         v10 * fx * (1 - fy) +
         v01 * (1 - fx) * fy +
         v11 * fx * fy;
}

export function encodeGeneral(text: string): EncodedData {
  const t0 = performance.now();
  const { matrix, flat } = parseNumeric(text);

  if (flat.length === 0) {
    throw new Error('No valid numeric data found. Expected CSV or JSON numbers.');
  }

  const SIZE = 256;
  const data = new Uint8ClampedArray(SIZE * SIZE * 4);

  let srcW: number;
  let srcH: number;
  let normalized: Float32Array;

  if (matrix.length > 0) {
    // Matrix input
    srcH = matrix.length;
    srcW = matrix[0].length;
    normalized = new Float32Array(srcW * srcH);
    let min = Infinity, max = -Infinity;

    for (let i = 0; i < flat.length; i++) {
      if (flat[i] < min) min = flat[i];
      if (flat[i] > max) max = flat[i];
    }

    const range = max - min || 1;
    for (let r = 0; r < srcH; r++) {
      for (let c = 0; c < srcW; c++) {
        normalized[r * srcW + c] = (matrix[r][c] - min) / range;
      }
    }
  } else {
    // Vector input: reshape to square
    const side = nearestSquare(flat.length);
    srcW = side;
    srcH = side;
    normalized = new Float32Array(side * side);

    let min = Infinity, max = -Infinity;
    for (const v of flat) {
      if (v < min) min = v;
      if (v > max) max = v;
    }

    const range = max - min || 1;
    for (let i = 0; i < flat.length; i++) {
      normalized[i] = (flat[i] - min) / range;
    }
    // Remaining cells stay zero (padding)
  }

  // Bilinear interpolation to 256x256
  for (let py = 0; py < SIZE; py++) {
    const v = py / SIZE;
    for (let px = 0; px < SIZE; px++) {
      const u = px / SIZE;
      const intensity = bilinearSample(normalized, srcW, srcH, u, v);
      const idx = (py * SIZE + px) * 4;
      const val = Math.round(Math.min(Math.max(intensity, 0), 1.0) * 255);
      data[idx + 0] = val;
      data[idx + 1] = val;
      data[idx + 2] = val;
      data[idx + 3] = 255;
    }
  }

  const imageData = new ImageData(data, SIZE, SIZE);
  const encodingTime = performance.now() - t0;

  return {
    imageData,
    metadata: {
      domain: 'general',
      originalSize: flat.length,
      encodingTime,
    },
  };
}

/** Example numeric matrix data for demonstration */
export const GENERAL_EXAMPLE = (() => {
  // Generate a 2D Gaussian + periodic pattern as demo matrix
  const lines: string[] = ['# 32x32 numeric matrix: Gaussian + periodic pattern'];
  const N = 32;
  for (let r = 0; r < N; r++) {
    const row: number[] = [];
    for (let c = 0; c < N; c++) {
      const x = (c / N - 0.5) * 2;
      const y = (r / N - 0.5) * 2;
      const gaussian = Math.exp(-(x * x + y * y) / 0.5);
      const periodic = 0.3 * Math.sin(8 * Math.PI * x) * Math.cos(6 * Math.PI * y);
      row.push(Number((gaussian + periodic).toFixed(4)));
    }
    lines.push(row.join(','));
  }
  return lines.join('\n');
})();
