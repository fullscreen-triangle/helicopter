import type { EncodedData } from './types';

/**
 * Genomic Encoder
 *
 * Parses FASTA or raw nucleotide sequence text, computes k-mer frequencies
 * (k=3, 64 trinucleotides), and builds a 256x256 image where:
 *   - 64 k-mers are arranged in an 8x8 grid
 *   - Each tile has intensity proportional to k-mer frequency
 *   - Tiles are expanded to 256x256 via bilinear interpolation
 */

const BASES = ['A', 'C', 'G', 'T'];

// Generate all 64 trinucleotides
function allTrimers(): string[] {
  const result: string[] = [];
  for (const a of BASES) {
    for (const b of BASES) {
      for (const c of BASES) {
        result.push(a + b + c);
      }
    }
  }
  return result;
}

const TRIMERS = allTrimers();
const TRIMER_INDEX = new Map(TRIMERS.map((t, i) => [t, i]));

function parseFasta(text: string): string {
  const lines = text.split(/\r?\n/);
  const seqLines: string[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('>') || trimmed.startsWith(';') || trimmed === '') continue;
    seqLines.push(trimmed.toUpperCase().replace(/[^ACGT]/g, ''));
  }
  return seqLines.join('');
}

function computeKmerFrequencies(seq: string): Float32Array {
  const counts = new Float32Array(64);
  let total = 0;

  for (let i = 0; i <= seq.length - 3; i++) {
    const kmer = seq.substring(i, i + 3);
    const idx = TRIMER_INDEX.get(kmer);
    if (idx !== undefined) {
      counts[idx]++;
      total++;
    }
  }

  // Normalize to frequencies
  if (total > 0) {
    for (let i = 0; i < 64; i++) {
      counts[i] /= total;
    }
  }

  return counts;
}

function bilinearSample(grid: Float32Array, gridW: number, gridH: number, u: number, v: number): number {
  // u, v in [0, 1]
  const x = u * (gridW - 1);
  const y = v * (gridH - 1);

  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, gridW - 1);
  const y1 = Math.min(y0 + 1, gridH - 1);

  const fx = x - x0;
  const fy = y - y0;

  const v00 = grid[y0 * gridW + x0];
  const v10 = grid[y0 * gridW + x1];
  const v01 = grid[y1 * gridW + x0];
  const v11 = grid[y1 * gridW + x1];

  return v00 * (1 - fx) * (1 - fy) +
         v10 * fx * (1 - fy) +
         v01 * (1 - fx) * fy +
         v11 * fx * fy;
}

export function encodeGenomic(text: string): EncodedData {
  const t0 = performance.now();
  const seq = parseFasta(text);

  if (seq.length < 3) {
    throw new Error('Sequence too short. Need at least 3 nucleotides (ACGT).');
  }

  const freqs = computeKmerFrequencies(seq);

  // Normalize frequencies for visualization
  let maxFreq = 0;
  for (let i = 0; i < 64; i++) {
    if (freqs[i] > maxFreq) maxFreq = freqs[i];
  }
  if (maxFreq > 0) {
    for (let i = 0; i < 64; i++) {
      freqs[i] /= maxFreq;
    }
  }

  // Build 256x256 image by bilinear interpolation of 8x8 grid
  const SIZE = 256;
  const data = new Uint8ClampedArray(SIZE * SIZE * 4);

  for (let py = 0; py < SIZE; py++) {
    const v = py / SIZE;
    for (let px = 0; px < SIZE; px++) {
      const u = px / SIZE;
      const intensity = bilinearSample(freqs, 8, 8, u, v);
      const idx = (py * SIZE + px) * 4;
      const val = Math.round(Math.min(intensity, 1.0) * 255);
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
      domain: 'genomic',
      originalSize: seq.length,
      encodingTime,
    },
  };
}

/** Example FASTA sequence for demonstration */
export const GENOMIC_EXAMPLE = `>Example_Sequence GFP coding region (partial)
ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAA
GTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGC
TGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAG
CAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTAC
AAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGA
CGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGG
CATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCC`;
