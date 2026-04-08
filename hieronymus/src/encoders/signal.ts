import type { EncodedData } from './types';

/**
 * Signal / Time Series Encoder
 *
 * Parses CSV/JSON time series data, computes windowed FFT (Short-Time Fourier Transform),
 * and builds a 256x256 spectrogram image where:
 *   x-axis = frequency bin
 *   y-axis = windowed time
 *   intensity = magnitude
 */

function parseTimeSeries(text: string): number[] {
  const trimmed = text.trim();
  const values: number[] = [];

  // Try JSON first
  try {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) {
      for (const item of parsed) {
        if (typeof item === 'number') {
          values.push(item);
        } else if (typeof item === 'object' && item !== null) {
          const v = item.value ?? item.v ?? item.y ?? item.amplitude ?? 0;
          values.push(Number(v));
        } else if (Array.isArray(item) && item.length >= 2) {
          values.push(Number(item[1])); // [time, value]
        }
      }
      if (values.length > 0) return values;
    }
  } catch {
    // Not JSON
  }

  // CSV: "time,value" or single column
  const lines = trimmed.split(/\r?\n/).filter((l) => l.trim() && !l.startsWith('#'));
  for (const line of lines) {
    const parts = line.split(/[,\t\s]+/).map(Number);
    if (parts.length >= 2 && !isNaN(parts[1])) {
      values.push(parts[1]); // second column = value
    } else if (parts.length === 1 && !isNaN(parts[0])) {
      values.push(parts[0]);
    }
  }

  return values;
}

/**
 * Simple radix-2 DFT for small N. Not a true FFT but works for arbitrary sizes.
 * Returns magnitudes for each frequency bin.
 */
function dft(signal: number[]): number[] {
  const N = signal.length;
  const magnitudes = new Array<number>(Math.floor(N / 2));

  for (let k = 0; k < magnitudes.length; k++) {
    let re = 0;
    let im = 0;
    for (let n = 0; n < N; n++) {
      const angle = (-2 * Math.PI * k * n) / N;
      re += signal[n] * Math.cos(angle);
      im += signal[n] * Math.sin(angle);
    }
    magnitudes[k] = Math.sqrt(re * re + im * im) / N;
  }

  return magnitudes;
}

/**
 * Hann window function
 */
function hannWindow(data: number[]): number[] {
  const N = data.length;
  return data.map((v, i) => v * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1))));
}

export function encodeSignal(text: string): EncodedData {
  const t0 = performance.now();
  const values = parseTimeSeries(text);

  if (values.length < 4) {
    throw new Error('Not enough data points. Need at least 4 numeric values.');
  }

  const SIZE = 256;
  const data = new Uint8ClampedArray(SIZE * SIZE * 4);

  // Compute short-time Fourier transform (STFT)
  const numWindows = SIZE; // number of time windows
  const windowSize = Math.min(Math.max(16, Math.floor(values.length / 4)), 256);
  const hopSize = Math.max(1, Math.floor((values.length - windowSize) / (numWindows - 1)));
  const numFreqBins = Math.floor(windowSize / 2);

  // Pre-compute all STFT windows
  const spectrogram: number[][] = [];
  let globalMax = 0;

  for (let w = 0; w < numWindows; w++) {
    const start = Math.min(w * hopSize, values.length - windowSize);
    if (start < 0) {
      spectrogram.push(new Array(numFreqBins).fill(0));
      continue;
    }

    const segment = values.slice(start, start + windowSize);
    const windowed = hannWindow(segment);
    const mags = dft(windowed);

    // Ensure we have exactly numFreqBins
    while (mags.length < numFreqBins) mags.push(0);
    const trimmed = mags.slice(0, numFreqBins);

    for (const m of trimmed) {
      if (m > globalMax) globalMax = m;
    }

    spectrogram.push(trimmed);
  }

  // Normalize and map to image
  if (globalMax === 0) globalMax = 1;

  for (let py = 0; py < SIZE; py++) {
    // Map y to time window index
    const wIdx = Math.floor((py / SIZE) * spectrogram.length);
    const row = spectrogram[Math.min(wIdx, spectrogram.length - 1)];

    for (let px = 0; px < SIZE; px++) {
      // Map x to frequency bin
      const fIdx = Math.floor((px / SIZE) * row.length);
      const mag = row[Math.min(fIdx, row.length - 1)] / globalMax;

      // Log-scale for better visualization
      const logMag = Math.log1p(mag * 10) / Math.log1p(10);

      const idx = (py * SIZE + px) * 4;
      const v = Math.round(Math.min(logMag, 1.0) * 255);
      data[idx + 0] = v;
      data[idx + 1] = v;
      data[idx + 2] = v;
      data[idx + 3] = 255;
    }
  }

  const imageData = new ImageData(data, SIZE, SIZE);
  const encodingTime = performance.now() - t0;

  return {
    imageData,
    metadata: {
      domain: 'signal',
      originalSize: values.length,
      encodingTime,
    },
  };
}

/** Example time series data for demonstration */
export const SIGNAL_EXAMPLE = (() => {
  // Generate a signal with multiple frequency components
  const lines: string[] = ['# Time series: 3 sine waves + noise'];
  const N = 512;
  for (let i = 0; i < N; i++) {
    const t = i / N;
    const v =
      0.6 * Math.sin(2 * Math.PI * 5 * t) +
      0.3 * Math.sin(2 * Math.PI * 20 * t) +
      0.15 * Math.sin(2 * Math.PI * 50 * t) +
      0.05 * (Math.random() - 0.5);
    lines.push(`${t.toFixed(4)},${v.toFixed(6)}`);
  }
  return lines.join('\n');
})();
