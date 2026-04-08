import type { EncodedData } from './types';

/**
 * Molecular / Spectral Encoder
 *
 * Parses frequency+amplitude data from CSV or JSON text and builds
 * a 256x256 spectral image where:
 *   x-axis = frequency (normalized to [0,1])
 *   y-axis = phase (0 to 2*pi)
 *   intensity = amplitude
 *
 * Each frequency component becomes a vertical Gaussian stripe.
 */

interface FreqComponent {
  freq: number;
  amp: number;
}

function parseFrequencies(text: string): FreqComponent[] {
  const trimmed = text.trim();
  const components: FreqComponent[] = [];

  // Try JSON first
  try {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) {
      for (const item of parsed) {
        if (typeof item === 'object' && item !== null) {
          const freq = item.freq ?? item.frequency ?? item.f ?? 0;
          const amp = item.amp ?? item.amplitude ?? item.a ?? item.intensity ?? 1;
          components.push({ freq: Number(freq), amp: Number(amp) });
        } else if (Array.isArray(item) && item.length >= 2) {
          components.push({ freq: Number(item[0]), amp: Number(item[1]) });
        }
      }
      if (components.length > 0) return components;
    }
  } catch {
    // Not JSON, try CSV
  }

  // CSV: each line is "frequency,amplitude" or just "frequency"
  const lines = trimmed.split(/\r?\n/).filter((l) => l.trim() && !l.startsWith('#'));
  for (const line of lines) {
    const parts = line.split(/[,\t\s]+/).map(Number);
    if (parts.length >= 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
      components.push({ freq: parts[0], amp: parts[1] });
    } else if (parts.length === 1 && !isNaN(parts[0])) {
      components.push({ freq: parts[0], amp: 1.0 });
    }
  }

  return components;
}

export function encodeMolecular(text: string): EncodedData {
  const t0 = performance.now();
  const components = parseFrequencies(text);

  if (components.length === 0) {
    throw new Error('No valid frequency data found. Expected CSV (freq,amp) or JSON array.');
  }

  const SIZE = 256;
  const data = new Uint8ClampedArray(SIZE * SIZE * 4);

  // Normalize frequencies and amplitudes
  const freqs = components.map((c) => c.freq);
  const amps = components.map((c) => c.amp);
  const fMin = Math.min(...freqs);
  const fMax = Math.max(...freqs);
  const fRange = fMax - fMin || 1;
  const aMax = Math.max(...amps, 1e-10);

  const normFreqs = freqs.map((f) => (f - fMin) / fRange);
  const normAmps = amps.map((a) => a / aMax);

  // Gaussian stripe width (in normalized coords)
  const sigma = Math.max(0.5 / components.length, 0.01);

  for (let y = 0; y < SIZE; y++) {
    const phase = (y / SIZE) * 2 * Math.PI;
    for (let x = 0; x < SIZE; x++) {
      const xNorm = x / SIZE;

      let intensity = 0;
      for (let k = 0; k < components.length; k++) {
        const dx = xNorm - normFreqs[k];
        const gaussian = Math.exp(-(dx * dx) / (2 * sigma * sigma));
        // Add phase modulation for visual richness
        const phaseMod = 0.5 + 0.5 * Math.sin(phase * (k + 1));
        intensity += normAmps[k] * gaussian * phaseMod;
      }

      intensity = Math.min(intensity, 1.0);
      const idx = (y * SIZE + x) * 4;
      const v = Math.round(intensity * 255);
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
      domain: 'molecular',
      originalSize: text.length,
      encodingTime,
    },
  };
}

/** Example frequency data for demonstration */
export const MOLECULAR_EXAMPLE = `# Raman spectral peaks (frequency cm-1, relative intensity)
250,0.12
420,0.35
520,0.78
680,0.45
785,0.92
1002,1.00
1150,0.65
1340,0.48
1450,0.72
1580,0.55
1620,0.88
1680,0.33
2850,0.42
2920,0.68
3050,0.25`;
