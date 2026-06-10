// Server-side TIFF/PNG fetch + decode → Float32Array JSON
// Handles CORS for BBBC, AllenCell, OpenCell, IDR

import { NextRequest, NextResponse } from 'next/server';

// Known dataset → URL map (BBBC images served from IDR / S3 mirrors)
const DATASET_URLS: Record<string, Record<string, string>> = {
  BBBC039: {
    'SiR_Actin_001.tif': 'https://data.broadinstitute.org/bbbc/BBBC039/images/SiR_Actin_001.tif',
    'SiR_Actin_002.tif': 'https://data.broadinstitute.org/bbbc/BBBC039/images/SiR_Actin_002.tif',
    'SiR_Actin_003.tif': 'https://data.broadinstitute.org/bbbc/BBBC039/images/SiR_Actin_003.tif',
  },
  BBBC006: {
    'v1_001.tif': 'https://data.broadinstitute.org/bbbc/BBBC006/images/v1_001.tif',
  },
  BBBC008: {
    'C57_01.tif': 'https://data.broadinstitute.org/bbbc/BBBC008/images/C57_01.tif',
  },
};

export async function GET(req: NextRequest) {
  const { searchParams } = req.nextUrl;
  const db = searchParams.get('db') ?? '';
  const dataset = searchParams.get('dataset') ?? '';
  const image = searchParams.get('image') ?? '';

  if (!db || !dataset || !image) {
    return NextResponse.json({ error: 'Missing db, dataset, or image param' }, { status: 400 });
  }

  // Resolve URL
  const url = DATASET_URLS[dataset]?.[image];
  if (!url) {
    // Fall through to synthetic image for unknown datasets
    return serveSynthetic(dataset, image);
  }

  try {
    const res = await fetch(url, { headers: { 'Accept': '*/*' }, next: { revalidate: 3600 } });
    if (!res.ok) return serveSynthetic(dataset, image);

    const buf = await res.arrayBuffer();
    const payload = decodeTiff(buf) ?? decodePng(buf);
    if (!payload) return serveSynthetic(dataset, image);
    return NextResponse.json(payload);
  } catch {
    return serveSynthetic(dataset, image);
  }
}

// Minimal TIFF decoder — handles 8-bit and 16-bit grayscale
function decodeTiff(buf: ArrayBuffer): { width: number; height: number; data: number[] } | null {
  try {
    const view = new DataView(buf);
    const littleEndian = view.getUint16(0) === 0x4949;
    const magic = view.getUint16(2, littleEndian);
    if (magic !== 42) return null;

    const ifdOffset = view.getUint32(4, littleEndian);
    const numEntries = view.getUint16(ifdOffset, littleEndian);

    let width = 0, height = 0, bitsPerSample = 8;
    let stripOffsets: number[] = [], stripByteCounts: number[] = [];

    for (let i = 0; i < numEntries; i++) {
      const base = ifdOffset + 2 + i * 12;
      const tag  = view.getUint16(base, littleEndian);
      const type = view.getUint16(base + 2, littleEndian);
      const count = view.getUint32(base + 4, littleEndian);
      const valOffset = base + 8;

      const readVal = (off: number) => type === 3
        ? view.getUint16(off, littleEndian)
        : view.getUint32(off, littleEndian);

      switch (tag) {
        case 256: width = readVal(valOffset); break;
        case 257: height = readVal(valOffset); break;
        case 258: bitsPerSample = readVal(valOffset); break;
        case 273:
          if (count === 1) {
            stripOffsets = [readVal(valOffset)];
          } else {
            const arrOff = view.getUint32(valOffset, littleEndian);
            for (let k = 0; k < count; k++)
              stripOffsets.push(view.getUint32(arrOff + k * 4, littleEndian));
          }
          break;
        case 279:
          if (count === 1) {
            stripByteCounts = [readVal(valOffset)];
          } else {
            const arrOff = view.getUint32(valOffset, littleEndian);
            for (let k = 0; k < count; k++)
              stripByteCounts.push(view.getUint32(arrOff + k * 4, littleEndian));
          }
          break;
      }
    }

    if (!width || !height) return null;

    const data: number[] = new Array(width * height);
    const maxVal = bitsPerSample === 16 ? 65535 : 255;
    let pixelIdx = 0;

    for (let s = 0; s < stripOffsets.length; s++) {
      let off = stripOffsets[s];
      const end = off + (stripByteCounts[s] ?? width * height * (bitsPerSample / 8));
      while (off < end && pixelIdx < data.length) {
        const raw = bitsPerSample === 16
          ? view.getUint16(off, littleEndian)
          : view.getUint8(off);
        data[pixelIdx++] = raw / maxVal;
        off += bitsPerSample === 16 ? 2 : 1;
      }
    }

    return { width, height, data };
  } catch {
    return null;
  }
}

function decodePng(_buf: ArrayBuffer): null {
  // PNG decode not implemented server-side without a library
  return null;
}

// Synthetic fallback: 256×256 two-Gaussian image
function serveSynthetic(dataset: string, image: string) {
  const W = 256, H = 256;
  const data: number[] = new Array(W * H).fill(0);

  // Two nucleus-like Gaussians
  const sources = [
    { cx: 80,  cy: 128, r: 25 },
    { cx: 176, cy: 128, r: 22 },
  ];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let v = 0.05; // background noise
      for (const s of sources) {
        const d2 = (x - s.cx) ** 2 + (y - s.cy) ** 2;
        v += 0.9 * Math.exp(-d2 / (2 * s.r ** 2));
      }
      // add slight Poisson noise
      v = Math.min(1, v + (Math.random() - 0.5) * 0.02);
      data[y * W + x] = Math.max(0, v);
    }
  }

  return NextResponse.json(
    { width: W, height: H, data, synthetic: true, dataset, image },
    { headers: { 'X-Scope-Synthetic': '1' } }
  );
}
