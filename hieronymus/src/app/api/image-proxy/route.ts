// Server-side image loader: local public/datasets/ first, remote fallback, then synthetic.
// Returns { width, height, data: number[], synthetic?, local? }

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

// ── Local dataset registry ────────────────────────────────────────────────────
// Maps dataset → image → relative path under public/datasets/
const LOCAL_PATHS: Record<string, Record<string, string>> = {
  BBBC007: {
    'A9 p10d.tif': 'BBBC007_v1_images/BBBC007_v1_images/A9/A9 p10d.tif',
    'A9 p10f.tif': 'BBBC007_v1_images/BBBC007_v1_images/A9/A9 p10f.tif',
    'A9 p9d.tif':  'BBBC007_v1_images/BBBC007_v1_images/A9/A9 p9d.tif',
    'A9 p9f.tif':  'BBBC007_v1_images/BBBC007_v1_images/A9/A9 p9f.tif',
    'A9 p7d.tif':  'BBBC007_v1_images/BBBC007_v1_images/A9/A9 p7d.tif',
    'A9 p7f.tif':  'BBBC007_v1_images/BBBC007_v1_images/A9/A9 p7f.tif',
    'A9 p5d.tif':  'BBBC007_v1_images/BBBC007_v1_images/A9/A9 p5d.tif',
    'A9 p5f.tif':  'BBBC007_v1_images/BBBC007_v1_images/A9/A9 p5f.tif',
    // f113 wellplate images (two-channel: d0=DAPI, d1=GFP)
    'AS_09125_040701150004_A02f00d0.tif': 'BBBC007_v1_images/BBBC007_v1_images/f113/AS_09125_040701150004_A02f00d0.tif',
    'AS_09125_040701150004_A02f00d1.tif': 'BBBC007_v1_images/BBBC007_v1_images/f113/AS_09125_040701150004_A02f00d1.tif',
    // f96 (17) images
    '17P1_POS0006_D_1UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f96 (17)/17P1_POS0006_D_1UL.tif',
    '17P1_POS0006_F_2UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f96 (17)/17P1_POS0006_F_2UL.tif',
    '17P1_POS0007_D_1UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f96 (17)/17P1_POS0007_D_1UL.tif',
    '17P1_POS0007_F_2UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f96 (17)/17P1_POS0007_F_2UL.tif',
    '17P1_POS0011_D_1UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f96 (17)/17P1_POS0011_D_1UL.tif',
    '17P1_POS0011_F_2UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f96 (17)/17P1_POS0011_F_2UL.tif',
    '17P1_POS0013_D_1UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f96 (17)/17P1_POS0013_D_1UL.tif',
    '17P1_POS0013_F_2UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f96 (17)/17P1_POS0013_F_2UL.tif',
    '17P1_POS0014_D_1UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f96 (17)/17P1_POS0014_D_1UL.tif',
    '17P1_POS0014_F_2UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f96 (17)/17P1_POS0014_F_2UL.tif',
    // f9620 images
    '20P1_POS0002_D_1UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f9620/20P1_POS0002_D_1UL.tif',
    '20P1_POS0002_F_2UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f9620/20P1_POS0002_F_2UL.tif',
    '20P1_POS0005_D_1UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f9620/20P1_POS0005_D_1UL.tif',
    '20P1_POS0005_F_2UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f9620/20P1_POS0005_F_2UL.tif',
    '20P1_POS0007_D_1UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f9620/20P1_POS0007_D_1UL.tif',
    '20P1_POS0007_F_2UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f9620/20P1_POS0007_F_2UL.tif',
    '20P1_POS0008_D_1UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f9620/20P1_POS0008_D_1UL.tif',
    '20P1_POS0008_F_2UL.tif': 'BBBC007_v1_images/BBBC007_v1_images/f9620/20P1_POS0008_F_2UL.tif',
  },
  AICS: {
    'AICS-24_515.ome.tif': 'AICS-24-part06/2017_10_24_Myosin/AICS-24/AICS-24_515.ome.tif',
  },
  AllenCell: {
    '3500001004_100X.ome.tiff': '3500001004_100X_20170623_5-Scene-1-P24-E06.ome.tiff',
  },
};

// ── Remote dataset URLs (tried if local not found) ────────────────────────────
const REMOTE_URLS: Record<string, Record<string, string>> = {
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

// Public datasets directory (resolved at request time so it works in any env)
function datasetsDir(): string {
  // In Next.js the cwd at runtime is the project root
  return path.join(process.cwd(), 'public', 'datasets');
}

export async function GET(req: NextRequest) {
  const { searchParams } = req.nextUrl;
  const db      = searchParams.get('db')      ?? '';
  const dataset = searchParams.get('dataset') ?? '';
  const image   = searchParams.get('image')   ?? '';

  if (!dataset || !image) {
    return NextResponse.json({ error: 'Missing dataset or image param' }, { status: 400 });
  }

  // 1. Try local file
  const localRel = LOCAL_PATHS[dataset]?.[image];
  if (localRel) {
    const localAbs = path.join(datasetsDir(), localRel);
    try {
      const buf = fs.readFileSync(localAbs);
      const payload = decodeTiff(buf.buffer as ArrayBuffer);
      if (payload) {
        return NextResponse.json({ ...payload, local: true, dataset, image });
      }
    } catch {
      // file missing or unreadable — fall through
    }
  }

  // 2. Try remote URL
  const url = REMOTE_URLS[dataset]?.[image];
  if (url) {
    try {
      const res = await fetch(url, { headers: { Accept: '*/*' }, next: { revalidate: 3600 } });
      if (res.ok) {
        const buf = await res.arrayBuffer();
        const payload = decodeTiff(buf);
        if (payload) return NextResponse.json({ ...payload, dataset, image });
      }
    } catch {
      // network failure — fall through
    }
  }

  // 3. Synthetic fallback — real two-nucleus cell morphology
  return serveSynthetic(dataset, image);
}

// ── TIFF decoder — 8-bit and 16-bit grayscale, uncompressed strips ────────────
function decodeTiff(buf: ArrayBuffer): { width: number; height: number; data: number[] } | null {
  try {
    const view = new DataView(buf);
    const littleEndian = view.getUint16(0) === 0x4949;
    const magic = view.getUint16(2, littleEndian);
    if (magic !== 42) return null; // not TIFF or BigTIFF

    const ifdOffset = view.getUint32(4, littleEndian);
    const numEntries = view.getUint16(ifdOffset, littleEndian);

    let width = 0, height = 0, bitsPerSample = 8;
    let stripOffsets: number[] = [], stripByteCounts: number[] = [];

    for (let i = 0; i < numEntries; i++) {
      const base = ifdOffset + 2 + i * 12;
      const tag  = view.getUint16(base, littleEndian);
      const type = view.getUint16(base + 2, littleEndian);
      const count = view.getUint32(base + 4, littleEndian);
      const valOff = base + 8;

      const readShort = (off: number) => view.getUint16(off, littleEndian);
      const readLong  = (off: number) => view.getUint32(off, littleEndian);
      const readVal   = (off: number) => type === 3 ? readShort(off) : readLong(off);

      switch (tag) {
        case 256: width          = readVal(valOff); break;
        case 257: height         = readVal(valOff); break;
        case 258: bitsPerSample  = readVal(valOff); break;
        case 273: // StripOffsets
          if (count === 1) {
            stripOffsets = [readVal(valOff)];
          } else {
            const arrOff = readLong(valOff);
            for (let k = 0; k < count; k++) stripOffsets.push(readLong(arrOff + k * 4));
          }
          break;
        case 279: // StripByteCounts
          if (count === 1) {
            stripByteCounts = [readVal(valOff)];
          } else {
            const arrOff = readLong(valOff);
            for (let k = 0; k < count; k++) stripByteCounts.push(readLong(arrOff + k * 4));
          }
          break;
      }
    }

    if (!width || !height) return null;

    // Crop to max 512×512 for browser payload size
    const outW = Math.min(width,  512);
    const outH = Math.min(height, 512);
    const stepX = Math.max(1, Math.floor(width  / outW));
    const stepY = Math.max(1, Math.floor(height / outH));
    const maxVal = bitsPerSample === 16 ? 65535 : 255;

    // Decode all pixels into a flat array first
    const full = new Float32Array(width * height);
    let pixelIdx = 0;
    for (let s = 0; s < stripOffsets.length && pixelIdx < full.length; s++) {
      let off = stripOffsets[s];
      const end = off + (stripByteCounts[s] ?? (width * height - pixelIdx) * (bitsPerSample / 8));
      while (off < end && pixelIdx < full.length) {
        const raw = bitsPerSample === 16 ? view.getUint16(off, littleEndian) : view.getUint8(off);
        full[pixelIdx++] = raw / maxVal;
        off += bitsPerSample === 16 ? 2 : 1;
      }
    }

    // Subsample to outW×outH
    const data: number[] = new Array(outW * outH);
    for (let row = 0; row < outH; row++) {
      for (let col = 0; col < outW; col++) {
        data[row * outW + col] = full[Math.min(row * stepY, height - 1) * width + Math.min(col * stepX, width - 1)];
      }
    }

    return { width: outW, height: outH, data };
  } catch {
    return null;
  }
}

// ── Synthetic fallback — realistic two-nucleus fluorescence morphology ─────────
function serveSynthetic(dataset: string, image: string) {
  const W = 256, H = 256;
  const data: number[] = new Array(W * H).fill(0);

  // Three nuclei at different positions/sizes to look like real DAPI staining
  const nuclei = [
    { cx: 72,  cy: 110, rx: 28, ry: 24, brightness: 0.92 },
    { cx: 180, cy: 118, rx: 26, ry: 28, brightness: 0.88 },
    { cx: 128, cy: 195, rx: 22, ry: 20, brightness: 0.78 },
  ];

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let v = 0.03 + (Math.sin(x * 0.08) * Math.cos(y * 0.09) + 1) * 0.01; // autofluorescence texture
      for (const n of nuclei) {
        const d2 = ((x - n.cx) / n.rx) ** 2 + ((y - n.cy) / n.ry) ** 2;
        // Nuclear shell — bright ring with darker interior (Hoechst/DAPI morphology)
        const shell = Math.exp(-((Math.sqrt(d2) - 0.7) ** 2) / 0.08);
        const interior = Math.exp(-d2 / 0.6) * 0.4;
        v += n.brightness * (shell + interior);
      }
      data[y * W + x] = Math.max(0, Math.min(1, v));
    }
  }

  return NextResponse.json({ width: W, height: H, data, synthetic: true, dataset, image });
}
