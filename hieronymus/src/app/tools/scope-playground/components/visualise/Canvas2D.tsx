'use client';

import React, { useEffect, useRef } from 'react';
import type { ScopeResult } from '@/lib/scope-runtime/runtime';

interface Props {
  result: ScopeResult | null;
  mode: string;
}

const VIRIDIS: [number, number, number][] = [
  [68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141],
  [41, 120, 142], [32, 144, 140], [34, 167, 132], [68, 190, 112],
  [121, 209, 81], [189, 222, 38], [253, 231, 37],
];

function viridis(t: number): [number, number, number] {
  const i = Math.min(VIRIDIS.length - 2, Math.max(0, Math.floor(t * (VIRIDIS.length - 1))));
  const f = t * (VIRIDIS.length - 1) - i;
  const a = VIRIDIS[i], b = VIRIDIS[i + 1];
  return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
}

export default function Canvas2D({ result, mode }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (!result) {
      ctx.fillStyle = '#1e1e1e';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#555';
      ctx.font = '13px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('No result — run a program first.', canvas.width / 2, canvas.height / 2);
      return;
    }

    const { visualData } = result;
    const W = visualData.width, H = visualData.height;
    canvas.width = W; canvas.height = H;

    if (mode === 'raw_image' || mode === 'overlay' || !visualData.rawImage) {
      drawRaw(ctx, visualData.rawImage, W, H);
    }

    if (mode === 'scale_field' && visualData.scaleField) {
      drawHeatmap(ctx, visualData.scaleField, W, H);
    }

    if (mode === 'segmentation' && visualData.rawImage) {
      drawRaw(ctx, visualData.rawImage, W, H);
      if (visualData.segmentationMask) drawMask(ctx, visualData.segmentationMask, W, H);
      if (visualData.segmentationContour) drawContour(ctx, visualData.segmentationContour, W, H);
    }

    if (mode === 'distance_map' && visualData.distanceMap) {
      drawHeatmap(ctx, visualData.distanceMap, W, H);
    }

    if (mode === 'geodesic') {
      if (visualData.rawImage) drawRaw(ctx, visualData.rawImage, W, H);
      if (visualData.segmentationContour) drawContour(ctx, visualData.segmentationContour, W, H);
      if (visualData.geodesicPath) drawGeodesic(ctx, visualData.geodesicPath);
    }
  }, [result, mode]);

  return (
    <div className="flex flex-col items-center gap-2">
      <canvas
        ref={canvasRef}
        width={256}
        height={256}
        className="border border-[#3a3a3a] rounded"
        style={{ imageRendering: 'pixelated', maxWidth: '100%', maxHeight: 320 }}
      />
      <div className="text-[#555] text-xs">{modeLabel(mode)}</div>
    </div>
  );
}

function drawRaw(ctx: CanvasRenderingContext2D, raw: Float32Array | undefined | null, W: number, H: number) {
  const imageData = ctx.createImageData(W, H);
  if (raw) {
    for (let i = 0; i < W * H; i++) {
      const v = Math.round(raw[i] * 255);
      imageData.data[i * 4]     = v;
      imageData.data[i * 4 + 1] = v;
      imageData.data[i * 4 + 2] = v;
      imageData.data[i * 4 + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

function drawHeatmap(ctx: CanvasRenderingContext2D, field: Float32Array, W: number, H: number) {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < field.length; i++) { if (field[i] < min) min = field[i]; if (field[i] > max) max = field[i]; }
  const range = max - min || 1;
  const imageData = ctx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const t = (field[i] - min) / range;
    const [r, g, b] = viridis(t);
    imageData.data[i * 4]     = r;
    imageData.data[i * 4 + 1] = g;
    imageData.data[i * 4 + 2] = b;
    imageData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

function drawMask(ctx: CanvasRenderingContext2D, mask: Uint8Array | Float32Array, W: number, H: number) {
  const imageData = ctx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    if (mask[i] > 0.5) {
      imageData.data[i * 4]     = 78;
      imageData.data[i * 4 + 1] = 201;
      imageData.data[i * 4 + 2] = 176;
      imageData.data[i * 4 + 3] = 80;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

function drawContour(ctx: CanvasRenderingContext2D, contour: Array<[number, number]>, W: number, H: number) {
  if (!contour.length) return;
  ctx.strokeStyle = '#4ec9b0';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(contour[0][0], contour[0][1]);
  for (let i = 1; i < contour.length; i++) ctx.lineTo(contour[i][0], contour[i][1]);
  ctx.closePath();
  ctx.stroke();
}

function drawGeodesic(ctx: CanvasRenderingContext2D, path: Array<[number, number]>) {
  if (!path.length) return;
  ctx.strokeStyle = '#ffd700';
  ctx.lineWidth = 2;
  ctx.shadowColor = '#ffd700';
  ctx.shadowBlur = 4;
  ctx.beginPath();
  ctx.moveTo(path[0][0], path[0][1]);
  for (let i = 1; i < path.length; i++) ctx.lineTo(path[i][0], path[i][1]);
  ctx.stroke();
  ctx.shadowBlur = 0;
  // Draw endpoint circles
  for (const pt of [path[0], path[path.length - 1]]) {
    ctx.fillStyle = '#ffd700';
    ctx.beginPath();
    ctx.arc(pt[0], pt[1], 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function modeLabel(mode: string) {
  const labels: Record<string, string> = {
    raw_image: 'Raw fluorescence image',
    scale_field: 'Scale field α(x,y) — viridis',
    segmentation: 'Segmentation mask + contour',
    distance_map: 'Fast-marching distance T(x,y)',
    geodesic: 'Geodesic path (gold) over raw image',
    overlay: 'Overlay',
  };
  return labels[mode] ?? mode;
}
