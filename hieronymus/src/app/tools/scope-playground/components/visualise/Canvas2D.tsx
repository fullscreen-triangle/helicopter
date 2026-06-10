'use client';

import React, { useEffect, useRef } from 'react';
import type { ScopeResult } from '@/lib/scope-runtime/runtime';

interface Props {
  result: ScopeResult | null;
  mode: string;
  preloadImage?: { data: Float32Array; width: number; height: number } | null;
}

// ── Viridis 11-stop ───────────────────────────────────────────────────────────
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

export default function Canvas2D({ result, mode, preloadImage }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // ── No result yet — show preload or placeholder ──────────────────────────
    if (!result) {
      if ((mode === 'raw_image' || mode === 'overlay') && preloadImage) {
        const { data, width, height } = preloadImage;
        canvas.width = width; canvas.height = height;
        drawRaw(ctx, data, width, height);
      } else {
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#3a3a3a';
        ctx.font = '12px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Run a program to see this view.', canvas.width / 2, canvas.height / 2);
      }
      return;
    }

    const { visualData } = result;
    const W = visualData.width;
    const H = visualData.height;
    canvas.width = W; canvas.height = H;

    switch (mode) {
      case 'raw_image':
        drawRaw(ctx, visualData.rawImage, W, H);
        break;

      case 'scale_field':
        if (visualData.scaleField) drawHeatmap(ctx, visualData.scaleField, W, H);
        else drawRaw(ctx, visualData.rawImage, W, H);
        break;

      case 'segmentation':
        // 1. raw image as base
        drawRaw(ctx, visualData.rawImage, W, H);
        // 2. nucleus overlays (label 1 = cyan, label 2 = magenta)
        if (visualData.segmentationMask) drawMaskOverlay(ctx, visualData.segmentationMask, W, H);
        // 3. contours on top
        if (visualData.segmentationContour?.length) drawContour(ctx, visualData.segmentationContour);
        break;

      case 'distance_map':
        if (visualData.distanceMap) drawHeatmap(ctx, visualData.distanceMap, W, H);
        else drawRaw(ctx, visualData.rawImage, W, H);
        break;

      case 'geodesic':
        drawRaw(ctx, visualData.rawImage, W, H);
        if (visualData.segmentationContour?.length) drawContour(ctx, visualData.segmentationContour);
        if (visualData.geodesicPath?.length) drawGeodesic(ctx, visualData.geodesicPath);
        break;

      case 'overlay':
        drawRaw(ctx, visualData.rawImage, W, H);
        if (visualData.scaleField) drawScaleFieldOverlay(ctx, visualData.scaleField, W, H);
        if (visualData.segmentationContour?.length) drawContour(ctx, visualData.segmentationContour);
        if (visualData.geodesicPath?.length) drawGeodesic(ctx, visualData.geodesicPath);
        break;

      default:
        drawRaw(ctx, visualData.rawImage, W, H);
    }
  }, [result, mode, preloadImage]);

  return (
    <div className="flex flex-col items-center gap-2">
      <canvas
        ref={canvasRef}
        width={256}
        height={256}
        className="border border-[#3a3a3a] rounded"
        style={{ imageRendering: 'pixelated', maxWidth: '100%', maxHeight: 360, width: '100%' }}
      />
      <div className="text-[#555] text-xs">{modeLabel(mode)}</div>
    </div>
  );
}

// ── Draw raw grayscale image ──────────────────────────────────────────────────
function drawRaw(ctx: CanvasRenderingContext2D, raw: Float32Array | undefined | null, W: number, H: number) {
  if (!raw || raw.length === 0) {
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, W, H);
    return;
  }
  const id = ctx.createImageData(W, H);
  const n = Math.min(raw.length, W * H);
  for (let i = 0; i < n; i++) {
    const v = Math.round(Math.max(0, Math.min(1, raw[i])) * 255);
    id.data[i * 4]     = v;
    id.data[i * 4 + 1] = v;
    id.data[i * 4 + 2] = v;
    id.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

// ── Draw viridis heatmap ──────────────────────────────────────────────────────
function drawHeatmap(ctx: CanvasRenderingContext2D, field: Float32Array, W: number, H: number) {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < field.length; i++) {
    if (isFinite(field[i])) { if (field[i] < min) min = field[i]; if (field[i] > max) max = field[i]; }
  }
  const range = (max - min) || 1;
  const id = ctx.createImageData(W, H);
  const n = Math.min(field.length, W * H);
  for (let i = 0; i < n; i++) {
    const t = (field[i] - min) / range;
    const [r, g, b] = viridis(isFinite(t) ? t : 0);
    id.data[i * 4]     = r;
    id.data[i * 4 + 1] = g;
    id.data[i * 4 + 2] = b;
    id.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

// ── Draw scale field as transparent viridis overlay on existing canvas ────────
function drawScaleFieldOverlay(ctx: CanvasRenderingContext2D, field: Float32Array, W: number, H: number) {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < field.length; i++) {
    if (isFinite(field[i])) { if (field[i] < min) min = field[i]; if (field[i] > max) max = field[i]; }
  }
  const range = (max - min) || 1;
  const id = ctx.createImageData(W, H);
  const n = Math.min(field.length, W * H);
  for (let i = 0; i < n; i++) {
    const t = (field[i] - min) / range;
    const [r, g, b] = viridis(isFinite(t) ? t : 0);
    id.data[i * 4]     = r;
    id.data[i * 4 + 1] = g;
    id.data[i * 4 + 2] = b;
    id.data[i * 4 + 3] = 80; // semi-transparent overlay
  }
  // Use putImageData into an offscreen canvas, then drawImage with globalAlpha
  const off = new OffscreenCanvas(W, H);
  const octx = off.getContext('2d')!;
  octx.putImageData(id, 0, 0);
  ctx.save();
  ctx.globalAlpha = 0.5;
  ctx.drawImage(off, 0, 0);
  ctx.restore();
}

// ── Draw nucleus mask overlay (label 1=cyan, label 2=magenta) ────────────────
// The mask is a Uint8Array where 1 = nucleus_a, 2 = nucleus_b, 0 = background.
// We composite over the existing raw image using an offscreen canvas.
function drawMaskOverlay(ctx: CanvasRenderingContext2D, mask: Uint8Array, W: number, H: number) {
  const id = ctx.createImageData(W, H);
  const n = Math.min(mask.length, W * H);
  for (let i = 0; i < n; i++) {
    if (mask[i] === 1) {
      id.data[i * 4]     = 78;   // cyan  #4ec9b0
      id.data[i * 4 + 1] = 201;
      id.data[i * 4 + 2] = 176;
      id.data[i * 4 + 3] = 120;  // semi-transparent
    } else if (mask[i] === 2) {
      id.data[i * 4]     = 197;  // magenta  #c586c0
      id.data[i * 4 + 1] = 134;
      id.data[i * 4 + 2] = 192;
      id.data[i * 4 + 3] = 120;
    }
    // 0 stays transparent
  }
  const off = new OffscreenCanvas(W, H);
  const octx = off.getContext('2d')!;
  octx.putImageData(id, 0, 0);
  ctx.drawImage(off, 0, 0);
}

// ── Draw segmentation contours ────────────────────────────────────────────────
function drawContour(ctx: CanvasRenderingContext2D, contour: Array<[number, number]>) {
  if (!contour.length) return;
  // Draw as individual pixels for precise contour (no path smoothing artifacts)
  ctx.fillStyle = '#4ec9b0';
  const step = Math.max(1, Math.floor(contour.length / 2000)); // limit to ~2k pixels
  for (let i = 0; i < contour.length; i += step) {
    ctx.fillRect(contour[i][0], contour[i][1], 1, 1);
  }
}

// ── Draw geodesic path ────────────────────────────────────────────────────────
function drawGeodesic(ctx: CanvasRenderingContext2D, path: Array<[number, number]>) {
  if (!path.length) return;
  ctx.save();
  ctx.strokeStyle = '#ffd700';
  ctx.lineWidth = 2;
  ctx.shadowColor = '#ffd700';
  ctx.shadowBlur = 5;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(path[0][0], path[0][1]);
  for (let i = 1; i < path.length; i++) ctx.lineTo(path[i][0], path[i][1]);
  ctx.stroke();
  ctx.shadowBlur = 0;
  // Endpoint dots
  for (const pt of [path[0], path[path.length - 1]]) {
    ctx.fillStyle = '#ffd700';
    ctx.beginPath();
    ctx.arc(pt[0], pt[1], 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function modeLabel(mode: string) {
  const labels: Record<string, string> = {
    raw_image:    'Raw fluorescence image',
    scale_field:  'Scale field α(x,y) — viridis heatmap',
    segmentation: 'Segmentation — cyan=nucleus_a  magenta=nucleus_b',
    distance_map: 'Fast-marching distance T(x,y)',
    geodesic:     'Geodesic path (gold) — shortest α-weighted route',
    overlay:      'Overlay: raw + scale field + contours + path',
  };
  return labels[mode] ?? mode;
}
