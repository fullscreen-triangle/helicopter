'use client';

import React, { useRef, useEffect } from 'react';
import type { MatchResult } from '@/engine/types';

interface MatchComparisonProps {
  result: MatchResult;
  previewA: string | null;
  previewB: string | null;
}

function ScoreGauge({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color =
    score > 0.8
      ? 'text-green-400'
      : score > 0.5
      ? 'text-amber-400'
      : 'text-red-400';
  const bgColor =
    score > 0.8
      ? 'bg-green-400'
      : score > 0.5
      ? 'bg-amber-400'
      : 'bg-red-400';

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-baseline">
        <span className="text-[10px] text-gray-500 uppercase tracking-widest">
          Match Score
        </span>
        <span className={`text-2xl font-bold ${color}`}>{pct}%</span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${bgColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function MetricRow({
  label,
  value,
  color = 'text-gray-300',
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="flex justify-between items-center py-1.5 border-b border-gray-800/50 last:border-0">
      <span className="text-[10px] text-gray-500 uppercase tracking-widest">
        {label}
      </span>
      <span className={`text-sm font-semibold ${color}`}>{value}</span>
    </div>
  );
}

function DualSimplexCanvas({
  imageA,
  imageB,
}: {
  imageA: { S_k: number; S_t: number; S_e: number };
  imageB: { S_k: number; S_t: number; S_e: number };
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const pad = 24;
    const by = H - pad;
    const cx = W / 2;
    const tri = [
      { x: cx, y: pad, label: 'Se', col: '#c084fc' },
      { x: pad, y: by, label: 'Sk', col: '#00d4ff' },
      { x: W - pad, y: by, label: 'St', col: '#ff6b35' },
    ];

    // Draw triangle
    ctx.beginPath();
    ctx.moveTo(tri[0].x, tri[0].y);
    ctx.lineTo(tri[1].x, tri[1].y);
    ctx.lineTo(tri[2].x, tri[2].y);
    ctx.closePath();
    ctx.strokeStyle = '#1c2535';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Grid lines
    for (let t = 0; t <= 1; t += 0.25) {
      for (let i = 0; i < 3; i++) {
        const j = (i + 1) % 3;
        const k = (i + 2) % 3;
        const px = tri[j].x * t + tri[k].x * (1 - t);
        const py = tri[j].y * t + tri[k].y * (1 - t);
        ctx.beginPath();
        ctx.moveTo(tri[i].x, tri[i].y);
        ctx.lineTo(px, py);
        ctx.strokeStyle = '#0d1117';
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }

    // Labels
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    tri.forEach((v) => {
      ctx.fillStyle = v.col;
      const offY = v.y < H / 2 ? -10 : 14;
      ctx.fillText(v.label, v.x, v.y + offY);
    });

    // Plot both points
    const toXY = (s: { S_k: number; S_t: number; S_e: number }) => {
      const total = s.S_k + s.S_t + s.S_e || 1;
      const sk = s.S_k / total;
      const st = s.S_t / total;
      const se = s.S_e / total;
      return {
        x: se * tri[0].x + sk * tri[1].x + st * tri[2].x,
        y: se * tri[0].y + sk * tri[1].y + st * tri[2].y,
      };
    };

    const ptA = toXY(imageA);
    const ptB = toXY(imageB);

    // Draw line between points
    ctx.beginPath();
    ctx.moveTo(ptA.x, ptA.y);
    ctx.lineTo(ptB.x, ptB.y);
    ctx.strokeStyle = 'rgba(100,116,139,0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Point A (cyan)
    ctx.beginPath();
    ctx.arc(ptA.x, ptA.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#00d4ff';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Point B (amber)
    ctx.beginPath();
    ctx.arc(ptB.x, ptB.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#ff6b35';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Legend
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#00d4ff';
    ctx.fillText('A', 8, H - 18);
    ctx.fillStyle = '#ff6b35';
    ctx.fillText('B', 20, H - 18);
    ctx.fillStyle = '#64748b';
    ctx.fillText(
      `d=${Math.sqrt(
        (imageA.S_k - imageB.S_k) ** 2 +
          (imageA.S_t - imageB.S_t) ** 2 +
          (imageA.S_e - imageB.S_e) ** 2
      ).toFixed(4)}`,
      34,
      H - 18
    );
  }, [imageA, imageB]);

  return (
    <canvas
      ref={canvasRef}
      width={220}
      height={220}
      className="w-full bg-gray-950 rounded"
      style={{ aspectRatio: '1' }}
    />
  );
}

function EntropyBar({
  label,
  valueA,
  valueB,
  colorA,
  colorB,
}: {
  label: string;
  valueA: number;
  valueB: number;
  colorA: string;
  colorB: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px]">
        <span className="text-gray-500 uppercase tracking-wider">{label}</span>
        <span className="text-gray-400">
          <span className={colorA}>{valueA.toFixed(4)}</span>
          {' / '}
          <span className={colorB}>{valueB.toFixed(4)}</span>
        </span>
      </div>
      <div className="relative h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`absolute h-full rounded-full opacity-70 ${colorA.replace('text-', 'bg-')}`}
          style={{ width: `${Math.min(valueA * 100, 100)}%` }}
        />
        <div
          className={`absolute h-full rounded-full opacity-40 ${colorB.replace('text-', 'bg-')}`}
          style={{ width: `${Math.min(valueB * 100, 100)}%` }}
        />
      </div>
    </div>
  );
}

export default function MatchComparison({
  result,
  previewA,
  previewB,
}: MatchComparisonProps) {
  return (
    <div className="space-y-6">
      {/* Side-by-side images */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <div className="text-[10px] text-cyan-400 uppercase tracking-widest">
            Image A
          </div>
          <div className="bg-gray-950 rounded-lg border border-gray-800/50 flex items-center justify-center min-h-[160px] overflow-hidden">
            {previewA ? (
              <img
                src={previewA}
                alt="Image A"
                className="max-w-full max-h-48 object-contain"
                style={{ imageRendering: 'pixelated' }}
              />
            ) : (
              <span className="text-gray-700 text-sm">No image</span>
            )}
          </div>
        </div>
        <div className="space-y-2">
          <div className="text-[10px] text-amber-400 uppercase tracking-widest">
            Image B
          </div>
          <div className="bg-gray-950 rounded-lg border border-gray-800/50 flex items-center justify-center min-h-[160px] overflow-hidden">
            {previewB ? (
              <img
                src={previewB}
                alt="Image B"
                className="max-w-full max-h-48 object-contain"
                style={{ imageRendering: 'pixelated' }}
              />
            ) : (
              <span className="text-gray-700 text-sm">No image</span>
            )}
          </div>
        </div>
      </div>

      {/* Score */}
      <ScoreGauge score={result.score} />

      {/* Metrics */}
      <div className="space-y-0">
        <MetricRow
          label="S-Distance"
          value={result.S_distance.toFixed(6)}
          color="text-amber-400"
        />
        <MetricRow
          label="Visibility"
          value={`${(result.visibility * 100).toFixed(1)}%`}
          color="text-green-400"
        />
        <MetricRow
          label="Circuit Depth Delta"
          value={result.circuits.toFixed(2)}
          color="text-cyan-400"
        />
        <MetricRow
          label="Elapsed"
          value={`${result.elapsed_ms.toFixed(1)} ms`}
          color="text-gray-400"
        />
      </div>

      {/* Per-image S-entropy comparison */}
      <div className="space-y-3">
        <h4 className="text-[10px] text-gray-500 uppercase tracking-widest">
          S-Entropy Comparison (A / B)
        </h4>
        <EntropyBar
          label="Sk (kinetic)"
          valueA={result.imageA.S_k}
          valueB={result.imageB.S_k}
          colorA="text-cyan-400"
          colorB="text-amber-400"
        />
        <EntropyBar
          label="St (thermal)"
          valueA={result.imageA.S_t}
          valueB={result.imageB.S_t}
          colorA="text-cyan-400"
          colorB="text-amber-400"
        />
        <EntropyBar
          label="Se (electronic)"
          valueA={result.imageA.S_e}
          valueB={result.imageB.S_e}
          colorA="text-cyan-400"
          colorB="text-amber-400"
        />
      </div>

      {/* Dual simplex */}
      <div className="space-y-2">
        <h4 className="text-[10px] text-gray-500 uppercase tracking-widest">
          Entropy Simplex
        </h4>
        <DualSimplexCanvas imageA={result.imageA} imageB={result.imageB} />
      </div>
    </div>
  );
}
