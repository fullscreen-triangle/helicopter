'use client';

import React, { useRef, useEffect } from 'react';

interface SimplexTriangleProps {
  S_k: number;
  S_t: number;
  S_e: number;
}

export default function SimplexTriangle({ S_k, S_t, S_e }: SimplexTriangleProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const historyRef = useRef<{ x: number; y: number }[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Triangle vertices (equilateral)
    const pad = 24;
    const by = H - pad;
    const cx = W / 2;
    const tri = [
      { x: cx, y: pad, label: 'Se', col: '#c084fc' }, // top
      { x: pad, y: by, label: 'Sk', col: '#00d4ff' }, // bottom-left
      { x: W - pad, y: by, label: 'St', col: '#ff6b35' }, // bottom-right
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

    // Convert barycentric (Sk, St, Se) -> cartesian
    const total = S_k + S_t + S_e || 1;
    const sk = S_k / total;
    const st = S_t / total;
    const se = S_e / total;
    const px = se * tri[0].x + sk * tri[1].x + st * tri[2].x;
    const py = se * tri[0].y + sk * tri[1].y + st * tri[2].y;

    // History trail
    const history = historyRef.current;
    history.push({ x: px, y: py });
    if (history.length > 80) history.shift();

    if (history.length > 1) {
      ctx.beginPath();
      ctx.moveTo(history[0].x, history[0].y);
      for (let i = 1; i < history.length; i++) {
        ctx.lineTo(history[i].x, history[i].y);
      }
      ctx.strokeStyle = 'rgba(100,116,139,0.4)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Current point
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Coordinates text
    ctx.fillStyle = '#64748b';
    ctx.font = '8px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(
      `Sk=${S_k.toFixed(3)} St=${S_t.toFixed(3)} Se=${S_e.toFixed(3)}`,
      8,
      H - 6
    );
  }, [S_k, S_t, S_e]);

  return (
    <div className="space-y-2">
      <h3 className="text-xs text-gray-600 uppercase tracking-widest">
        Entropy Simplex
      </h3>
      <canvas
        ref={canvasRef}
        width={220}
        height={220}
        className="w-full bg-gray-950 rounded"
        style={{ aspectRatio: '1' }}
      />
    </div>
  );
}
