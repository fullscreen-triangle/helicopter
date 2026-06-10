'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3';

// ── Types ─────────────────────────────────────────────────────────────────────
interface CellData {
  distances: number[];
  organelleSizes: number[];
  cellDensity: number[];
  intensityOverTime: Array<{ t: number; intensity: number }>;
  organellePositions: Array<{ x: number; y: number; label: string; r: number }>;
}

interface ImagePayload {
  data: Float32Array;
  width: number;
  height: number;
  synthetic?: boolean;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// Generate realistic synthetic cell data for demo
function generateCellData(seed = 1): CellData {
  const rng = (s: number) => { let x = Math.sin(s) * 10000; return x - Math.floor(x); };
  const distances = Array.from({ length: 40 }, (_, i) => 2 + rng(seed + i) * 18);
  const organelleSizes = Array.from({ length: 30 }, (_, i) => 0.2 + rng(seed + 100 + i) * 2.8);
  const cellDensity = Array.from({ length: 64 }, (_, i) => {
    const cx = (i % 8) / 8, cy = Math.floor(i / 8) / 8;
    return 0.1 + 0.9 * Math.exp(-((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / 0.08) + rng(seed + 200 + i) * 0.15;
  });
  const intensityOverTime = Array.from({ length: 60 }, (_, i) => ({
    t: i,
    intensity: 0.4 + 0.35 * Math.sin(i / 8) + 0.15 * rng(seed + 300 + i),
  }));
  const organellePositions = [
    { x: 0.55, y: 0.50, label: 'Nucleus',      r: 0.14 },
    { x: 0.35, y: 0.35, label: 'Mitochondria', r: 0.05 },
    { x: 0.70, y: 0.38, label: 'Mitochondria', r: 0.04 },
    { x: 0.42, y: 0.65, label: 'Golgi',        r: 0.06 },
    { x: 0.72, y: 0.62, label: 'Lysosome',     r: 0.04 },
    { x: 0.28, y: 0.55, label: 'Ribosome',     r: 0.03 },
    { x: 0.60, y: 0.28, label: 'ER',           r: 0.05 },
  ];
  return { distances, organelleSizes, cellDensity, intensityOverTime, organellePositions };
}

// ── Individual chart components ────────────────────────────────────────────────

function DistanceHistogram({ distances }: { distances: number[] }) {
  const ref = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!ref.current || !distances.length) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();
    const W = ref.current.clientWidth || 340, H = 220;
    const m = { top: 16, right: 16, bottom: 40, left: 44 };
    const iW = W - m.left - m.right, iH = H - m.top - m.bottom;
    svg.attr('viewBox', `0 0 ${W} ${H}`);
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
    g.append('rect').attr('width', iW).attr('height', iH).attr('fill', '#0d1117');

    const x = d3.scaleLinear().domain([0, d3.max(distances)! * 1.05]).range([0, iW]);
    const bins = d3.bin().domain(x.domain() as [number, number]).thresholds(12)(distances);
    const y = d3.scaleLinear().domain([0, d3.max(bins, d => d.length)! * 1.15]).range([iH, 0]);

    g.selectAll('.bar').data(bins).join('rect').attr('class', 'bar')
      .attr('x', d => x(d.x0!)).attr('y', d => y(d.length))
      .attr('width', d => Math.max(0, x(d.x1!) - x(d.x0!) - 1))
      .attr('height', d => iH - y(d.length))
      .attr('fill', '#569cd6').attr('opacity', 0.85);

    g.append('g').attr('transform', `translate(0,${iH})`).call(d3.axisBottom(x).ticks(6, '.1f'))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 10);
    g.append('g').call(d3.axisLeft(y).ticks(4))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 10);
    g.selectAll('.domain').attr('stroke', '#3a3a3a');

    g.append('text').attr('x', iW / 2).attr('y', iH + 34).attr('fill', '#858585')
      .attr('font-size', 10).attr('text-anchor', 'middle').text('Distance (µm)');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -iH / 2).attr('y', -32)
      .attr('fill', '#858585').attr('font-size', 10).attr('text-anchor', 'middle').text('Count');
  }, [distances]);
  return (
    <div className="bg-[#0d1117] border border-[#3a3a3a] rounded p-2">
      <div className="text-[#858585] text-xs mb-1">Organelle Distance Distribution</div>
      <svg ref={ref} className="w-full" style={{ height: 220 }} />
    </div>
  );
}

function OrganelleSizeBoxPlot({ sizes }: { sizes: number[] }) {
  const ref = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!ref.current || !sizes.length) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();
    const W = ref.current.clientWidth || 340, H = 220;
    const m = { top: 16, right: 16, bottom: 40, left: 50 };
    const iW = W - m.left - m.right, iH = H - m.top - m.bottom;
    svg.attr('viewBox', `0 0 ${W} ${H}`);
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
    g.append('rect').attr('width', iW).attr('height', iH).attr('fill', '#0d1117');

    const sorted = [...sizes].sort(d3.ascending);
    const q1 = d3.quantile(sorted, 0.25)!;
    const median = d3.quantile(sorted, 0.5)!;
    const q3 = d3.quantile(sorted, 0.75)!;
    const iqr = q3 - q1;
    const lo = Math.max(sorted[0], q1 - 1.5 * iqr);
    const hi = Math.min(sorted[sorted.length - 1], q3 + 1.5 * iqr);

    const y = d3.scaleLinear().domain([0, d3.max(sizes)! * 1.1]).range([iH, 0]);
    const cx = iW / 2, bw = iW * 0.35;

    // Whiskers
    g.append('line').attr('x1', cx).attr('x2', cx).attr('y1', y(lo)).attr('y2', y(q1))
      .attr('stroke', '#4ec9b0').attr('stroke-width', 1.5);
    g.append('line').attr('x1', cx).attr('x2', cx).attr('y1', y(q3)).attr('y2', y(hi))
      .attr('stroke', '#4ec9b0').attr('stroke-width', 1.5);
    [lo, hi].forEach(v => g.append('line').attr('x1', cx - bw / 4).attr('x2', cx + bw / 4)
      .attr('y1', y(v)).attr('y2', y(v)).attr('stroke', '#4ec9b0').attr('stroke-width', 1.5));

    // Box
    g.append('rect').attr('x', cx - bw / 2).attr('y', y(q3)).attr('width', bw)
      .attr('height', y(q1) - y(q3)).attr('fill', '#4ec9b030').attr('stroke', '#4ec9b0').attr('stroke-width', 1.5);

    // Median
    g.append('line').attr('x1', cx - bw / 2).attr('x2', cx + bw / 2)
      .attr('y1', y(median)).attr('y2', y(median)).attr('stroke', '#ffd700').attr('stroke-width', 2);

    // Outliers
    sorted.filter(v => v < lo || v > hi).forEach(v =>
      g.append('circle').attr('cx', cx + (Math.random() - 0.5) * 10).attr('cy', y(v))
        .attr('r', 3).attr('fill', '#c586c0').attr('opacity', 0.8));

    g.append('g').attr('transform', `translate(0,${iH})`).call(d3.axisBottom(d3.scalePoint().domain(['size']).range([0, iW])))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 10);
    g.append('g').call(d3.axisLeft(y).ticks(5, '.2f'))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 10);
    g.selectAll('.domain').attr('stroke', '#3a3a3a');

    g.append('text').attr('transform', 'rotate(-90)').attr('x', -iH / 2).attr('y', -38)
      .attr('fill', '#858585').attr('font-size', 10).attr('text-anchor', 'middle').text('Size (µm)');
  }, [sizes]);
  return (
    <div className="bg-[#0d1117] border border-[#3a3a3a] rounded p-2">
      <div className="text-[#858585] text-xs mb-1">Organelle Size Distribution</div>
      <svg ref={ref} className="w-full" style={{ height: 220 }} />
    </div>
  );
}

function CellDensityHeatmap({ density }: { density: number[] }) {
  const ref = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!ref.current || !density.length) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();
    const W = ref.current.clientWidth || 340, H = 220;
    const m = { top: 16, right: 16, bottom: 40, left: 44 };
    const iW = W - m.left - m.right, iH = H - m.top - m.bottom;
    svg.attr('viewBox', `0 0 ${W} ${H}`);
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const side = Math.round(Math.sqrt(density.length));
    const cellW = iW / side, cellH = iH / side;
    const color = d3.scaleSequential(d3.interpolateInferno).domain([0, d3.max(density)!]);

    g.selectAll('rect').data(density).join('rect')
      .attr('x', (_, i) => (i % side) * cellW)
      .attr('y', (_, i) => Math.floor(i / side) * cellH)
      .attr('width', cellW).attr('height', cellH)
      .attr('fill', d => color(d));

    // Colour legend
    const lgW = 10, lgH = iH * 0.7, lgX = iW + 4, lgY = iH * 0.15;
    const lgScale = d3.scaleLinear().domain([0, d3.max(density)!]).range([lgH, 0]);
    const lgGrad = svg.append('defs').append('linearGradient')
      .attr('id', 'hmLeg').attr('x1', '0%').attr('y1', '100%').attr('x2', '0%').attr('y2', '0%');
    [0, 0.25, 0.5, 0.75, 1].forEach(t =>
      lgGrad.append('stop').attr('offset', `${t * 100}%`).attr('stop-color', color(t * d3.max(density)!)));
    g.append('rect').attr('x', lgX).attr('y', lgY).attr('width', lgW).attr('height', lgH).attr('fill', 'url(#hmLeg)');
    g.append('g').attr('transform', `translate(${lgX + lgW},${lgY})`).call(d3.axisRight(lgScale).ticks(3, '.1f'))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 9);

    g.append('text').attr('x', iW / 2).attr('y', iH + 32).attr('fill', '#858585')
      .attr('font-size', 10).attr('text-anchor', 'middle').text('Cell Density Map');
  }, [density]);
  return (
    <div className="bg-[#0d1117] border border-[#3a3a3a] rounded p-2">
      <div className="text-[#858585] text-xs mb-1">Cell Density Heatmap</div>
      <svg ref={ref} className="w-full" style={{ height: 220 }} />
    </div>
  );
}

function IntensityLineChart({ data }: { data: Array<{ t: number; intensity: number }> }) {
  const ref = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!ref.current || !data.length) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();
    const W = ref.current.clientWidth || 340, H = 220;
    const m = { top: 16, right: 16, bottom: 40, left: 44 };
    const iW = W - m.left - m.right, iH = H - m.top - m.bottom;
    svg.attr('viewBox', `0 0 ${W} ${H}`);
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
    g.append('rect').attr('width', iW).attr('height', iH).attr('fill', '#0d1117');

    const x = d3.scaleLinear().domain([0, data.length - 1]).range([0, iW]);
    const y = d3.scaleLinear().domain([0, d3.max(data, d => d.intensity)! * 1.1]).range([iH, 0]);

    // Area fill
    g.append('path').datum(data)
      .attr('fill', '#569cd620')
      .attr('d', d3.area<typeof data[0]>().x((_, i) => x(i)).y0(iH).y1(d => y(d.intensity)).curve(d3.curveMonotoneX));

    // Line
    g.append('path').datum(data)
      .attr('fill', 'none').attr('stroke', '#569cd6').attr('stroke-width', 2)
      .attr('d', d3.line<typeof data[0]>().x((_, i) => x(i)).y(d => y(d.intensity)).curve(d3.curveMonotoneX));

    g.append('g').attr('transform', `translate(0,${iH})`).call(d3.axisBottom(x).ticks(8, 'd'))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 10);
    g.append('g').call(d3.axisLeft(y).ticks(4, '.2f'))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 10);
    g.selectAll('.domain').attr('stroke', '#3a3a3a');

    g.append('text').attr('x', iW / 2).attr('y', iH + 34).attr('fill', '#858585')
      .attr('font-size', 10).attr('text-anchor', 'middle').text('Time (frames)');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -iH / 2).attr('y', -32)
      .attr('fill', '#858585').attr('font-size', 10).attr('text-anchor', 'middle').text('Intensity');
  }, [data]);
  return (
    <div className="bg-[#0d1117] border border-[#3a3a3a] rounded p-2">
      <div className="text-[#858585] text-xs mb-1">Fluorescence Intensity Over Time</div>
      <svg ref={ref} className="w-full" style={{ height: 220 }} />
    </div>
  );
}

function OrganelleScatterPlot({ organelles, image }: {
  organelles: CellData['organellePositions'];
  image: ImagePayload | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const W = 340, H = 340;
    canvas.width = W; canvas.height = H;

    // Draw background or actual cell image
    if (image) {
      const id = ctx.createImageData(image.width, image.height);
      for (let i = 0; i < image.width * image.height; i++) {
        const v = Math.round(Math.max(0, Math.min(1, image.data[i])) * 255);
        id.data[i * 4] = v; id.data[i * 4 + 1] = v; id.data[i * 4 + 2] = v; id.data[i * 4 + 3] = 255;
      }
      // Draw image scaled to canvas
      const offscreen = new OffscreenCanvas(image.width, image.height);
      offscreen.getContext('2d')!.putImageData(id, 0, 0);
      ctx.drawImage(offscreen, 0, 0, W, H);
    } else {
      ctx.fillStyle = '#0d1117';
      ctx.fillRect(0, 0, W, H);
    }

    // Draw organelle overlay
    const COLORS: Record<string, string> = {
      Nucleus: '#4ec9b080', Mitochondria: '#ffd700cc', Golgi: '#c586c0cc',
      Lysosome: '#f44336cc', Ribosome: '#569cd6cc', ER: '#dcdcaacc',
    };
    for (const org of organelles) {
      const cx = org.x * W, cy = org.y * H, r = org.r * W;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = COLORS[org.label] ?? '#ffffff80';
      ctx.fill();
      ctx.strokeStyle = (COLORS[org.label] ?? '#fff').replace(/[0-9a-f]{2}$/, 'ff');
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(org.label, cx, cy - r - 4);
    }
  }, [organelles, image]);

  return (
    <div className="bg-[#0d1117] border border-[#3a3a3a] rounded p-2">
      <div className="text-[#858585] text-xs mb-1">Organelle Positions — annotated cell image</div>
      <canvas ref={canvasRef} className="w-full rounded" style={{ maxHeight: 340, imageRendering: 'pixelated' }} />
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

const LOCAL_IMAGES = [
  { dataset: 'BBBC007', image: 'A9 p10d.tif', label: 'A9 p10 DAPI' },
  { dataset: 'BBBC007', image: 'A9 p9d.tif',  label: 'A9 p9 DAPI'  },
  { dataset: 'BBBC007', image: 'A9 p7d.tif',  label: 'A9 p7 DAPI'  },
  { dataset: 'BBBC007', image: 'A9 p5d.tif',  label: 'A9 p5 DAPI'  },
  { dataset: 'BBBC007', image: '17P1_POS0006_D_1UL.tif', label: 'f96 POS0006' },
  { dataset: 'BBBC007', image: '20P1_POS0002_D_1UL.tif', label: 'f9620 POS0002' },
];

export default function MicroscopyViewer() {
  const [image, setImage]       = useState<ImagePayload | null>(null);
  const [selIdx, setSelIdx]     = useState(0);
  const [loading, setLoading]   = useState(false);
  const [cellData, setCellData] = useState<CellData>(generateCellData(1));
  const fileRef = useRef<HTMLInputElement>(null);

  const fetchImage = useCallback(async (dataset: string, img: string, idx: number) => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ db: 'BBBC', dataset, image: img });
      const res = await fetch(`/api/image-proxy?${params}`);
      const json = await res.json();
      if (!json.error) {
        setImage({ data: new Float32Array(json.data), width: json.width, height: json.height, synthetic: json.synthetic });
        setCellData(generateCellData(idx + 1));
        setSelIdx(idx);
      }
    } finally { setLoading(false); }
  }, []);

  // Load first image on mount
  useEffect(() => { fetchImage(LOCAL_IMAGES[0].dataset, LOCAL_IMAGES[0].image, 0); }, [fetchImage]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    try {
      const bitmap = await createImageBitmap(file);
      const W = Math.min(bitmap.width, 512), H = Math.min(bitmap.height, 512);
      const off = new OffscreenCanvas(W, H);
      const ctx = off.getContext('2d')!;
      ctx.drawImage(bitmap, 0, 0, W, H);
      const id = ctx.getImageData(0, 0, W, H);
      const f32 = new Float32Array(W * H);
      for (let i = 0; i < W * H; i++)
        f32[i] = (id.data[i * 4] * 0.299 + id.data[i * 4 + 1] * 0.587 + id.data[i * 4 + 2] * 0.114) / 255;
      setImage({ data: f32, width: W, height: H });
      setCellData(generateCellData(Math.floor(Math.random() * 100)));
    } finally { setLoading(false); }
  };

  return (
    <div className="min-h-screen bg-[#1e1e1e] text-[#d4d4d4] font-mono text-sm">
      {/* Header */}
      <div className="flex items-center gap-4 px-6 py-3 bg-[#252526] border-b border-[#3a3a3a]">
        <span className="text-white font-semibold tracking-wide text-base">MICROSCOPY VISUALIZATION MODULE</span>
        <div className="flex-1" />
        <span className="text-[#858585] text-xs">5-panel analysis · BBBC007 HeLa A9</span>
      </div>

      <div className="flex gap-3 p-4">
        {/* Sidebar: image selector */}
        <div className="w-52 shrink-0 space-y-2">
          <div className="text-[#858585] text-xs uppercase tracking-wider mb-2">Dataset</div>
          {LOCAL_IMAGES.map((entry, i) => (
            <button key={i} onClick={() => fetchImage(entry.dataset, entry.image, i)}
              className={`w-full text-left px-3 py-2 rounded text-xs border ${
                selIdx === i ? 'border-[#007acc] bg-[#37373d] text-white' : 'border-[#3a3a3a] hover:bg-[#2a2a2a] text-[#858585]'
              }`}>
              {entry.label}
            </button>
          ))}
          <div className="border-t border-[#3a3a3a] pt-2 mt-2">
            <button onClick={() => fileRef.current?.click()}
              className="w-full px-3 py-2 rounded text-xs border border-[#3a3a3a] hover:bg-[#2a2a2a] text-[#858585]">
              Upload image…
            </button>
            <input ref={fileRef} type="file" accept="image/*,.tif,.tiff" className="hidden" onChange={handleUpload} />
          </div>
          {loading && <div className="text-[#ffa500] text-xs py-2">Loading…</div>}

          {/* Quick stats */}
          {image && (
            <div className="border border-[#3a3a3a] rounded p-2 text-[0.68rem] space-y-1 mt-2">
              <div className="text-[#858585]">Image info</div>
              <div>{image.width}×{image.height} px</div>
              <div className="text-[#4ec9b0]">{image.synthetic ? 'synthetic' : 'real TIFF'}</div>
              <div className="text-[#858585]">n={cellData.distances.length} distances</div>
              <div className="text-[#858585]">n={cellData.organelleSizes.length} organelles</div>
            </div>
          )}
        </div>

        {/* Main: 5-panel grid */}
        <div className="flex-1 grid grid-cols-2 gap-3">
          {/* Panel 1: Organelle scatter / annotated cell image */}
          <div className="col-span-1 row-span-2">
            <OrganelleScatterPlot organelles={cellData.organellePositions} image={image} />
          </div>

          {/* Panel 2: Distance histogram */}
          <DistanceHistogram distances={cellData.distances} />

          {/* Panel 3: Size box plot */}
          <OrganelleSizeBoxPlot sizes={cellData.organelleSizes} />

          {/* Panel 4: Density heatmap */}
          <CellDensityHeatmap density={cellData.cellDensity} />

          {/* Panel 5: Intensity over time */}
          <IntensityLineChart data={cellData.intensityOverTime} />
        </div>
      </div>
    </div>
  );
}
