'use client';

import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface Point { freq: number; energy: number; }
interface Props { data: Point[]; exponent: number; }

export default function SpectralPowerChart({ data, exponent }: Props) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current) return;

    // Guard: need at least 2 finite, positive points for a log-log chart
    const pts = data.filter(d => d.freq > 0 && d.energy > 0 && isFinite(d.freq) && isFinite(d.energy));
    if (pts.length < 2) {
      d3.select(ref.current).selectAll('*').remove();
      d3.select(ref.current).append('text')
        .attr('x', '50%').attr('y', '50%').attr('text-anchor', 'middle')
        .attr('fill', '#555').attr('font-size', 12).text('No spectral data');
      return;
    }

    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const W = ref.current.clientWidth || 380;
    const H = 220;
    const m = { top: 16, right: 24, bottom: 40, left: 58 };
    const iW = W - m.left - m.right;
    const iH = H - m.top - m.bottom;

    svg.attr('viewBox', `0 0 ${W} ${H}`);

    const fMin = d3.min(pts, d => d.freq)!;
    const fMax = d3.max(pts, d => d.freq)!;
    const eMin = d3.min(pts, d => d.energy)!;
    const eMax = d3.max(pts, d => d.energy)!;

    const x = d3.scaleLog().domain([fMin, fMax]).range([0, iW]).clamp(true);
    const y = d3.scaleLog().domain([eMin * 0.5, eMax * 2]).range([iH, 0]).clamp(true);

    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    // Background
    g.append('rect').attr('width', iW).attr('height', iH).attr('fill', '#0d1117');

    // Grid lines
    g.append('g').attr('class', 'grid')
      .call(d3.axisLeft(y).ticks(4, '.0e').tickSize(-iW))
      .selectAll('line').attr('stroke', '#2a2a2a').attr('stroke-dasharray', '2,2');
    g.selectAll('.grid .domain').remove();
    g.selectAll('.grid text').remove();

    // Power-law fit line
    const eMid = pts[Math.floor(pts.length / 2)]?.energy ?? 1;
    const fMid = Math.sqrt(fMin * fMax);
    const fitPts = pts.map(d => ({ freq: d.freq, energy: eMid * Math.pow(d.freq / fMid, exponent) }))
      .filter(d => d.energy > 0 && isFinite(d.energy));
    if (fitPts.length >= 2) {
      g.append('path')
        .datum(fitPts)
        .attr('fill', 'none')
        .attr('stroke', '#3a3a3a')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4,3')
        .attr('d', d3.line<typeof fitPts[0]>()
          .x(d => x(d.freq)).y(d => y(d.energy))
          .defined(d => d.energy > 0 && isFinite(d.energy)));
    }

    // Data line
    g.append('path')
      .datum(pts)
      .attr('fill', 'none')
      .attr('stroke', '#569cd6')
      .attr('stroke-width', 2)
      .attr('d', d3.line<Point>()
        .x(d => x(d.freq)).y(d => y(d.energy))
        .defined(d => d.energy > 0 && isFinite(d.energy)));

    // Data dots
    g.selectAll('.dot').data(pts.filter((_, i) => i % 4 === 0))
      .join('circle').attr('class', 'dot')
      .attr('cx', d => x(d.freq)).attr('cy', d => y(d.energy))
      .attr('r', 2).attr('fill', '#569cd6');

    // Axes
    g.append('g').attr('transform', `translate(0,${iH})`)
      .call(d3.axisBottom(x).ticks(5, '.1e'))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 10);
    g.append('g')
      .call(d3.axisLeft(y).ticks(4, '.1e'))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 10);
    g.selectAll('.domain').attr('stroke', '#3a3a3a');

    // Labels
    g.append('text').attr('x', iW / 2).attr('y', iH + 34)
      .attr('fill', '#858585').attr('font-size', 10).attr('text-anchor', 'middle')
      .text('Frequency (cycles/px)');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -iH / 2).attr('y', -46)
      .attr('fill', '#858585').attr('font-size', 10).attr('text-anchor', 'middle')
      .text('Spectral Power');
    g.append('text').attr('x', iW - 4).attr('y', 14)
      .attr('fill', '#9cdcfe').attr('font-size', 10).attr('text-anchor', 'end')
      .text(`β = ${exponent.toFixed(2)}`);
  }, [data, exponent]);

  return (
    <div className="bg-[#0d1117] border border-[#3a3a3a] rounded p-2">
      <div className="text-[#858585] text-xs mb-1">Spectral Power  |û(k)|² vs k  (log-log)</div>
      <svg ref={ref} className="w-full" style={{ height: 220 }} />
    </div>
  );
}
