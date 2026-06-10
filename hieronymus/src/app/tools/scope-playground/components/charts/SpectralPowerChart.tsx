'use client';

import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface Point { freq: number; energy: number; }
interface Props { data: Point[]; exponent: number; }

export default function SpectralPowerChart({ data, exponent }: Props) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current || !data.length) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const W = ref.current.clientWidth || 380;
    const H = 220;
    const m = { top: 16, right: 20, bottom: 40, left: 52 };
    const iW = W - m.left - m.right;
    const iH = H - m.top - m.bottom;

    svg.attr('viewBox', `0 0 ${W} ${H}`);

    const x = d3.scaleLog()
      .domain(d3.extent(data, d => d.freq) as [number,number])
      .range([0, iW]);
    const y = d3.scaleLog()
      .domain([d3.min(data, d => d.energy)! * 0.5, d3.max(data, d => d.energy)! * 2])
      .range([iH, 0]);

    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    // Grid
    g.append('g').call(d3.axisLeft(y).ticks(4, '.0e').tickSize(-iW))
      .selectAll('line').attr('stroke', '#3a3a3a');
    g.selectAll('.domain').remove();

    // Axes
    g.append('g').attr('transform', `translate(0,${iH})`)
      .call(d3.axisBottom(x).ticks(5, '.0e'))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 10);
    g.append('g').call(d3.axisLeft(y).ticks(4, '.0e'))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 10);
    g.selectAll('.domain').attr('stroke', '#3a3a3a');

    // Power law fit line
    const [fMin, fMax] = d3.extent(data, d => d.freq) as [number,number];
    const fMid = Math.sqrt(fMin * fMax);
    const eMid = data[Math.floor(data.length / 2)]?.energy ?? 1;
    const fitLine = [[fMin, eMid * (fMin / fMid) ** exponent], [fMax, eMid * (fMax / fMid) ** exponent]] as [number,number][];
    g.append('path')
      .datum(fitLine)
      .attr('fill', 'none')
      .attr('stroke', '#4a4a4a')
      .attr('stroke-dasharray', '4,3')
      .attr('d', d3.line<[number,number]>().x(d => x(d[0])).y(d => y(d[1])));

    // Data line
    g.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', '#569cd6')
      .attr('stroke-width', 1.5)
      .attr('d', d3.line<Point>().x(d => x(d.freq)).y(d => y(d.energy)));

    // Labels
    g.append('text').attr('x', iW / 2).attr('y', iH + 34).attr('fill', '#858585')
      .attr('font-size', 10).attr('text-anchor', 'middle').text('Frequency (cycles/px)');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -iH / 2).attr('y', -42)
      .attr('fill', '#858585').attr('font-size', 10).attr('text-anchor', 'middle').text('Power');
    g.append('text').attr('x', iW - 4).attr('y', 12).attr('fill', '#858585')
      .attr('font-size', 10).attr('text-anchor', 'end').text(`β = ${exponent.toFixed(2)}`);
  }, [data, exponent]);

  return (
    <div className="bg-[#1e1e1e] border border-[#3a3a3a] rounded p-2">
      <div className="text-[#858585] text-xs mb-1">Spectral Power (log-log)</div>
      <svg ref={ref} className="w-full" style={{ height: 220 }} />
    </div>
  );
}
