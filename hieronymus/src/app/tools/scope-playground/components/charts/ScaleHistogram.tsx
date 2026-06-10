'use client';

import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface Bin { alpha: number; count: number; }
interface Props { data: Bin[]; mean: number; }

export default function ScaleHistogram({ data, mean }: Props) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current || !data.length) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const W = ref.current.clientWidth || 380;
    const H = 200;
    const m = { top: 16, right: 20, bottom: 40, left: 52 };
    const iW = W - m.left - m.right;
    const iH = H - m.top - m.bottom;

    svg.attr('viewBox', `0 0 ${W} ${H}`);

    const x = d3.scaleLinear()
      .domain([d3.min(data, d => d.alpha)! * 0.9, d3.max(data, d => d.alpha)! * 1.1])
      .range([0, iW]);
    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.count)! * 1.15])
      .range([iH, 0]);

    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(-iW))
      .selectAll('line').attr('stroke', '#2a2a2a');
    g.selectAll('.domain').remove();

    const bw = iW / data.length * 0.85;
    g.selectAll('.bar')
      .data(data)
      .join('rect')
      .attr('class', 'bar')
      .attr('x', d => x(d.alpha) - bw / 2)
      .attr('y', d => y(d.count))
      .attr('width', bw)
      .attr('height', d => iH - y(d.count))
      .attr('fill', '#4ec9b0')
      .attr('opacity', 0.7);

    // Mean line
    g.append('line')
      .attr('x1', x(mean)).attr('x2', x(mean))
      .attr('y1', 0).attr('y2', iH)
      .attr('stroke', '#ffd700').attr('stroke-width', 1.5).attr('stroke-dasharray', '4,3');
    g.append('text')
      .attr('x', x(mean) + 4).attr('y', 10)
      .attr('fill', '#ffd700').attr('font-size', 9)
      .text(`ᾱ=${mean.toFixed(3)}`);

    // Axes
    g.append('g').attr('transform', `translate(0,${iH})`)
      .call(d3.axisBottom(x).ticks(6, '.2f'))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 10);
    g.append('g').call(d3.axisLeft(y).ticks(4))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 10);
    g.selectAll('.domain').attr('stroke', '#3a3a3a');

    g.append('text').attr('x', iW / 2).attr('y', iH + 34).attr('fill', '#858585')
      .attr('font-size', 10).attr('text-anchor', 'middle').text('α (scale field)');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -iH / 2).attr('y', -40)
      .attr('fill', '#858585').attr('font-size', 10).attr('text-anchor', 'middle').text('Count');
  }, [data, mean]);

  return (
    <div className="bg-[#1e1e1e] border border-[#3a3a3a] rounded p-2">
      <div className="text-[#858585] text-xs mb-1">Scale field α(x,y) distribution</div>
      <svg ref={ref} className="w-full" style={{ height: 200 }} />
    </div>
  );
}
