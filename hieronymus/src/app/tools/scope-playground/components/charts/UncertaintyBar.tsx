'use client';

import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface Props {
  data: { d: number; deltaD: number; goals: Array<{ threshold: number; unit: string; op: string }> };
}

export default function UncertaintyBar({ data }: Props) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const W = ref.current.clientWidth || 380;
    const H = 160;
    const m = { top: 24, right: 20, bottom: 40, left: 52 };
    const iW = W - m.left - m.right;
    const iH = H - m.top - m.bottom;

    svg.attr('viewBox', `0 0 ${W} ${H}`);

    const { d, deltaD, goals } = data;
    const maxY = Math.max(d + deltaD * 2, ...goals.map(g => g.threshold)) * 1.2;

    const x = d3.scaleBand().domain(['distance', 'δd']).range([0, iW]).padding(0.35);
    const y = d3.scaleLinear().domain([0, maxY]).range([iH, 0]);

    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(-iW))
      .selectAll('line').attr('stroke', '#2a2a2a');
    g.selectAll('.domain').remove();

    // Main distance bar
    g.append('rect')
      .attr('x', x('distance')!).attr('y', y(d))
      .attr('width', x.bandwidth()).attr('height', iH - y(d))
      .attr('fill', '#569cd6').attr('opacity', 0.8);

    // Error bar for δd
    const bx = x('δd')! + x.bandwidth() / 2;
    g.append('rect')
      .attr('x', x('δd')!).attr('y', y(deltaD))
      .attr('width', x.bandwidth()).attr('height', iH - y(deltaD))
      .attr('fill', '#c586c0').attr('opacity', 0.8);

    // Goal threshold lines
    const colors = ['#4caf50', '#ffa500', '#f44336'];
    goals.slice(0, 3).forEach((goal, i) => {
      const cy = y(goal.threshold);
      g.append('line')
        .attr('x1', 0).attr('x2', iW)
        .attr('y1', cy).attr('y2', cy)
        .attr('stroke', colors[i % colors.length])
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '5,3');
      g.append('text')
        .attr('x', iW - 2).attr('y', cy - 3)
        .attr('fill', colors[i % colors.length])
        .attr('font-size', 9).attr('text-anchor', 'end')
        .text(`goal: δd ${goal.op} ${goal.threshold}${goal.unit}`);
    });

    // Value labels
    g.append('text').attr('x', x('distance')! + x.bandwidth() / 2).attr('y', y(d) - 4)
      .attr('fill', '#d4d4d4').attr('font-size', 9).attr('text-anchor', 'middle')
      .text(`${d.toFixed(3)} µm`);
    g.append('text').attr('x', x('δd')! + x.bandwidth() / 2).attr('y', y(deltaD) - 4)
      .attr('fill', '#d4d4d4').attr('font-size', 9).attr('text-anchor', 'middle')
      .text(`±${deltaD.toFixed(3)}`);

    // Axes
    g.append('g').attr('transform', `translate(0,${iH})`)
      .call(d3.axisBottom(x))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 10);
    g.append('g').call(d3.axisLeft(y).ticks(4, '.2f'))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 10);
    g.selectAll('.domain').attr('stroke', '#3a3a3a');

    g.append('text').attr('transform', 'rotate(-90)').attr('x', -iH / 2).attr('y', -40)
      .attr('fill', '#858585').attr('font-size', 10).attr('text-anchor', 'middle').text('µm');
  }, [data]);

  return (
    <div className="bg-[#1e1e1e] border border-[#3a3a3a] rounded p-2">
      <div className="text-[#858585] text-xs mb-1">Distance ± Uncertainty vs Goal thresholds</div>
      <svg ref={ref} className="w-full" style={{ height: 160 }} />
    </div>
  );
}
