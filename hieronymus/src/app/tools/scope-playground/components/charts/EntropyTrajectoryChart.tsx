'use client';

import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface Point { phase: string; sk: number; st: number; se: number; }
interface Props { data: Point[]; }

const COLORS = { sk: '#4ec9b0', st: '#569cd6', se: '#c586c0' };

export default function EntropyTrajectoryChart({ data }: Props) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current || !data.length) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const W = ref.current.clientWidth || 380;
    const H = 220;
    const m = { top: 16, right: 80, bottom: 40, left: 40 };
    const iW = W - m.left - m.right;
    const iH = H - m.top - m.bottom;

    svg.attr('viewBox', `0 0 ${W} ${H}`);

    const phases = data.map(d => d.phase);
    const x = d3.scalePoint().domain(phases).range([0, iW]).padding(0.3);
    const y = d3.scaleLinear().domain([0, 1]).range([iH, 0]);

    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    // Grid
    g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(-iW))
      .selectAll('line').attr('stroke', '#2a2a2a');
    g.selectAll('.domain').remove();

    // Stacked area
    const stack = d3.stack<Point>().keys(['sk', 'st', 'se']);
    const layers = stack(data);

    const area = d3.area<d3.SeriesPoint<Point>>()
      .x((_d, i) => x(phases[i])!)
      .y0(d => y(d[0]))
      .y1(d => y(d[1]));

    const colorMap: Record<string, string> = COLORS;
    g.selectAll('.layer')
      .data(layers)
      .join('path')
      .attr('fill', d => colorMap[d.key] + '55')
      .attr('stroke', d => colorMap[d.key])
      .attr('stroke-width', 1)
      .attr('d', area as any);

    // Axes
    g.append('g').attr('transform', `translate(0,${iH})`)
      .call(d3.axisBottom(x))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 9);
    g.append('g').call(d3.axisLeft(y).ticks(4, '.1f'))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 10);
    g.selectAll('.domain').attr('stroke', '#3a3a3a');

    // Legend
    const leg = svg.append('g').attr('transform', `translate(${m.left + iW + 8},${m.top})`);
    [['S_k','sk'],['S_t','st'],['S_e','se']].forEach(([label, key], i) => {
      leg.append('rect').attr('x', 0).attr('y', i * 18).attr('width', 10).attr('height', 10)
        .attr('fill', colorMap[key]);
      leg.append('text').attr('x', 14).attr('y', i * 18 + 9).attr('fill', '#858585')
        .attr('font-size', 10).text(label);
    });

    g.append('text').attr('x', iW / 2).attr('y', iH + 34).attr('fill', '#858585')
      .attr('font-size', 10).attr('text-anchor', 'middle').text('Phase');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -iH / 2).attr('y', -30)
      .attr('fill', '#858585').attr('font-size', 10).attr('text-anchor', 'middle').text('S-entropy');
  }, [data]);

  return (
    <div className="bg-[#1e1e1e] border border-[#3a3a3a] rounded p-2">
      <div className="text-[#858585] text-xs mb-1">S-entropy trajectory (S_k + S_t + S_e = 1)</div>
      <svg ref={ref} className="w-full" style={{ height: 220 }} />
    </div>
  );
}
