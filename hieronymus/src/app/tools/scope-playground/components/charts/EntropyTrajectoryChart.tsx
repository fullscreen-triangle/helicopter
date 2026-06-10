'use client';

import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface Point { phase: string; sk: number; st: number; se: number; }
interface Props { data: Point[]; }

const SK_COLOR = '#4ec9b0';
const ST_COLOR = '#569cd6';
const SE_COLOR = '#c586c0';

export default function EntropyTrajectoryChart({ data }: Props) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current || !data.length) return;

    // Normalise each row to sum exactly to 1 (floating-point drift guard)
    const pts: Point[] = data.map(d => {
      const sum = (d.sk + d.st + d.se) || 1;
      return { phase: d.phase, sk: d.sk / sum, st: d.st / sum, se: d.se / sum };
    });

    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const W = ref.current.clientWidth || 380;
    const H = 220;
    const m = { top: 16, right: 88, bottom: 44, left: 40 };
    const iW = W - m.left - m.right;
    const iH = H - m.top - m.bottom;

    svg.attr('viewBox', `0 0 ${W} ${H}`);

    const phases = pts.map(d => d.phase);
    const x = d3.scalePoint<string>().domain(phases).range([0, iW]).padding(0.35);
    const y = d3.scaleLinear().domain([0, 1]).range([iH, 0]);

    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    // Background
    g.append('rect').attr('width', iW).attr('height', iH).attr('fill', '#0d1117');

    // Grid
    g.append('g').attr('class', 'grid')
      .call(d3.axisLeft(y).ticks(5).tickSize(-iW))
      .selectAll('line').attr('stroke', '#1e2a2a').attr('stroke-dasharray', '2,2');
    g.selectAll('.grid .domain').remove();
    g.selectAll('.grid text').remove();

    // Stacked areas
    type StackRow = { phase: string; sk: number; st: number; se: number; [k: string]: string | number };
    const stackData: StackRow[] = pts as StackRow[];
    const stack = d3.stack<StackRow>().keys(['sk', 'st', 'se']).order(d3.stackOrderNone).offset(d3.stackOffsetNone);
    const layers = stack(stackData);

    const area = d3.area<d3.SeriesPoint<StackRow>>()
      .x((_d, i) => x(phases[i]) ?? 0)
      .y0(d => y(Math.max(0, Math.min(1, d[0]))))
      .y1(d => y(Math.max(0, Math.min(1, d[1]))))
      .curve(d3.curveMonotoneX);

    const colors: Record<string, string> = { sk: SK_COLOR, st: ST_COLOR, se: SE_COLOR };

    g.selectAll('.layer')
      .data(layers)
      .join('path')
      .attr('class', 'layer')
      .attr('fill', d => colors[d.key] + '50')
      .attr('stroke', d => colors[d.key])
      .attr('stroke-width', 1.5)
      .attr('d', area as any);

    // Individual lines for each component
    const lineGen = d3.line<Point>().x(d => x(d.phase) ?? 0).curve(d3.curveMonotoneX);

    // Draw sk, st, se as running cumulative values (matching stacked area top edges)
    const cumulPts = pts.map(d => ({ phase: d.phase, sk: d.sk, skst: d.sk + d.st, all: 1.0 }));
    const lineData: Array<{ key: string; color: string; vals: Array<{ phase: string; v: number }> }> = [
      { key: 'S_k', color: SK_COLOR, vals: cumulPts.map(d => ({ phase: d.phase, v: d.sk })) },
      { key: 'S_k+S_t', color: ST_COLOR, vals: cumulPts.map(d => ({ phase: d.phase, v: d.skst })) },
    ];
    for (const { color, vals } of lineData) {
      g.append('path')
        .datum(vals)
        .attr('fill', 'none')
        .attr('stroke', color)
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,2')
        .attr('d', d3.line<typeof vals[0]>().x(d => x(d.phase) ?? 0).y(d => y(d.v)).curve(d3.curveMonotoneX));
    }

    // Phase dots on S_k boundary
    g.selectAll('.sk-dot').data(pts)
      .join('circle').attr('class', 'sk-dot')
      .attr('cx', d => x(d.phase) ?? 0).attr('cy', d => y(d.sk))
      .attr('r', 3).attr('fill', SK_COLOR);

    // Axes
    g.append('g').attr('transform', `translate(0,${iH})`)
      .call(d3.axisBottom(x))
      .selectAll('text')
      .attr('fill', '#858585').attr('font-size', 9)
      .attr('transform', 'rotate(-20)').attr('text-anchor', 'end');
    g.append('g')
      .call(d3.axisLeft(y).ticks(5, '.2f'))
      .selectAll('text').attr('fill', '#858585').attr('font-size', 10);
    g.selectAll('.domain').attr('stroke', '#3a3a3a');

    // Legend
    const leg = svg.append('g').attr('transform', `translate(${m.left + iW + 10},${m.top})`);
    [['S_k', SK_COLOR], ['S_t', ST_COLOR], ['S_e', SE_COLOR]].forEach(([label, color], i) => {
      leg.append('rect').attr('x', 0).attr('y', i * 20).attr('width', 10).attr('height', 10).attr('fill', color);
      leg.append('text').attr('x', 14).attr('y', i * 20 + 9).attr('fill', '#858585').attr('font-size', 10).text(label);
    });

    // Axis labels
    g.append('text').attr('x', iW / 2).attr('y', iH + 38)
      .attr('fill', '#858585').attr('font-size', 10).attr('text-anchor', 'middle').text('Phase');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -iH / 2).attr('y', -30)
      .attr('fill', '#858585').attr('font-size', 10).attr('text-anchor', 'middle').text('S-entropy');
  }, [data]);

  return (
    <div className="bg-[#0d1117] border border-[#3a3a3a] rounded p-2">
      <div className="text-[#858585] text-xs mb-1">S-entropy conservation  S_k + S_t + S_e = 1</div>
      <svg ref={ref} className="w-full" style={{ height: 220 }} />
    </div>
  );
}
