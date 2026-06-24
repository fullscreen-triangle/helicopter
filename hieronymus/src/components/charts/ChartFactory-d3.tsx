'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { motion } from 'framer-motion';

export interface ChartConfig {
  id: string;
  type: 'line' | 'bar' | 'scatter' | 'area' | 'pie' | 'histogram' | 'parallel';
  title: string;
  data: any[];
  dataKey?: string;
  xAxisKey?: string;
  yAxisKey?: string;
  series?: Array<{ name: string; dataKey: string; color?: string }>;
  colors?: string[];
  options?: Record<string, any>;
}

interface D3ChartProps {
  config: ChartConfig;
  onBrush?: (selection: any[]) => void;
  highlightedIndices?: number[];
  width?: number;
  height?: number;
}

// Margin conventions for d3 charts
const MARGIN = { top: 20, right: 30, bottom: 30, left: 60 };

/**
 * Line Chart with d3.js
 * Supports linked brushing via onBrush callback
 */
export const LineChart: React.FC<D3ChartProps> = ({
  config,
  onBrush,
  highlightedIndices = [],
  width = 500,
  height = 300,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const brushRef = useRef<any>(null);

  useEffect(() => {
    if (!svgRef.current || !config.data.length) return;

    const w = width - MARGIN.left - MARGIN.right;
    const h = height - MARGIN.top - MARGIN.bottom;

    // Select and clear previous content
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Scales
    const xScale = d3
      .scaleLinear()
      .domain(
        d3.extent(config.data, (d: any) =>
          parseFloat(d[config.xAxisKey || 'x'])
        ) as [number, number]
      )
      .range([0, w]);

    const yScale = d3
      .scaleLinear()
      .domain([
        0,
        d3.max(config.data, (d: any) =>
          parseFloat(d[config.yAxisKey || config.dataKey || 'y'])
        ) || 1,
      ] as [number, number])
      .range([h, 0]);

    // Line generator
    const line = d3
      .line<any>()
      .x((d) => xScale(parseFloat(d[config.xAxisKey || 'x'])))
      .y((d) => yScale(parseFloat(d[config.yAxisKey || config.dataKey || 'y'])));

    // Main group
    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // X Axis
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale))
      .style('font-size', '10px');

    // Y Axis
    g.append('g')
      .call(d3.axisLeft(yScale))
      .style('font-size', '10px');

    // Grid lines
    g.append('g')
      .attr('stroke', '#333')
      .attr('stroke-opacity', 0.1)
      .call(
        d3
          .axisLeft(yScale)
          .tickSize(-w)
          .tickFormat(() => '')
      );

    // Path
    g.append('path')
      .datum(config.data)
      .attr('fill', 'none')
      .attr('stroke', config.colors?.[0] || '#06b6d4')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Brush (for linked selection)
    const brush = d3
      .brushX()
      .extent([
        [0, 0],
        [w, h],
      ])
      .on('end', (event) => {
        if (!event.selection) return;

        const [x0, x1] = event.selection.map(xScale.invert);
        const selected = config.data.filter((d: any) => {
          const val = parseFloat(d[config.xAxisKey || 'x']);
          return val >= x0 && val <= x1;
        });

        onBrush?.(selected);
      });

    g.append('g').call(brush);

    // Highlight points
    g.selectAll('.point')
      .data(config.data, (_d, i) => i)
      .enter()
      .append('circle')
      .attr('cx', (d: any) => xScale(parseFloat(d[config.xAxisKey || 'x'])))
      .attr('cy', (d: any) =>
        yScale(parseFloat(d[config.yAxisKey || config.dataKey || 'y']))
      )
      .attr('r', 3)
      .attr('fill', (_d, i) =>
        highlightedIndices.includes(i) ? '#ff6b6b' : 'transparent'
      )
      .attr('opacity', 0.7);
  }, [config, onBrush, highlightedIndices, width, height]);

  return (
    <div className="bg-[#0f1420] rounded p-2">
      <div className="text-sm font-semibold text-cyan-400 mb-2">
        {config.title}
      </div>
      <svg ref={svgRef} className="w-full" />
    </div>
  );
};

/**
 * Scatter Plot with d3.js
 * Supports linked brushing with brush selection
 */
export const ScatterChart: React.FC<D3ChartProps> = ({
  config,
  onBrush,
  highlightedIndices = [],
  width = 500,
  height = 300,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !config.data.length) return;

    const w = width - MARGIN.left - MARGIN.right;
    const h = height - MARGIN.top - MARGIN.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Scales
    const xScale = d3
      .scaleLinear()
      .domain(
        d3.extent(config.data, (d: any) =>
          parseFloat(d[config.xAxisKey || 'x'])
        ) as [number, number]
      )
      .range([0, w]);

    const yScale = d3
      .scaleLinear()
      .domain(
        d3.extent(config.data, (d: any) =>
          parseFloat(d[config.yAxisKey || 'y'])
        ) as [number, number]
      )
      .range([h, 0]);

    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale))
      .style('font-size', '10px');

    g.append('g')
      .call(d3.axisLeft(yScale))
      .style('font-size', '10px');

    // Brush
    const brush = d3
      .brush()
      .extent([
        [0, 0],
        [w, h],
      ])
      .on('end', (event) => {
        if (!event.selection) return;

        const [[x0, y0], [x1, y1]] = event.selection;
        const selected = config.data.filter((d: any) => {
          const x = xScale(parseFloat(d[config.xAxisKey || 'x']));
          const y = yScale(parseFloat(d[config.yAxisKey || 'y']));
          return x >= x0 && x <= x1 && y >= y0 && y <= y1;
        });

        onBrush?.(selected);
      });

    g.append('g').call(brush);

    // Points
    g.selectAll('.point')
      .data(config.data, (_d, i) => i)
      .enter()
      .append('circle')
      .attr('cx', (d: any) => xScale(parseFloat(d[config.xAxisKey || 'x'])))
      .attr('cy', (d: any) => yScale(parseFloat(d[config.yAxisKey || 'y'])))
      .attr('r', 4)
      .attr('fill', (_d, i) =>
        highlightedIndices.includes(i)
          ? '#ff6b6b'
          : config.colors?.[0] || '#06b6d4'
      )
      .attr('opacity', 0.6);
  }, [config, onBrush, highlightedIndices, width, height]);

  return (
    <div className="bg-[#0f1420] rounded p-2">
      <div className="text-sm font-semibold text-cyan-400 mb-2">
        {config.title}
      </div>
      <svg ref={svgRef} className="w-full" />
    </div>
  );
};

/**
 * Bar Chart with d3.js
 * Supports linked brushing
 */
export const BarChart: React.FC<D3ChartProps> = ({
  config,
  onBrush,
  highlightedIndices = [],
  width = 500,
  height = 300,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !config.data.length) return;

    const w = width - MARGIN.left - MARGIN.right;
    const h = height - MARGIN.top - MARGIN.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Scales
    const xScale = d3
      .scaleBand()
      .domain(config.data.map((_d: any, i: number) => i.toString()))
      .range([0, w])
      .padding(0.1);

    const yScale = d3
      .scaleLinear()
      .domain([
        0,
        d3.max(config.data, (d: any) =>
          parseFloat(d[config.dataKey || 'value'])
        ) || 1,
      ] as [number, number])
      .range([h, 0]);

    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale))
      .style('font-size', '10px');

    g.append('g')
      .call(d3.axisLeft(yScale))
      .style('font-size', '10px');

    // Bars
    g.selectAll('.bar')
      .data(config.data)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', (_d: any, i: number) => xScale(i.toString()) || 0)
      .attr('y', (d: any) => yScale(parseFloat(d[config.dataKey || 'value'])))
      .attr('width', xScale.bandwidth())
      .attr('height', (d: any) =>
        h - yScale(parseFloat(d[config.dataKey || 'value']))
      )
      .attr('fill', (_d, i) =>
        highlightedIndices.includes(i)
          ? '#ff6b6b'
          : config.colors?.[0] || '#06b6d4'
      )
      .attr('opacity', 0.7)
      .on('click', (_event: any, d: any) => {
        // Single bar selection
        onBrush?.([d]);
      });
  }, [config, onBrush, highlightedIndices, width, height]);

  return (
    <div className="bg-[#0f1420] rounded p-2">
      <div className="text-sm font-semibold text-cyan-400 mb-2">
        {config.title}
      </div>
      <svg ref={svgRef} className="w-full" />
    </div>
  );
};

/**
 * Histogram with d3.js
 * Supports brush selection on bins
 */
export const Histogram: React.FC<D3ChartProps> = ({
  config,
  onBrush,
  highlightedIndices = [],
  width = 500,
  height = 300,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !config.data.length) return;

    const w = width - MARGIN.left - MARGIN.right;
    const h = height - MARGIN.top - MARGIN.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const values = config.data.map((d: any) =>
      parseFloat(d[config.dataKey || 'value'])
    );

    // Create histogram bins
    const histogram = d3
      .histogram()
      .domain([d3.min(values)!, d3.max(values)!])
      .thresholds(15);

    const bins = histogram(values);

    const xScale = d3
      .scaleLinear()
      .domain([d3.min(values)!, d3.max(values)!] as [number, number])
      .range([0, w]);

    const yScale = d3
      .scaleLinear()
      .domain([0, d3.max(bins, (d) => d.length) || 1] as [number, number])
      .range([h, 0]);

    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale))
      .style('font-size', '10px');

    g.append('g')
      .call(d3.axisLeft(yScale))
      .style('font-size', '10px');

    // Bars
    g.selectAll('.bar')
      .data(bins)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', (d) => xScale(d.x0!))
      .attr('y', (d) => yScale(d.length))
      .attr('width', (d) => Math.max(0, xScale(d.x1!) - xScale(d.x0!)))
      .attr('height', (d) => h - yScale(d.length))
      .attr('fill', config.colors?.[0] || '#06b6d4')
      .attr('opacity', 0.7)
      .on('click', (_event: any, d: any) => {
        // Get data points in this bin
        const binData = config.data.filter(
          (row: any) =>
            parseFloat(row[config.dataKey || 'value']) >= d.x0 &&
            parseFloat(row[config.dataKey || 'value']) < d.x1
        );
        onBrush?.(binData);
      });
  }, [config, onBrush, highlightedIndices, width, height]);

  return (
    <div className="bg-[#0f1420] rounded p-2">
      <div className="text-sm font-semibold text-cyan-400 mb-2">
        {config.title}
      </div>
      <svg ref={svgRef} className="w-full" />
    </div>
  );
};

/**
 * Pie Chart with d3.js
 */
export const PieChart: React.FC<D3ChartProps> = ({
  config,
  onBrush,
  width = 500,
  height = 300,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !config.data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const radius = Math.min(width, height) / 2 - 20;

    const pie = d3
      .pie<any>()
      .value((d) => parseFloat(d[config.dataKey || 'value']));

    const arc = d3
      .arc<any>()
      .innerRadius(0)
      .outerRadius(radius);

    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${width / 2},${height / 2})`);

    const data = pie(config.data);

    g.selectAll('.arc')
      .data(data)
      .enter()
      .append('path')
      .attr('class', 'arc')
      .attr('d', arc)
      .attr('fill', (_d, i) => config.colors?.[i % config.colors.length] || d3.schemeCategory10[i % 10])
      .attr('opacity', 0.7)
      .on('click', (_event: any, d: any) => {
        onBrush?.([d.data]);
      });
  }, [config, onBrush, width, height]);

  return (
    <div className="bg-[#0f1420] rounded p-2">
      <div className="text-sm font-semibold text-cyan-400 mb-2">
        {config.title}
      </div>
      <svg ref={svgRef} className="w-full" />
    </div>
  );
};

/**
 * Factory component that selects the right chart type
 */
export const D3ChartComponent: React.FC<D3ChartProps> = (props) => {
  const { config } = props;

  switch (config.type) {
    case 'line':
      return <LineChart {...props} />;
    case 'scatter':
      return <ScatterChart {...props} />;
    case 'bar':
      return <BarChart {...props} />;
    case 'histogram':
      return <Histogram {...props} />;
    case 'pie':
      return <PieChart {...props} />;
    default:
      return <LineChart {...props} />;
  }
};
