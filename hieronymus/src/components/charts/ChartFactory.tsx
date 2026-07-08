'use client';

import React from 'react';
import { LineChart, ScatterChart, BarChart, Histogram, PieChart, D3ChartComponent, ChartConfig } from './ChartFactory-d3';

export type ChartType = 'line' | 'bar' | 'scatter' | 'area' | 'pie' | 'histogram' | 'parallel' | 'radar' | 'treemap' | 'composed';
export type { ChartConfig };

interface ChartComponentProps {
  config: ChartConfig;
  onBrush?: (selection: any[]) => void;
  highlightedIndices?: number[];
}

export function ChartComponent({ config, onBrush, highlightedIndices = [] }: ChartComponentProps) {
  return (
    <D3ChartComponent
      config={config}
      onBrush={onBrush}
      highlightedIndices={highlightedIndices}
      width={450}
      height={300}
    />
  );
}

export default ChartComponent;
