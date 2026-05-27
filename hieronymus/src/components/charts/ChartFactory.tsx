'use client';

import React, { ReactNode } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  AreaChart,
  Area,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Treemap,
} from 'recharts';

export type ChartType =
  | 'line'
  | 'bar'
  | 'scatter'
  | 'area'
  | 'pie'
  | 'radar'
  | 'treemap'
  | 'composed';

export interface ChartConfig {
  id: string;
  type: ChartType;
  title: string;
  data: any[];
  dataKey?: string;
  xAxisKey?: string;
  yAxisKey?: string;
  series?: Array<{
    name: string;
    dataKey: string;
    color?: string;
  }>;
  colors?: string[];
  options?: Record<string, any>;
}

const COLORS = [
  '#06b6d4', // cyan
  '#ec4899', // pink
  '#f59e0b', // amber
  '#10b981', // emerald
  '#8b5cf6', // violet
  '#ef4444', // red
  '#3b82f6', // blue
  '#14b8a6', // teal
];

const GridConfig = {
  strokeDasharray: '3 3',
  stroke: '#374151',
  vertical: true,
  horizontal: true,
};

const TooltipConfig = {
  contentStyle: {
    backgroundColor: '#1f2937',
    border: '1px solid #4b5563',
    borderRadius: '4px',
    color: '#e5e7eb',
  },
};

function LineChartComponent({ config }: { config: ChartConfig }) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={config.data} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
        <CartesianGrid {...GridConfig} />
        <XAxis dataKey={config.xAxisKey || 'name'} stroke="#9ca3af" />
        <YAxis stroke="#9ca3af" />
        <Tooltip {...TooltipConfig} />
        <Legend />
        {config.series ? (
          config.series.map((s, i) => (
            <Line
              key={s.dataKey}
              type="monotone"
              dataKey={s.dataKey}
              stroke={s.color || COLORS[i % COLORS.length]}
              dot={false}
              strokeWidth={2}
            />
          ))
        ) : (
          <Line
            type="monotone"
            dataKey={config.dataKey || 'value'}
            stroke={COLORS[0]}
            dot={false}
            strokeWidth={2}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}

function BarChartComponent({ config }: { config: ChartConfig }) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={config.data} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
        <CartesianGrid {...GridConfig} />
        <XAxis dataKey={config.xAxisKey || 'name'} stroke="#9ca3af" />
        <YAxis stroke="#9ca3af" />
        <Tooltip {...TooltipConfig} />
        <Legend />
        {config.series ? (
          config.series.map((s, i) => (
            <Bar
              key={s.dataKey}
              dataKey={s.dataKey}
              fill={s.color || COLORS[i % COLORS.length]}
            />
          ))
        ) : (
          <Bar dataKey={config.dataKey || 'value'} fill={COLORS[0]} />
        )}
      </BarChart>
    </ResponsiveContainer>
  );
}

function ScatterChartComponent({ config }: { config: ChartConfig }) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart data={config.data} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
        <CartesianGrid {...GridConfig} />
        <XAxis dataKey={config.xAxisKey || 'x'} stroke="#9ca3af" />
        <YAxis dataKey={config.yAxisKey || 'y'} stroke="#9ca3af" />
        <Tooltip {...TooltipConfig} />
        {config.series ? (
          config.series.map((s, i) => (
            <Scatter
              key={s.dataKey}
              dataKey={s.dataKey}
              fill={s.color || COLORS[i % COLORS.length]}
            />
          ))
        ) : (
          <Scatter dataKey={config.dataKey || 'value'} fill={COLORS[0]} />
        )}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function AreaChartComponent({ config }: { config: ChartConfig }) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <AreaChart data={config.data} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
        <CartesianGrid {...GridConfig} />
        <XAxis dataKey={config.xAxisKey || 'name'} stroke="#9ca3af" />
        <YAxis stroke="#9ca3af" />
        <Tooltip {...TooltipConfig} />
        <Legend />
        {config.series ? (
          config.series.map((s, i) => (
            <Area
              key={s.dataKey}
              type="monotone"
              dataKey={s.dataKey}
              stroke={s.color || COLORS[i % COLORS.length]}
              fill={s.color || COLORS[i % COLORS.length]}
              fillOpacity={0.4}
            />
          ))
        ) : (
          <Area
            type="monotone"
            dataKey={config.dataKey || 'value'}
            stroke={COLORS[0]}
            fill={COLORS[0]}
            fillOpacity={0.4}
          />
        )}
      </AreaChart>
    </ResponsiveContainer>
  );
}

function PieChartComponent({ config }: { config: ChartConfig }) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <PieChart>
        <Pie
          data={config.data}
          dataKey={config.dataKey || 'value'}
          nameKey={config.xAxisKey || 'name'}
          cx="50%"
          cy="50%"
          outerRadius={120}
          label
        >
          {config.data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip {...TooltipConfig} />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
}

function RadarChartComponent({ config }: { config: ChartConfig }) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <RadarChart data={config.data}>
        <PolarGrid stroke="#4b5563" />
        <PolarAngleAxis dataKey={config.xAxisKey || 'name'} stroke="#9ca3af" />
        <PolarRadiusAxis stroke="#9ca3af" />
        <Radar
          name={config.series?.[0]?.name || 'Value'}
          dataKey={config.dataKey || 'value'}
          stroke={COLORS[0]}
          fill={COLORS[0]}
          fillOpacity={0.5}
        />
        <Tooltip {...TooltipConfig} />
        <Legend />
      </RadarChart>
    </ResponsiveContainer>
  );
}

function TreemapChartComponent({ config }: { config: ChartConfig }) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <Treemap
        data={config.data}
        dataKey={config.dataKey || 'value'}
        nameKey={config.xAxisKey || 'name'}
        stroke="#fff"
        fill="#8884d8"
      >
        {config.data.map((entry, index) => (
          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
        ))}
      </Treemap>
    </ResponsiveContainer>
  );
}

function ComposedChartComponent({ config }: { config: ChartConfig }) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <ComposedChart data={config.data} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
        <CartesianGrid {...GridConfig} />
        <XAxis dataKey={config.xAxisKey || 'name'} stroke="#9ca3af" />
        <YAxis stroke="#9ca3af" />
        <Tooltip {...TooltipConfig} />
        <Legend />
        {config.series?.map((s, i) => {
          const color = s.color || COLORS[i % COLORS.length];
          if (config.options?.seriesType?.[s.dataKey] === 'bar') {
            return <Bar key={s.dataKey} dataKey={s.dataKey} fill={color} />;
          }
          return <Line key={s.dataKey} type="monotone" dataKey={s.dataKey} stroke={color} />;
        })}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

interface ChartComponentProps {
  config: ChartConfig;
}

export function ChartComponent({ config }: ChartComponentProps): ReactNode {
  const commonProps = { config };

  switch (config.type) {
    case 'line':
      return <LineChartComponent {...commonProps} />;
    case 'bar':
      return <BarChartComponent {...commonProps} />;
    case 'scatter':
      return <ScatterChartComponent {...commonProps} />;
    case 'area':
      return <AreaChartComponent {...commonProps} />;
    case 'pie':
      return <PieChartComponent {...commonProps} />;
    case 'radar':
      return <RadarChartComponent {...commonProps} />;
    case 'treemap':
      return <TreemapChartComponent {...commonProps} />;
    case 'composed':
      return <ComposedChartComponent {...commonProps} />;
    default:
      return <LineChartComponent {...commonProps} />;
  }
}

export default ChartComponent;
