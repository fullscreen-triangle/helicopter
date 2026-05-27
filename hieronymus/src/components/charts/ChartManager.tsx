'use client';

import React, { createContext, useContext, useCallback, useState } from 'react';
import { ChartConfig, ChartType } from './ChartFactory';

interface ChartManagerContextType {
  charts: Map<string, ChartConfig>;
  addChart: (config: ChartConfig) => void;
  updateChart: (id: string, data: any[], options?: Record<string, any>) => void;
  removeChart: (id: string) => void;
  clearCharts: () => void;
  getChart: (id: string) => ChartConfig | undefined;
  getCharts: () => ChartConfig[];
}

const ChartManagerContext = createContext<ChartManagerContextType | null>(null);

export function ChartManagerProvider({ children }: { children: React.ReactNode }) {
  const [charts, setCharts] = useState<Map<string, ChartConfig>>(new Map());

  const addChart = useCallback((config: ChartConfig) => {
    setCharts((prev) => {
      const next = new Map(prev);
      next.set(config.id, config);
      return next;
    });
  }, []);

  const updateChart = useCallback(
    (id: string, data: any[], options?: Record<string, any>) => {
      setCharts((prev) => {
        const next = new Map(prev);
        const chart = next.get(id);
        if (chart) {
          next.set(id, {
            ...chart,
            data,
            options: { ...chart.options, ...options },
          });
        }
        return next;
      });
    },
    []
  );

  const removeChart = useCallback((id: string) => {
    setCharts((prev) => {
      const next = new Map(prev);
      next.delete(id);
      return next;
    });
  }, []);

  const clearCharts = useCallback(() => {
    setCharts(new Map());
  }, []);

  const getChart = useCallback(
    (id: string) => charts.get(id),
    [charts]
  );

  const getCharts = useCallback(() => Array.from(charts.values()), [charts]);

  return (
    <ChartManagerContext.Provider
      value={{
        charts,
        addChart,
        updateChart,
        removeChart,
        clearCharts,
        getChart,
        getCharts,
      }}
    >
      {children}
    </ChartManagerContext.Provider>
  );
}

export function useChartManager() {
  const context = useContext(ChartManagerContext);
  if (!context) {
    throw new Error('useChartManager must be used within ChartManagerProvider');
  }
  return context;
}

/**
 * Simple DSL for chart creation within script execution
 * Usage in code:
 *
 * const c = createChartBuilder(chartManager);
 * c.line('my_chart')
 *   .title('Revenue Trend')
 *   .data([{month: 'Jan', value: 100}, ...])
 *   .x('month')
 *   .y('value')
 *   .build();
 *
 * c.bar('another_chart')
 *   .data(data)
 *   .series([{name: 'Sales', key: 'sales'}, {name: 'Costs', key: 'costs'}])
 *   .build();
 */
export class ChartBuilder {
  private config: Partial<ChartConfig>;
  private manager: ChartManagerContextType;

  constructor(id: string, type: ChartType, manager: ChartManagerContextType) {
    this.manager = manager;
    this.config = {
      id,
      type,
      title: id,
      data: [],
      xAxisKey: 'name',
      dataKey: 'value',
    };
  }

  title(title: string): this {
    this.config.title = title;
    return this;
  }

  data(data: any[]): this {
    this.config.data = data;
    return this;
  }

  x(key: string): this {
    this.config.xAxisKey = key;
    return this;
  }

  y(key: string): this {
    this.config.yAxisKey = key;
    this.config.dataKey = key;
    return this;
  }

  key(key: string): this {
    this.config.dataKey = key;
    return this;
  }

  series(
    series: Array<{
      name: string;
      dataKey: string;
      color?: string;
    }>
  ): this {
    this.config.series = series;
    return this;
  }

  colors(colors: string[]): this {
    this.config.colors = colors;
    return this;
  }

  options(options: Record<string, any>): this {
    this.config.options = options;
    return this;
  }

  build(): ChartConfig {
    const config = this.config as ChartConfig;
    this.manager.addChart(config);
    return config;
  }

  update(data: any[], options?: Record<string, any>): this {
    this.manager.updateChart(this.config.id as string, data, options);
    return this;
  }
}

export function createChartBuilder(manager: ChartManagerContextType) {
  return {
    line: (id: string) => new ChartBuilder(id, 'line', manager),
    bar: (id: string) => new ChartBuilder(id, 'bar', manager),
    scatter: (id: string) => new ChartBuilder(id, 'scatter', manager),
    area: (id: string) => new ChartBuilder(id, 'area', manager),
    pie: (id: string) => new ChartBuilder(id, 'pie', manager),
    radar: (id: string) => new ChartBuilder(id, 'radar', manager),
    treemap: (id: string) => new ChartBuilder(id, 'treemap', manager),
    composed: (id: string) => new ChartBuilder(id, 'composed', manager),
  };
}
