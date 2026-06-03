'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChartComponent } from './ChartFactory';
import { useChartManager } from './ChartManager';

export default function ChartGrid() {
  const { getCharts, highlightedIndices, onChartBrush } = useChartManager();
  const charts = getCharts();

  if (charts.length === 0) {
    return (
      <div className="h-full flex items-center justify-center bg-[#0a0e27]">
        <div className="text-center">
          <div className="text-gray-600 text-lg font-semibold mb-2">
            No charts yet
          </div>
          <div className="text-gray-700 text-sm">
            Run your analysis script to generate charts
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full w-full overflow-auto bg-[#0a0e27] p-4">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 auto-rows-max">
        <AnimatePresence>
          {charts.map((chart) => (
            <motion.div
              key={chart.id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.3 }}
              className="bg-[#0f1420] border border-gray-800/50 rounded-lg p-4 overflow-hidden shadow-lg hover:border-gray-700 transition-colors"
            >
              {/* Chart header */}
              <div className="mb-4">
                <h3 className="text-sm font-semibold text-cyan-400 tracking-wide">
                  {chart.title}
                </h3>
                <div className="text-[9px] text-gray-600 mt-1">
                  {chart.type.charAt(0).toUpperCase() + chart.type.slice(1)} •{' '}
                  {chart.data.length} records
                </div>
              </div>

              {/* Chart */}
              <div className="bg-[#050810] rounded p-2">
                <ChartComponent
                  config={chart}
                  onBrush={onChartBrush}
                  highlightedIndices={highlightedIndices}
                />
              </div>

              {/* Chart info */}
              {chart.series && (
                <div className="mt-3 pt-3 border-t border-gray-800/50">
                  <div className="text-[8px] text-gray-600 mb-2">Series</div>
                  <div className="flex flex-wrap gap-2">
                    {chart.series.map((s, i) => (
                      <div
                        key={s.dataKey}
                        className="px-2 py-1 bg-gray-900 rounded text-[8px] text-gray-400 flex items-center gap-1"
                      >
                        <div
                          className="w-2 h-2 rounded-full"
                          style={{ backgroundColor: s.color || '#06b6d4' }}
                        />
                        {s.name}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
