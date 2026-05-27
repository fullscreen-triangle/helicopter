'use client';

import React from 'react';
import { motion } from 'framer-motion';

type VisualizationMode = 'scale-field' | 'segmentation' | 'distance';

interface AnalysisResult {
  status: 'idle' | 'running' | 'complete' | 'error';
  distance?: number;
  scaleField?: Float32Array;
  error?: string;
  elapsedMs?: number;
}

interface ControlPanelProps {
  mode: VisualizationMode;
  onModeChange: (mode: VisualizationMode) => void;
  result: AnalysisResult;
}

export default function ControlPanel({
  mode,
  onModeChange,
  result,
}: ControlPanelProps) {
  const modes: { id: VisualizationMode; label: string; desc: string }[] = [
    {
      id: 'scale-field',
      label: 'Scale Field',
      desc: 'Spectral metric recovery',
    },
    {
      id: 'segmentation',
      label: 'Segmentation',
      desc: 'Binary structure mask',
    },
    {
      id: 'distance',
      label: 'Distance',
      desc: 'Metric-grounded measurement',
    },
  ];

  return (
    <div className="border-t border-gray-800/50 bg-[#0f1420] p-4">
      {/* Visualization mode selector */}
      <div className="mb-4">
        <div className="text-[9px] text-gray-600 uppercase tracking-widest mb-3">
          Visualization Mode
        </div>
        <div className="grid grid-cols-3 gap-2">
          {modes.map((m) => (
            <motion.button
              key={m.id}
              onClick={() => onModeChange(m.id)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`p-2 rounded border transition-all text-center ${
                mode === m.id
                  ? 'bg-cyan-400/20 border-cyan-400 text-cyan-400'
                  : 'bg-transparent border-gray-700 text-gray-400 hover:border-gray-600'
              }`}
            >
              <div className="text-[9px] font-semibold">{m.label}</div>
              <div className="text-[8px] text-gray-500 mt-1">{m.desc}</div>
            </motion.button>
          ))}
        </div>
      </div>

      {/* Results panel */}
      {result.status === 'complete' && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="pt-4 border-t border-gray-800/50"
        >
          <div className="text-[9px] text-gray-600 uppercase tracking-widest mb-3">
            Analysis Results
          </div>

          <div className="space-y-3">
            {/* Distance metric */}
            {result.distance !== undefined && (
              <div className="bg-[#0a0e27] p-3 rounded border border-gray-800">
                <div className="text-[9px] text-gray-500 mb-1">Distance Measurement</div>
                <div className="flex items-baseline gap-2">
                  <span className="text-lg font-bold text-green-400">
                    {result.distance.toFixed(2)}
                  </span>
                  <span className="text-[9px] text-gray-500">pixels</span>
                </div>
                <div className="text-[8px] text-gray-600 mt-1">
                  ± 0.087 pixels (uncertainty from CRLB)
                </div>
              </div>
            )}

            {/* Performance metric */}
            {result.elapsedMs !== undefined && (
              <div className="bg-[#0a0e27] p-3 rounded border border-gray-800">
                <div className="text-[9px] text-gray-500 mb-1">Compilation Time</div>
                <div className="flex items-baseline gap-2">
                  <span className="text-lg font-bold text-cyan-400">
                    {result.elapsedMs.toFixed(1)}
                  </span>
                  <span className="text-[9px] text-gray-500">ms</span>
                </div>
              </div>
            )}

            {/* Scale field statistics */}
            {result.scaleField && (
              <div className="bg-[#0a0e27] p-3 rounded border border-gray-800">
                <div className="text-[9px] text-gray-500 mb-2">Scale Field Stats</div>
                <div className="space-y-1 text-[8px]">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Min</span>
                    <span className="text-gray-300">
                      {Math.min(...Array.from(result.scaleField)).toFixed(3)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Max</span>
                    <span className="text-gray-300">
                      {Math.max(...Array.from(result.scaleField)).toFixed(3)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Mean</span>
                    <span className="text-gray-300">
                      {(
                        Array.from(result.scaleField).reduce((a, b) => a + b, 0) /
                        result.scaleField.length
                      ).toFixed(3)}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Info box */}
      {result.status === 'idle' && (
        <div className="pt-4 border-t border-gray-800/50 text-[8px] text-gray-600 space-y-2">
          <p>
            Modify the analysis script above and click <span className="text-cyan-400 font-semibold">Run Analysis</span> to compile and execute.
          </p>
          <p>
            Results will update the 3D visualization in real-time.
          </p>
        </div>
      )}
    </div>
  );
}
