'use client';

import React from 'react';
import type { ObservationResult } from '@/engine/types';

interface ObservationPanelProps {
  result: ObservationResult | null;
}

function MetricBar({
  label,
  value,
  color,
  max = 1,
}: {
  label: string;
  value: number;
  color: string;
  max?: number;
}) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-gray-500 uppercase tracking-wider">{label}</span>
        <span className={color}>{value.toFixed(4)}</span>
      </div>
      <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color.replace('text-', 'bg-')}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function StatCell({
  label,
  value,
  color = 'text-gray-300',
  wide = false,
}: {
  label: string;
  value: string;
  color?: string;
  wide?: boolean;
}) {
  return (
    <div className={`p-3 bg-gray-900/50 ${wide ? 'col-span-2' : ''}`}>
      <div className="text-[9px] text-gray-600 uppercase tracking-widest mb-1">
        {label}
      </div>
      <div className={`text-sm font-semibold ${color}`}>{value}</div>
    </div>
  );
}

export default function ObservationPanel({ result }: ObservationPanelProps) {
  if (!result) {
    return (
      <div className="space-y-4">
        <h3 className="text-xs text-gray-600 uppercase tracking-widest">
          Metrics
        </h3>
        <div className="text-gray-700 text-sm text-center py-8">
          Drop an image to begin observation
        </div>
      </div>
    );
  }

  const conservationOk = Math.abs(result.conservation - 1.0) < 1e-3;

  return (
    <div className="space-y-4">
      <h3 className="text-xs text-gray-600 uppercase tracking-widest">
        Entropy Coordinates
      </h3>

      <MetricBar label="Sk (kinetic)" value={result.S_k} color="text-cyan-400" />
      <MetricBar label="St (thermal)" value={result.S_t} color="text-amber-400" />
      <MetricBar label="Se (electronic)" value={result.S_e} color="text-violet-400" />

      <h3 className="text-xs text-gray-600 uppercase tracking-widest mt-6">
        Quality Metrics
      </h3>

      <div className="grid grid-cols-2 gap-[1px] bg-gray-800 rounded overflow-hidden">
        <StatCell
          label="Conservation"
          value={result.conservation.toFixed(4)}
          color={conservationOk ? 'text-green-400' : 'text-red-400'}
        />
        <StatCell
          label="Coherence"
          value={`${(result.coherence * 100).toFixed(1)}%`}
          color="text-green-400"
        />
        <StatCell
          label="Partition Depth"
          value={result.partitionDepth.toFixed(1)}
          color="text-cyan-400"
        />
        <StatCell
          label="Sharpness"
          value={result.sharpness.toFixed(3)}
          color="text-gray-300"
        />
        <StatCell
          label="Noise"
          value={result.noise.toFixed(4)}
          color="text-gray-300"
        />
        <StatCell
          label="Visibility"
          value={`${(result.visibility * 100).toFixed(1)}%`}
          color="text-green-400"
        />
        <StatCell
          label="Elapsed"
          value={`${result.elapsed_ms.toFixed(1)} ms`}
          color="text-gray-400"
          wide
        />
      </div>
    </div>
  );
}
