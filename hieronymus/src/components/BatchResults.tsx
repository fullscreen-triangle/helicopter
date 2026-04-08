'use client';

import React from 'react';

export interface BatchResultItem {
  id: string;
  name: string;
  domain: string;
  distance: number;
  S_k: number;
  S_t: number;
  S_e: number;
}

interface BatchResultsProps {
  results: BatchResultItem[];
  onSelect?: (item: BatchResultItem) => void;
}

function MiniEntropyBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="h-1 w-12 bg-gray-800 rounded-full overflow-hidden inline-block ml-1">
      <div
        className={`h-full rounded-full ${color}`}
        style={{ width: `${Math.min(value * 100, 100)}%` }}
      />
    </div>
  );
}

export default function BatchResults({ results, onSelect }: BatchResultsProps) {
  if (results.length === 0) {
    return (
      <div className="text-gray-700 text-sm text-center py-8">
        No results yet. Store some observations and then match against the
        database.
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <div className="grid grid-cols-[40px_1fr_100px_80px_150px] gap-2 px-3 py-2 text-[9px] text-gray-600 uppercase tracking-widest border-b border-gray-800/50">
        <span>Rank</span>
        <span>Name</span>
        <span>Domain</span>
        <span>S-Dist</span>
        <span>S-Entropy</span>
      </div>

      {results.map((item, idx) => {
        const isTop = idx === 0;
        return (
          <button
            key={item.id}
            onClick={() => onSelect?.(item)}
            className={`w-full grid grid-cols-[40px_1fr_100px_80px_150px] gap-2 px-3 py-2.5 text-left rounded transition-colors duration-150
              ${
                isTop
                  ? 'bg-amber-400/5 border border-amber-400/20 hover:bg-amber-400/10'
                  : 'hover:bg-gray-900/50 border border-transparent'
              }`}
          >
            <span
              className={`text-sm font-bold ${
                isTop ? 'text-amber-400' : 'text-gray-600'
              }`}
            >
              {idx + 1}
            </span>
            <span
              className={`text-sm truncate ${
                isTop ? 'text-amber-400' : 'text-gray-300'
              }`}
            >
              {item.name}
            </span>
            <span className="text-xs text-gray-500 truncate">{item.domain}</span>
            <span
              className={`text-xs font-mono ${
                isTop ? 'text-green-400' : 'text-gray-400'
              }`}
            >
              {item.distance.toFixed(4)}
            </span>
            <div className="flex items-center gap-1 text-[10px]">
              <span className="text-cyan-400">{item.S_k.toFixed(2)}</span>
              <MiniEntropyBar value={item.S_k} color="bg-cyan-400" />
              <span className="text-amber-400">{item.S_t.toFixed(2)}</span>
              <MiniEntropyBar value={item.S_t} color="bg-amber-400" />
              <span className="text-violet-400">{item.S_e.toFixed(2)}</span>
              <MiniEntropyBar value={item.S_e} color="bg-violet-400" />
            </div>
          </button>
        );
      })}
    </div>
  );
}
