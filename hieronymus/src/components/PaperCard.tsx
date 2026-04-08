'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import type { Paper } from '@/data/papers';

const accentColors = [
  'border-cyan-500/40 hover:border-cyan-400',
  'border-amber-500/40 hover:border-amber-400',
  'border-violet-500/40 hover:border-violet-400',
  'border-emerald-500/40 hover:border-emerald-400',
  'border-rose-500/40 hover:border-rose-400',
];

const numberColors = [
  'text-cyan-400',
  'text-amber-400',
  'text-violet-400',
  'text-emerald-400',
  'text-rose-400',
];

export default function PaperCard({
  paper,
  index,
}: {
  paper: Paper;
  index: number;
}) {
  const accent = accentColors[index % accentColors.length];
  const numColor = numberColors[index % numberColors.length];

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1, duration: 0.5 }}
    >
      <Link
        href={`/publications/${paper.slug}`}
        className={`block p-6 rounded-lg border bg-gray-900/50 backdrop-blur-sm transition-all duration-300 hover:bg-gray-900/80 hover:shadow-lg hover:shadow-black/20 hover:-translate-y-1 ${accent}`}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <span
            className={`text-xs font-mono tracking-widest uppercase ${numColor}`}
          >
            Paper {paper.number}
          </span>
          <span className="text-xs text-gray-600 font-mono">
            {paper.lines.toLocaleString()} lines
          </span>
        </div>

        {/* Title */}
        <h3 className="text-lg font-semibold text-light mb-1 leading-snug">
          {paper.title}
        </h3>
        <p className="text-xs text-gray-500 mb-4 leading-relaxed">
          {paper.subtitle}
        </p>

        {/* Key Result */}
        <div className="mb-4 pl-3 border-l-2 border-gray-700">
          <p className="text-sm text-gray-400 italic leading-relaxed">
            {paper.keyResult}
          </p>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-4 gap-2 text-center">
          <div>
            <span className="block text-sm font-semibold text-light">
              {paper.theorems}
            </span>
            <span className="text-[10px] text-gray-600 uppercase tracking-wider">
              Theorems
            </span>
          </div>
          <div>
            <span className="block text-sm font-semibold text-light">
              {paper.refs}
            </span>
            <span className="text-[10px] text-gray-600 uppercase tracking-wider">
              References
            </span>
          </div>
          <div>
            <span className="block text-sm font-semibold text-light">
              {paper.panels}
            </span>
            <span className="text-[10px] text-gray-600 uppercase tracking-wider">
              Panels
            </span>
          </div>
          <div>
            <span className="block text-sm font-semibold text-light">
              {paper.results}
            </span>
            <span className="text-[10px] text-gray-600 uppercase tracking-wider">
              Results
            </span>
          </div>
        </div>

        {/* Read more */}
        <div className="mt-4 pt-3 border-t border-gray-800/60 text-right">
          <span className={`text-xs tracking-wider ${numColor}`}>
            Read paper details &rarr;
          </span>
        </div>
      </Link>
    </motion.div>
  );
}
