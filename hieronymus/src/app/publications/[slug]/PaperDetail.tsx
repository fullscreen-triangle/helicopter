'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import type { Paper } from '@/data/papers';

const accentColors = [
  'text-cyan-400 border-cyan-400/40',
  'text-amber-400 border-amber-400/40',
  'text-violet-400 border-violet-400/40',
  'text-emerald-400 border-emerald-400/40',
  'text-rose-400 border-rose-400/40',
];

const badgeColors = [
  'bg-cyan-400/10 text-cyan-400',
  'bg-amber-400/10 text-amber-400',
  'bg-violet-400/10 text-violet-400',
  'bg-emerald-400/10 text-emerald-400',
  'bg-rose-400/10 text-rose-400',
];

const fade = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.5 },
  }),
};

export default function PaperDetail({ paper }: { paper: Paper }) {
  const ci = paper.number - 1;
  const accent = accentColors[ci % accentColors.length];
  const badge = badgeColors[ci % badgeColors.length];

  return (
    <div className="min-h-[calc(100vh-140px)] px-8 py-16 max-w-4xl mx-auto sm:px-4">
      {/* Back link */}
      <motion.div initial="hidden" animate="visible" variants={fade} custom={0}>
        <Link
          href="/publications"
          className="text-sm text-gray-500 hover:text-gray-300 transition-colors"
        >
          &larr; All publications
        </Link>
      </motion.div>

      {/* Paper number badge */}
      <motion.div
        className="mt-6 mb-4"
        initial="hidden"
        animate="visible"
        variants={fade}
        custom={1}
      >
        <span
          className={`inline-block px-3 py-1 rounded text-xs font-mono tracking-widest uppercase ${badge}`}
        >
          Paper {paper.number} of 5
        </span>
      </motion.div>

      {/* Title */}
      <motion.h1
        className="text-3xl font-bold mb-2 leading-tight sm:text-2xl"
        initial="hidden"
        animate="visible"
        variants={fade}
        custom={2}
      >
        {paper.title}
      </motion.h1>
      <motion.p
        className="text-gray-400 mb-2 leading-relaxed"
        initial="hidden"
        animate="visible"
        variants={fade}
        custom={3}
      >
        {paper.subtitle}
      </motion.p>
      <motion.p
        className="text-sm text-gray-600 mb-8"
        initial="hidden"
        animate="visible"
        variants={fade}
        custom={3}
      >
        Kundai Farai Sachikonye &middot; AIMe Registry for Artificial
        Intelligence
      </motion.p>

      {/* Metrics row */}
      <motion.div
        className="grid grid-cols-5 gap-4 mb-10 py-4 px-4 rounded-lg border border-gray-800/60 bg-gray-900/30 sm:grid-cols-3"
        initial="hidden"
        animate="visible"
        variants={fade}
        custom={4}
      >
        {[
          { v: paper.lines.toLocaleString(), l: 'Lines' },
          { v: paper.theorems, l: 'Theorems' },
          { v: paper.refs, l: 'References' },
          { v: paper.panels, l: 'Panels' },
          { v: paper.results, l: 'Result Files' },
        ].map((m) => (
          <div key={m.l} className="text-center">
            <span className={`block text-lg font-semibold ${accent.split(' ')[0]}`}>
              {m.v}
            </span>
            <span className="text-[10px] text-gray-600 uppercase tracking-wider">
              {m.l}
            </span>
          </div>
        ))}
      </motion.div>

      {/* Key Result */}
      <motion.div
        className="mb-10"
        initial="hidden"
        animate="visible"
        variants={fade}
        custom={5}
      >
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Key Result
        </h2>
        <div className={`pl-4 border-l-2 ${accent.split(' ')[1]}`}>
          <p className="text-light leading-relaxed">{paper.keyResult}</p>
        </div>
      </motion.div>

      {/* Abstract */}
      <motion.div
        className="mb-10"
        initial="hidden"
        animate="visible"
        variants={fade}
        custom={6}
      >
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Abstract
        </h2>
        <p className="text-gray-300 leading-relaxed text-sm">{paper.abstract}</p>
      </motion.div>

      {/* Key Theorems */}
      <motion.div
        className="mb-10"
        initial="hidden"
        animate="visible"
        variants={fade}
        custom={7}
      >
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Key Theorems
        </h2>
        <ul className="space-y-3">
          {paper.keyTheorems.map((thm, i) => (
            <li key={i} className="flex items-start gap-3">
              <span
                className={`flex-shrink-0 w-6 h-6 rounded flex items-center justify-center text-xs font-mono ${badge}`}
              >
                {i + 1}
              </span>
              <span className="text-gray-300 text-sm leading-relaxed">
                {thm}
              </span>
            </li>
          ))}
        </ul>
      </motion.div>

      {/* Validation Results */}
      <motion.div
        className="mb-10"
        initial="hidden"
        animate="visible"
        variants={fade}
        custom={8}
      >
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Validation Results
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-1">
          {Object.entries(paper.validation).map(([key, value]) => (
            <div
              key={key}
              className="flex items-center justify-between px-4 py-3 rounded border border-gray-800/60 bg-gray-900/30"
            >
              <span className="text-xs text-gray-500 font-mono">{key}</span>
              <span className="text-sm font-semibold text-light font-mono">
                {value}
              </span>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Panels */}
      <motion.div
        className="mb-10"
        initial="hidden"
        animate="visible"
        variants={fade}
        custom={9}
      >
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Figure Panels
        </h2>
        <div className="grid grid-cols-5 gap-3 sm:grid-cols-2">
          {paper.panelNames.map((name, i) => (
            <div
              key={i}
              className="flex flex-col items-center justify-center p-4 rounded border border-gray-800/60 bg-gray-900/30 text-center"
            >
              <span className={`text-2xl font-bold mb-1 ${accent.split(' ')[0]}`}>
                {i + 1}
              </span>
              <span className="text-[10px] text-gray-500 leading-tight">
                {name}
              </span>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Navigation */}
      <motion.div
        className="flex items-center justify-between pt-8 border-t border-gray-800/60"
        initial="hidden"
        animate="visible"
        variants={fade}
        custom={10}
      >
        <Link
          href="/publications"
          className="text-sm text-gray-500 hover:text-gray-300 transition-colors"
        >
          &larr; All publications
        </Link>
        <div className="flex gap-4">
          {paper.number > 1 && (
            <Link
              href={`/publications/${
                ['measurement-modalities-stereogram', 'image-harmonic-coupling', 'universal-spectral-matching', 'gpu-observation-architecture', 'ray-tracing-cellular-computing'][paper.number - 2]
              }`}
              className="text-sm text-gray-500 hover:text-gray-300 transition-colors"
            >
              &larr; Previous
            </Link>
          )}
          {paper.number < 5 && (
            <Link
              href={`/publications/${
                ['measurement-modalities-stereogram', 'image-harmonic-coupling', 'universal-spectral-matching', 'gpu-observation-architecture', 'ray-tracing-cellular-computing'][paper.number]
              }`}
              className={`text-sm hover:text-gray-300 transition-colors ${accent.split(' ')[0]}`}
            >
              Next &rarr;
            </Link>
          )}
        </div>
      </motion.div>
    </div>
  );
}
