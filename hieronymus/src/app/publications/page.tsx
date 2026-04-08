'use client';

import { motion } from 'framer-motion';
import PaperCard from '@/components/PaperCard';
import { papers, totalStats } from '@/data/papers';

const statItems = [
  { label: 'Papers', value: totalStats.papers },
  { label: 'Lines', value: totalStats.lines.toLocaleString() },
  { label: 'Theorems', value: totalStats.theorems },
  { label: 'References', value: totalStats.refs },
  { label: 'Panels', value: totalStats.panels },
  { label: 'Result Files', value: totalStats.results },
];

export default function PublicationsPage() {
  return (
    <div className="min-h-[calc(100vh-140px)] px-8 py-16 max-w-6xl mx-auto sm:px-4">
      {/* Header */}
      <motion.div
        className="text-center mb-12"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-4xl font-bold mb-3 sm:text-3xl">Publications</h1>
        <p className="text-gray-500 max-w-2xl mx-auto mb-8 leading-relaxed">
          Five papers establishing the mathematical foundations and engineering
          architecture for universal observation through GPU fragment shaders.
          Every theorem is derived from two axioms: bounded phase space and
          categorical observation.
        </p>

        {/* Aggregate stats bar */}
        <motion.div
          className="flex flex-wrap justify-center gap-6 py-4 px-6 rounded-lg border border-gray-800/60 bg-gray-900/30 max-w-3xl mx-auto sm:gap-3"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          {statItems.map((s) => (
            <div key={s.label} className="text-center">
              <span className="block text-lg font-semibold text-cyan-400 sm:text-base">
                {s.value}
              </span>
              <span className="text-[10px] text-gray-600 uppercase tracking-wider">
                {s.label}
              </span>
            </div>
          ))}
        </motion.div>
      </motion.div>

      {/* Papers grid */}
      <div className="grid grid-cols-2 gap-6 lg:grid-cols-1 lg:max-w-2xl lg:mx-auto">
        {papers.map((paper, i) => (
          <PaperCard key={paper.slug} paper={paper} index={i} />
        ))}
      </div>

      {/* Bottom note */}
      <motion.p
        className="text-center text-gray-600 text-xs mt-12 tracking-wider"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
      >
        All papers by Kundai Farai Sachikonye &middot; AIMe Registry for
        Artificial Intelligence
      </motion.p>
    </div>
  );
}
