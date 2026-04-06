'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';

const encoders = [
  {
    name: 'Microscopy',
    href: '/observe/microscopy',
    status: 'Active',
    color: 'border-cyan-400 text-cyan-400',
    description:
      'Fluorescence, brightfield, and phase-contrast microscopy images. Maps local entropy to partition coordinates.',
  },
  {
    name: 'Spectroscopy',
    href: '/observe',
    status: 'Coming Soon',
    color: 'border-amber-400/30 text-amber-500/50',
    description:
      'UV-Vis, Raman, and IR spectral data. Encodes absorption peaks as harmonic oscillator quantum numbers.',
  },
  {
    name: 'Mass Spectrometry',
    href: '/observe',
    status: 'Coming Soon',
    color: 'border-violet-400/30 text-violet-500/50',
    description:
      'm/z fragmentation patterns encoded as partition coordinates for compound matching.',
  },
];

export default function ObservePage() {
  return (
    <div className="min-h-[calc(100vh-140px)] flex flex-col items-center justify-center px-8 py-16">
      <motion.h1
        className="text-4xl font-bold mb-2"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        Observation Workspace
      </motion.h1>
      <motion.p
        className="text-gray-500 mb-12"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        Select a domain encoder to begin.
      </motion.p>

      <div className="grid grid-cols-3 gap-6 max-w-5xl w-full lg:grid-cols-1 lg:max-w-lg">
        {encoders.map((enc, i) => (
          <motion.div
            key={enc.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 * i }}
          >
            <Link
              href={enc.href}
              className={`block p-6 border rounded-lg ${enc.color} bg-dark/50 hover:bg-gray-900/80 transition-colors`}
            >
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-semibold">{enc.name}</h3>
                <span className="text-xs tracking-wider opacity-60">
                  {enc.status}
                </span>
              </div>
              <p className="text-gray-500 text-sm leading-relaxed">
                {enc.description}
              </p>
            </Link>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
