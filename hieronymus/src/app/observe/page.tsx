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
      'Fluorescence, brightfield, and phase-contrast microscopy images. Maps local entropy, gradients, and Laplacian to partition coordinates (n,l,m,s).',
  },
  {
    name: 'Molecular',
    href: '/observe/molecular',
    status: 'Active',
    color: 'border-amber-400 text-amber-400',
    description:
      'Spectral frequency data: Raman, IR, UV-Vis absorption peaks. Encodes frequency-amplitude pairs as spectral images for GPU observation.',
  },
  {
    name: 'Genomic',
    href: '/observe/genomic',
    status: 'Active',
    color: 'border-green-400 text-green-400',
    description:
      'Nucleotide sequences in FASTA or raw format. Computes trinucleotide (k=3) frequencies and maps k-mer distribution to partition coordinates.',
  },
  {
    name: 'Signal',
    href: '/observe/signal',
    status: 'Active',
    color: 'border-violet-400 text-violet-400',
    description:
      'Time series and waveform data. Computes windowed FFT spectrogram and maps frequency-time power distribution through the observation pipeline.',
  },
  {
    name: 'General',
    href: '/observe/general',
    status: 'Active',
    color: 'border-rose-400 text-rose-400',
    description:
      'Any numeric data: CSV matrices, JSON arrays, or vectors. Reshapes and normalizes to a 2D image for domain-agnostic observation.',
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
