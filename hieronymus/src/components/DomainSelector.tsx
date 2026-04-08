'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import React from 'react';

const domains = [
  {
    name: 'Microscopy',
    href: '/observe/microscopy',
    color: 'cyan',
    description: 'Fluorescence, brightfield, and phase-contrast microscopy images.',
  },
  {
    name: 'Molecular',
    href: '/observe/molecular',
    color: 'amber',
    description: 'Spectral frequency data: Raman, IR, UV-Vis absorption peaks.',
  },
  {
    name: 'Genomic',
    href: '/observe/genomic',
    color: 'green',
    description: 'Nucleotide sequences: FASTA, raw ACGT. K-mer frequency analysis.',
  },
  {
    name: 'Signal',
    href: '/observe/signal',
    color: 'violet',
    description: 'Time series data: sensor readings, waveforms, temporal signals.',
  },
  {
    name: 'General',
    href: '/observe/general',
    color: 'rose',
    description: 'Any numeric data: matrices, vectors, CSV tables.',
  },
];

const colorMap: Record<string, { active: string; inactive: string; border: string }> = {
  cyan: {
    active: 'text-cyan-400 border-cyan-400 bg-cyan-400/10',
    inactive: 'text-gray-500 border-gray-800 hover:border-gray-600 hover:text-gray-400',
    border: 'border-cyan-400',
  },
  amber: {
    active: 'text-amber-400 border-amber-400 bg-amber-400/10',
    inactive: 'text-gray-500 border-gray-800 hover:border-gray-600 hover:text-gray-400',
    border: 'border-amber-400',
  },
  green: {
    active: 'text-green-400 border-green-400 bg-green-400/10',
    inactive: 'text-gray-500 border-gray-800 hover:border-gray-600 hover:text-gray-400',
    border: 'border-green-400',
  },
  violet: {
    active: 'text-violet-400 border-violet-400 bg-violet-400/10',
    inactive: 'text-gray-500 border-gray-800 hover:border-gray-600 hover:text-gray-400',
    border: 'border-violet-400',
  },
  rose: {
    active: 'text-rose-400 border-rose-400 bg-rose-400/10',
    inactive: 'text-gray-500 border-gray-800 hover:border-gray-600 hover:text-gray-400',
    border: 'border-rose-400',
  },
};

export default function DomainSelector() {
  const pathname = usePathname();

  return (
    <div className="flex flex-wrap gap-2">
      {domains.map((d) => {
        const isActive = pathname === d.href;
        const colors = colorMap[d.color];
        return (
          <Link
            key={d.name}
            href={d.href}
            className={`px-4 py-2 border rounded-lg text-xs tracking-wider transition-colors duration-200 ${
              isActive ? colors.active : colors.inactive
            }`}
            title={d.description}
          >
            {d.name.toUpperCase()}
          </Link>
        );
      })}
    </div>
  );
}

export { domains };
