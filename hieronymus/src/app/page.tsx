'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.15, duration: 0.6, ease: 'easeOut' },
  }),
};

const features = [
  {
    title: 'Observe',
    href: '/observe/microscopy',
    color: 'text-cyan-400 border-cyan-400/30',
    description:
      'Drop an image. The GPU encodes it into partition coordinates (n,l,m,s) and computes entropy conservation in real time.',
  },
  {
    title: 'Match',
    href: '/observe',
    color: 'text-amber-400 border-amber-400/30',
    description:
      'Compare two images through their spectral signatures. Measure visibility, interference circuits, and S-distance on the entropy simplex.',
  },
  {
    title: 'Diagnose',
    href: '/observe',
    color: 'text-violet-400 border-violet-400/30',
    description:
      'Validate measurement quality through the stereogram consistency check. Every pixel must satisfy the conservation law.',
  },
];

export default function HomePage() {
  return (
    <div className="min-h-[calc(100vh-140px)] flex flex-col items-center justify-center px-8 py-16">
      {/* Hero */}
      <motion.div
        className="text-center max-w-3xl mx-auto"
        initial="hidden"
        animate="visible"
      >
        <motion.h1
          className="text-7xl font-bold tracking-tight mb-4 xl:text-5xl sm:text-4xl"
          variants={fadeUp}
          custom={0}
        >
          Hieronymus
        </motion.h1>
        <motion.p
          className="text-xl text-gray-400 mb-2 tracking-widest uppercase text-sm"
          variants={fadeUp}
          custom={1}
        >
          Universal Observation Platform
        </motion.p>
        <motion.p
          className="text-gray-500 max-w-xl mx-auto mb-10 leading-relaxed"
          variants={fadeUp}
          custom={2}
        >
          A client-side GPU engine that transforms any image into its
          thermodynamic coordinates. No server. No upload. Your data never
          leaves your browser.
        </motion.p>
        <motion.div variants={fadeUp} custom={3}>
          <Link
            href="/observe/microscopy"
            className="inline-block px-8 py-3 border border-cyan-400 text-cyan-400 rounded
                       hover:bg-cyan-400 hover:text-dark transition-colors duration-200
                       tracking-widest uppercase text-sm font-semibold"
          >
            Begin Observation
          </Link>
        </motion.div>
      </motion.div>

      {/* Feature cards */}
      <motion.div
        className="grid grid-cols-3 gap-6 mt-20 max-w-5xl w-full lg:grid-cols-1 lg:max-w-lg"
        initial="hidden"
        animate="visible"
      >
        {features.map((f, i) => (
          <motion.div key={f.title} variants={fadeUp} custom={4 + i}>
            <Link
              href={f.href}
              className={`block p-6 border rounded-lg ${f.color} bg-dark/50 hover:bg-gray-900/80 transition-colors`}
            >
              <h3 className="text-lg font-semibold mb-2">{f.title}</h3>
              <p className="text-gray-500 text-sm leading-relaxed">
                {f.description}
              </p>
            </Link>
          </motion.div>
        ))}
      </motion.div>

      {/* Tags */}
      <motion.div
        className="flex gap-3 mt-12 flex-wrap justify-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
      >
        {['WebGL2', '4-Pass Pipeline', 'Fragment Shaders', 'Sk + St + Se = 1'].map(
          (tag) => (
            <span
              key={tag}
              className="text-xs tracking-wider text-gray-600 border border-gray-800 px-3 py-1 rounded"
            >
              {tag}
            </span>
          ),
        )}
      </motion.div>
    </div>
  );
}
