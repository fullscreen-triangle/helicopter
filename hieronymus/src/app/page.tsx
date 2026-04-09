'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import dynamic from 'next/dynamic';

const LandingScene = dynamic(
  () => import('@/components/landing/LandingScene'),
  { ssr: false }
);

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
    accent: 'text-cyan-400 border-cyan-400/30 hover:border-cyan-400/60',
    dotColor: 'bg-cyan-400',
    description:
      'Drop an image. The GPU encodes it into partition coordinates (n,l,m,s) and computes entropy conservation in real time.',
  },
  {
    title: 'Match',
    href: '/match',
    accent: 'text-amber-400 border-amber-400/30 hover:border-amber-400/60',
    dotColor: 'bg-amber-400',
    description:
      'Compare two images through their spectral signatures. Measure visibility, interference circuits, and S-distance on the entropy simplex.',
  },
  {
    title: 'Diagnose',
    href: '/observe',
    accent: 'text-violet-400 border-violet-400/30 hover:border-violet-400/60',
    dotColor: 'bg-violet-400',
    description:
      'Validate measurement quality through the stereogram consistency check. Every pixel must satisfy the conservation law.',
  },
];

const stats = [
  { label: 'Papers', value: '5' },
  { label: 'Theorems', value: '211' },
  { label: 'Panels', value: '25' },
];

export default function HomePage() {
  return (
    <div className="min-h-screen bg-[#080c10] overflow-hidden">
      {/* ─── Hero Section ─── */}
      <section className="relative min-h-screen flex flex-col lg:flex-col">
        <div className="flex flex-1 lg:flex-col">
          {/* Left text panel — 40% */}
          <motion.div
            className="w-[40%] lg:w-full flex flex-col justify-center px-12 py-16 xl:px-8 md:px-6 lg:py-10 lg:items-center lg:text-center relative z-10"
            initial="hidden"
            animate="visible"
          >
            <motion.p
              className="text-cyan-400/60 text-xs tracking-[0.35em] uppercase mb-4 font-medium"
              variants={fadeUp}
              custom={0}
            >
              Universal Observation Platform
            </motion.p>

            <motion.h1
              className="text-7xl xl:text-6xl md:text-5xl sm:text-4xl font-bold tracking-widest uppercase mb-6 text-light"
              variants={fadeUp}
              custom={1}
            >
              Hieronymus
            </motion.h1>

            <motion.div
              className="w-16 h-px bg-cyan-400/40 mb-6 lg:mx-auto"
              variants={fadeUp}
              custom={2}
            />

            <motion.p
              className="text-gray-400 leading-relaxed max-w-md mb-10 text-sm lg:mx-auto"
              variants={fadeUp}
              custom={3}
            >
              The fragment shader IS the observation apparatus. The rendered
              texture IS the computed result.
            </motion.p>

            <motion.div
              className="flex gap-4 sm:flex-col sm:w-full"
              variants={fadeUp}
              custom={4}
            >
              <Link
                href="/observe/microscopy"
                className="inline-flex items-center justify-center px-8 py-3 bg-cyan-400/10 border border-cyan-400/40 text-cyan-400
                           hover:bg-cyan-400 hover:text-[#080c10] transition-all duration-300
                           tracking-widest uppercase text-xs font-semibold rounded sm:w-full"
              >
                Start Observing
              </Link>
              <Link
                href="/publications"
                className="inline-flex items-center justify-center px-8 py-3 border border-gray-700 text-gray-400
                           hover:border-gray-500 hover:text-light transition-all duration-300
                           tracking-widest uppercase text-xs font-semibold rounded sm:w-full"
              >
                Read Papers
              </Link>
            </motion.div>
          </motion.div>

          {/* Right 3D canvas — 60% */}
          <div className="w-[60%] lg:w-full h-screen lg:h-[50vh] relative">
            {/* Gradient overlay to blend into the dark bg on the left edge */}
            <div className="absolute inset-y-0 left-0 w-24 bg-gradient-to-r from-[#080c10] to-transparent z-10 pointer-events-none lg:hidden" />
            <LandingScene />
          </div>
        </div>
      </section>

      {/* ─── Feature Cards ─── */}
      <section className="relative z-10 px-12 xl:px-8 md:px-6 pb-16 -mt-12 lg:mt-0">
        <motion.div
          className="grid grid-cols-3 gap-6 max-w-6xl mx-auto lg:grid-cols-1 lg:max-w-lg"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
        >
          {features.map((f, i) => (
            <motion.div key={f.title} variants={fadeUp} custom={i}>
              <Link
                href={f.href}
                className={`block p-6 border rounded-lg ${f.accent} bg-[#0c1118]/80 backdrop-blur-sm transition-all duration-300 hover:translate-y-[-2px]`}
              >
                <div className="flex items-center gap-2 mb-3">
                  <span className={`w-2 h-2 rounded-full ${f.dotColor}`} />
                  <h3 className="text-lg font-semibold tracking-wide">
                    {f.title}
                  </h3>
                </div>
                <p className="text-gray-500 text-sm leading-relaxed">
                  {f.description}
                </p>
              </Link>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* ─── Stats Row ─── */}
      <section className="relative z-10 border-t border-gray-800/50">
        <motion.div
          className="max-w-6xl mx-auto px-12 xl:px-8 md:px-6 py-12 flex justify-center gap-16 md:gap-8 sm:gap-6"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3, duration: 0.6 }}
        >
          {stats.map((s, i) => (
            <div key={s.label} className="text-center">
              <p className="text-3xl md:text-2xl font-bold text-light tracking-tight">
                {s.value}
              </p>
              <p className="text-xs text-gray-500 uppercase tracking-widest mt-1">
                {s.label}
              </p>
              {i < stats.length - 1 && (
                <span className="hidden" />
              )}
            </div>
          ))}
        </motion.div>
      </section>

      {/* ─── Tech Tags ─── */}
      <section className="relative z-10 pb-16">
        <motion.div
          className="flex gap-3 flex-wrap justify-center px-6"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.5 }}
        >
          {[
            'WebGL2',
            '4-Pass Pipeline',
            'Fragment Shaders',
            'Sk + St + Se = 1',
          ].map((tag) => (
            <span
              key={tag}
              className="text-xs tracking-wider text-gray-600 border border-gray-800 px-3 py-1 rounded"
            >
              {tag}
            </span>
          ))}
        </motion.div>
      </section>
    </div>
  );
}
