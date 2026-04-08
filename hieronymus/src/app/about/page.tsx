'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';

const fade = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.12, duration: 0.5 },
  }),
};

const pipelineSteps = [
  {
    label: 'Data Input',
    detail: 'Image, spectrum, sequence, or any bounded signal',
    color: 'text-cyan-400',
  },
  {
    label: 'Domain Encoder',
    detail:
      'CPU preprocessor + GLSL shader converts raw data to partition coordinates (n, l, m, s)',
    color: 'text-amber-400',
  },
  {
    label: 'Partition Pass',
    detail:
      'Fragment shader computes visible and invisible pixel signatures from partition coordinates',
    color: 'text-violet-400',
  },
  {
    label: 'Interference Pass',
    detail:
      'Cross-modal correlation, consistency check, and coupling strength between modalities',
    color: 'text-emerald-400',
  },
  {
    label: 'Entropy Pass',
    detail:
      'Computes S_k (kinetic), S_t (thermal), S_e (emission) and enforces S_k + S_t + S_e = 1',
    color: 'text-rose-400',
  },
  {
    label: 'Result',
    detail:
      'Conservation score, partition depth, sharpness, coherence, and visibility — all from one frame',
    color: 'text-cyan-400',
  },
];

const techStack = [
  { name: 'WebGL 2', desc: 'Fragment shaders as observation apparatus' },
  {
    name: 'OffscreenCanvas',
    desc: 'GPU computation in a Web Worker — no main-thread blocking',
  },
  {
    name: 'Web Workers',
    desc: 'Isolated thread for the entire observation pipeline',
  },
  {
    name: 'O(1) Memory',
    desc: '~13 MB working set, constant regardless of data size',
  },
  {
    name: 'Zero Upload',
    desc: 'All computation on your GPU — no data leaves the browser',
  },
];

export default function AboutPage() {
  return (
    <div className="min-h-[calc(100vh-140px)] px-8 py-16 max-w-4xl mx-auto sm:px-4">
      <motion.div initial="hidden" animate="visible">
        {/* Title */}
        <motion.h1
          className="text-4xl font-bold mb-3 sm:text-3xl"
          variants={fade}
          custom={0}
        >
          About Hieronymus
        </motion.h1>
        <motion.p
          className="text-gray-500 mb-12 max-w-2xl"
          variants={fade}
          custom={1}
        >
          A universal observation platform built on the mathematical identity
          between rendering and measurement.
        </motion.p>

        {/* Section 1: What is this? */}
        <motion.section className="mb-12" variants={fade} custom={2}>
          <h2 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider mb-3">
            What is this?
          </h2>
          <div className="text-gray-300 space-y-3 leading-relaxed">
            <p>
              Hieronymus is a client-side GPU observation engine that transforms
              any image — microscopy, spectroscopy, or arbitrary scientific
              imaging — into its thermodynamic coordinates using WebGL2 fragment
              shaders running entirely in your browser.
            </p>
            <p>
              Instead of applying algorithms to data, Hieronymus treats the GPU
              fragment shader as a physical observation apparatus. When the
              shader writes a pixel value, it performs a measurement. The
              rendered texture is not a picture of the result — it is the result
              itself, expressed in categorical representation.
            </p>
          </div>
        </motion.section>

        {/* Section 2: Core Principle */}
        <motion.section className="mb-12" variants={fade} custom={3}>
          <h2 className="text-sm font-semibold text-amber-400 uppercase tracking-wider mb-3">
            Core Principle
          </h2>
          <div className="p-6 rounded-lg border border-gray-800/60 bg-gray-900/30 mb-4">
            <p className="text-center text-lg font-semibold text-light mb-4">
              The Triple Equivalence
            </p>
            <div className="flex items-center justify-center gap-4 flex-wrap text-center">
              <span className="text-cyan-400 font-semibold">Observation</span>
              <span className="text-gray-600">=</span>
              <span className="text-amber-400 font-semibold">Computation</span>
              <span className="text-gray-600">=</span>
              <span className="text-violet-400 font-semibold">Processing</span>
            </div>
            <p className="text-center text-sm text-gray-500 mt-4">
              Oscillatory, categorical, and partitional descriptions yield
              identical state counts and entropies, connected by explicit
              bijective maps.
            </p>
          </div>
          <p className="text-gray-300 leading-relaxed">
            Every persistent dynamical system confined to bounded phase space
            necessarily oscillates. Every oscillator has a spectral
            decomposition. Every spectrum maps to a 2D image. Therefore every
            comparison problem — matching molecules, signals, sequences, or
            arbitrary data — reduces to a computer vision problem. This is not
            an analogy; it is a chain of mathematical identities.
          </p>
        </motion.section>

        {/* Section 3: How it works */}
        <motion.section className="mb-12" variants={fade} custom={4}>
          <h2 className="text-sm font-semibold text-violet-400 uppercase tracking-wider mb-3">
            How It Works
          </h2>
          <p className="text-gray-300 leading-relaxed mb-6">
            The observation pipeline runs entirely on the GPU as a sequence of
            fragment shader passes. Each pass reads from the previous
            pass&apos;s texture and writes to the next, with the final pass
            producing the observation result.
          </p>
          <div className="space-y-3">
            {pipelineSteps.map((step, i) => (
              <div
                key={step.label}
                className="flex items-start gap-4 p-3 rounded border border-gray-800/40 bg-gray-900/20"
              >
                <span
                  className={`flex-shrink-0 w-7 h-7 rounded flex items-center justify-center text-xs font-mono font-bold ${step.color} bg-gray-900`}
                >
                  {i + 1}
                </span>
                <div>
                  <span className={`text-sm font-semibold ${step.color}`}>
                    {step.label}
                  </span>
                  <p className="text-xs text-gray-500 mt-0.5">
                    {step.detail}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </motion.section>

        {/* Section 4: Technology */}
        <motion.section className="mb-12" variants={fade} custom={5}>
          <h2 className="text-sm font-semibold text-emerald-400 uppercase tracking-wider mb-3">
            Technology
          </h2>
          <p className="text-gray-300 leading-relaxed mb-6">
            No server computation. No data upload. No cloud dependency.
            Everything runs on the GPU already in your device — including
            integrated GPUs like Intel UHD, AMD Radeon, and Apple M-series.
          </p>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-1">
            {techStack.map((t) => (
              <div
                key={t.name}
                className="p-4 rounded border border-gray-800/60 bg-gray-900/30"
              >
                <span className="text-sm font-semibold text-light">
                  {t.name}
                </span>
                <p className="text-xs text-gray-500 mt-1">{t.desc}</p>
              </div>
            ))}
          </div>
        </motion.section>

        {/* Section 5: Papers */}
        <motion.section className="mb-8" variants={fade} custom={6}>
          <h2 className="text-sm font-semibold text-rose-400 uppercase tracking-wider mb-3">
            Papers
          </h2>
          <p className="text-gray-300 leading-relaxed mb-4">
            The mathematical foundations are established across five papers
            totalling 11,862 lines, 211 theorems, and 294 references. Every
            theorem derives from two axioms: bounded phase space and categorical
            observation.
          </p>
          <Link
            href="/publications"
            className="inline-block px-6 py-2.5 border border-rose-400/40 text-rose-400 rounded hover:bg-rose-400/10 transition-colors text-sm tracking-wider"
          >
            View all publications &rarr;
          </Link>
        </motion.section>

        {/* Conservation law footer */}
        <motion.div
          className="text-center pt-8 border-t border-gray-800/40"
          variants={fade}
          custom={7}
        >
          <p className="text-gray-600 font-mono text-sm tracking-wider">
            S<sub>k</sub> + S<sub>t</sub> + S<sub>e</sub> = 1
          </p>
          <p className="text-gray-700 text-xs mt-1">
            The conservation law that every observation must satisfy
          </p>
        </motion.div>
      </motion.div>
    </div>
  );
}
