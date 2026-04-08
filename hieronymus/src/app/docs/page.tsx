'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';

const fade = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.5 },
  }),
};

const sections = [
  {
    title: 'Engine API',
    href: '#engine-api',
    color: 'text-cyan-400 border-cyan-400/40',
    description:
      'The ObservationEngine class, useObservation hook, worker message protocol, and result types.',
    items: [
      'ObservationEngine — WebGL2 pipeline manager',
      'useObservation() — React hook for observation lifecycle',
      'WorkerInput / WorkerOutput — typed message protocol',
      'ObservationResult / MatchResult — output types',
    ],
  },
  {
    title: 'Writing Encoders',
    href: '/docs/encoders',
    color: 'text-amber-400 border-amber-400/40',
    description:
      'How to write a domain encoder that maps raw data to partition coordinates for the GPU pipeline.',
    items: [
      'CPU preprocessor (TypeScript) — data normalization',
      'GPU shader (GLSL) — per-pixel encoding',
      'The S-entropy contract: S_k + S_t + S_e = 1',
      'Example: microscopy encoder walkthrough',
    ],
  },
  {
    title: 'Shader Reference',
    href: '/docs/shaders',
    color: 'text-violet-400 border-violet-400/40',
    description:
      'Complete reference for the 6-pass GPU shader pipeline with GLSL code and uniforms.',
    items: [
      'Pass 1: Encode — raw data to partition features',
      'Pass 2: Partition — feature map to (n,l,m,s) coordinates',
      'Pass 3: Interference — cross-modal consistency check',
      'Pass 4: Entropy — S_k, S_t, S_e computation',
      'Pass 5: Display — false-colour visualization',
      'Uniforms reference table',
    ],
  },
];

export default function DocsPage() {
  return (
    <div className="min-h-[calc(100vh-140px)] px-8 py-16 max-w-4xl mx-auto sm:px-4">
      <motion.div initial="hidden" animate="visible">
        <motion.h1
          className="text-4xl font-bold mb-3 sm:text-3xl"
          variants={fade}
          custom={0}
        >
          Documentation
        </motion.h1>
        <motion.p
          className="text-gray-500 mb-12 max-w-2xl"
          variants={fade}
          custom={1}
        >
          Technical reference for the Hieronymus observation engine, domain
          encoders, and shader pipeline.
        </motion.p>

        {/* Section cards */}
        <div className="space-y-6">
          {sections.map((sec, i) => (
            <motion.div key={sec.title} variants={fade} custom={i + 2}>
              {sec.href.startsWith('/') ? (
                <Link href={sec.href} className="block group">
                  <SectionCard section={sec} />
                </Link>
              ) : (
                <div>
                  <SectionCard section={sec} />
                </div>
              )}
            </motion.div>
          ))}
        </div>

        {/* Engine API inline section */}
        <motion.div
          id="engine-api"
          className="mt-16 pt-8 border-t border-gray-800/40"
          variants={fade}
          custom={5}
        >
          <h2 className="text-2xl font-bold mb-6">Engine API</h2>

          {/* ObservationEngine */}
          <div className="mb-8">
            <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider mb-3">
              ObservationEngine
            </h3>
            <p className="text-gray-300 text-sm leading-relaxed mb-4">
              The core class that manages the WebGL2 pipeline. It compiles all
              shaders, creates framebuffers, and runs the multi-pass observation
              sequence.
            </p>
            <div className="rounded border border-gray-800/60 bg-gray-900/50 p-4 font-mono text-xs text-gray-300 overflow-x-auto">
              <pre>{`class ObservationEngine {
  constructor(canvas: OffscreenCanvas | HTMLCanvasElement, width?: number, height?: number)
  observe(imageData: ArrayBuffer, width: number, height: number): ObservationResult
  match(imageDataA: ArrayBuffer, imageDataB: ArrayBuffer, w: number, h: number): MatchResult
  setUniforms(uniforms: Record<string, number>): void
  destroy(): void
}`}</pre>
            </div>
          </div>

          {/* useObservation hook */}
          <div className="mb-8">
            <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider mb-3">
              useObservation Hook
            </h3>
            <p className="text-gray-300 text-sm leading-relaxed mb-4">
              React hook that manages the Web Worker lifecycle. Initialises the
              observation engine in a worker thread and provides methods to
              observe images and match image pairs.
            </p>
            <div className="rounded border border-gray-800/60 bg-gray-900/50 p-4 font-mono text-xs text-gray-300 overflow-x-auto">
              <pre>{`const {
  ready,           // boolean — engine initialised
  loading,         // boolean — observation in progress
  result,          // ObservationResult | null
  matchResult,     // MatchResult | null
  error,           // string | null
  observe,         // (imageData: ArrayBuffer, w: number, h: number, encoder?: string) => void
  match,           // (dataA: ArrayBuffer, dataB: ArrayBuffer, w: number, h: number) => void
} = useObservation();`}</pre>
            </div>
          </div>

          {/* Types */}
          <div className="mb-8">
            <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider mb-3">
              Result Types
            </h3>
            <div className="rounded border border-gray-800/60 bg-gray-900/50 p-4 font-mono text-xs text-gray-300 overflow-x-auto">
              <pre>{`interface ObservationResult {
  S_k: number;           // Kinetic entropy component
  S_t: number;           // Thermal entropy component
  S_e: number;           // Emission entropy component
  conservation: number;  // |S_k + S_t + S_e - 1| (should be ~0)
  partitionDepth: number;
  sharpness: number;
  noise: number;
  coherence: number;
  visibility: number;
  elapsed_ms: number;
}

interface MatchResult {
  score: number;         // Overall match score [0,1]
  visibility: number;    // Interference visibility
  circuits: number;      // Number of matching circuits detected
  S_distance: number;    // Distance on entropy simplex
  elapsed_ms: number;
  imageA: { S_k: number; S_t: number; S_e: number };
  imageB: { S_k: number; S_t: number; S_e: number };
}`}</pre>
            </div>
          </div>

          {/* Uniforms */}
          <div className="mb-8">
            <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider mb-3">
              Default Uniforms
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="py-2 px-3 text-gray-500 font-mono text-xs">
                      Uniform
                    </th>
                    <th className="py-2 px-3 text-gray-500 font-mono text-xs">
                      Default
                    </th>
                    <th className="py-2 px-3 text-gray-500 text-xs">
                      Description
                    </th>
                  </tr>
                </thead>
                <tbody className="text-gray-400 text-xs">
                  <tr className="border-b border-gray-800/40">
                    <td className="py-2 px-3 font-mono">epsilon</td>
                    <td className="py-2 px-3 font-mono">0.15</td>
                    <td className="py-2 px-3">
                      Consistency threshold for dual-pixel agreement
                    </td>
                  </tr>
                  <tr className="border-b border-gray-800/40">
                    <td className="py-2 px-3 font-mono">J</td>
                    <td className="py-2 px-3 font-mono">1.0</td>
                    <td className="py-2 px-3">
                      Coupling constant for inter-modal interaction
                    </td>
                  </tr>
                  <tr className="border-b border-gray-800/40">
                    <td className="py-2 px-3 font-mono">beta</td>
                    <td className="py-2 px-3 font-mono">2.0</td>
                    <td className="py-2 px-3">
                      Inverse temperature for Boltzmann weighting
                    </td>
                  </tr>
                  <tr className="border-b border-gray-800/40">
                    <td className="py-2 px-3 font-mono">nmax</td>
                    <td className="py-2 px-3 font-mono">8</td>
                    <td className="py-2 px-3">
                      Maximum principal quantum number for partition depth
                    </td>
                  </tr>
                  <tr className="border-b border-gray-800/40">
                    <td className="py-2 px-3 font-mono">Aeg</td>
                    <td className="py-2 px-3 font-mono">2.58</td>
                    <td className="py-2 px-3">
                      Einstein A coefficient for spontaneous emission
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-mono">alpha</td>
                    <td className="py-2 px-3 font-mono">0.5</td>
                    <td className="py-2 px-3">
                      Information transfer efficiency between modalities
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}

function SectionCard({
  section,
}: {
  section: (typeof sections)[number];
}) {
  return (
    <div
      className={`p-6 rounded-lg border bg-gray-900/50 transition-all duration-300 hover:bg-gray-900/80 ${section.color.split(' ')[1]}`}
    >
      <h3 className={`text-lg font-semibold mb-2 ${section.color.split(' ')[0]}`}>
        {section.title}
      </h3>
      <p className="text-gray-400 text-sm mb-4">{section.description}</p>
      <ul className="space-y-1.5">
        {section.items.map((item) => (
          <li key={item} className="flex items-start gap-2 text-xs text-gray-500">
            <span className="text-gray-700 mt-0.5">&bull;</span>
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}
