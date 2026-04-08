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

export default function EncodersPage() {
  return (
    <div className="min-h-[calc(100vh-140px)] px-8 py-16 max-w-4xl mx-auto sm:px-4">
      <motion.div initial="hidden" animate="visible">
        {/* Back link */}
        <motion.div variants={fade} custom={0}>
          <Link
            href="/docs"
            className="text-sm text-gray-500 hover:text-gray-300 transition-colors"
          >
            &larr; Documentation
          </Link>
        </motion.div>

        <motion.h1
          className="text-3xl font-bold mt-6 mb-3 sm:text-2xl"
          variants={fade}
          custom={1}
        >
          Writing Domain Encoders
        </motion.h1>
        <motion.p
          className="text-gray-500 mb-10 max-w-2xl"
          variants={fade}
          custom={2}
        >
          A domain encoder is the only component that changes between
          application domains. It converts raw data into partition coordinates
          that the GPU pipeline can process.
        </motion.p>

        {/* Architecture overview */}
        <motion.section className="mb-10" variants={fade} custom={3}>
          <h2 className="text-sm font-semibold text-amber-400 uppercase tracking-wider mb-3">
            Encoder Architecture
          </h2>
          <p className="text-gray-300 text-sm leading-relaxed mb-4">
            Every encoder has two parts that work together: a CPU preprocessor
            written in TypeScript and a GPU shader written in GLSL. The CPU
            part normalises and loads the data, the GPU part does the per-pixel
            encoding at massively parallel throughput.
          </p>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-1">
            <div className="p-4 rounded border border-amber-400/20 bg-gray-900/30">
              <h3 className="text-sm font-semibold text-amber-400 mb-2">
                CPU Preprocessor
              </h3>
              <ul className="space-y-1.5 text-xs text-gray-400">
                <li>&bull; Load and validate input data</li>
                <li>&bull; Normalise to [0, 1] range</li>
                <li>&bull; Convert to RGBA pixel array</li>
                <li>&bull; Upload as WebGL texture</li>
              </ul>
            </div>
            <div className="p-4 rounded border border-amber-400/20 bg-gray-900/30">
              <h3 className="text-sm font-semibold text-amber-400 mb-2">
                GPU Shader (GLSL)
              </h3>
              <ul className="space-y-1.5 text-xs text-gray-400">
                <li>&bull; Read texel at current UV coordinate</li>
                <li>&bull; Compute local statistics (mean, variance)</li>
                <li>&bull; Derive partition features</li>
                <li>&bull; Output encoded vec4 for next pass</li>
              </ul>
            </div>
          </div>
        </motion.section>

        {/* S-Entropy Contract */}
        <motion.section className="mb-10" variants={fade} custom={4}>
          <h2 className="text-sm font-semibold text-rose-400 uppercase tracking-wider mb-3">
            The S-Entropy Contract
          </h2>
          <div className="p-6 rounded-lg border border-rose-400/20 bg-gray-900/30 mb-4">
            <p className="text-center font-mono text-lg text-light mb-2">
              S<sub>k</sub> + S<sub>t</sub> + S<sub>e</sub> = 1
            </p>
            <p className="text-center text-sm text-gray-500">
              Every encoder must produce outputs that satisfy this conservation
              law.
            </p>
          </div>
          <div className="space-y-3 text-sm text-gray-300 leading-relaxed">
            <p>
              The three entropy components partition the total information
              content of each pixel:
            </p>
            <ul className="space-y-2 pl-4">
              <li>
                <span className="text-cyan-400 font-semibold">S_k</span>{' '}
                (kinetic) — gradient magnitude, translational disorder. How
                rapidly the signal changes in the local neighbourhood.
              </li>
              <li>
                <span className="text-amber-400 font-semibold">S_t</span>{' '}
                (thermal) — local variance, configurational disorder. How much
                the signal varies within a local window.
              </li>
              <li>
                <span className="text-rose-400 font-semibold">S_e</span>{' '}
                (emission) — ternary state probability from the oxygen model.
                The fraction of information encoded in the invisible modality.
              </li>
            </ul>
            <p>
              The encoder does not need to compute S_k, S_t, S_e directly — that
              happens in the entropy pass (pass 4). But the encoder must output
              features from which these values can be derived, and the resulting
              triple must sum to unity.
            </p>
          </div>
        </motion.section>

        {/* Microscopy encoder example */}
        <motion.section className="mb-10" variants={fade} custom={5}>
          <h2 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider mb-3">
            Example: Microscopy Encoder
          </h2>
          <p className="text-gray-300 text-sm leading-relaxed mb-4">
            The built-in microscopy encoder (
            <code className="text-xs bg-gray-800 px-1.5 py-0.5 rounded font-mono">
              encode_microscopy.glsl
            </code>
            ) demonstrates the full pattern. It computes luminance, local
            statistics, and Shannon entropy from a 3x3 neighbourhood.
          </p>

          {/* GLSL code snippet */}
          <div className="rounded border border-gray-800/60 bg-gray-900/50 p-4 font-mono text-xs text-gray-300 overflow-x-auto mb-4">
            <p className="text-gray-600 mb-2">
              encode_microscopy.glsl &mdash; key functions
            </p>
            <pre>{`#version 300 es
precision highp float;
uniform sampler2D u_image;
uniform float u_nmax;
in vec2 v_uv;
out vec4 fragColor;

float luminance(vec3 c) {
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Sample 3x3 neighbourhood for local statistics
void neighbourhood(sampler2D tex, vec2 uv, vec2 texelSize,
                   out float mean, out float variance) {
  float sum = 0.0, sum2 = 0.0;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      float v = luminance(
        texture(tex, uv + vec2(float(dx), float(dy)) * texelSize).rgb
      );
      sum  += v;
      sum2 += v * v;
    }
  }
  mean     = sum / 9.0;
  variance = sum2 / 9.0 - mean * mean;
}

// Shannon entropy from local 3x3 histogram (4 bins)
float localEntropy(sampler2D tex, vec2 uv, vec2 ts) {
  // ... bin pixel values, compute -sum(p * log2(p))
}`}</pre>
          </div>

          <p className="text-gray-400 text-sm leading-relaxed">
            The encoder outputs a <code className="text-xs bg-gray-800 px-1.5 py-0.5 rounded font-mono">vec4</code> containing:
          </p>
          <ul className="mt-2 space-y-1 text-xs text-gray-500 pl-4">
            <li>
              <span className="font-mono text-gray-400">R:</span> quantised
              principal level n (partition depth)
            </li>
            <li>
              <span className="font-mono text-gray-400">G:</span> angular level
              l from local variance
            </li>
            <li>
              <span className="font-mono text-gray-400">B:</span> magnetic
              number m from gradient direction
            </li>
            <li>
              <span className="font-mono text-gray-400">A:</span> spin s from
              Shannon entropy
            </li>
          </ul>
        </motion.section>

        {/* Writing your own */}
        <motion.section className="mb-10" variants={fade} custom={6}>
          <h2 className="text-sm font-semibold text-emerald-400 uppercase tracking-wider mb-3">
            Writing Your Own Encoder
          </h2>
          <div className="space-y-4 text-sm text-gray-300 leading-relaxed">
            <p>
              To add a new domain (e.g. spectroscopy, chromatography, genomics),
              you need to:
            </p>
            <ol className="space-y-3 pl-4">
              <li>
                <span className="text-emerald-400 font-semibold">1.</span>{' '}
                Create a GLSL file{' '}
                <code className="text-xs bg-gray-800 px-1.5 py-0.5 rounded font-mono">
                  encode_&lt;domain&gt;.glsl
                </code>{' '}
                in <code className="text-xs bg-gray-800 px-1.5 py-0.5 rounded font-mono">src/shaders/</code>.
                It must accept <code className="text-xs bg-gray-800 px-1.5 py-0.5 rounded font-mono">u_image</code> (sampler2D) and{' '}
                <code className="text-xs bg-gray-800 px-1.5 py-0.5 rounded font-mono">u_nmax</code> (float), and output a{' '}
                <code className="text-xs bg-gray-800 px-1.5 py-0.5 rounded font-mono">vec4</code> of partition features.
              </li>
              <li>
                <span className="text-emerald-400 font-semibold">2.</span>{' '}
                Export the shader from{' '}
                <code className="text-xs bg-gray-800 px-1.5 py-0.5 rounded font-mono">
                  src/shaders/index.ts
                </code>.
              </li>
              <li>
                <span className="text-emerald-400 font-semibold">3.</span>{' '}
                Import the shader in{' '}
                <code className="text-xs bg-gray-800 px-1.5 py-0.5 rounded font-mono">
                  ObservationEngine.ts
                </code>{' '}
                and add it to the shader compilation map.
              </li>
              <li>
                <span className="text-emerald-400 font-semibold">4.</span>{' '}
                Write a CPU preprocessor function that converts your domain data
                into an RGBA pixel array (Uint8Array or Float32Array) suitable
                for texture upload.
              </li>
              <li>
                <span className="text-emerald-400 font-semibold">5.</span>{' '}
                Pass the encoder name to{' '}
                <code className="text-xs bg-gray-800 px-1.5 py-0.5 rounded font-mono">
                  observe()
                </code>{' '}
                to select it at runtime.
              </li>
            </ol>
            <p>
              The rest of the pipeline — partition, interference, entropy, and
              display passes — is universal and does not change between domains.
              Only the encoder changes.
            </p>
          </div>
        </motion.section>

        {/* Bottom nav */}
        <motion.div
          className="flex items-center justify-between pt-8 border-t border-gray-800/40"
          variants={fade}
          custom={7}
        >
          <Link
            href="/docs"
            className="text-sm text-gray-500 hover:text-gray-300 transition-colors"
          >
            &larr; Documentation
          </Link>
          <Link
            href="/docs/shaders"
            className="text-sm text-violet-400 hover:text-violet-300 transition-colors"
          >
            Shader Reference &rarr;
          </Link>
        </motion.div>
      </motion.div>
    </div>
  );
}
