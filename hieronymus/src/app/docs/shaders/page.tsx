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

const passes = [
  {
    number: 1,
    name: 'Encode',
    file: 'encode_microscopy.glsl',
    color: 'text-cyan-400 border-cyan-400/20',
    inputs: 'u_image (raw image texture), u_nmax',
    outputs: 'vec4(n, l, m, s) — partition features per pixel',
    description:
      'Converts the raw image into partition feature space. Computes luminance, local mean and variance over a 3x3 neighbourhood, gradient direction, and Shannon entropy from a 4-bin local histogram. Outputs are quantised to the range [0, n_max].',
    glsl: `float I = luminance(texture(u_image, v_uv).rgb);
float mean, variance;
neighbourhood(u_image, v_uv, texelSize, mean, variance);
float H = localEntropy(u_image, v_uv, texelSize);
float n = floor(clamp(I * u_nmax, 1.0, u_nmax));
float l = clamp(sqrt(variance) * u_nmax, 0.0, n - 1.0);
fragColor = vec4(n / u_nmax, l / u_nmax, m_norm, H);`,
  },
  {
    number: 2,
    name: 'Partition',
    file: 'partition.glsl',
    color: 'text-amber-400 border-amber-400/20',
    inputs: 'u_image (raw), u_pass1 (encode output), u_Aeg, u_nmax',
    outputs: 'vec4(n_vis, P0, P1, P2) — visible partition + ternary O2 states',
    description:
      'Computes the invisible pixel via the oxygen model. Uses Beer-Lambert to estimate O2 concentration from image intensity. Solves the Einstein rate equations at steady state to compute the three ternary probabilities: P0 (ground/reference), P1 (absorbing/detector), P2 (emitting/source). Combines with the visible partition coordinates from pass 1.',
    glsl: `float OD = -log(max(I, 0.001) / I0);
float O2conc = clamp(OD / 2.0, 0.0, 1.0);
float u_nu = I * 0.5;
float Bge = u_Aeg * 2.0;
float Beg = u_Aeg * 1.5;
float denom = u_Aeg + (Beg + Bge) * u_nu;
float P2 = (Bge * u_nu) / max(denom, 1e-6);
float P0 = u_Aeg / max(denom, 1e-6);
float P1 = 1.0 - P0 - P2;`,
  },
  {
    number: 3,
    name: 'Interference',
    file: 'interference.glsl',
    color: 'text-violet-400 border-violet-400/20',
    inputs: 'u_pass1 (visible), u_pass2 (invisible), u_epsilon, u_J, u_beta, u_nmax',
    outputs: 'vec4(correlation, consistency, coupling, coherence)',
    description:
      'Cross-modal consistency check between visible and invisible modalities. Computes inter-modality correlation over a 5x5 neighbourhood, dual-pixel consistency (whether the two paths agree within threshold epsilon), coupling strength via Boltzmann-weighted energy, and local phase coherence from gradient alignment.',
    glsl: `float corr = interModalCorr(v_uv, texelSize);
float vis_n = texture(u_pass1, v_uv).r;
float inv_n = texture(u_pass2, v_uv).a;
float diff = abs(vis_n - inv_n);
float consistent = step(diff, u_epsilon) ? 1.0 : 0.0;
float E = -u_J * corr;
float coupling = exp(-u_beta * E);
fragColor = vec4(corr, consistent, coupling, coherence);`,
  },
  {
    number: 4,
    name: 'Entropy',
    file: 'entropy.glsl',
    color: 'text-emerald-400 border-emerald-400/20',
    inputs: 'u_image, u_pass2 (ternary), u_pass3 (consistency), u_alpha, u_nmax',
    outputs: 'vec4(S_k, S_t, S_e, conservation)',
    description:
      'Computes the three entropy coordinates and enforces the conservation law. S_k (kinetic) from gradient magnitude measures translational disorder. S_t (thermal) from local 5x5 variance measures configurational disorder. S_e (emission) from the ternary state probabilities measures information in the invisible channel. The three are normalised so S_k + S_t + S_e = 1.',
    glsl: `// Kinetic entropy: gradient magnitude
float gx = I_right - I_left;
float gy = I_up - I_down;
float grad = sqrt(gx*gx + gy*gy);
float Sk = clamp(grad * 4.0, 0.0, 1.0);

// Thermal entropy: local variance
float var = localVariance(v_uv, ts);
float St = clamp(sqrt(var) * 3.0, 0.0, 1.0);

// Emission entropy: ternary information
float Se = clamp(ternaryEntropy * u_alpha, 0.0, 1.0);

// Normalise to conservation law
float total = Sk + St + Se;
fragColor = vec4(Sk/total, St/total, Se/total, abs(total - 1.0));`,
  },
  {
    number: 5,
    name: 'Display',
    file: 'display.glsl',
    color: 'text-rose-400 border-rose-400/20',
    inputs: 'u_tex (any pass output), u_pass (which pass to visualise)',
    outputs: 'vec4(r, g, b, 1.0) — false-colour visualisation',
    description:
      'Renders any intermediate or final pass as a false-colour image for human inspection. Uses the inferno colourmap for scalar quantities and custom colour mapping for vector quantities. Switchable via u_pass uniform: 0 = partition depth (cyan-magenta), 1 = ternary states (RGB), 2 = consistency (green/red), 3 = entropy triple (RGB mapped to S_k/S_t/S_e).',
    glsl: `vec3 inferno(float t) {
  // Perceptually uniform colourmap
  const vec3 c0 = vec3(0.0002, 0.0016, 0.0140);
  const vec3 c5 = vec3(0.9882, 0.9922, 0.7490);
  // ... 6-segment piecewise linear interpolation
}

if (u_pass == 0) {
  fragColor = vec4(mix(cyan, magenta, tex.r), 1.0);
} else if (u_pass == 3) {
  fragColor = vec4(tex.r, tex.g, tex.b, 1.0); // S_k=R, S_t=G, S_e=B
}`,
  },
];

const uniforms = [
  {
    name: 'u_image',
    type: 'sampler2D',
    passes: '1, 2, 4',
    description: 'The original input image texture',
  },
  {
    name: 'u_pass1',
    type: 'sampler2D',
    passes: '2, 3',
    description: 'Output of encode pass (partition features)',
  },
  {
    name: 'u_pass2',
    type: 'sampler2D',
    passes: '3, 4',
    description: 'Output of partition pass (ternary states)',
  },
  {
    name: 'u_pass3',
    type: 'sampler2D',
    passes: '4',
    description: 'Output of interference pass (consistency data)',
  },
  {
    name: 'u_tex',
    type: 'sampler2D',
    passes: '5',
    description: 'Any pass output for display visualisation',
  },
  {
    name: 'u_nmax',
    type: 'float',
    passes: '1-4',
    description: 'Maximum principal quantum number (default: 8)',
  },
  {
    name: 'u_epsilon',
    type: 'float',
    passes: '3',
    description: 'Consistency threshold (default: 0.15)',
  },
  {
    name: 'u_J',
    type: 'float',
    passes: '3',
    description: 'Inter-modal coupling constant (default: 1.0)',
  },
  {
    name: 'u_beta',
    type: 'float',
    passes: '3',
    description: 'Inverse temperature for Boltzmann weighting (default: 2.0)',
  },
  {
    name: 'u_Aeg',
    type: 'float',
    passes: '2',
    description: 'Einstein A coefficient for spontaneous emission (default: 2.58)',
  },
  {
    name: 'u_alpha',
    type: 'float',
    passes: '4',
    description: 'Information transfer efficiency (default: 0.5)',
  },
  {
    name: 'u_pass',
    type: 'int',
    passes: '5',
    description: 'Which pass to visualise in display shader (0-3)',
  },
];

export default function ShadersPage() {
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
          Shader Pipeline Reference
        </motion.h1>
        <motion.p
          className="text-gray-500 mb-10 max-w-2xl"
          variants={fade}
          custom={2}
        >
          The observation pipeline consists of 5 fragment shader passes executed
          sequentially via framebuffer ping-pong. Each pass reads from the
          previous pass&apos;s output texture and writes to the next.
        </motion.p>

        {/* Pipeline overview */}
        <motion.div
          className="flex items-center justify-center gap-2 mb-12 flex-wrap"
          variants={fade}
          custom={3}
        >
          {passes.map((p, i) => (
            <span key={p.name} className="flex items-center gap-2">
              <span
                className={`px-3 py-1.5 rounded border text-xs font-mono ${p.color}`}
              >
                {p.name}
              </span>
              {i < passes.length - 1 && (
                <span className="text-gray-700">&rarr;</span>
              )}
            </span>
          ))}
        </motion.div>

        {/* Pass details */}
        <div className="space-y-8">
          {passes.map((p, i) => (
            <motion.section
              key={p.name}
              className={`p-6 rounded-lg border bg-gray-900/30 ${p.color.split(' ')[1]}`}
              variants={fade}
              custom={4 + i}
            >
              <div className="flex items-start justify-between mb-3 sm:flex-col sm:gap-1">
                <h2 className={`text-lg font-semibold ${p.color.split(' ')[0]}`}>
                  Pass {p.number}: {p.name}
                </h2>
                <span className="text-xs text-gray-600 font-mono">{p.file}</span>
              </div>

              <p className="text-gray-300 text-sm leading-relaxed mb-4">
                {p.description}
              </p>

              {/* I/O */}
              <div className="grid grid-cols-2 gap-3 mb-4 sm:grid-cols-1">
                <div className="p-3 rounded bg-gray-900/50">
                  <span className="text-[10px] text-gray-600 uppercase tracking-wider block mb-1">
                    Inputs
                  </span>
                  <span className="text-xs text-gray-400 font-mono">
                    {p.inputs}
                  </span>
                </div>
                <div className="p-3 rounded bg-gray-900/50">
                  <span className="text-[10px] text-gray-600 uppercase tracking-wider block mb-1">
                    Outputs
                  </span>
                  <span className="text-xs text-gray-400 font-mono">
                    {p.outputs}
                  </span>
                </div>
              </div>

              {/* GLSL snippet */}
              <div className="rounded border border-gray-800/60 bg-gray-950/50 p-4 font-mono text-xs text-gray-400 overflow-x-auto">
                <pre>{p.glsl}</pre>
              </div>
            </motion.section>
          ))}
        </div>

        {/* Uniforms reference table */}
        <motion.section className="mt-12" variants={fade} custom={10}>
          <h2 className="text-2xl font-bold mb-6">Uniforms Reference</h2>
          <p className="text-gray-400 text-sm mb-6 leading-relaxed">
            All uniforms used across the pipeline. Each shader only binds the
            uniforms it needs — unused uniforms are silently ignored by WebGL2.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="py-2 px-3 text-gray-500 font-mono text-xs">
                    Name
                  </th>
                  <th className="py-2 px-3 text-gray-500 font-mono text-xs">
                    Type
                  </th>
                  <th className="py-2 px-3 text-gray-500 text-xs">Passes</th>
                  <th className="py-2 px-3 text-gray-500 text-xs">
                    Description
                  </th>
                </tr>
              </thead>
              <tbody className="text-gray-400 text-xs">
                {uniforms.map((u) => (
                  <tr key={u.name} className="border-b border-gray-800/40">
                    <td className="py-2 px-3 font-mono text-light">
                      {u.name}
                    </td>
                    <td className="py-2 px-3 font-mono">{u.type}</td>
                    <td className="py-2 px-3">{u.passes}</td>
                    <td className="py-2 px-3">{u.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.section>

        {/* Framebuffer layout */}
        <motion.section className="mt-12" variants={fade} custom={11}>
          <h2 className="text-lg font-semibold text-gray-300 mb-4">
            Framebuffer Layout
          </h2>
          <p className="text-gray-400 text-sm leading-relaxed mb-4">
            The pipeline uses RGBA32F textures for all intermediate framebuffers.
            Each pass writes to a dedicated FBO, and the output becomes the input
            texture for the next pass. The final display pass renders to the
            default framebuffer (canvas).
          </p>
          <div className="rounded border border-gray-800/60 bg-gray-900/50 p-4 font-mono text-xs text-gray-300 overflow-x-auto">
            <pre>{`FBO 0 (encode)       → tex0  [RGBA32F, W x H]
FBO 1 (partition)    → tex1  [RGBA32F, W x H]
FBO 2 (interference) → tex2  [RGBA32F, W x H]
FBO 3 (entropy)      → tex3  [RGBA32F, W x H]
Default FBO (display) → canvas`}</pre>
          </div>
        </motion.section>

        {/* Bottom nav */}
        <motion.div
          className="flex items-center justify-between pt-8 mt-8 border-t border-gray-800/40"
          variants={fade}
          custom={12}
        >
          <Link
            href="/docs/encoders"
            className="text-sm text-amber-400 hover:text-amber-300 transition-colors"
          >
            &larr; Writing Encoders
          </Link>
          <Link
            href="/docs"
            className="text-sm text-gray-500 hover:text-gray-300 transition-colors"
          >
            Documentation home
          </Link>
        </motion.div>
      </motion.div>
    </div>
  );
}
