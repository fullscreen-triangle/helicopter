# Hieronymus

**Universal Observation Platform** --- Client-side GPU computation through fragment shaders as observation apparatus.

Hieronymus transforms any data (microscopy images, molecular spectra, genomic sequences, time series) into categorical partition coordinates, then uses GPU fragment shaders to perform observation, matching, and diagnostic computation entirely in the browser. No server-side computation. No Python runtime. O(1) GPU memory.

## Core Principle

> **Observation = Computation = Processing**

The GPU fragment shader is not rendering a visualization. It is performing a physical observation. The rendered texture IS the computed result in categorical representation, not a picture of it. This identity --- the Rendering--Measurement Identity --- means that the act of rendering IS the act of computing.

## How It Works

1. **Drop data** --- microscopy image, spectrum, sequence, or any numeric data
2. **GPU observes** --- OffscreenCanvas + WebGL2 fragment shaders compute partition coordinates (n, l, m, s) and S-entropy (S_k, S_t, S_e) per pixel, all in a Web Worker
3. **Results appear** --- conservation-compliant metrics, quality observables, match scores, coherence diagnostics

The GPU holds the observation apparatus (shaders), not the data. Stream and observe on demand. Memory usage: ~25 MB regardless of database size.

## Architecture

```
Browser (client-side):
  OffscreenCanvas + WebGL2    ← observation apparatus
  Web Worker                  ← background computation
  React + Next.js 14          ← UI

Edge (serverless):
  KV store                    ← S-entropy coordinates (12 bytes/item)
  Edge function               ← S-distance matching
```

### Shader Pipeline

| Pass | Operation | Input | Output |
|------|-----------|-------|--------|
| 0 | **Encode** | Raw data | S-entropy texture (S_k, S_t, S_e) |
| 1 | **Partition** | S-entropy | Partition coordinates (n, l, m, s) |
| 2 | **Interfere** | Two partition textures | Visibility map + match score |
| 3 | **Entropy** | All textures | Conservation check + quality metrics |
| 4 | **Ray March** | 3D volume | Triple observation (optical + chromatographic + circuit) |
| 5 | **Multi-Ray** | N phase maps | Coherence index (health diagnostic) |

Passes 0--3 run for all observations (~5 ms). Passes 4--5 run only for 3D diagnostics (~43 ms total).

### Domain Encoders

The observation engine is universal. Only the encoder (data to S-entropy) is domain-specific:

| Domain | S_k | S_t | S_e |
|--------|-----|-----|-----|
| Microscopy | Local Shannon entropy | Gradient magnitude | Conservation residual |
| Molecular | Spectral entropy | Frequency centroid | Spectral spread |
| Genomic | k-mer entropy | Positional complexity | Repeat depth |
| Signal | Power spectrum entropy | Autocorrelation | Spectral centroid |

## Theoretical Foundation

Built on five self-contained papers, each deriving everything from two axioms (Bounded Phase Space + Categorical Observation):

| Paper | Key Result |
|-------|------------|
| **Measurement-Modality Stereogram** | Every pixel is a dual object: visible (optical) + invisible (O2 categorical). Dual-pixel consistency provides cross-validation. |
| **Image Harmonic Matching Circuits** | Image comparison IS interference. Matching circuits are wallless resonant cavities. Eliminates von Neumann bottleneck. |
| **Universal Spectral Matching** | ALL comparison reduces to computer vision. Spectra are images. GPU interference is the universal comparator. |
| **GPU Observation Architecture** | Fragment shader IS observation apparatus. O(1) memory. GPU-supervised training with physical observables. Integrated GPU sufficient. |
| **Ray-Tracing Cellular Computing** | Single ray march simultaneously computes optical absorption, chromatographic retention, and circuit current. 8 oscillator classes. Coherence = health. |

### Key Theorems

- **Triple Equivalence**: Oscillation = Category = Partition. S = k_B M ln(n).
- **Rendering--Measurement Identity**: R(F, S) = M(O, S). Rendering IS measurement.
- **Storage Redundancy**: Observations can be re-performed in O(1). Storing them is redundant.
- **Triple Observation Identity**: mu_a proportional to 1/(tau * d_S) proportional to G * RT. One ray march step = three simultaneous observations.
- **Coherence--Health Identity**: V_cell = eta_cell. Interference visibility = cellular coherence index.

## Running Locally

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Navigate to `/observe/microscopy` and drop an image.

Requirements: Node.js 18+, any modern browser with WebGL2 (97%+ support).

## Project Structure

```
hieronymus/
├── src/
│   ├── app/                    # Next.js 14 App Router pages
│   │   ├── page.tsx            # Landing page
│   │   ├── observe/            # Observation workspaces
│   │   │   └── microscopy/     # Microscopy encoder (live)
│   │   ├── match/              # Spectral matching
│   │   ├── diagnose/           # Cellular diagnostics + 3D
│   │   └── publications/       # Paper gallery
│   ├── engine/                 # GPU observation engine (core)
│   │   ├── ObservationEngine.ts
│   │   ├── ShaderCompiler.ts
│   │   ├── TextureManager.ts
│   │   ├── observation.worker.ts
│   │   └── useObservation.ts   # React hook
│   ├── shaders/                # GLSL fragment shaders
│   │   ├── encode_microscopy.glsl
│   │   ├── partition.glsl
│   │   ├── interference.glsl
│   │   ├── entropy.glsl
│   │   └── display.glsl
│   └── components/             # UI components
│       ├── DropZone.tsx
│       ├── ObservationPanel.tsx
│       └── SimplexTriangle.tsx
├── publications/               # 5 papers + source documents
├── shaders.html                # Standalone WebGL2 demo
└── architecture.md             # Detailed implementation blueprint
```

## Performance

| Metric | Value |
|--------|-------|
| GPU memory | ~25 MB (constant, independent of database size) |
| Pipeline latency (2D) | ~5 ms per observation |
| Pipeline latency (3D) | ~43 ms (23 FPS) |
| Bundle size (core) | <200 KB gzipped |
| Batch matching (1K items) | <1 s via edge function |
| Hardware required | Any laptop with integrated GPU |

## Publications

All papers are in `publications/` with validation experiments, results (JSON/CSV), and panel figures:

- `measurement-modalities-stereogram/` --- 1,572 lines, 53 theorems, 5 panels
- `image-harmonic-coupling/` --- 1,393 lines, 46 theorems, 5 panels
- `universal-spectral-matching/` --- 2,525 lines, 42 theorems, 5 panels
- `gpu-observation-architecture/` --- 3,268 lines, 52 theorems, 5 panels
- `ray-tracing-cellular-computing/` --- 3,104 lines, 18 theorems, 5 panels

Total: 11,862 lines of LaTeX, 211 formal theorem environments, 25 panel figures, 67 result files.

## License

See [LICENSE.md](LICENSE.md).
