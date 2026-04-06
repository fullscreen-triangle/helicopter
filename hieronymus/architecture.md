# Hieronymus — Website Architecture

> **Purpose**: A client-side GPU observation platform where collaborators drop data (images, spectra, sequences, time series) and receive computed results — partition signatures, match scores, coherence diagnostics, 3D reconstructions — all running on their integrated GPU via OffscreenCanvas + WebGL2/WebGPU. No Python. No server-side computation. Edge functions handle only database lookups.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Technology Stack](#2-technology-stack)
3. [Site Map & Routes](#3-site-map--routes)
4. [GPU Observation Engine](#4-gpu-observation-engine)
5. [Shader Pipeline](#5-shader-pipeline)
6. [OffscreenCanvas Worker Architecture](#6-offscreencanvas-worker-architecture)
7. [Domain Encoders](#7-domain-encoders)
8. [3D Visualization Layer](#8-3d-visualization-layer)
9. [Edge Functions & Database](#9-edge-functions--database)
10. [UI Components](#10-ui-components)
11. [Data Flow](#11-data-flow)
12. [File Structure](#12-file-structure)
13. [Implementation Phases](#13-implementation-phases)
14. [Performance Budget](#14-performance-budget)
15. [Papers & Theory Reference](#15-papers--theory-reference)

---

## 1. Design Principles

- [ ] **Observation = Computation**: The GPU fragment shader IS the computation. Textures are results, not pictures. Display is optional.
- [ ] **O(1) GPU Memory**: Stream and observe on demand. Never store observation results. GPU holds the apparatus (shaders), not the data.
- [ ] **Client-Side First**: All GPU computation runs in the user's browser on their integrated GPU. Zero server-side compute.
- [ ] **No Python Runtime**: Python exists only in the repo for offline validation/panel generation. Production runtime is 100% JS/TS + GLSL.
- [ ] **Serverless Backend**: Edge functions (Vercel/Cloudflare) for database lookups only. S-entropy coordinates are 12 bytes per item — trivial payload.
- [ ] **Progressive Enhancement**: Core functionality works with WebGL2 (95%+ browser support). WebGPU provides compute shaders for advanced features when available.
- [ ] **Domain-Agnostic Core**: The observation engine is universal. Only the encoder (data → S-entropy coordinates) is domain-specific. Everything else is shared.

---

## 2. Technology Stack

### Frontend (Client)
- [x] **Framework**: Next.js 14+ (App Router) — upgraded from 13 to 14.2
- [x] **UI**: React 18 + Tailwind CSS 3 + Framer Motion 10 — kept from template
- [x] **GPU**: WebGL2 (primary) + WebGPU (progressive enhancement)
- [ ] **3D Visualization**: Three.js (only for optional display, NOT for computation)
- [x] **Worker**: OffscreenCanvas in Web Worker for background GPU computation
- [x] **State**: React Context + useReducer (lightweight, no Redux)
- [ ] **Charts**: Lightweight — either visx, recharts, or raw Canvas2D

### Backend (Edge)
- [ ] **Edge Functions**: Vercel Edge Functions (or Cloudflare Workers)
- [ ] **Database**: Vercel KV or Cloudflare KV for S-entropy coordinate storage
- [ ] **Metadata**: Vercel Postgres or Cloudflare D1 for item metadata (names, labels, provenance)
- [ ] **No Python**: No Flask, no FastAPI, no server-side ML

### Build & Deploy
- [ ] **Build**: Next.js build with static export where possible
- [ ] **Deploy**: Vercel (zero-config for Next.js)
- [ ] **CI**: GitHub Actions for lint + type check

---

## 3. Site Map & Routes

```
/                           Landing page — hero, value proposition, demo teaser
/observe                    Main observation workspace (drop data, run pipeline)
/observe/microscopy         Microscopy-specific encoder + dual-pixel pipeline
/observe/molecular          Molecular spectroscopy encoder
/observe/genomic            Genomic sequence encoder
/observe/signal             Time series / signal encoder
/observe/general            General-purpose encoder (any numeric data)
/match                      Spectral matching workspace (compare two items)
/match/batch                Batch matching against a database
/diagnose                   Cellular diagnostic workspace (coherence index)
/diagnose/3d                3D holographic reconstruction + ray tracing
/publications               Paper gallery (5 papers + source docs)
/publications/[slug]        Individual paper page with panels, results, abstract
/about                      About the framework, theory overview
/docs                       API documentation for the observation engine
/docs/encoders              How to write a domain encoder
/docs/shaders               Shader pipeline reference
```

### Route Implementation Status
- [ ] `/` — Landing page
- [ ] `/observe` — Main workspace
- [ ] `/observe/microscopy` — Microscopy encoder
- [ ] `/observe/molecular` — Molecular encoder
- [ ] `/observe/genomic` — Genomic encoder
- [ ] `/observe/signal` — Signal encoder
- [ ] `/observe/general` — General encoder
- [ ] `/match` — Matching workspace
- [ ] `/match/batch` — Batch matching
- [ ] `/diagnose` — Diagnostic workspace
- [ ] `/diagnose/3d` — 3D reconstruction
- [ ] `/publications` — Paper gallery
- [ ] `/publications/[slug]` — Individual paper
- [ ] `/about` — About page
- [ ] `/docs` — Documentation
- [ ] `/docs/encoders` — Encoder docs
- [ ] `/docs/shaders` — Shader docs

---

## 4. GPU Observation Engine

The core engine that runs inside a Web Worker with OffscreenCanvas. This is the heart of the application.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Main Thread (UI)                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │ React UI │  │ Drop Zone│  │ Results Display       │  │
│  └────┬─────┘  └────┬─────┘  └──────────▲───────────┘  │
│       │              │                    │              │
│       │    postMessage(imageData)         │ postMessage  │
│       │              │                    │ (results)    │
├───────┼──────────────┼────────────────────┼──────────────┤
│  Web Worker Thread                                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │  ObservationEngine                                 │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  OffscreenCanvas + WebGL2 Context           │  │  │
│  │  │                                             │  │  │
│  │  │  ┌──────────┐ ┌──────────┐ ┌────────────┐ │  │  │
│  │  │  │ Shader   │ │ Shader   │ │ Shader     │ │  │  │
│  │  │  │ Pass 0   │ │ Pass 1   │ │ Pass 2...  │ │  │  │
│  │  │  │ Encode   │ │ Partition│ │ Interfere  │ │  │  │
│  │  │  └──────────┘ └──────────┘ └────────────┘ │  │  │
│  │  │                                             │  │  │
│  │  │  ┌──────────────────────────────────────┐  │  │  │
│  │  │  │ Framebuffer Objects (FBOs)           │  │  │  │
│  │  │  │ - Partition texture (RGBA32F)        │  │  │  │
│  │  │  │ - Interference texture (RGBA32F)     │  │  │  │
│  │  │  │ - Quality metrics texture (RGBA32F)  │  │  │  │
│  │  │  └──────────────────────────────────────┘  │  │  │
│  │  │                                             │  │  │
│  │  │  Readback: 5 floats (S_k, S_t, S_e, V, η) │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Engine API (Worker Message Protocol)

```typescript
// Main → Worker messages
type WorkerInput =
  | { type: "init" }                           // Initialize WebGL context
  | { type: "observe"; data: ImageData; encoder: string }  // Run observation
  | { type: "match"; dataA: ImageData; dataB: ImageData }  // Compare two
  | { type: "batch"; query: ImageData; dbIds: string[] }   // Batch match
  | { type: "diagnose"; data: ImageData; mode: "2d" | "3d" } // Health diagnostic
  | { type: "setUniforms"; uniforms: Record<string, number> } // Update params

// Worker → Main messages
type WorkerOutput =
  | { type: "ready" }
  | { type: "result"; payload: ObservationResult }
  | { type: "matchResult"; payload: MatchResult }
  | { type: "batchResult"; payload: BatchResult }
  | { type: "diagnoseResult"; payload: DiagnoseResult }
  | { type: "error"; message: string }
  | { type: "progress"; stage: string; percent: number }

interface ObservationResult {
  S_k: number;             // Knowledge entropy [0,1]
  S_t: number;             // Temporal entropy [0,1]
  S_e: number;             // Evolution entropy [0,1]
  conservation: number;    // |S_k + S_t + S_e - 1| (should be ~0)
  partitionDepth: number;  // Mean principal depth n
  sharpness: number;       // Partition sharpness (quality metric)
  noise: number;           // Observation noise (quality metric)
  coherence: number;       // Phase coherence (quality metric)
  visibility: number;      // Interference visibility
  elapsed_ms: number;      // Pipeline time
  // Optional: full texture data for visualization
  partitionTexture?: Float32Array;
  entropyTexture?: Float32Array;
}

interface MatchResult {
  score: number;           // Match score [0,1]
  visibility: number;      // Interference visibility
  circuits: number;        // Number of matching circuits detected
  S_distance: number;      // Categorical distance d_S
  elapsed_ms: number;
}

interface DiagnoseResult {
  eta_cell: number;        // Coherence index [0,1]
  healthy: boolean;        // eta_cell > 0.7
  decoherent_classes: string[];  // Which oscillator classes are decoherent
  retention: number;       // Total chromatographic retention
  current: number;         // Total circuit current
  flow_velocity: number;   // Mean cytoplasmic flow velocity
  elapsed_ms: number;
  // Optional: 3D volume for visualization
  volumeTexture?: Float32Array;
}
```

### Engine Implementation Checklist
- [x] `src/engine/ObservationEngine.ts` — Main engine class
- [x] `src/engine/ShaderCompiler.ts` — Compile + link GLSL programs
- [x] `src/engine/TextureManager.ts` — FBO creation, texture lifecycle
- [x] `src/engine/observation.worker.ts` — Web Worker entry point
- [x] `src/engine/useObservation.ts` — React hook wrapping the worker

---

## 5. Shader Pipeline

Six shader programs, each implementing one observation pass. All shaders read from the same theoretical framework (Papers 1-5).

### Pass 0: Domain Encoder (per-domain, swappable)
- **Input**: Raw data texture (image, spectrum, sequence)
- **Output**: S-entropy coordinates texture (RGBA32F: R=S_k, G=S_t, B=S_e, A=unused)
- **Theory**: S-entropy coordinate definition (Paper 3, Sec. 6; Paper 4, Sec. 7)

### Pass 1: Partition Observation
- **Input**: S-entropy texture from Pass 0
- **Output**: Partition coordinates texture (RGBA32F: R=n/n_max, G=ℓ/n_max, B=m_norm, A=s)
- **Theory**: Partition coordinates from hierarchical subdivision, C(n)=2n² (all papers, Sec. 3-4)
- **Source**: Adapted from `shaders.html` Pass 1

### Pass 2: Interference / Consistency
- **Input**: Two partition textures (query + target, or visible + invisible)
- **Output**: Interference map (RGBA32F: R=d_cat, G=validity_mask, B=correlation_rho, A=fused_n)
- **Theory**: Dual-pixel consistency (Paper 1), interference visibility (Paper 2), spectral matching (Paper 3)
- **Source**: Adapted from `shaders.html` Pass 3

### Pass 3: Entropy Conservation & Quality
- **Input**: S-entropy texture, partition texture, interference texture
- **Output**: Quality metrics (RGBA32F: R=Sk_color, G=St_color, B=Se_color, A=sum_check)
- **Theory**: S_k + S_t + S_e = 1 conservation (all papers); physical observables (Paper 4, Sec. 16)
- **Source**: Adapted from `shaders.html` Pass 4

### Pass 4: Ray March (3D diagnostic mode only)
- **Input**: 3D volume texture (from holographic back-propagation)
- **Output**: Ray accumulation (RGBA32F per ray: R=color, G=retention, B=current, A=phase)
- **Theory**: Triple observation identity (Paper 5, Sec. 12), Algorithm 1
- **Source**: Paper 5, Listing 1

### Pass 5: Multi-Ray Interference (3D diagnostic mode only)
- **Input**: N phase maps from Pass 4
- **Output**: Coherence map (RGBA32F: R=eta_cell, G=V_cell)
- **Theory**: Coherence-health identity V_cell = eta_cell (Paper 5, Sec. 15)
- **Source**: Paper 5, Listing 2

### Shader Implementation Checklist
- [x] `src/shaders/vertex.glsl` — Universal fullscreen quad vertex shader
- [x] `src/shaders/encode_microscopy.glsl` — Microscopy domain encoder (Pass 0)
- [ ] `src/shaders/encode_spectral.glsl` — Molecular spectral encoder (Pass 0)
- [ ] `src/shaders/encode_genomic.glsl` — Genomic k-mer encoder (Pass 0)
- [ ] `src/shaders/encode_signal.glsl` — Time series encoder (Pass 0)
- [ ] `src/shaders/encode_general.glsl` — General-purpose encoder (Pass 0)
- [x] `src/shaders/partition.glsl` — Partition observation (Pass 1)
- [x] `src/shaders/interference.glsl` — Interference / consistency (Pass 2)
- [x] `src/shaders/entropy.glsl` — Entropy conservation + quality (Pass 3)
- [ ] `src/shaders/ray_march.glsl` — Triple-observation ray march (Pass 4)
- [ ] `src/shaders/multi_ray.glsl` — Multi-ray interference (Pass 5)
- [x] `src/shaders/display.glsl` — Optional display shader (colormaps)

---

## 6. OffscreenCanvas Worker Architecture

### Why OffscreenCanvas
- Fragment shaders compute WITHOUT rendering to screen
- Runs in Web Worker — main thread never blocked
- GPU writes to FBO textures, we readback only scalars (5-20 floats)
- User sees results (numbers, charts) while GPU works invisibly

### Worker Lifecycle
```
1. Main thread creates Worker
2. Worker creates OffscreenCanvas(256, 256) — or (512,512) for high-res
3. Worker gets WebGL2 context from OffscreenCanvas
4. Worker compiles all shader programs
5. Worker signals "ready"
6. Main thread sends data via postMessage (transferable ArrayBuffer)
7. Worker runs pipeline passes on GPU
8. Worker reads back scalars via gl.readPixels (tiny: 4-80 bytes)
9. Worker posts results back to main thread
10. Main thread updates UI
```

### Worker Checklist
- [ ] `src/engine/observation.worker.ts` — Worker entry, message dispatch
- [ ] `src/engine/useObservation.ts` — React hook: `const { observe, match, diagnose, ready } = useObservation()`
- [ ] Handle Worker instantiation/termination on component mount/unmount
- [ ] Handle transferable ArrayBuffers for zero-copy image passing
- [ ] Handle WebGL2 context loss + recovery in Worker
- [ ] Graceful fallback if OffscreenCanvas unsupported (run on main thread canvas)

---

## 7. Domain Encoders

Each encoder maps domain-specific data to universal S-entropy coordinates (S_k, S_t, S_e) in [0,1]^3. This is the ONLY domain-specific component.

### Microscopy Encoder
- **Input**: Image (PNG/JPG/TIFF)
- **S_k**: From local Shannon entropy (partition configuration complexity)
- **S_t**: From gradient magnitude (spatial frequency content)
- **S_e**: Conservation residual: 1 - S_k - S_t
- **Source**: `shaders.html` Pass 1 logic (already implemented)

### Molecular Spectral Encoder
- **Input**: Frequency list {ω_k, A_k} (CSV, JSON, or pasted text)
- **S_k**: Normalized spectral entropy: -Σ p_k log p_k / log(N)
- **S_t**: Frequency centroid: Σ(ω_k * A_k) / Σ(A_k), normalized
- **S_e**: Spectral spread: std(ω_k * A_k) / max(ω_k), normalized
- **Theory**: Paper 3, Sec. 23a

### Genomic Encoder
- **Input**: Nucleotide sequence (FASTA, pasted text)
- **S_k**: k-mer entropy (k=3): Shannon entropy of trinucleotide frequencies
- **S_t**: Positional complexity: entropy of k-mer positions along sequence
- **S_e**: Repeat depth: fraction of sequence covered by tandem repeats
- **Theory**: Paper 3, Sec. 23c; Paper 4, Sec. 23

### Signal / Time Series Encoder
- **Input**: Numeric time series (CSV, JSON)
- **S_k**: Power spectrum entropy (from FFT)
- **S_t**: Temporal autocorrelation at lag 1
- **S_e**: Spectral centroid / Nyquist, normalized
- **Theory**: Paper 3, Sec. 23d

### General Encoder
- **Input**: Any numeric vector or matrix (CSV, JSON)
- **S_k**: Shannon entropy of value distribution
- **S_t**: Mean pairwise correlation (for matrix) or autocorrelation (for vector)
- **S_e**: Conservation residual: 1 - S_k - S_t

### Encoder Checklist
- [ ] `src/encoders/microscopy.ts` — CPU preprocessing for microscopy
- [ ] `src/encoders/molecular.ts` — Parse frequency data, compute S-entropy
- [ ] `src/encoders/genomic.ts` — Parse FASTA, compute k-mer frequencies
- [ ] `src/encoders/signal.ts` — Parse CSV/JSON time series, FFT
- [ ] `src/encoders/general.ts` — Generic numeric encoder
- [ ] `src/encoders/types.ts` — Shared types (SEntropyCoords, EncoderInput, etc.)

---

## 8. 3D Visualization Layer (Optional Display)

Three.js is used ONLY for optional user-facing visualization. It is never part of the computation path.

### Features
- [ ] Interactive 3D volume rendering (from ray march texture)
- [ ] Holographic reconstruction viewer (depth-slice scrubber)
- [ ] S-entropy simplex triangle (2D Canvas, already in `shaders.html`)
- [ ] Partition depth colormap on input image
- [ ] Interference visibility map overlay
- [ ] Coherence index gauge / dial

### Visualization Checklist
- [ ] `src/viz/VolumeViewer.tsx` — Three.js volume renderer (lazy loaded)
- [ ] `src/viz/SimplexTriangle.tsx` — S-entropy simplex (Canvas2D, port from shaders.html)
- [ ] `src/viz/PartitionOverlay.tsx` — Partition colormap overlay on image
- [ ] `src/viz/CoherenceGauge.tsx` — Circular gauge for eta_cell
- [ ] `src/viz/MatchHeatmap.tsx` — Heatmap for batch match results

---

## 9. Edge Functions & Database

### Edge Function: `/api/match`
```typescript
// Receives: query S-entropy (12 bytes: 3 x float32)
// Returns: top-K matching items from database
export default async function handler(req) {
  const { S_k, S_t, S_e } = await req.json();
  const results = await kv.scan("sentropy:*");
  // Compute S-distance to each item (trivial arithmetic)
  // Return top-K ranked by S-distance
  return Response.json(topK);
}
```

### Edge Function: `/api/store`
```typescript
// Receives: S-entropy coords + metadata
// Stores in KV
export default async function handler(req) {
  const { S_k, S_t, S_e, name, domain, metadata } = await req.json();
  await kv.set(`sentropy:${id}`, { S_k, S_t, S_e });
  await db.insert("items", { id, name, domain, metadata });
  return Response.json({ id });
}
```

### Database Schema
```
KV Store (Vercel KV / Cloudflare KV):
  key: "sentropy:{uuid}"
  value: { S_k: float, S_t: float, S_e: float }  // 12 bytes

Relational (Vercel Postgres / Cloudflare D1):
  items:
    id: uuid (PK)
    name: text
    domain: enum(microscopy, molecular, genomic, signal, general)
    created_at: timestamp
    metadata: jsonb
```

### Edge Checklist
- [ ] `src/app/api/match/route.ts` — Match endpoint
- [ ] `src/app/api/store/route.ts` — Store endpoint
- [ ] `src/app/api/lookup/route.ts` — Metadata lookup
- [ ] Database schema migration script
- [ ] KV store setup

---

## 10. UI Components

### Existing (from template, keep/adapt)
- [ ] `Navbar.js` → `Navbar.tsx` — Add observation/match/diagnose nav items
- [ ] `Layout.js` → `Layout.tsx` — Keep wrapper, add worker provider
- [ ] `Footer.js` → `Footer.tsx` — Update links
- [ ] `TransitionEffect.js` → keep for page transitions
- [ ] `AnimatedText.js` → keep for headings
- [ ] Theme switcher (light/dark) — keep

### New Components
- [ ] `src/components/DropZone.tsx` — Universal file drop zone (images, CSV, FASTA, JSON)
- [ ] `src/components/ObservationPanel.tsx` — Real-time metrics display (S_k, S_t, S_e, quality)
- [ ] `src/components/PassSelector.tsx` — Shader pass selector (from shaders.html)
- [ ] `src/components/UniformSliders.tsx` — Parameter sliders (epsilon, J, beta, n_max, A_eg, alpha)
- [ ] `src/components/ResultsTable.tsx` — Tabular results display
- [ ] `src/components/PipelineLog.tsx` — Real-time pipeline execution log
- [ ] `src/components/MatchComparison.tsx` — Side-by-side comparison view
- [ ] `src/components/BatchResults.tsx` — Ranked results list for batch matching
- [ ] `src/components/PaperCard.tsx` — Publication card with abstract, panels, results
- [ ] `src/components/DomainSelector.tsx` — Encoder domain selection tabs
- [ ] `src/components/ExportButton.tsx` — Export results as JSON/CSV

---

## 11. Data Flow

### Observe Flow (single item)
```
User drops image/file
  → Main thread: parse file type, select encoder
  → Main thread: send ImageData/ArrayBuffer to Worker (transferable)
  → Worker: run encoder (Pass 0) → S-entropy texture
  → Worker: run partition (Pass 1) → partition texture
  → Worker: run entropy check (Pass 3) → quality texture
  → Worker: readback 10 floats (S_k, S_t, S_e, n_mean, sharpness, noise, coherence, V, elapsed, conservation)
  → Worker: postMessage results to Main
  → Main thread: update ObservationPanel, SimplexTriangle, PartitionOverlay
  → (Optional) User clicks "Store": Main sends S-entropy to /api/store edge function
```

### Match Flow (two items)
```
User drops two files (or one file + selects database item)
  → Main thread: send both to Worker
  → Worker: encode both (Pass 0 × 2)
  → Worker: partition both (Pass 1 × 2)
  → Worker: interference (Pass 2) with both partition textures
  → Worker: readback (score, visibility, circuits, d_S)
  → Worker: postMessage to Main
  → Main: update MatchComparison view
```

### Batch Match Flow
```
User drops query file + selects "Match against database"
  → Main: encode query via Worker → get S-entropy
  → Main: send S-entropy to /api/match edge function
  → Edge: scan KV, rank by S-distance, return top-K metadata
  → Main: display ranked results in BatchResults
  → (Optional) User clicks item: fetch full observation for detailed comparison
```

### Diagnose Flow (3D)
```
User drops microscopy image + selects "3D Diagnostic"
  → Worker: Pass 0-1 (encode + partition) → 2D observation
  → Worker: Holographic back-propagation (Pass 1b) → 3D volume texture
  → Worker: Ray march (Pass 4) × N directions → phase maps
  → Worker: Multi-ray interference (Pass 5) → coherence map
  → Worker: readback (eta_cell, retention, current, flow_velocity)
  → Worker: postMessage results + volume texture to Main
  → Main: display DiagnoseResult + launch VolumeViewer (Three.js, lazy)
```

---

## 12. File Structure

```
hieronymus/
├── architecture.md                  ← This file
├── shaders.html                     ← Existing standalone demo (keep as reference)
├── next.config.js                   ← Next.js config (upgrade)
├── tailwind.config.js               ← Tailwind config (adapt)
├── postcss.config.js                ← PostCSS config (keep)
├── package.json                     ← Dependencies (upgrade)
├── tsconfig.json                    ← TypeScript config (add)
│
├── public/
│   ├── fonts/                       ← Montserrat, Space Mono
│   └── images/                      ← Static assets
│
├── publications/                    ← Papers (existing, keep as-is)
│   ├── measurement-modalities-stereogram/
│   ├── image-harmonic-coupling/
│   ├── universal-spectral-matching/
│   ├── gpu-observation-architecture/
│   ├── ray-tracing-cellular-computing/
│   └── sources/
│
├── src/
│   ├── app/                         ← Next.js App Router pages
│   │   ├── layout.tsx               ← Root layout (Navbar, Footer, WorkerProvider)
│   │   ├── page.tsx                 ← Landing page /
│   │   ├── observe/
│   │   │   ├── page.tsx             ← Main observation workspace
│   │   │   ├── microscopy/page.tsx
│   │   │   ├── molecular/page.tsx
│   │   │   ├── genomic/page.tsx
│   │   │   ├── signal/page.tsx
│   │   │   └── general/page.tsx
│   │   ├── match/
│   │   │   ├── page.tsx             ← Matching workspace
│   │   │   └── batch/page.tsx       ← Batch matching
│   │   ├── diagnose/
│   │   │   ├── page.tsx             ← 2D diagnostic
│   │   │   └── 3d/page.tsx          ← 3D holographic + ray tracing
│   │   ├── publications/
│   │   │   ├── page.tsx             ← Paper gallery
│   │   │   └── [slug]/page.tsx      ← Individual paper
│   │   ├── about/page.tsx
│   │   ├── docs/
│   │   │   ├── page.tsx
│   │   │   ├── encoders/page.tsx
│   │   │   └── shaders/page.tsx
│   │   └── api/
│   │       ├── match/route.ts       ← Edge: S-distance matching
│   │       ├── store/route.ts       ← Edge: store S-entropy
│   │       └── lookup/route.ts      ← Edge: metadata lookup
│   │
│   ├── engine/                      ← GPU observation engine (core)
│   │   ├── ObservationEngine.ts     ← Main engine class
│   │   ├── ShaderCompiler.ts        ← GLSL compile + link
│   │   ├── TextureManager.ts        ← FBO + texture lifecycle
│   │   ├── Readback.ts              ← GPU → CPU extraction
│   │   ├── observation.worker.ts    ← Web Worker entry
│   │   └── useObservation.ts        ← React hook
│   │
│   ├── shaders/                     ← GLSL source files
│   │   ├── vertex.glsl
│   │   ├── encode_microscopy.glsl
│   │   ├── encode_spectral.glsl
│   │   ├── encode_genomic.glsl
│   │   ├── encode_signal.glsl
│   │   ├── encode_general.glsl
│   │   ├── partition.glsl
│   │   ├── interference.glsl
│   │   ├── entropy.glsl
│   │   ├── ray_march.glsl
│   │   ├── multi_ray.glsl
│   │   └── display.glsl
│   │
│   ├── encoders/                    ← Domain-specific CPU preprocessing
│   │   ├── types.ts
│   │   ├── microscopy.ts
│   │   ├── molecular.ts
│   │   ├── genomic.ts
│   │   ├── signal.ts
│   │   └── general.ts
│   │
│   ├── viz/                         ← Optional visualization (Three.js)
│   │   ├── VolumeViewer.tsx
│   │   ├── SimplexTriangle.tsx
│   │   ├── PartitionOverlay.tsx
│   │   ├── CoherenceGauge.tsx
│   │   └── MatchHeatmap.tsx
│   │
│   ├── components/                  ← UI components
│   │   ├── Navbar.tsx               ← Adapted from template
│   │   ├── Layout.tsx               ← Adapted from template
│   │   ├── Footer.tsx               ← Adapted from template
│   │   ├── AnimatedText.tsx         ← From template
│   │   ├── TransitionEffect.tsx     ← From template
│   │   ├── DropZone.tsx             ← New: file drop zone
│   │   ├── ObservationPanel.tsx     ← New: real-time metrics
│   │   ├── PassSelector.tsx         ← New: shader pass selector
│   │   ├── UniformSliders.tsx       ← New: parameter controls
│   │   ├── ResultsTable.tsx         ← New: tabular results
│   │   ├── PipelineLog.tsx          ← New: execution log
│   │   ├── MatchComparison.tsx      ← New: side-by-side
│   │   ├── BatchResults.tsx         ← New: ranked list
│   │   ├── PaperCard.tsx            ← New: publication card
│   │   ├── DomainSelector.tsx       ← New: encoder tabs
│   │   └── ExportButton.tsx         ← New: JSON/CSV export
│   │
│   └── styles/
│       └── globals.css              ← Tailwind directives + custom vars
```

---

## 13. Implementation Phases

### Phase 1: Foundation (Engine + Microscopy) --- COMPLETE
> Get the core GPU observation engine working with microscopy images.

- [x] Upgrade Next.js to 14+ App Router
- [x] Add TypeScript configuration
- [x] Implement `ObservationEngine.ts` with WebGL2 context on OffscreenCanvas
- [x] Port `shaders.html` Pass 1-4 shaders to standalone GLSL files
- [x] Implement `observation.worker.ts`
- [x] Implement `useObservation.ts` hook
- [x] Build `/observe/microscopy` page with DropZone + ObservationPanel
- [ ] Verify S-entropy conservation holds to machine precision
- [ ] Verify pipeline runs in < 50ms on integrated GPU
- [x] Port SimplexTriangle from `shaders.html`

### Phase 2: Matching
> Enable comparing two observations and batch matching.

- [ ] Implement interference shader (Pass 2)
- [ ] Build `/match` page with dual DropZone + MatchComparison
- [ ] Implement edge function `/api/match`
- [ ] Implement edge function `/api/store`
- [ ] Set up KV store for S-entropy database
- [ ] Build `/match/batch` with BatchResults
- [ ] Verify match score symmetry: |Match(A,B) - Match(B,A)| < 1e-6

### Phase 3: Multi-Domain Encoders
> Extend beyond microscopy to molecular, genomic, signal, general data.

- [ ] Implement `encode_spectral.glsl` + `molecular.ts` encoder
- [ ] Implement `encode_genomic.glsl` + `genomic.ts` encoder
- [ ] Implement `encode_signal.glsl` + `signal.ts` encoder
- [ ] Implement `encode_general.glsl` + `general.ts` encoder
- [ ] Build `/observe/molecular`, `/observe/genomic`, `/observe/signal`, `/observe/general` pages
- [ ] Verify cross-domain matching: within-domain > cross-domain scores
- [ ] Build DomainSelector component

### Phase 4: 3D Diagnostic (Ray Tracing)
> Holographic reconstruction + triple-observation ray march.

- [ ] Implement holographic back-propagation (angular spectrum, FFT in shader or JS)
- [ ] Implement `ray_march.glsl` (Pass 4)
- [ ] Implement `multi_ray.glsl` (Pass 5)
- [ ] Build `/diagnose` page with coherence gauge
- [ ] Build `/diagnose/3d` with VolumeViewer (Three.js, lazy loaded)
- [ ] Verify eta_cell correctly separates healthy/diseased synthetic cells
- [ ] Verify triple observation consistency: r > 0.95 between mu_a, 1/(tau*d_S), G*RT

### Phase 5: Publications & Documentation
> Paper gallery and API documentation.

- [ ] Build `/publications` gallery page with PaperCard components
- [ ] Build `/publications/[slug]` with abstract, panels, results for each paper
- [ ] Build `/docs` with engine API reference
- [ ] Build `/docs/encoders` with guide for writing new encoders
- [ ] Build `/docs/shaders` with shader pipeline reference
- [ ] Build `/about` page with framework overview

### Phase 6: Polish & Deploy
> Performance, accessibility, deployment.

- [ ] Lighthouse audit: target 90+ on all metrics
- [ ] Responsive design verification (mobile, tablet, desktop)
- [ ] Error boundaries for WebGL context loss
- [ ] Loading states and progress indicators
- [ ] SEO metadata for all pages
- [ ] Deploy to Vercel
- [ ] Set up custom domain
- [ ] GitHub Actions CI (lint + type check + build)

---

## 14. Performance Budget

| Metric | Target | Rationale |
|--------|--------|-----------|
| GPU memory (observation) | < 25 MB | Paper 4: O(1) memory, 13 MB working set |
| Pipeline latency (2D) | < 50 ms | Paper 4: ~5ms per pass × 4 passes + overhead |
| Pipeline latency (3D) | < 100 ms | Paper 5: ~43ms for 5-pass pipeline |
| Time to first observation | < 2 s | Worker init + shader compile + first pass |
| Bundle size (core) | < 200 KB gzipped | Engine + shaders (no Three.js in core) |
| Bundle size (3D viz) | < 500 KB gzipped | Three.js lazy loaded only for /diagnose/3d |
| Edge function latency | < 50 ms | KV lookup + S-distance computation |
| Batch match (1000 items) | < 1 s | Edge: 1000 × S-distance is trivial arithmetic |

---

## 15. Papers & Theory Reference

Quick reference for which paper grounds which component:

| Component | Primary Paper | Key Theorem/Section |
|-----------|--------------|-------------------|
| Partition coordinates (n,ℓ,m,s) | All papers | C(n) = 2n², Axioms 1-2 |
| S-entropy conservation | All papers | S_k + S_t + S_e = 1 |
| Commutation [Ô_cat, Ô_phys] = 0 | All papers | Zero backaction |
| Visible + invisible pixel | Paper 1 | Dual-Pixel Consistency Theorem |
| Interference matching | Paper 2 | Matching circuits, V = similarity |
| Spectral images, universal matching | Paper 3 | Spectral Image Theorem, Universal Reduction |
| Rendering = measurement, O(1) memory | Paper 4 | Rendering-Measurement Identity, Storage Redundancy |
| GPU-supervised training | Paper 4 | Black-Box Oracle Convergence |
| Physical observables as quality metrics | Paper 4 | Sharpness, noise, coherence, visibility |
| Ray march triple observation | Paper 5 | Triple Observation Identity: μ_a ∝ 1/(τ·d_S) ∝ G·RT |
| 8 oscillator classes | Paper 5 | Protein→Circadian, 10^14 to 10^-5 Hz |
| Coherence = health | Paper 5 | V_cell = η_cell |
| Holographic 3D reconstruction | Paper 5 | Angular spectrum back-propagation |
| Cytoplasmic fluid dynamics | Paper 5 | Kirchhoff → Stokes, viscosity from partition lag |
| Domain encoders | Paper 3 + 4 | S-entropy from domain data |
| Existing shader demo | `shaders.html` | 4-pass WebGL2 pipeline (microscopy) |

---

## Notes

- The existing `shaders.html` is a working proof-of-concept of the 4-pass pipeline. It should be preserved as a standalone demo and used as reference for porting shaders into the modular engine.
- The existing Next.js template (pages router, JS) will be migrated to App Router + TypeScript. Existing components (Navbar, Footer, AnimatedText, etc.) are kept and adapted.
- Three.js is ONLY for the optional 3D visualization layer. It is never loaded unless the user navigates to `/diagnose/3d`. All computation uses raw WebGL2 via OffscreenCanvas.
- The architecture supports future WebGPU adoption: the engine abstraction (`ObservationEngine.ts`) can be extended with a WebGPU backend while keeping the same worker message protocol and React hook API.
