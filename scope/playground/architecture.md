# SCOPE Playground — Web Tool Architecture

## What the playground is

The SCOPE Playground is a browser-based interactive tool for writing and executing SCOPE programs against real public microscopy image databases. It compiles and runs the full SCOPE five-phase pipeline on still images fetched from BBBC, Allen Cell, OpenCell, and IDR, and returns world-space measurements with formal uncertainty bounds.

The Rust desktop version handles streaming acquisition hardware, GPU-accelerated wavelet pipelines, z-stack 3D fields, and full HDF5 export. The playground handles everything that can be done in a browser — which is most of the interesting science.

---

## Stack

| Layer | Technology | Reason |
|---|---|---|
| App framework | Next.js 14 (App Router) | Existing project convention |
| Language | TypeScript (strict) | In-browser compiler is pure TS |
| Styling | Tailwind CSS, dark theme (`bg-dark text-light`) | Existing project convention |
| Animation | Framer Motion | Existing project convention |
| 2D visualisation | Canvas 2D API | Scale field heatmap, segmentation overlay, distance geodesic |
| 3D visualisation | Three.js + React Three Fiber | Point cloud, isosurface, entropy sphere, distance tube |
| Charts | D3.js | Spectral power-law plot, entropy trajectory, uncertainty bar, histogram |
| HTTP fetching | `fetch` + Next.js API route `/api/image-proxy` | BBBC, Allen Cell, IDR require server-side CORS proxy |
| In-browser compilation | Hand-written recursive-descent parser → AST → interpreter | No Wasm, no eval, purely deterministic |
| State management | React `useReducer` | Result state is complex enough to need a reducer |

---

## Page layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCOPE PLAYGROUND          [Run ▶]  [Clear]  [Export JSON]  [Export PNG]    │
├─────────────────────────┬───────────────────────────────────────────────────┤
│                         │  TABS: [Dataset] [Visualise] [Charts] [3D]        │
│   CODE EDITOR           ├───────────────────────────────────────────────────┤
│                         │                                                   │
│   scope nuclear_sep {   │  ── DATASET tab ──────────────────────────────── │
│     channels { ... }    │  ┌─────────────────────────────────────────────┐ │
│     coordinate_space {  │  │  BBBC039   HeLa Cells (Hoechst + Actin)     │ │
│       field 100x100 µm  │  │  BBBC006   CHO Cells (Tubulin)              │ │
│       depth 10          │  │  BBBC008   Drosophila (GFP + DAPI)          │ │
│       ...               │  │  AllenCell 3D Structures                    │ │
│     }                   │  │  IDR       Time-lapse                        │ │
│     nucleus_pair = ...  │  │  ─────────────────────────────────────────  │ │
│     dispatch { ... }    │  │  Selected: BBBC039 / SiR_Actin_001.tif      │ │
│   }                     │  │  Resolution: 0.1 µm/px  Size: 1024×1024     │ │
│                         │  │  Channels: DAPI, Actin                      │ │
│                         │  └─────────────────────────────────────────────┘ │
│                         │                                                   │
│                         │  ── VISUALISE tab ─────────────────────────────  │
│                         │  [Raw Image] [Scale Field] [Segmentation]         │
│                         │  [Distance Map] [Geodesic Path] [Overlay]         │
│                         │  ┌─────────────────────────────────────────────┐ │
│                         │  │                                             │ │
│                         │  │         Canvas 2D  (512×512 px)             │ │
│                         │  │   heatmap / overlay / geodesic path         │ │
│                         │  │                                             │ │
│                         │  └─────────────────────────────────────────────┘ │
│                         │                                                   │
│                         │  ── CHARTS tab ─────────────────────────────────  │
│                         │  [Spectral Power] [Entropy Trajectory]            │
│                         │  [Uncertainty Bar] [Scale Histogram]              │
│                         │  ┌─────────────────────────────────────────────┐ │
│                         │  │   D3.js chart panel (400×250 px)            │ │
│                         │  └─────────────────────────────────────────────┘ │
│                         │                                                   │
│                         │  ── 3D tab ─────────────────────────────────────  │
│                         │  [Scale Field] [Point Cloud] [Entropy Sphere]     │
│                         │  [Distance Tube] [Partition Tree]                 │
│                         │  ┌─────────────────────────────────────────────┐ │
│                         │  │   Three.js / R3F canvas  (full height)      │ │
│                         │  │   orbit / zoom / pan enabled                │ │
│                         │  └─────────────────────────────────────────────┘ │
│                         │                                                   │
│                         │  RESULTS PANEL                                    │
│                         │  distance: 14.312 ± 0.157 µm  (1.10%)            │
│                         │  goal: ✓ δd < 0.5 µm  ✗ δd < 0.1 µm            │
│                         │  S_k=0.412  S_t=0.281  S_e=0.307  Σ=1.000 ✓    │
│                         │  CRLB: 0.019 px  SNR: 10.2  H: 2.08 bits        │
├─────────────────────────┴───────────────────────────────────────────────────┤
│  CONSOLE                                                                     │
│  [COMPILE]  cell=PROPHASE  n=10  S_t: 0.500→0.200                           │
│  [MEASURE]  ᾱ=0.988 µm/px  power_law=-0.410  bilateral σ_s=5 σ_r=0.3 ✓   │
│  [EXECUTE]  d=14.312 µm  δd=0.157 µm  ε_weighted=0.016                     │
│  [EMIT]     S_k+S_t+S_e = 1.000000000000000 ✓                               │
│  [GOAL]     distance_uncertainty=0.157 µm  < 0.5 µm ✓  < 0.1 µm ✗         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module map

```
src/
  app/
    tools/
      scope-playground/
        page.tsx                    ← root page, reducer, wires all panels
        layout.tsx                  ← scoped layout (no extra navbar)
  api/
    image-proxy/
      route.ts                      ← server-side TIFF fetch + decode
  components/
    scope-playground/
      CodeEditor.tsx                ← textarea + keyword highlighting
      DatasetBrowser.tsx            ← database list + image selector
      TabPanel.tsx                  ← tab switcher (Dataset/Visualise/Charts/3D)
      visualise/
        Canvas2D.tsx                ← raw image, heatmap, segmentation, geodesic overlay
        ScaleFieldHeatmap.tsx       ← α(x,y) false-colour with colorbar
        SegmentationOverlay.tsx     ← binary mask + contour over raw image
        DistanceMap.tsx             ← T(x,y) fast-marching result + geodesic path line
        GeodesicPath.tsx            ← highlighted shortest path between two points
      charts/
        SpectralPowerChart.tsx      ← D3 log-log |û(k)| vs k, power-law fit line
        EntropyTrajectoryChart.tsx  ← D3 line chart S_k / S_t / S_e across phases
        UncertaintyBar.tsx          ← D3 horizontal bar: δd with goal threshold markers
        ScaleHistogram.tsx          ← D3 histogram of α(x,y) values
        ChannelCapacityChart.tsx    ← D3 curve C = ½log₂(1+SNR) with operating point
      threed/
        SceneRoot.tsx               ← R3F Canvas, camera, lighting, OrbitControls
        ScaleFieldSurface.tsx       ← α(x,y) extruded as a 3D height map mesh
        PointCloud.tsx              ← image pixels as coloured 3D points (z = intensity)
        EntropySphere.tsx           ← sphere subdivided by S_k / S_t / S_e sectors
        DistanceTube.tsx            ← geodesic path rendered as a tube in 3D
        PartitionTree.tsx           ← tree of (n,ℓ,m,s) states as 3D graph
        IsoSurface.tsx              ← marching-squares iso-contours lifted into 3D
      ResultsPanel.tsx              ← distance, goal status, S-entropy, CRLB, SNR
      ConsoleLog.tsx                ← per-phase structured log lines
      GoalStatusBar.tsx             ← goal pass/fail chips with threshold values
  lib/
    scope-engine/
      index.ts                      ← public API: compile(), typeCheck(), execute()
      lexer.ts                      ← tokeniser
      parser.ts                     ← recursive-descent → AST
      type-checker.ts               ← five invariants (including goal reachability)
      phases/
        compile.ts                  ← Phase 1: timing → cell label
        measure.ts                  ← Phase 2: spectral scale field → Φ
        execute.ts                  ← Phase 3: morphism chain interpreter
        emit.ts                     ← Phase 4: assemble Result
      databases/
        bbbc.ts
        allen.ts
        idr.ts
        opencell.ts
      mic/
        scale-field.ts              ← windowed FFT → bilateral α(x,y)
        fast-marching.ts            ← geodesic distance T(x,y)
        entropy.ts                  ← Shannon H, Fisher F, CRLB
        segmentation.ts             ← level-set active contour, Otsu threshold
      visualisation/
        colormap.ts                 ← viridis / plasma / RdBu palettes as Float32Array
        geodesic-path.ts            ← backtrack from T(x,y) to extract path pixels
        spectral-analysis.ts        ← radial power spectrum for D3 chart
        entropy-trajectory.ts       ← per-phase S_k/S_t/S_e arrays for D3 chart
```

---

## Data flow through the five phases

```
User selects dataset + image
        │
        ▼
/api/image-proxy fetches .tif server-side → decodes TIFF → returns
  { width, height, data: Float32Array (normalised [0,1]) }
        │
        ▼
Phase 1 — COMPILE  (compile.ts)
  Input:  pixel histogram → synthetic ΔP events
          channels { cell } timing bounds from AST
  Output: cell_label (n, ℓ, m, s)   trajectory τ
  Emits:  ChartData.entropyTrajectory[0] = { S_k, S_t, S_e }
        │
        ▼
Phase 2 — MEASURE  (measure.ts)
  Input:  image Float32Array
  Stage1: windowed 2D FFT → |û(k)|  → SpectralPowerChart data
  Stage2: spectral gradient → α(x,y) raw  → ScaleHistogram data
  Stage3: bilateral filter → α(x,y) smooth
  Output: CoordField Φ  scaleField α
  Emits:  ChartData.spectralPower[]  ChartData.scaleHistogram[]
        │
        ▼
Phase 3 — EXECUTE  (execute.ts)
  Input:  morphism chain  Φ  image
  Steps:
    observe      → Σ₀ = (n,0,0,+½)
    catalyze     → ε·w applied  partition box shrunk
    access       → segmentation → centroid  fuzzy threshold applied
    measure_distance → fast-marching T(x,y) → d + geodesic path pixels
    fuse         → weighted average
  Output: Σ_f  d  δd  geodesic path pixels
  Emits:  DistanceMap T(x,y)  GeodesicPath pixels
          ChartData.uncertaintyBar { d, δd, goals[] }
        │
        ▼
Phase 4 — EMIT  (emit.ts)
  Input:  Σ_f  d  δd  s_entropy
  Output: Result { structure, position, distance, uncertainty, s_entropy,
                   goalStatus[], chartData, visualData }
  Verify: S_k + S_t + S_e = 1 ± 1e-12
  Emits:  ChartData.entropyTrajectory[4] = final entropy
        │
        ├──→ ResultsPanel (numbers)
        ├──→ GoalStatusBar (pass/fail chips)
        ├──→ Canvas2D (heatmap / segmentation / distance map)
        ├──→ D3 charts (spectral power, entropy trajectory, uncertainty, histogram)
        └──→ Three.js scene (scale field surface, point cloud, entropy sphere,
                             distance tube, partition tree)
```

---

## Visualisation outputs — what each view shows

### Canvas 2D views

| Tab | What is rendered | Data source |
|---|---|---|
| Raw Image | Normalised [0,1] grayscale with false-colour LUT | `image Float32Array` |
| Scale Field | α(x,y) as a heatmap (viridis), colorbar on right | `scaleField.alpha` |
| Segmentation | Binary mask (cyan) overlaid on raw image + contour | `segmentation.mask` |
| Distance Map | T(x,y) fast-marching result (plasma LUT) + source dot + target dot | `distanceMap` |
| Geodesic Path | Raw image with the shortest path drawn as a 2px white line | `geodesicPath.pixels` |
| Overlay | Raw + scale field + segmentation boundary + path combined | all above |

### D3 chart views

| Chart | X axis | Y axis | Key annotation |
|---|---|---|---|
| Spectral Power | log frequency (cycles/image) | log spectral energy | Power-law fit line, measured exponent α |
| Entropy Trajectory | Phase (COMPILE → EMIT) | S_k / S_t / S_e stacked | Vertical line at phase boundary, Σ=1 constraint |
| Uncertainty Bar | — | δd value | Horizontal threshold lines from `goal {}` block |
| Scale Histogram | α value (µm/px) | Pixel count | Mean ᾱ vertical line, ±1σ band |
| Channel Capacity | SNR (linear) | C = ½log₂(1+SNR) bits | Operating point dot for this image's SNR |

### Three.js 3D views

| Scene | What is shown | Controls |
|---|---|---|
| Scale Field Surface | α(x,y) extruded as height map mesh, viridis colour | Orbit, zoom, pan |
| Point Cloud | Every image pixel as a point; x,y = pixel coords, z = intensity; colour = α | Orbit, zoom |
| Entropy Sphere | Unit sphere sliced into three sectors proportional to S_k / S_t / S_e; sectors labelled | Orbit |
| Distance Tube | The geodesic path between the two measured points rendered as a glowing tube in 3D; tube radius = δd | Orbit |
| Partition Tree | Hierarchical graph of (n,ℓ,m,s) states visited during EXECUTE; nodes coloured by entropy contribution | Orbit, click node to inspect |
| Iso-Surface | Marching-squares contours at 0.25, 0.5, 0.75 of α(x,y), lifted into 3D at those heights | Orbit |

---

## Image proxy

```
GET /api/image-proxy?url=<encoded-image-url>
```

Fetches the `.tif` server-side, decodes via the `tiff` npm package (supports 8-bit and 16-bit), normalises to `Float32Array` in `[0,1]`, returns `{ width, height, data: number[] }` as JSON. Images are cached in `sessionStorage` by URL hash for the browser session — a 1024×1024 float32 image is ~4MB serialised, well within quota.

---

## Playground vs desktop (Rust)

| Feature | Playground (TypeScript) | Desktop (Rust) |
|---|---|---|
| Image size | ≤ 1024×1024 | Unlimited (mmap) |
| Scale field | Windowed FFT O(N·W²) | GPU wavelet WGPU O(N log N) |
| Fast marching | JS min-heap O(N log N) | Rust priority queue ~40× faster |
| Temporal acquisition | Synthetic from histogram | Real hardware serial/USB |
| Streaming frames | No | Yes, ring buffer |
| 3D coordinate fields | 2D slice only | Full z-stack |
| Fuzzy access threshold | Yes | Yes |
| Goal block | Yes | Yes |
| Confidence-weighted ε | Yes | Yes |
| Export | JSON + PNG + SVG charts | HDF5 / TIFF / CSV |
| S-entropy precision | 1e-12 | f64 (~1e-16) |
| 3D visualisation | Three.js in browser | Native OpenGL/Vulkan |
