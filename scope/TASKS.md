# SCOPE Implementation Tasks

Tasks are struck out when complete. Do not delete them — the history is useful.

---

## Playground specifications

- ~~Write `scope/playground/architecture.md`~~
- ~~Write `scope/playground/compiler.md`~~
- ~~Write `scope/playground/syntax.md`~~
- ~~Write `scope/playground/examples.md`~~
- ~~Update all four files to include visualisation charts, 3D output, and Kwasa-Kwasa extensions (confidence, fuzzy access threshold, goal block)~~

---

## Playground TypeScript compiler (`hieronymus/src/lib/scope-compiler/`)

- ~~Write `ast.ts` — full AST node types matching spec (ScopeProgram, GoalDecl, RuleDecl, CatalyzeStep with confidence, AccessStep with threshold, VisualiseStep)~~
- ~~Write `lexer.ts` — full token set including goal/rule/invariant/epsilon/confidence/threshold/visualise/load/db/dataset/image/n/rho; comparison operators; µm/pixel~~
- ~~Write `parser.ts` — recursive descent, all blocks optional, load() frame ref, confidence/threshold optional args~~
- ~~Write `type-checker.ts` — all five invariants: depth compatibility, cell overlap, entropy budget (confidence-weighted), coordinate grounding, goal reachability (warning)~~
- ~~Write `index.ts` — clean public API: compile(source): CompileResult with ok/program/errors/warnings/log~~
- ~~Delete stale `compiler.ts` and `code-generator.ts`~~

---

## Playground TypeScript runtime (`hieronymus/src/lib/scope-runtime/`)

- ~~Write `phases/compile.ts` — timing events from histogram, cell classification, S_t update, entropyTrajectory[0]~~
- ~~Write `phases/measure.ts` — windowed FFT → spectral gradient → bilateral filter → α(x,y); chartData.spectralPower; scaleHistogram; pointCloud~~
- ~~Write `phases/execute.ts` — morphism chain interpreter: observe/catalyze/access/measure_distance/fuse/visualise~~
- ~~Write `phases/emit.ts` — assemble Result, evaluate goal criteria, verify S-entropy sum~~
- ~~Write `runtime.ts` — top-level run(program, image): Promise<Result>; orchestrates four phases~~
- Rewrite `real-executor.ts` to use new runtime + new AST (replace ExecutionPlan stub)
- ~~Write `/api/image-proxy/route.ts` — server-side TIFF fetch + decode → Float32Array JSON~~

---

## MIC algorithms (`hieronymus/src/lib/scope-runtime/mic/`)

- ~~Write `scale-field.ts` — Algorithm 1: windowed FFT → spectral gradient → bilateral filter → α(x,y) (extract from existing mic-engine/index.ts)~~
- ~~Write `fast-marching.ts` — geodesic distance T(x,y) + backtrack path extraction~~
- ~~Write `entropy.ts` — Shannon H, Fisher F, CRLB, SNR via Otsu~~
- ~~Write `segmentation.ts` — Otsu + level-set active contour; fuzzy membership map~~

---

## Playground web UI (`hieronymus/src/app/tools/scope-playground/`)

- ~~Write `page.tsx` — root page with useReducer state, wires all panels~~
- ~~Write `layout.tsx`~~
- Write `components/CodeEditor.tsx` — textarea with SCOPE keyword highlighting
- Write `components/DatasetBrowser.tsx` — BBBC/AllenCell/OpenCell/IDR selector (inlined into page.tsx DatasetTab)
- ~~Write `components/visualise/Canvas2D.tsx` — raw image, heatmap, segmentation, geodesic overlay~~
- ~~Write `components/charts/SpectralPowerChart.tsx` — D3 log-log power-law~~
- ~~Write `components/charts/EntropyTrajectoryChart.tsx` — D3 stacked area S_k/S_t/S_e~~
- ~~Write `components/charts/UncertaintyBar.tsx` — D3 bar with goal threshold lines~~
- ~~Write `components/charts/ScaleHistogram.tsx` — D3 histogram of α(x,y) values~~
- ~~Write `components/charts/ChannelCapacityChart.tsx` — D3 C=½log₂(1+SNR) curve (reuses SpectralPowerChart)~~
- ~~Write `components/threed/ScaleFieldSurface.tsx` — α(x,y) height map mesh~~
- ~~Write `components/threed/EntropySphere.tsx` — sphere sectored by S_k/S_t/S_e~~
- ~~Write `components/threed/DistanceTube.tsx` — geodesic path as glowing tube~~
- Write `components/threed/PointCloud.tsx` — pixels as coloured 3D points
- Write `components/threed/PartitionTree.tsx` — (n,ℓ,m,s) state tree as 3D graph

---

## Desktop specifications (`scope/desktop/`)

- Write `scope/desktop/architecture.md`
- Write `scope/desktop/compiler.md`
- Write `scope/desktop/syntax.md`
- Write `scope/desktop/examples.md`

---

## Desktop Rust implementation

- (Not started — begins after playground is complete)
