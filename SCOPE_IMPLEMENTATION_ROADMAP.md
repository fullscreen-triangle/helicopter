# SCOPE Implementation Roadmap

## Status Summary

✅ **Completed Documents**:
- [x] architecture.md — Updated with SCOPE compiler architecture & five-phase execution model
- [x] publications/studio/compiler.md — Complete compiler specification (lexer, parser, type-checker, code-generator)
- [x] publications/studio/examples.md — 6 progressive tutorial examples with real BBBC data

✅ **Existing IDE**:
- [x] Browser sandbox at `/tools/analysis-studio` — Three-column IDE (file browser, code editor, output)
- [x] VSCode-style UI with tabs, syntax highlighting (basic)
- [x] "Compile & Execute" button wired to compilation + execution

---

## Phase 1: SCOPE Compiler Implementation

### Tasks (In Priority Order)

#### 1.1 Lexer (`src/lib/scope-compiler/lexer.ts`)
- [ ] Implement tokenization (keywords, symbols, numbers, identifiers)
- [ ] Handle comments (// and /* */)
- [ ] Return Token[] with line/col for error reporting
- [ ] Test: `lex("scope hello { ... }")` returns correct token stream

#### 1.2 Parser (`src/lib/scope-compiler/parser.ts`)
- [ ] Implement recursive descent parser
- [ ] Build AST nodes for Program, Channels, CoordinateSpace, Morphisms, Dispatch
- [ ] Parse morphism chains with |> operator
- [ ] Test: `parse(tokens)` produces valid AST for all 6 examples

#### 1.3 Type Checker (`src/lib/scope-compiler/type-checker.ts`)
- [ ] Verify partition depth consistency (all observe() use declared depth n)
- [ ] Verify S-entropy balance (catalyze vs access counts)
- [ ] Verify dispatch completeness (all cells have dispatch rules)
- [ ] Verify chain existence (all action refs exist)
- [ ] Return errors + warnings

#### 1.4 Code Generator (`src/lib/scope-compiler/code-generator.ts`)
- [ ] Convert AST to ExecutionPlan JSON
- [ ] Serialize all steps (observe, catalyze, fuse, measure, access)
- [ ] Include coordinate_space and channels metadata
- [ ] Test: generated plan matches expected schema

#### 1.5 Main Entry (`src/lib/scope-compiler/index.ts`)
- [ ] Export `compileScope(sourceCode: string) → CompileResult`
- [ ] Orchestrate all four stages
- [ ] Return `{ success, ir, errors, warnings }`

---

## Phase 2: SCOPE Executor Implementation

### Tasks (In Priority Order)

#### 2.1 Five-Phase Orchestration (`src/lib/scope-runtime/five-phase-executor.ts`)
- [ ] Phase 1 COMPILE: Accept timing events, accumulate into trajectory
- [ ] Phase 2 ASSIGN: Classify trajectory, select morphism by cell bounds
- [ ] Phase 3 MEASURE: Run spectral pipeline, return coordinate field Φ
- [ ] Phase 4 EXECUTE: Run morphism chain with Φ available
  - [ ] observe(frame, n) → create partition state
  - [ ] catalyze(constraint) → reduce categorical distance
  - [ ] fuse(chain, rho) → blend observations
  - [ ] measure_distance(A, B) → ||Φ(u_A) - Φ(u_B)|| in µm
  - [ ] access(structure) → narrow partition space
- [ ] Phase 5 EMIT: Return ObservationResult with world-space coordinates

#### 2.2 Coordinate Field Extraction (`src/lib/scope-runtime/spectral-pipeline.ts`)
- [ ] Implement Phase 3 MEASURE: spectral decomposition
- [ ] Load real image from `/datasets/{dataset}/` or use synthetic
- [ ] Run FFT → dyadic scales → coherence enforcement
- [ ] Output: CoordinateField { φ: (u, v) → Vector3, α: (u, v) → number }

#### 2.3 Morphism Chain Executor (`src/lib/scope-runtime/morphism-executor.ts`)
- [ ] Load ExecutionPlan for a morphism
- [ ] Execute steps in sequence with phase 4 GPU shaders
- [ ] Thread coordinate field Φ through steps
- [ ] Compute S-entropy updates at each step
- [ ] Return final partition state + measurements

#### 2.4 Integration with IDE (`src/app/tools/analysis-studio/ScopeIDEMain.tsx`)
- [ ] On "Compile & Execute":
  1. Call `compileScope(code)` → get ExecutionPlan
  2. Log compilation success/errors
  3. Call `executeSCOPE(plan, imageData, dataSource)` → get ObservationResult
  4. Log execution phases + timing
  5. Display result: position, distance (if applicable), S-entropy

---

## Phase 3: Real Data Integration

### Tasks

#### 3.1 Dataset Discovery (`src/lib/dataset-fetcher.ts`)
- [ ] Scan `/public/datasets/` for manifest.json files (already created)
- [ ] List available datasets dynamically
- [ ] Update IDE file browser to show discovered datasets

#### 3.2 Image Loading (`src/lib/scope-runtime/image-loader.ts`)
- [ ] Load .tif/.tiff files from `/datasets/{dataset}/{image}`
- [ ] Parse metadata (dimensions, channels, bit depth)
- [ ] Convert to ImageData for Phase 3 spectral pipeline

#### 3.3 Spectral Metric Reconstruction (`src/shaders/spectral-pipeline.glsl`)
- [ ] FFT decomposition (or FFT.js library)
- [ ] Dyadic scale reconstruction
- [ ] Coherence enforcement (bilateral filtering)
- [ ] Output coordinate field Φ as texture

---

## Phase 4: Validation & Testing

### Unit Tests

- [ ] Compiler: lexer → tokens
- [ ] Compiler: parser → AST for all 6 examples
- [ ] Compiler: type-checker → errors & warnings
- [ ] Compiler: code-generator → ExecutionPlan schema
- [ ] Executor: Phase 1 COMPILE on synthetic timing events
- [ ] Executor: Phase 3 MEASURE on real BBBC007 image
- [ ] Executor: Phase 4 EXECUTE morphism chain with Φ
- [ ] Executor: S-entropy conservation verification (sum ≈ 1.0 ± 10⁻¹⁵)

### Integration Tests

- [ ] Tutorial Example 1: Synthetic hello_world → compiles + executes
- [ ] Tutorial Example 2: BBBC007 observation → loads real image + returns position
- [ ] Tutorial Example 3: Nuclear separation → measure_distance returns µm with uncertainty
- [ ] Tutorial Example 4: Constrained measurement → catalyze reduces uncertainty
- [ ] Tutorial Example 5: Fused measurement → fuse blends observations
- [ ] Tutorial Example 6: Hierarchy analysis → access narrows partition space

### End-to-End Validation

- [ ] Run all 6 examples in IDE, capture screenshots
- [ ] Verify S-entropy sums match expected values (≈1.0)
- [ ] Verify distances match manual measurement (±15% tolerance)
- [ ] Verify world-space coordinates are in micrometers

---

## File Checklist

```
src/
├── lib/
│   ├── scope-compiler/
│   │   ├── index.ts                 ← [ ] Main entry: compileScope()
│   │   ├── lexer.ts                 ← [ ] Tokenizer
│   │   ├── parser.ts                ← [ ] AST builder
│   │   ├── type-checker.ts          ← [ ] Invariant verification
│   │   └── code-generator.ts        ← [ ] ExecutionPlan emitter
│   │
│   ├── scope-runtime/
│   │   ├── five-phase-executor.ts   ← [ ] Orchestrates all 5 phases
│   │   ├── spectral-pipeline.ts     ← [ ] Phase 3: Coordinate field
│   │   ├── morphism-executor.ts     ← [ ] Phase 4: Chain execution
│   │   ├── image-loader.ts          ← [ ] Load real microscopy images
│   │   └── api-clients.ts           ← [ ] (Existing; may extend)
│   │
│   ├── dataset-fetcher.ts           ← [ ] Discover & list datasets
│   ├── scope-examples.ts            ← [ ] (Existing; may extend)
│   └── scope-client.ts              ← [ ] (Existing client utilities)
│
├── shaders/
│   └── spectral-pipeline.glsl       ← [ ] Phase 3 shader
│
└── app/
    └── tools/
        └── analysis-studio/
            └── ScopeIDEMain.tsx     ← [ ] Wire Compile & Execute button

publications/
└── studio/
    ├── architecture.md              ← [x] Updated
    ├── compiler.md                  ← [x] Created
    └── examples.md                  ← [x] Created
```

---

## Expected Outcomes

### After Phase 1 (Compiler)
- `compileScope(code)` successfully parses all 6 examples
- Type checking catches depth/entropy/dispatch errors
- ExecutionPlan JSON is serializable and complete

### After Phase 2 (Executor)
- Five phases execute in order
- Phase 3 generates real coordinate field Φ from BBBC007 image
- Phase 4 executes measure_distance() with Φ to return µm distances
- S-entropy conservation verified (sum = 1.0 ± 10⁻¹⁵)

### After Phase 3 (Real Data)
- IDE shows discovered datasets in file browser
- Images load from `/public/datasets/`
- Real measurements on BBBC data (not synthetic)

### After Phase 4 (Validation)
- All 6 examples produce expected outputs
- Distances match manual measurements (±15%)
- No synthetic data in production (only tutorials for learning)

---

## Next Immediate Steps

1. **Start with Phase 1.1** — Implement lexer
   - Simple regex-based tokenizer
   - Return Token[] with line/col
   - Test on Example 1 code

2. **Then Phase 1.2** — Implement parser
   - Recursive descent for each BNF rule
   - Build AST nodes
   - Test on all 6 examples

3. **Then Phases 1.3-1.5** — Complete compiler pipeline
   - Type checking catches errors
   - Code generator produces ExecutionPlan

4. **Then Phase 2** — Wire executor to use compiled plans
   - Run phases 1-2 on CPU
   - Run phase 3 via GPU shader
   - Run phase 4 via morphism executor

5. **Then Phases 3-4** — Real data + validation
