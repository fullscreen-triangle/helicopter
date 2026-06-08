# SCOPE Implementation Roadmap

## Status Summary

🎯 **WORKING END-TO-END**: Example 1 (hello_world) compiles and executes in IDE

✅ **Completed Documents**:
- [x] architecture.md — Updated with SCOPE compiler architecture & five-phase execution model
- [x] publications/studio/compiler.md — Complete compiler specification (lexer, parser, type-checker, code-generator)
- [x] publications/studio/examples.md — 6 progressive tutorial examples with real BBBC data

✅ **Existing IDE**:
- [x] Browser sandbox at `/tools/analysis-studio` — Three-column IDE (file browser, code editor, output)
- [x] VSCode-style UI with tabs, syntax highlighting (basic)
- [x] "Compile & Execute" button wired to compilation + execution

---

## ✅ What's Working NOW

**Examples 1-3: Core pipeline with world-space measurements**

Example 1: hello_world
```
Compiles → Phases 1-5 → Position (50.0, 50.0, 0.0) µm
S-Entropy: (0.330, 0.330, 0.340) sum=1.000 ✓
```

Example 3: nuclear_separation (NEW)
```
Compiles → Phase 3 generates Φ → Phase 4 measure_distance() → 
Distance: 50.0 µm ± 0.8 µm
S-Entropy conserved ✓
```

**Full Chain**:
```
SCOPE source 
  → Compiler (lexer+parser+type-checker+codegen) 
  → ExecutionPlan JSON 
  → Minimal Executor:
      Phase 3: generateCoordinateField(Φ)
      Phase 4: steps (observe, measure_distance, catalyze, access)
  → ObservationResult (position, distance, S-entropy)
  → IDE displays result ✓
```

---

## Phase 1: SCOPE Compiler Implementation

### Tasks (In Priority Order)

#### 1.1 Lexer (`src/lib/scope-compiler/lexer.ts`)
- [x] Implement tokenization (keywords, symbols, numbers, identifiers)
- [x] Handle comments (// and /* */)
- [x] Return Token[] with line/col for error reporting
- [x] Test: `lex("scope hello { ... }")` returns correct token stream

#### 1.2 Parser (`src/lib/scope-compiler/parser.ts`)
- [x] Implement recursive descent parser
- [x] Build AST nodes for Program, Channels, CoordinateSpace, Morphisms, Dispatch
- [x] Parse morphism chains with |> operator
- [x] Test: `parse(tokens)` produces valid AST for all 6 examples

#### 1.3 Type Checker (`src/lib/scope-compiler/type-checker.ts`)
- [x] Verify partition depth consistency (all observe() use declared depth n)
- [x] Verify S-entropy balance (catalyze vs access counts)
- [x] Verify dispatch completeness (all cells have dispatch rules)
- [x] Verify chain existence (all action refs exist)
- [x] Return errors + warnings

#### 1.4 Code Generator (`src/lib/scope-compiler/code-generator.ts`)
- [x] Convert AST to ExecutionPlan JSON
- [x] Serialize all steps (observe, catalyze, fuse, measure, access)
- [x] Include coordinate_space and channels metadata
- [x] Test: generated plan matches expected schema

#### 1.5 Main Entry (`src/lib/scope-compiler/index.ts`)
- [x] Export `compileScope(sourceCode: string) → CompileResult`
- [x] Orchestrate all four stages
- [x] Return `{ success, ir, errors, warnings }`

---

## Phase 2: SCOPE Executor Implementation

### 2.0 Minimal Executor (DONE ✓)
- [x] `src/lib/scope-runtime/minimal-executor.ts` — Synthetic version for Examples 1-3
  - [x] Phases 1-5 with coordinate field generation
  - [x] Returns valid ObservationResult with S-entropy conservation
  - [x] Wired to IDE (run button works)
  - [x] Example 1: Position (50, 50, 0) µm, S-entropy (0.33, 0.33, 0.34)
  - [x] Example 3: measure_distance returns distance in µm ± uncertainty

### 2.1 Spectral Pipeline (DONE ✓ - Minimal Version)
- [x] `src/lib/scope-runtime/spectral-pipeline.ts` — Coordinate field generation
  - [x] generateSyntheticCoordinateField() — Map pixels to world-space (µm)
  - [x] generateCoordinateFieldFromImage() — Placeholder for real images
  - [x] measureDistance() — World-space distance with uncertainty
  - [x] CoordinateField type: φ and α functions

### Next: Expand Minimal Executor → Real Data Support

#### 2.2 Real Image Loading

#### 2.2 Morphism Chain Executor (Phase 4)
- [ ] Load ExecutionPlan morphism
- [ ] Execute steps: observe → catalyze → fuse → measure → access
- [ ] Update S-entropy at each step
- [ ] Use Φ for world-space measurements

#### 2.3 Real Image Support (Phase 3)
- [ ] Load `.tif` from `/datasets/{dataset}/`
- [ ] Extract pixel data
- [ ] Pass to spectral pipeline

#### 2.4 Multi-Example Support
- [ ] Example 2: Real BBBC007 data
- [ ] Example 3: measure_distance() returns µm
- [ ] Example 4: catalyze() reduces uncertainty
- [ ] Examples 5-6: fuse and access

---

## 🎯 Checkpoint: Working Pipeline (2026-06-07)

**PHASE 1-3 COMPLETE & TESTED:**
- [x] Compiler: Full 4-stage pipeline (lex→parse→typecheck→codegen)
- [x] Executor: Phases 1-5 with coordinate field Φ
- [x] IDE: Compile & Execute button wired
- [x] Examples 1-3: Compile and run successfully

**Test Examples:**
```
Example 1 (hello_world):
  Input: scope block
  Output: Position (50.0, 50.0, 0.0) µm, S-entropy (0.330, 0.330, 0.340)
  
Example 3 (nuclear_separation):
  Input: measure_distance(nucleus_a, nucleus_b)
  Output: Distance 50.0 µm ± 0.8 µm, S-entropy conserved
```

**Current Limitations (synthetic data):**
- Uses generated Φ, not real spectral analysis
- Pixel positions hardcoded
- No real image loading yet

**Ready for Examples 4-6:**
- Real image loading → Example 2 (BBBC007)
- Catalyze reducing uncertainty → Example 4
- Fuse and access steps → Examples 5-6

---

## Phase 4: Real Data Integration

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
