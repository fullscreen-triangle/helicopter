# SCOPE Playground — Compiler Specification

The TypeScript playground compiler is a single-pass recursive-descent parser that produces an AST, runs five static checks, and then hands the AST to the runtime interpreter. There is no code generation step; the AST is interpreted directly. The compiler also produces chart data and 3D scene data as side-outputs alongside the measurement result.

---

## Source → AST → Type-check → Interpret → Visualise

```
Source text (string)
        │
        ▼
  Lexer (tokenise)
        │  Token[]
        ▼
  Parser (recursive descent)
        │  ScopeProgram AST node
        ▼
  Type Checker (five invariants)
        │  TypeCheckResult { ok: boolean, errors: TypeError[], warnings: TypeWarning[] }
        ▼
  Interpreter (five-phase runtime)
        │  Result record + ChartData + VisualData
        ▼
  Visualisation layer (Canvas 2D + D3 + Three.js)
```

---

## Tokens

```
keyword   ::= 'scope' | 'channels' | 'sync' | 'cell' | 'at' | 'freq'
            | 'bounds' | 'action' | 'coordinate_space' | 'field'
            | 'depth' | 'lambda_s' | 'lambda_t'
            | 'dispatch' | 'when' | 'do' | 'execute' | 'emit'
            | 'observe' | 'catalyze' | 'fuse' | 'measure_distance'
            | 'access' | 'rho' | 'confidence' | 'threshold'
            | 'goal' | 'rule' | 'invariant' | 'epsilon' | 'within'
            | 'load' | 'db' | 'dataset' | 'image'
            | 'visualise' | 'chart' | 'plot'

ident     ::= [a-zA-Z_][a-zA-Z0-9_]*

nat       ::= [0-9]+

real      ::= [0-9]+ ('.' [0-9]+)? ([eE] [+-]? [0-9]+)?

string    ::= '"' [^"]* '"'

unit      ::= 'µm' | 'nm' | 'px' | 'freq' | 's' | 'µm/pixel' | 'bits' | '%'

punct     ::= '{' | '}' | '(' | ')' | ',' | '=' | '|>' | 'x' | '<' | '>' | '<='

comment   ::= '//' [^\n]* '\n'
```

---

## Grammar (BNF)

Extensions beyond the original paper grammar are marked `[ext]`.

```bnf
program ::=
  'scope' ident '{'
    channels_decl?
    coordinate_space_decl?
    goal_decl?              [ext]
    rule_decl*              [ext]
    morphism_decl*
    dispatch_decl?
  '}'

channels_decl ::=
  'channels' '{' channel_item* '}'

channel_item ::=
  sync_item | cell_item

sync_item ::=
  'sync' ident 'at' real unit

cell_item ::=
  'cell' ident 'bounds' '(' real ',' real ')' 'action' ident

coordinate_space_decl ::=
  'coordinate_space' '{'
    'field' real 'x' real unit
    'depth' nat
    'lambda_s' real
    'lambda_t' real
  '}'

-- [ext] Goal block: declares success criteria evaluated at EMIT
goal_decl ::=
  'goal' '{'
    goal_criterion*
  '}'

goal_criterion ::=
  ident comparison_op real unit    -- e.g. distance_uncertainty < 0.5 µm
  | ident comparison_op real       -- e.g. s_entropy_conservation < 1e-12

comparison_op ::= '<' | '<=' | '>' | '>=' | '=='

-- [ext] Named constraint rule declarations
rule_decl ::=
  'rule' ident '(' ident ')' '{'
    'invariant' ':' string
    'epsilon'   ':' real
  '}'

morphism_decl ::=
  ident '=' morphism_expr

morphism_expr ::=
  observe_expr ('|>' morphism_step)*

observe_expr ::=
  'observe' '(' frame_ref ',' 'n' '=' nat ')'

frame_ref ::=
  ident
  | 'load' '(' db_ref ')'

db_ref ::=
  'db' '=' string ',' 'dataset' '=' string ',' 'image' '=' string

morphism_step ::=
  catalyze_step
  | fuse_step
  | measure_distance_step
  | access_step
  | visualise_step          [ext]

catalyze_step ::=
  'catalyze' '(' constraint (',' 'confidence' '=' real)? ')'   [confidence ext]

constraint ::=
  ident '(' ident ')'
  | ident '(' string ')'
  | ident             -- name of a declared rule

fuse_step ::=
  'fuse' '(' ident ',' 'rho' '=' real ')'

measure_distance_step ::=
  'measure_distance' '(' target ',' target ')'

target ::= ident

access_step ::=
  'access' '(' ident (',' 'threshold' '=' real)? ')'   [threshold ext]

-- [ext] Inline visualisation hint — tells the visualiser what to show after this step
visualise_step ::=
  'visualise' '(' vis_mode ')'

vis_mode ::=
  'scale_field' | 'segmentation' | 'distance_map' | 'geodesic'
  | 'point_cloud' | 'entropy_sphere' | 'partition_tree'
  | 'spectral_power' | 'entropy_trajectory' | 'uncertainty_bar'

dispatch_decl ::=
  'dispatch' '{' when_stmt* '}'

when_stmt ::=
  'when' ident 'do' action

action ::=
  'execute' '(' ident ')'
  | 'emit' ident
  | '{' action* '}'
```

---

## AST node types (TypeScript interfaces)

```typescript
interface ScopeProgram {
  kind: 'ScopeProgram';
  name: string;
  channels?: ChannelsDecl;
  coordinateSpace?: CoordinateSpaceDecl;
  goal?: GoalDecl;              // [ext]
  rules: RuleDecl[];            // [ext]
  morphisms: MorphismDecl[];
  dispatch?: DispatchDecl;
}

// ── channels ──────────────────────────────────────────────────────

interface ChannelsDecl { kind: 'ChannelsDecl'; items: ChannelItem[]; }
type ChannelItem = SyncItem | CellItem;

interface SyncItem {
  kind: 'SyncItem'; name: string; value: number; unit: string;
}
interface CellItem {
  kind: 'CellItem'; name: string;
  boundsLow: number; boundsHigh: number; action: string;
}

// ── coordinate_space ──────────────────────────────────────────────

interface CoordinateSpaceDecl {
  kind: 'CoordinateSpaceDecl';
  fieldX: number; fieldY: number; unit: string;
  depth: number; lambdaS: number; lambdaT: number;
}

// ── goal [ext] ───────────────────────────────────────────────────

interface GoalDecl {
  kind: 'GoalDecl';
  criteria: GoalCriterion[];
}
interface GoalCriterion {
  kind: 'GoalCriterion';
  metric: string;          // e.g. "distance_uncertainty", "s_entropy_conservation"
  op: '<' | '<=' | '>' | '>=' | '==';
  threshold: number;
  unit: string;            // may be empty
}

// ── rule [ext] ───────────────────────────────────────────────────

interface RuleDecl {
  kind: 'RuleDecl';
  name: string;            // e.g. "conservation"
  argument: string;        // e.g. "dna_mass"
  invariant: string;       // human-readable description
  epsilon: number;         // explicit ε (overrides default lookup table)
}

// ── morphisms ────────────────────────────────────────────────────

interface MorphismDecl {
  kind: 'MorphismDecl'; name: string; expr: MorphismExpr;
}
interface MorphismExpr {
  kind: 'MorphismExpr'; observe: ObserveExpr; steps: MorphismStep[];
}
interface ObserveExpr {
  kind: 'ObserveExpr'; frame: FrameRef; depth: number;
}
type FrameRef =
  | { kind: 'ChannelRef'; name: string }
  | { kind: 'LoadRef'; db: string; dataset: string; image: string };

type MorphismStep =
  | CatalyzeStep | FuseStep | MeasureDistanceStep | AccessStep | VisualiseStep;

interface CatalyzeStep {
  kind: 'CatalyzeStep';
  constraint: string;      // e.g. "conservation(dna_mass)"
  epsilon: number;         // resolved from rule_decl or default table
  confidence: number;      // [ext] 0–1, default 1.0 (no confidence discount)
}
interface FuseStep {
  kind: 'FuseStep'; morphismRef: string; rho: number;
}
interface MeasureDistanceStep {
  kind: 'MeasureDistanceStep'; target1: string; target2: string;
}
interface AccessStep {
  kind: 'AccessStep'; target: string;
  threshold: number;       // [ext] segmentation threshold 0–1, default 0.5
}
interface VisualiseStep {
  kind: 'VisualiseStep'; mode: string;  // [ext]
}

// ── dispatch ─────────────────────────────────────────────────────

interface DispatchDecl { kind: 'DispatchDecl'; rules: WhenStmt[]; }
interface WhenStmt { kind: 'WhenStmt'; cell: string; action: Action; }
type Action =
  | { kind: 'ExecuteAction'; morphismRef: string }
  | { kind: 'EmitAction'; name: string }
  | { kind: 'BlockAction'; actions: Action[] };
```

---

## Type checker — five invariants

All five are checked before any execution begins. Failure halts with a typed error; partial execution never happens.

### Invariant 1: Depth Compatibility

Every `observe(frame, n=N)` must match `coordinate_space.depth`. If no `coordinate_space` block, the depth from the first `observe` is propagated to all.

```
∀ morphism m: m.expr.observe.depth == program.coordinateSpace.depth
```

Error: `DepthMismatch { morphism, got, expected }`

### Invariant 2: Cell Partition Consistency

Cell bounds must be disjoint. Gaps are allowed (fall through to default handler).

```
∀ cells ci, cj (i ≠ j):
  ci.boundsHigh <= cj.boundsLow  OR  cj.boundsHigh <= ci.boundsLow
```

Error: `CellOverlap { cell1, cell2, overlap: [lo, hi] }`

### Invariant 3: Entropy Budget

Total confidence-weighted epsilon across all `catalyze` steps in a morphism must not exceed the entropy budget:

```
Σ (εᵢ · wᵢ)  <  1 - S_t_initial - S_e_minimum
```

where `S_t_initial = 0.5`, `S_e_minimum = 0.1`, and `wᵢ = 1 - confidence_i` (full-confidence catalyst costs its full ε; zero-confidence catalyst costs nothing).

Default ε (overridden by `rule` declarations):

| Constraint family | Default ε |
|---|---|
| `conservation(*)` | 0.008 |
| `phase_lock(*)` | 0.010 |
| `thermal(*)` | 0.005 |
| `symmetry(*)` | 0.006 |
| any declared `rule` | from `rule.epsilon` |

Error: `EntropyBudgetExceeded { morphism, weightedEpsilon, budget }`

### Invariant 4: Coordinate Grounding

Every `measure_distance(t1, t2)` must be preceded in the same morphism by `access(t1)` and `access(t2)` (unless the target is a declared channel name).

```
∀ measure_distance(t1, t2) in m:
  (∃ access(t1) before it  OR  t1 ∈ channelNames)
  AND
  (∃ access(t2) before it  OR  t2 ∈ channelNames)
```

Error: `UngroundedDistance { morphism, target }`

### Invariant 5: Goal Reachability [ext]

If a `goal {}` block is present, check statically that the declared metric thresholds are achievable given `coordinate_space` parameters, using the analytic uncertainty bound from Theorem 2:

```
δd_min = α_nominal · (fieldSize / 2^depth) · (1 + Σ ε_min)
```

where `α_nominal = 1.0 µm/px` and `Σ ε_min` is the minimum total epsilon (all catalysts at full confidence). If `δd_min > goal.distance_uncertainty`, the goal cannot be met at this depth — emit a warning (not a hard error, since α varies spatially and the actual δd may be lower):

Warning: `GoalUnreachableAtDepth { metric, requiredDepth, currentDepth, predictedMin, goalThreshold }`

This warning is shown in the console with a suggested minimum depth:

```
[GOAL WARNING]  distance_uncertainty < 0.10 µm unreachable at depth=6
                predicted δd_min = 0.156 µm  (need depth ≥ 10 for 100×100 µm field)
                Suggestion: increase coordinate_space.depth to 10
```

---

## Interpreter (runtime)

The interpreter is a set of pure functions threaded through a `RuntimeState`. Each phase appends to `chartData` and `visualData` for the visualisation layer.

```typescript
interface RuntimeState {
  // image
  image: Float32Array;
  width: number;
  height: number;
  // phase outputs
  timingEvents: TimingDeviation[];
  cellLabel: PartitionState | null;
  coordField: CoordField | null;
  partitionState: PartitionState;
  accessedTargets: Record<string, { x: number; y: number; membershipMap: Float32Array }>;
  // entropy
  sk: number; st: number; se: number;
  // result accumulation
  distance: number | null;
  uncertainty: number | null;
  geodesicPath: Array<[number, number]>;
  distanceMap: Float32Array | null;
  // visualisation side-outputs
  chartData: ChartData;
  visualData: VisualData;
  // console
  log: LogLine[];
}

interface ChartData {
  spectralPower: Array<{ freq: number; energy: number }>;   // for SpectralPowerChart
  powerLawExponent: number;
  scaleHistogram: Array<{ alpha: number; count: number }>; // for ScaleHistogram
  entropyTrajectory: Array<{ phase: string; sk: number; st: number; se: number }>;
  uncertaintyBar: { d: number; deltaD: number; goals: GoalThreshold[] };
  channelCapacity: { snr: number; capacity: number };
}

interface VisualData {
  scaleField: Float32Array;           // α(x,y) for Canvas2D heatmap + 3D surface
  segmentationMask: Uint8Array;       // binary mask for Canvas2D overlay
  segmentationContour: Array<[number, number]>; // contour pixels
  distanceMap: Float32Array;          // T(x,y) for Canvas2D distance view
  geodesicPath: Array<[number, number]>; // path pixels for Canvas2D + 3D tube
  pointCloud: Float32Array;           // [x,y,z, r,g,b] × N for 3D point cloud
  partitionStates: PartitionStateNode[]; // tree for 3D partition tree
}

// Phase functions
function compilePhase(state: RuntimeState, program: ScopeProgram): RuntimeState
function measurePhase(state: RuntimeState): RuntimeState
function executePhase(state: RuntimeState, chain: MorphismDecl): RuntimeState
function emitPhase(state: RuntimeState, program: ScopeProgram): Result
```

### Phase 1 — COMPILE

1. Derive synthetic timing events from pixel intensity histogram: `ΔP = (bin/255 − 0.5) × 4e-6 s`.
2. Classify the dominant ΔP into the declared timing cells (if any).
3. Update entropy: `ΔS_t = −(H_before − H_after)`, `ΔS_k = +|ΔS_t|`.
4. Append `entropyTrajectory[0]` to `chartData`.

### Phase 2 — MEASURE

Runs the three-stage MIC spectral pipeline:

1. **FFT**: windowed 2D FFT → magnitude spectrum. Build `chartData.spectralPower` (radial average of `|û(k)|²` vs `k`). Fit power-law slope via log-log linear regression → `powerLawExponent`.
2. **Dyadic decomposition**: `α(x,y) = 2ω₀ / (−∂_ω log|û_local|)` at each window. Build `chartData.scaleHistogram`.
3. **Coherence enforcement**: bilateral filter `σ_s=5, σ_r=0.3`. Output `visualData.scaleField`.

Build `visualData.pointCloud` from `(x, y, intensity)` with colour from α(x,y).

Append `entropyTrajectory[1]` (no change — deterministic bijection).

### Phase 3 — EXECUTE

Execute morphism steps left-to-right:

- **`observe(frame, n)`**: `Σ = (n,0,0,+½)`. No entropy change.
- **`catalyze(constraint, confidence=w)`**: effective ε = `ε_base × (1 − w × 0.5)`. Higher confidence means the constraint is well-validated, reducing effective cost. `ΔS_k += ε_eff`. Partition bounding box shrinks by `ε_eff × width` on each side (visual feedback).
- **`access(target, threshold=t)`**: Otsu threshold if `t=0.5` (default), otherwise hard threshold at `t`. Level-set refinement for 20 iterations. Store centroid + membership map (per-pixel confidence that this pixel belongs to `target`). `ΔS_k += 0.05`. Append to `visualData.segmentationMask`.
- **`measure_distance(t1, t2)`**: Fast marching from `accessedTargets[t1].centroid` over `α(x,y)` cost function → `T(x,y)`. Extract distance `d = T[t2.centroid]`. Backtrack gradient descent on `T` to get geodesic path pixels. Uncertainty: `δd = ᾱ · (fieldSize / 2^n) · (1 + Σ εᵢ_eff)`. Store `distanceMap`, `geodesicPath`. Build `chartData.uncertaintyBar`.
- **`fuse(ref, rho)`**: `d_final = rho × d_current + (1−rho) × d_ref`. `δd_final = sqrt((rho·δd₁)² + ((1−rho)·δd₂)²)`.
- **`visualise(mode)`**: sets the active visualisation tab to `mode` after this step completes.

Append `entropyTrajectory[2,3]`.

### Phase 4 — EMIT

1. Assemble `Result`.
2. Evaluate all `goal {}` criteria against the result: `distance_uncertainty` → compare `δd` against threshold; `s_entropy_conservation` → check `|S_k+S_t+S_e−1|`.
3. Verify `S_k + S_t + S_e = 1 ± 1e-12`. Log warning if violated.
4. Append `entropyTrajectory[4]`.
5. Compute `chartData.channelCapacity = { snr, capacity: 0.5 * log2(1+snr) }`.

---

## Result record

```typescript
interface Result {
  structure: string;
  position: [number, number, number];  // world-space centroid in µm
  distance: number;                    // µm
  uncertainty: number;                 // µm
  relativeUncertainty: number;         // fraction
  sEntropy: { sk: number; st: number; se: number; sum: number };
  goalStatus: GoalStatus[];
  chartData: ChartData;
  visualData: VisualData;
  log: LogLine[];
}

interface GoalStatus {
  metric: string;
  op: string;
  threshold: number;
  unit: string;
  actual: number;
  passed: boolean;
}
```

---

## Error and warning types

```typescript
type CompileError =
  | { kind: 'ParseError'; message: string; line: number; col: number }
  | { kind: 'DepthMismatch'; morphism: string; got: number; expected: number }
  | { kind: 'CellOverlap'; cell1: string; cell2: string; overlap: [number, number] }
  | { kind: 'EntropyBudgetExceeded'; morphism: string; weightedEpsilon: number; budget: number }
  | { kind: 'UngroundedDistance'; morphism: string; target: string }
  | { kind: 'UnknownMorphismRef'; name: string }
  | { kind: 'UnknownCellRef'; name: string }
  | { kind: 'UnknownRuleRef'; name: string };

type CompileWarning =
  | { kind: 'GoalUnreachableAtDepth'; metric: string; requiredDepth: number;
      currentDepth: number; predictedMin: number; goalThreshold: number }
  | { kind: 'VacuousConstraint'; morphism: string; constraint: string; reason: string }
  | { kind: 'ConfidenceDiscountLarge'; morphism: string; constraint: string;
      confidence: number; effectiveEpsilon: number };

type RuntimeError =
  | { kind: 'ImageFetchFailed'; url: string; status: number }
  | { kind: 'TargetNotFound'; target: string }
  | { kind: 'EntropyConservationViolation'; sum: number }
  | { kind: 'FastMarchingDiverged'; source: [number, number] };
```

---

## Console output format

```
[COMPILE]  cell=PROPHASE  ΔP_mean=-1.40e-6s  n=10  S_t: 0.500→0.200  S_k: 0.300→0.600
[ASSIGN]   morphism=nucleus_pair_measurement
[MEASURE]  ᾱ=0.988 µm/px  σ_α=0.152  power_law=-0.410  C=1.83 bits/px  bilateral ✓
[EXECUTE]  observe n=10  Σ=(10,0,0,+½)
[EXECUTE]  catalyze(conservation(dna_mass))  ε=0.008  conf=0.9  ε_eff=0.0044  S_k→0.604
[EXECUTE]  catalyze(phase_lock(chromatin))   ε=0.010  conf=1.0  ε_eff=0.010   S_k→0.614
[EXECUTE]  access(nucleus_a, threshold=0.5)  centroid=(312,487)  mask_area=1820px
[EXECUTE]  access(nucleus_b, threshold=0.5)  centroid=(589,502)  mask_area=1754px
[EXECUTE]  measure_distance(nucleus_a, nucleus_b)  fast-marching 1024×1024
[EXECUTE]  d=14.312 µm  δd=0.157 µm  (1.10%)  path_length=283px
[EMIT]     S_k=0.412  S_t=0.281  S_e=0.307  sum=1.000000000000000 ✓
[GOAL]     distance_uncertainty=0.157 µm  < 0.5 µm ✓
[GOAL]     distance_uncertainty=0.157 µm  < 0.1 µm ✗  (need depth ≥ 13)
[GOAL]     s_entropy_conservation=1.1e-16  < 1e-12 ✓
```
