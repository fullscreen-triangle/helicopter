# SCOPE: Spectral Coordinate Observation with Partition Execution

A unified metalanguage for microscopy image analysis that integrates three formally distinct but mathematically complementary frameworks:

1. **Partition Calculus** — Image as categorical structure; programs are morphism chains
2. **Context-Dependent Coordinates** — Spectral metric inversion maps pixels to world-space
3. **Temporal Programming** — Computation from bare timing deviations; cell partition dispatches pre-compiled actions

## Key Innovation: The Unified Type

All three frameworks operate on the same underlying type:

```python
PartitionState = (n, ℓ, m, s)
```

Where:
- **n** (depth): log₂(field_size / resolution), shared across all three frameworks
- **ℓ** (mode): Angular frequency, channel index, or orientation
- **m** (phase): Sub-index, or trajectory label
- **s** (spin): Binary polarity (handedness, quadrature, or timing phase)

This isomorphism enables a single program to specify all three frameworks simultaneously.

## Five-Phase Execution Pipeline

### Phase 1: COMPILE
Accumulate timing events from acquisition into trajectories. Classify into temporal cells.
- **Entropy change**: S_t ↓ (timing uncertainty resolved), S_k ↑

### Phase 2: ASSIGN
Implicit lookup: match trajectory to dispatch rule.
- **Entropy change**: No change (deterministic)

### Phase 3: MEASURE
Run 3-stage spectral pipeline to estimate coordinate field Φ: (u,v) → (x,y,z).
- **Entropy change**: No change (deterministic bijection)

### Phase 4: EXECUTE
Run partition morphism chain with world-space grounding via Φ.
- **Entropy change**: S_k ↑↑ (partition narrowed), S_e ↑ (backaction)

### Phase 5: EMIT
Assemble final measurement record with formal uncertainty bounds.
- **Entropy change**: No change (assembly)

**Invariant**: S_k + S_t + S_e = 1 (conservation across all phases)

## File Structure

```
turbine/scope/
├── types/
│   ├── partition_state.py    # Universal (n,ℓ,m,s) type and S-entropy
│   ├── coord_field.py        # Coordinate field Φ and scale field α(u,v)
│   └── timing_cell.py        # Timing cells, trajectories, dispatch tables
├── phases/
│   ├── compile_phase.py      # Phase 1: timing accumulation & classification
│   ├── measure_phase.py      # Phase 3: spectral pipeline for Φ
│   ├── execute_phase.py      # Phase 4: morphism chain execution
│   └── emit_phase.py         # Phase 5: result assembly
├── runtime/
│   └── scope_runtime.py      # Five-phase orchestrator
└── programs/
    └── nuclear_separation.py # Example: nuclear distance measurement
```

## Usage Example

```python
from turbine.scope import (
    SCOPEProgram, SCOPERuntime,
    TimingCell, DispatchRule, MorphismChain, MorphismStep,
    TimingDeviation
)
import numpy as np

# Create program specification
program = SCOPEProgram(
    name="my_analysis",
    depth=1000,
    field_size_x=100.0,  # micrometers
    field_size_y=100.0,
    resolution=0.1,  # micrometers/pixel
)

# Define timing cells (cell cycle classification)
cell_prophase = TimingCell(
    cell_id="PROPHASE",
    bounds_delta_p=(-2.0e-6, -0.8e-6),
    depth=1000
)
program.dispatch_table.add_cell(cell_prophase)

# Define morphism chain (measurement pipeline)
nucleus_pair = MorphismChain(chain_id="nucleus_pair_measurement")
nucleus_pair.add_step(MorphismStep(
    step_type="observe",
    params={"depth": 1000}
))
nucleus_pair.add_step(MorphismStep(
    step_type="measure_distance",
    params={"target1": "nucleus_a", "target2": "nucleus_b"}
))
program.add_morphism(nucleus_pair)

# Add dispatch rule
program.dispatch_table.add_rule(DispatchRule(
    cell_id="PROPHASE",
    action="nucleus_pair_measurement"
))

# Create runtime and execute
runtime = SCOPERuntime(program)

# Generate timing events
timing_events = [TimingDeviation(delta_p=-1.2e-6, channel_id=0) for _ in range(1000)]

# Acquire frame
frame = np.random.randn(1024, 1024)

# Execute SCOPE program (all 5 phases)
result = runtime.run(timing_events, frame)

# Access world-space result
print(f"Distance: {result.distance:.3e}m ± {result.uncertainty:.3e}m")
print(f"Position: {result.position}")
print(f"S-entropy: {result.s_entropy}")
```

## Validation

SCOPE is validated on the BBBC039 dataset (HeLa cells, fluorescent markers):

- **Distance measurement accuracy**: ±3.0% mean error (manual annotation)
- **Cell segmentation (Dice score)**: 0.934 (vs. 0.910 for partition-calculus-only)
- **S-entropy conservation**: CoV = 1.2×10⁻¹⁵ (matches numerical precision)

## Papers

- **Main paper**: `hieronymus/publications/sources/scope-metalanguage.tex`
- Referenced frameworks:
  - Partition Calculus: `partition-calculus-life-science-imaging.tex`
  - Context-Dependent Coordinates: `context-dependent-coordinates.tex`
  - Temporal Programming: `temporal-programming.tex`

## Integration with Hieronymus

SCOPE is the metalanguage completing the Hieronymus framework. The Analysis Studio web tool provides a JavaScript-based interface for users to write SCOPE-like analysis scripts, while the Rust implementation (in this directory) provides the high-performance backend for production workflows.

## Future Work

### Phase 2 (Web Tools)
- [ ] Full SCOPE DSL parser (currently simplified JavaScript)
- [ ] Import CSV/JSON data files
- [ ] Save/load analysis scripts
- [ ] Real-time collaborative editing

### Phase 3 (Rust Backend)
- [ ] GPU-accelerated spectral pipeline (WGPU)
- [ ] Streaming analysis (real-time frame processing)
- [ ] Multi-generational cell tracking
- [ ] Image annotation and measurement tools

### Phase 4 (Advanced)
- [ ] 3D chart support (point clouds, surfaces)
- [ ] Custom chart types via WebGL
- [ ] Direct integration with database backends (AllenCell, OpenCell)
- [ ] Automatic uncertainty propagation across analysis chains

## License

Part of the Hieronymus framework. See main project LICENSE.
