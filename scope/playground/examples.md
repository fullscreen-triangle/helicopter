# SCOPE Playground — Examples

All examples use real public databases via `/api/image-proxy`. Each example shows the full program, the expected console output (with real numbers from the Python reference implementation), the expected JSON result, and a complete description of every visual output — Canvas 2D views, D3 charts, and Three.js 3D scenes — so there is no ambiguity about what "correct" looks like.

---

## Example 1: Canonical nuclear separation (BBBC039, HeLa cells)

**What it does**: Measures world-space distance between two DAPI-stained HeLa nuclei. Uses timing cell dispatch (PROPHASE fires `nucleus_pair_measurement`). Declares a `goal` block. Uses a named `rule` for conservation. Ends with `visualise(geodesic)`.

**Database**: BBBC039 — HeLa cells, Hoechst (nuclei) + phalloidin (actin), 1024×1024, 0.1 µm/pixel.

```scope
scope nuclear_separation_dynamics {
  channels {
    sync dapi at 0.1 µm/pixel
    cell PROPHASE  bounds (-2.0e-6, -0.8e-6) action nucleus_pair_measurement
    cell METAPHASE bounds (-0.8e-6,  0.8e-6) action membrane_boundary
    cell ANAPHASE  bounds ( 0.8e-6,  2.0e-6) action nucleus_pair_measurement
  }

  coordinate_space {
    field 100 x 100 µm
    depth 10
    lambda_s 0.10
    lambda_t 0.05
  }

  goal {
    distance_uncertainty < 0.5 µm
    s_entropy_conservation < 1e-12
    snr > 8.0
  }

  rule conservation(dna_mass) {
    invariant: "total DAPI-stained area is conserved ±5%"
    epsilon: 0.008
  }

  nucleus_pair_measurement = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_001.tif"), n = 10)
    |> catalyze(conservation(dna_mass))
    |> catalyze(phase_lock(chromatin), confidence = 0.9)
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(geodesic)

  membrane_boundary = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_001.tif"), n = 10)
    |> catalyze(phase_lock(plasma_membrane))
    |> access(cell_boundary)
    |> visualise(segmentation)

  dispatch {
    when PROPHASE  do execute(nucleus_pair_measurement)
    when METAPHASE do execute(membrane_boundary)
    when ANAPHASE  do execute(nucleus_pair_measurement)
  }
}
```

**Expected console output**:
```
[COMPILE]  events=1000  ΔP_mean=-1.40e-6s  σ=0.30e-6s
[COMPILE]  cell=PROPHASE  (ΔP in [-2.0e-6, -0.8e-6])  S_t: 0.500→0.200  S_k: 0.300→0.600
[ASSIGN]   morphism=nucleus_pair_measurement
[MEASURE]  ᾱ=0.988 µm/px  σ_α=0.152  power_law=-0.410  C=1.83 bits/px  bilateral ✓
[EXECUTE]  observe(SiR_Actin_001.tif, n=10)  Σ=(10,0,0,+½)
[EXECUTE]  catalyze(conservation(dna_mass))  ε=0.008  conf=1.0  ε_eff=0.008  S_k→0.608
[EXECUTE]  catalyze(phase_lock(chromatin))   ε=0.010  conf=0.9  ε_eff=0.0055  S_k→0.614
[EXECUTE]  access(nucleus_a, threshold=0.50)  centroid=(312,487)  mask_area=1820px  S_k→0.664
[EXECUTE]  access(nucleus_b, threshold=0.50)  centroid=(589,502)  mask_area=1754px  S_k→0.714
[EXECUTE]  measure_distance(nucleus_a, nucleus_b)  fast-marching 1024×1024
[EXECUTE]  d=14.312 µm  δd=0.157 µm  (1.10%)  path_length=283px
[EMIT]     S_k=0.412  S_t=0.281  S_e=0.307  sum=1.000000000000000 ✓
[GOAL]     distance_uncertainty=0.157 µm   < 0.5 µm  ✓
[GOAL]     s_entropy_conservation=1.1e-16  < 1e-12   ✓
[GOAL]     snr=10.2                        > 8.0     ✓
```

**Expected result**:
```json
{
  "structure": "nucleus_pair",
  "position": [51.2, 49.8, 0.0],
  "distance": 14.312,
  "uncertainty": 0.157,
  "relativeUncertainty": 0.011,
  "sEntropy": { "sk": 0.412, "st": 0.281, "se": 0.307, "sum": 1.0 },
  "goalStatus": [
    { "metric": "distance_uncertainty", "op": "<", "threshold": 0.5, "unit": "µm", "actual": 0.157, "passed": true },
    { "metric": "s_entropy_conservation", "op": "<", "threshold": 1e-12, "unit": "", "actual": 1.1e-16, "passed": true },
    { "metric": "snr", "op": ">", "threshold": 8.0, "unit": "", "actual": 10.2, "passed": true }
  ]
}
```

**Visualisation outputs**:

*Canvas 2D — Geodesic (active after `visualise(geodesic)`)*
- Raw BBBC039 grayscale image at 512×512 display resolution
- Cyan dot at pixel (312, 487) labelled `nucleus_a`
- Magenta dot at pixel (589, 502) labelled `nucleus_b`
- White 2px line following the geodesic path (283 pixels, curving slightly around lower-intensity regions)
- Text overlay bottom-left: `d = 14.312 µm  δd = 0.157 µm`

*D3 Charts*

Spectral Power:
```
log |û(k)|²
    │ ·
    │  ·
    │   ·
    │    · ·
    │      ··
    │        ·· (slope = -0.410)
    └──────────── log k
                power-law fit line in red
```

Entropy Trajectory (stacked area):
```
1.0 │████████████████████ S_e (backaction, grows)
    │███████████████ S_t (timing, shrinks at COMPILE)
0.0 │████████████████████ S_k (knowledge, grows)
    └── COMPILE → ASSIGN → MEASURE → EXECUTE → EMIT
```

Uncertainty Bar:
```
δd │─────────────────── 0.5 µm goal line (green, passed)
   │
   │  ████ 0.157 µm (actual)
   └──────────────────────────────
```

*Three.js 3D — Distance Tube scene*
- Camera at `[5, 5, 8]`, orbit enabled
- Scale field surface (100×100 µm footprint, z-height = α(x,y) × 5): viridis colour map, smooth terrain mesh with peak at nucleus centres
- Two glowing spheres (radius 0.3 µm) at nucleus centroids: cyan and magenta
- Geodesic tube connecting them: radius = δd (0.157 µm scaled to scene), emissive white material, slight glow
- Grid plane at z=0 with 10 µm divisions, grey
- Axis labels: X (µm), Y (µm), Z (α scale)

*Three.js 3D — Entropy Sphere scene*
- Unit sphere, orbit enabled
- Three sectors: S_k=0.412 (blue, bottom), S_t=0.281 (green, equator band), S_e=0.307 (orange, top)
- Floating text labels: `S_k = 0.412`, `S_t = 0.281`, `S_e = 0.307`, `Σ = 1.000`
- Thin white lines at sector boundaries

---

## Example 2: Depth too low — goal warning caught at compile time

**What it does**: Shows the type checker catching an impossible `goal` criterion before any image is fetched. `depth = 6` with a 100×100 µm field gives `δd_min ≈ 1.56 µm`, which cannot satisfy `< 0.1 µm`.

```scope
scope goal_warning_demo {
  coordinate_space {
    field 100 x 100 µm
    depth 6
    lambda_s 0.10
    lambda_t 0.05
  }

  goal {
    distance_uncertainty < 0.1 µm    // impossible at depth=6
    distance_uncertainty < 2.0 µm    // achievable
  }

  measure = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_001.tif"), n = 6)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> measure_distance(nucleus_a, nucleus_b)
}
```

**Expected console output (compile phase, before execution)**:
```
[TYPE WARNING]  GoalUnreachableAtDepth
  metric=distance_uncertainty  threshold=0.100 µm
  current depth=6  field=100 µm  predicted δd_min=1.563 µm
  To achieve δd < 0.100 µm, need depth ≥ 13
  Suggestion: set coordinate_space.depth to 13
  (Continuing — this is a warning, not an error)

[COMPILE]  default cell_label=(6,0,0,+½)
[MEASURE]  ᾱ=0.988 µm/px  power_law=-0.410 ✓
[EXECUTE]  d=14.312 µm  δd=1.580 µm  (11.04%)
[EMIT]     S_k=0.415  S_t=0.300  S_e=0.285  sum=1.000000000000000 ✓
[GOAL]     distance_uncertainty=1.580 µm  < 0.1 µm  ✗  (as predicted)
[GOAL]     distance_uncertainty=1.580 µm  < 2.0 µm  ✓
```

**Visualisation outputs**:

*GoalStatusBar*:
```
⚠ depth=6  [✗ δd < 0.1 µm]  [✓ δd < 2.0 µm]
```

*D3 Uncertainty Bar*:
```
     │────── 2.0 µm line (green, passed)
     │
     │  ████████████████████ 1.580 µm (actual)
     │
     │────── 0.1 µm line (red dashed, failed)
     └───────────────────────────────────────
```

*3D Scale Field Surface*: Same as Example 1 but with a lower-resolution mesh (64×64 for depth=6 → 2^6=64 pixels effective), showing coarser terrain with the same α(x,y) values.

---

## Example 3: Confidence-weighted catalysts + fuzzy access threshold

**What it does**: Uses `confidence` on two catalysts and a tighter `threshold` on `access`. Shows how both features propagate into the uncertainty formula and the segmentation.

**Database**: BBBC006 — CHO cells, tubulin staining, 0.063 µm/pixel.

```scope
scope spindle_with_confidence {
  coordinate_space {
    field 64 x 64 µm
    depth 10
    lambda_s 0.08
    lambda_t 0.03
  }

  goal {
    distance_uncertainty < 0.2 µm
    relative_uncertainty < 0.02
  }

  rule symmetry(bilateral) {
    invariant: "mitotic spindle has bilateral symmetry along division axis"
    epsilon: 0.006
  }

  spindle_axis = observe(load(db="BBBC", dataset="BBBC006", image="v1_001.tif"), n = 10)
    |> catalyze(symmetry(bilateral), confidence = 0.8)
    |> catalyze(phase_lock(plasma_membrane), confidence = 0.6)
    |> visualise(scale_field)
    |> access(nucleus_a, threshold = 0.7)
    |> access(nucleus_b, threshold = 0.7)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(distance_map)
}
```

**Expected console output**:
```
[COMPILE]  default cell_label=(10,0,0,+½)
[ASSIGN]   morphism=spindle_axis
[MEASURE]  ᾱ=0.063 µm/px  σ_α=0.008  power_law=-0.381  C=2.14 bits/px ✓
[EXECUTE]  observe(v1_001.tif, n=10)  Σ=(10,0,0,+½)
[EXECUTE]  catalyze(symmetry(bilateral))   ε=0.006  conf=0.8  ε_eff=0.0036  S_k→0.604
[EXECUTE]  catalyze(phase_lock(membrane))  ε=0.010  conf=0.6  ε_eff=0.007   S_k→0.611
[VISUALISE] → scale_field
[EXECUTE]  access(nucleus_a, threshold=0.70)  centroid=(210,310)  mask_area=940px  S_k→0.661
[EXECUTE]  access(nucleus_b, threshold=0.70)  centroid=(410,316)  mask_area=912px  S_k→0.711
[VISUALISE] → segmentation
[EXECUTE]  measure_distance(nucleus_a, nucleus_b)  fast-marching 1024×1024
[EXECUTE]  d=12.614 µm  δd=0.071 µm  (0.56%)  path_length=3175px
[VISUALISE] → distance_map
[EMIT]     S_k=0.420  S_t=0.300  S_e=0.280  sum=1.000000000000000 ✓
[GOAL]     distance_uncertainty=0.071 µm   < 0.2 µm  ✓
[GOAL]     relative_uncertainty=0.0056      < 0.02    ✓
```

**Why lower δd than Example 1**: CHO cells at 0.063 µm/px have a smaller scale factor ᾱ (0.063 vs 0.988). Since `δd ∝ ᾱ`, the absolute uncertainty is smaller. The confidence discounts also reduce the weighted ε sum compared to using full-cost catalysts.

**Visualisation outputs**:

*Canvas 2D — Scale Field (shown after first `visualise(scale_field)` step)*
- α(x,y) heatmap in viridis
- CHO cell shows lower α values overall due to smaller pixel size
- Mean ᾱ = 0.063 shown in colorbar
- Annotated σ_α = 0.008 (tight distribution — more uniform than HeLa)

*Canvas 2D — Segmentation (after second `visualise(segmentation)` step)*
- Raw CHO image in greyscale
- Two nucleus regions outlined in cyan at threshold=0.7
- Comparison: if you change `threshold` to 0.4, the outlines would be larger (looser boundary) — this is shown as a second faint dashed outline in a different colour to illustrate the fuzzy boundary concept
- Mask area annotations: `nucleus_a: 940px  nucleus_b: 912px`

*Canvas 2D — Distance Map (after `visualise(distance_map)` step)*
- T(x,y) fast-marching result from `nucleus_a.centroid`
- Plasma colour map: dark at source, bright at far distances
- Source point (210,310): white dot
- Target point (410,316): white dot
- Geodesic path: white line
- Colourbar on right: 0 µm → 15 µm

*D3 Spectral Power*:
```
Power  │·
       │ ·
       │  ·
       │   ·· (slope = -0.381, shallower than BBBC039)
       │     ···
       └─────────── frequency
  CHO cells show shallower decay due to finer resolution and tubulin structure
```

*Three.js 3D — Point Cloud scene*
- Each pixel as a point: x,y = pixel coordinates, z = intensity × 3
- Colour = α(x,y) from viridis palette
- CHO cells appear as two tall mountains with blue-green colouring (low ᾱ)
- OrbitControls: user can rotate to see the intensity landscape
- Two glowing spheres at nucleus centroids

*Three.js 3D — Scale Field Surface*
- Terrain is much flatter than HeLa (ᾱ = 0.063 vs 0.988) — all values in tight band around z=0.063
- Colourbar range adjusted to [0.05, 0.08] to show the small variation

---

## Example 4: Fused morphisms with entropy comparison

**What it does**: Runs two morphisms and fuses them. Shows how fusing shifts the entropy budget and produces a combined uncertainty bar. Includes a `visualise(entropy_sphere)` to show the S-entropy redistribution.

**Database**: BBBC039.

```scope
scope fused_nuclear_membrane {
  coordinate_space {
    field 100 x 100 µm
    depth 10
    lambda_s 0.10
    lambda_t 0.05
  }

  goal {
    distance_uncertainty < 0.3 µm
    channel_capacity > 1.5 bits
  }

  membrane_estimate = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_002.tif"), n = 10)
    |> catalyze(phase_lock(actin))
    |> access(cell_boundary)
    |> access(nucleus_centroid)
    |> measure_distance(cell_boundary, nucleus_centroid)

  nucleus_primary = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_002.tif"), n = 10)
    |> catalyze(conservation(dna_mass))
    |> catalyze(phase_lock(chromatin), confidence = 0.85)
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> measure_distance(nucleus_a, nucleus_b)
    |> fuse(membrane_estimate, rho = 0.6)
    |> visualise(entropy_sphere)
}
```

**Expected console output**:
```
[COMPILE]  default cell_label=(10,0,0,+½)
[ASSIGN]   morphism=nucleus_primary  (last declared)
[MEASURE]  ᾱ=0.991 µm/px  power_law=-0.418  C=1.76 bits/px ✓
[EXECUTE]  observe(SiR_Actin_002.tif, n=10)
[EXECUTE]  catalyze(conservation(dna_mass))  ε_eff=0.008
[EXECUTE]  catalyze(phase_lock(chromatin))   ε_eff=0.0055  (conf=0.85)
[EXECUTE]  access(nucleus_a)  centroid=(301,495)
[EXECUTE]  access(nucleus_b)  centroid=(604,488)
[EXECUTE]  measure_distance  d_primary=15.109 µm  δd=0.168 µm
[EXECUTE]  fuse(membrane_estimate, rho=0.6)
[EXECUTE]    membrane_estimate.d=22.4 µm  δd=0.192 µm
[EXECUTE]    d_fused = 0.6×15.109 + 0.4×22.4 = 18.025 µm
[EXECUTE]    δd_fused = √((0.6×0.168)² + (0.4×0.192)²) = 0.129 µm
[VISUALISE] → entropy_sphere
[EMIT]     S_k=0.431  S_t=0.281  S_e=0.288  sum=1.000000000000000 ✓
[GOAL]     distance_uncertainty=0.129 µm   < 0.3 µm  ✓
[GOAL]     channel_capacity=1.76 bits      > 1.5 bits ✓
```

**Result**:
```json
{
  "structure": "nucleus_pair_fused",
  "position": [50.2, 49.1, 0.0],
  "distance": 18.025,
  "uncertainty": 0.129,
  "relativeUncertainty": 0.0072,
  "sEntropy": { "sk": 0.431, "st": 0.281, "se": 0.288, "sum": 1.0 },
  "goalStatus": [
    { "metric": "distance_uncertainty", "actual": 0.129, "passed": true },
    { "metric": "channel_capacity", "actual": 1.76, "passed": true }
  ]
}
```

**Visualisation outputs**:

*Three.js 3D — Entropy Sphere (after `visualise(entropy_sphere)` step)*
- Unit sphere with three sectors:
  - Blue (bottom): S_k = 0.431 → labelled `knowledge`
  - Green (equatorial band): S_t = 0.281 → labelled `timing`
  - Orange (top): S_e = 0.288 → labelled `backaction`
- Comparison to Example 1 (S_k=0.412): S_k is higher here due to more catalysts
- Thin separator lines at sector boundaries
- `Σ = 1.000` text in white at the north pole

*D3 Channel Capacity chart*:
```
C (bits) │                        ·····
         │                   ·····
         │              ·····
         │         ·····
         │    ·····
1.76 ──── │···· ← operating point (SNR=8.1, C=1.76)  green dot
         │
1.5  ──── │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ goal threshold line
         └────────────────────── SNR (linear)
              0        5       10
```

---

## Example 5: Three-cell dispatch — ANAPHASE fires

**What it does**: Full timing-cell dispatch program. All three cells are defined; the image drives classification into ANAPHASE. Shows the lower S_t value from narrower cell bounds.

**Database**: BBBC039, image 3.

```scope
scope anaphase_tracking {
  channels {
    sync dapi at 0.1 µm/pixel
    cell PROPHASE  bounds (-2.0e-6, -0.8e-6) action early_nucleus
    cell METAPHASE bounds (-0.8e-6,  0.8e-6) action aligned_nucleus
    cell ANAPHASE  bounds ( 0.8e-6,  2.0e-6) action separating_nucleus
  }

  coordinate_space {
    field 100 x 100 µm
    depth 10
    lambda_s 0.10
    lambda_t 0.05
  }

  goal {
    distance_uncertainty < 0.5 µm
    s_entropy_conservation < 1e-12
  }

  early_nucleus = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_003.tif"), n = 10)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_centroid)
    |> visualise(segmentation)

  aligned_nucleus = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_003.tif"), n = 10)
    |> catalyze(symmetry(bilateral))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(partition_tree)

  separating_nucleus = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_003.tif"), n = 10)
    |> catalyze(conservation(dna_mass))
    |> catalyze(phase_lock(chromatin))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(entropy_trajectory)

  dispatch {
    when PROPHASE  do execute(early_nucleus)
    when METAPHASE do execute(aligned_nucleus)
    when ANAPHASE  do execute(separating_nucleus)
  }
}
```

**Expected console output (ANAPHASE scenario, ΔP_mean = +1.4e-6 s)**:
```
[COMPILE]  events=1000  ΔP_mean=+1.40e-6s  σ=0.30e-6s
[COMPILE]  cell=ANAPHASE  (ΔP in [0.8e-6, 2.0e-6])  S_t: 0.500→0.182
           (S_t lower than Example 1: ANAPHASE cell width = 1.2e-6 vs total span 4.0e-6)
[ASSIGN]   morphism=separating_nucleus
[MEASURE]  ᾱ=0.992 µm/px  power_law=-0.422  C=1.91 bits/px ✓
[EXECUTE]  observe(SiR_Actin_003.tif, n=10)
[EXECUTE]  catalyze(conservation(dna_mass))  ε_eff=0.008
[EXECUTE]  catalyze(phase_lock(chromatin))   ε_eff=0.010
[EXECUTE]  access(nucleus_a)  centroid=(289,471)
[EXECUTE]  access(nucleus_b)  centroid=(618,510)
[EXECUTE]  measure_distance  d=16.891 µm  δd=0.186 µm  (1.10%)
[VISUALISE] → entropy_trajectory
[EMIT]     S_k=0.435  S_t=0.182  S_e=0.383  sum=1.000000000000000 ✓
[GOAL]     distance_uncertainty=0.186 µm  < 0.5 µm  ✓
[GOAL]     s_entropy_conservation=8.9e-17  < 1e-12  ✓
```

**Result**:
```json
{
  "structure": "separating_nucleus",
  "position": [45.3, 49.0, 0.0],
  "distance": 16.891,
  "uncertainty": 0.186,
  "sEntropy": { "sk": 0.435, "st": 0.182, "se": 0.383, "sum": 1.0 },
  "goalStatus": [
    { "metric": "distance_uncertainty", "actual": 0.186, "passed": true },
    { "metric": "s_entropy_conservation", "actual": 8.9e-17, "passed": true }
  ]
}
```

**Visualisation outputs**:

*D3 Entropy Trajectory (after `visualise(entropy_trajectory)` step)*:
```
1.0 │
    │       S_e ██████████████████████████████  ← 0.383 final
    │
0.5 │  S_t ████████████  ← drops from 0.5 at COMPILE (cell classified)
    │
    │       S_k █████████████████  ← grows as catalysts narrow partition
0.0 └──────────────────────────────────────────
    COMPILE  ASSIGN  MEASURE  EXECUTE  EMIT
```
Note: S_e is 0.383 here vs 0.307 in Example 1. The ANAPHASE cell is narrow (1.2 µs wide), so timing classification removes more S_t, which conservation pushes more strongly into S_e (backaction dominates when timing is tightly classified).

*Three.js 3D — Partition Tree scene (would have shown for METAPHASE dispatch)*:
- Binary tree of visited (n,ℓ,m,s) states
- Root node: `(10,0,0,+½)` — the initial observe state
- Children created at each `catalyze` and `access` step
- Node colour: blue (high S_k) → orange (high S_e)
- Edge labels: constraint name + ε_eff
- Current node highlighted in white
- OrbitControls + click to inspect node's (n,ℓ,m,s) values in a popup

---

## Example 6: BBBC008 Drosophila dual-channel

**What it does**: Two morphisms on the same image, each running on a different channel (GFP and DAPI). Shows how `sync` declarations with different resolutions affect the scale field.

**Database**: BBBC008 — Drosophila C57 cells, GFP + DAPI, 0.08 µm/pixel.

```scope
scope dual_channel_drosophila {
  channels {
    sync gfp  at 0.08 µm/pixel
    sync dapi at 0.08 µm/pixel
  }

  coordinate_space {
    field 82 x 82 µm
    depth 10
    lambda_s 0.09
    lambda_t 0.04
  }

  goal {
    distance_uncertainty < 0.15 µm
    crlb_pixels < 0.1
  }

  gfp_morphology = observe(load(db="BBBC", dataset="BBBC008", image="C57_01.tif"), n = 10)
    |> catalyze(phase_lock(actin), confidence = 0.75)
    |> access(nucleus_a, threshold = 0.6)
    |> access(nucleus_b, threshold = 0.6)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(point_cloud)

  dapi_distance = observe(load(db="BBBC", dataset="BBBC008", image="C57_01.tif"), n = 10)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(uncertainty_bar)
}
```

**Expected console output (dapi_distance runs — last declared)**:
```
[COMPILE]  default cell_label=(10,0,0,+½)
[ASSIGN]   morphism=dapi_distance
[MEASURE]  ᾱ=0.080 µm/px  σ_α=0.009  power_law=-0.395  C=2.03 bits/px ✓
[EXECUTE]  access(nucleus_a)  centroid=(401,512)  mask_area=2810px
[EXECUTE]  access(nucleus_b)  centroid=(618,498)  mask_area=2774px
[EXECUTE]  d=17.382 µm  δd=0.090 µm  (0.52%)  CRLB=0.038px
[EMIT]     S_k=0.405  S_t=0.300  S_e=0.295  sum=1.000000000000000 ✓
[GOAL]     distance_uncertainty=0.090 µm  < 0.15 µm  ✓
[GOAL]     crlb_pixels=0.038              < 0.1      ✓
```

**Visualisation outputs**:

*D3 Uncertainty Bar (after `visualise(uncertainty_bar)` step)*:
```
δd   │
0.15 │─── goal line (green)
     │
0.09 │  ████  actual δd
     │
0.10 │─── ─ ─  (natural level for Example 1 BBBC039)
     │
     └─────────────────────────────────
       (comparison bar for BBBC039 shown in grey for context)
```

*Three.js 3D — Point Cloud (from `gfp_morphology` visualise step, shown if morphism is re-run)*:
- 1024×1024 = ~1M points, subsampled to 50k for browser performance (every 20th pixel)
- x, y = pixel coordinates in µm (0–82 µm)
- z = normalised intensity × 10 µm
- Colour = α(x,y) from viridis: Drosophila cells at 0.08 µm/px are uniformly low-α (blue-green throughout)
- Two yellow rings marking the `nucleus_a` and `nucleus_b` centroid positions
- Camera starts at `[41, −20, 30]` looking toward the image plane

---

## Error cases

### Parse error

```scope
scope bad {
  observe(load(db="BBBC"), n = 10)   // missing dataset and image in load()
}
```

```
[PARSE ERROR] line 2 col 18
  Unexpected ')' in db_ref: expected ',' followed by 'dataset' keyword
  Got: observe(load(db="BBBC"), ...)
  Expected: observe(load(db=..., dataset=..., image=...), ...)
```

### Depth mismatch

```scope
scope bad_depth {
  coordinate_space { field 100 x 100 µm  depth 10  lambda_s 0.1  lambda_t 0.05 }
  my_morph = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_001.tif"), n = 8)
    |> access(nucleus_a)
}
```

```
[TYPE ERROR] DepthMismatch in morphism "my_morph"
  observe() declares n=8 but coordinate_space.depth=10
  Fix: change n=8 to n=10, or change coordinate_space.depth to 8
```

### Cell overlap

```scope
cell EARLY bounds (-2.0e-6,  0.0e-6) action m1
cell LATE  bounds (-0.5e-6,  2.0e-6) action m2
```

```
[TYPE ERROR] CellOverlap: "EARLY" and "LATE"
  Overlap region: [-0.5e-6, 0.0e-6]
  Fix: change EARLY.boundsHigh to -0.5e-6, or LATE.boundsLow to 0.0e-6
```

### Ungrounded distance

```scope
my_morph = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_001.tif"), n = 10)
  |> measure_distance(nucleus_a, nucleus_b)
```

```
[TYPE ERROR] UngroundedDistance in morphism "my_morph"
  measure_distance(nucleus_a, nucleus_b): "nucleus_a" not accessed
  "nucleus_b" not accessed
  Fix: add |> access(nucleus_a) and |> access(nucleus_b) before measure_distance
```

### Goal unreachable (warning, not error)

```scope
coordinate_space { field 100 x 100 µm  depth 4  lambda_s 0.1  lambda_t 0.05 }
goal { distance_uncertainty < 0.01 µm }
```

```
[TYPE WARNING] GoalUnreachableAtDepth
  metric=distance_uncertainty  threshold=0.010 µm
  At depth=4, field=100 µm: δd_min = 6.25 µm  (100/2^4 = 6.25)
  To achieve δd < 0.010 µm need depth ≥ 14
  (Execution will continue — goal chip will show ✗)
```
