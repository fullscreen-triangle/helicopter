# SCOPE Playground — Language Syntax Reference

SCOPE programs unify three microscopy frameworks — partition calculus, context-dependent coordinates, and temporal programming — through a single program structure and shared `(n, ℓ, m, s)` type. Extensions beyond the published paper (marked `[ext]`) are fully implemented in the playground and will be incorporated into the next paper revision.

---

## Program structure

```scope
scope <name> {
  channels { ... }           // optional: timing cells and acquisition sync
  coordinate_space { ... }   // optional: physical field parameters
  goal { ... }               // optional [ext]: success criteria
  rule <name>(<arg>) { ... } // optional [ext]: named constraint declarations
  <name> = observe(...) ...  // morphism declarations (one or more)
  dispatch { ... }           // optional: timing cell → morphism routing
}
```

---

## `channels` block

### `sync` — acquisition reference

```scope
sync <name> at <value> <unit>
```

| Unit | Meaning |
|---|---|
| `µm/pixel` | spatial resolution |
| `freq` | acquisition frequency in Hz |
| `nm` | illumination wavelength |

```scope
sync dapi   at 0.1  µm/pixel
sync gfp    at 0.065 µm/pixel
sync laser  at 488  nm
```

### `cell` — timing cell for dispatch

```scope
cell <name> bounds (<low>, <high>) action <morphism_name>
```

Defines a Borel set in ΔP (timing deviation in seconds) space. The timing classification drives dispatch. Cell bounds must be disjoint — the type checker errors if they overlap.

```scope
cell PROPHASE  bounds (-2.0e-6, -0.8e-6) action nucleus_pair_measurement
cell METAPHASE bounds (-0.8e-6,  0.8e-6) action membrane_boundary
cell ANAPHASE  bounds ( 0.8e-6,  2.0e-6) action nucleus_pair_measurement
```

---

## `coordinate_space` block

```scope
coordinate_space {
  field <width> x <height> µm
  depth <n>
  lambda_s <value>
  lambda_t <value>
}
```

`depth n` is the central parameter: it equals `log₂(field_size_pixels)`. For a 1024×1024 image at 0.1 µm/px over a 100×100 µm field: `n = log₂(1000) ≈ 10`.

`lambda_s` and `lambda_t` control the spatial and temporal coherence weights in the bilateral filter during MEASURE.

---

## `goal` block [ext]

Declares success criteria evaluated at EMIT. The type checker statically warns if any criterion is analytically unreachable given `coordinate_space.depth` and field size.

```scope
goal {
  distance_uncertainty   < 0.5  µm
  distance_uncertainty   < 0.1  µm
  s_entropy_conservation < 1e-12
  relative_uncertainty   < 0.02
  snr                    > 8.0
  channel_capacity       > 1.5  bits
}
```

Supported metrics:

| Metric name | What it measures | Unit |
|---|---|---|
| `distance_uncertainty` | δd from Theorem 2 | µm |
| `relative_uncertainty` | δd / d | (fraction) |
| `s_entropy_conservation` | \|S_k+S_t+S_e−1\| | (dimensionless) |
| `snr` | signal-to-noise ratio of the image | (dimensionless) |
| `channel_capacity` | C = ½log₂(1+SNR) | bits |
| `crlb_pixels` | Cramér-Rao lower bound on position | px |

The GoalStatusBar shows each criterion as a green ✓ or red ✗ chip after execution. The uncertainty bar chart draws a vertical line at each threshold so you can see how close the result is.

---

## `rule` declarations [ext]

Named constraint declarations replace magic strings with explicit epsilon values and human-readable invariants. Once declared, the name is used inside `catalyze()`.

```scope
rule conservation(dna_mass) {
  invariant: "total DAPI-stained area is conserved ±5% across frames"
  epsilon: 0.008
}

rule phase_lock(chromatin) {
  invariant: "chromatin conformation is coherent to reference image"
  epsilon: 0.010
}

rule thermal(cytoplasm) {
  invariant: "diffusion limited by thermal model at 37°C"
  epsilon: 0.005
}
```

When a `rule` is declared, its `epsilon` overrides the default lookup table. The type checker verifies that every `catalyze(name)` call references a declared rule or a built-in constraint family.

---

## Morphism declarations

```scope
<name> = observe(<frame_ref>, n = <nat>) |> <step> |> <step> ...
```

### `observe` — entry point

```scope
observe(<frame_ref>, n = <nat>)
```

Creates initial partition state `Σ₀ = (n, 0, 0, +½)`. The frame ref is either a channel name or a `load(...)` expression.

```scope
observe(dapi, n = 10)
observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_001.tif"), n = 10)
```

### `catalyze` — apply physical constraint [confidence ext]

```scope
|> catalyze(<constraint>)
|> catalyze(<constraint>, confidence = <real>)
```

Applies a physical constraint, consuming `ε_eff = ε_base × (1 − confidence × 0.5)` from the entropy budget. Default `confidence = 1.0` (full cost). A `confidence < 1.0` means the constraint is uncertain — it still narrows the partition but at reduced entropy cost, and the uncertainty formula uses the discounted ε.

The `confidence` parameter comes from Kwasa-Kwasa's probabilistic evidence model: constraints backed by strong experimental evidence get `confidence = 0.95`; poorly validated constraints use lower values like `0.5`.

```scope
|> catalyze(conservation(dna_mass))                    // ε_eff = 0.008
|> catalyze(phase_lock(chromatin), confidence = 0.9)   // ε_eff = 0.0044
|> catalyze(thermal(cytoplasm), confidence = 0.6)      // ε_eff = 0.003
```

Built-in constraint families (default ε, overridden by `rule` declarations):

| Constraint | Default ε | Semantics |
|---|---|---|
| `conservation(dna_mass)` | 0.008 | Total DAPI area conserved |
| `conservation(cell_count)` | 0.008 | Segmented cell count conserved |
| `phase_lock(chromatin)` | 0.010 | Chromatin coherent to reference |
| `phase_lock(plasma_membrane)` | 0.010 | Membrane boundary coherent |
| `phase_lock(actin)` | 0.010 | Actin cytoskeleton coherent |
| `thermal(temperature)` | 0.005 | Diffusion within thermal model |
| `symmetry(bilateral)` | 0.006 | Bilateral symmetry |
| `symmetry(radial)` | 0.006 | Radial symmetry |

### `access` — traverse to sub-structure [threshold ext]

```scope
|> access(<structure_name>)
|> access(<structure_name>, threshold = <real>)
```

Segments the image to locate a named structure and records its centroid and per-pixel membership map. `threshold` (0–1, default 0.5) controls the segmentation boundary: `0.5` uses Otsu, higher values require stronger foreground signal.

The membership map is a fuzzy boundary — each pixel has a degree `[0,1]` of belonging to the structure rather than a hard binary. This directly implements Kwasa-Kwasa's fuzzy unit boundary concept. A `threshold = 0.8` produces tighter, more confident segmentation at the cost of possibly missing low-contrast periphery.

```scope
|> access(nucleus_a)                         // Otsu (default 0.5)
|> access(nucleus_b, threshold = 0.75)       // tighter boundary
|> access(cell_boundary, threshold = 0.3)    // loose, catches faint edges
```

Standard structure names:

| Name | What is found |
|---|---|
| `nucleus_a` | First (largest) detected nucleus |
| `nucleus_b` | Second detected nucleus |
| `nucleus_centroid` | Centroid of all nuclei combined |
| `cell_boundary` | Outermost cell contour |
| `partition_boundary` | Morphological reconstruction result |
| `separation_vector` | Line between two most distant nuclei |
| `spindle_midpoint` | Midpoint of mitotic spindle axis |

### `measure_distance` — world-space geodesic distance

```scope
|> measure_distance(<target1>, <target2>)
```

Runs fast marching over `α(x,y)` from `target1.centroid` and reads off the geodesic distance to `target2.centroid`. Backtrack gradient descent on the distance map to extract the geodesic path pixels for visualisation.

```
d = ∫_γ α(γ(t)) |γ̇(t)| dt

δd = ᾱ · (L_field / 2^n) · (1 + Σ εᵢ_eff)
```

The geodesic path is rendered as a white overlay line in the Canvas2D view and as a glowing tube in the 3D Distance Tube scene.

### `fuse` — combine morphisms

```scope
|> fuse(<morphism_name>, rho = <real>)
```

Weighted combination: `d_final = rho × d_current + (1−rho) × d_ref`. Uncertainty combines in quadrature: `δd_final = √((rho·δd₁)² + ((1−rho)·δd₂)²)`.

### `visualise` — inline visualisation hint [ext]

```scope
|> visualise(scale_field)
|> visualise(segmentation)
|> visualise(distance_map)
|> visualise(geodesic)
|> visualise(point_cloud)
|> visualise(entropy_sphere)
|> visualise(partition_tree)
|> visualise(spectral_power)
|> visualise(entropy_trajectory)
|> visualise(uncertainty_bar)
```

Tells the playground which view to switch to after this step completes. Useful for guiding a reader through a multi-step program where different steps have different interesting outputs.

---

## `dispatch` block

Routes classified timing cells to morphisms.

```scope
dispatch {
  when PROPHASE  do execute(nucleus_pair_measurement)
  when METAPHASE do { execute(membrane_boundary) emit membrane_result }
  when ANAPHASE  do execute(nucleus_pair_measurement)
}
```

---

## `load` expression

Fetches an image from a public database at runtime via the CORS proxy.

```scope
observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_001.tif"), n = 10)
observe(load(db="AllenCell", dataset="allencell_3d", image="histone_001"), n = 10)
observe(load(db="IDR", dataset="idr_timelapse", image="idr0001/cell_001"), n = 10)
observe(load(db="OpenCell", dataset="opencell_proteins", image="LMNB1_001"), n = 10)
```

Supported databases:

| `db` | Dataset IDs | Channels | Resolution |
|---|---|---|---|
| `"BBBC"` | `"BBBC039"`, `"BBBC006"`, `"BBBC008"` | DAPI, Actin, GFP, Tubulin | 0.063–0.1 µm/px |
| `"AllenCell"` | `"allencell_3d"` | GFP, DAPI | 0.065 µm/px |
| `"OpenCell"` | `"opencell_proteins"` | GFP, DAPI | 0.08 µm/px |
| `"IDR"` | `"idr_timelapse"` | GFP, DAPI, RFP | 0.1 µm/px |

---

## Complete minimal program

```scope
scope hello {
  goal {
    distance_uncertainty < 0.5 µm
    s_entropy_conservation < 1e-12
  }

  coordinate_space {
    field 100 x 100 µm
    depth 10
    lambda_s 0.10
    lambda_t 0.05
  }

  measure_nuclei = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_001.tif"), n = 10)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(geodesic)
}
```

Expected output:
```json
{
  "structure": "nucleus_pair",
  "position": [51.2, 49.8, 0.0],
  "distance": 14.312,
  "uncertainty": 0.157,
  "s_entropy": { "S_k": 0.412, "S_t": 0.281, "S_e": 0.307 },
  "goalStatus": [
    { "metric": "distance_uncertainty", "op": "<", "threshold": 0.5, "actual": 0.157, "passed": true },
    { "metric": "s_entropy_conservation", "op": "<", "threshold": 1e-12, "actual": 1.1e-16, "passed": true }
  ]
}
```

Visualisation after execution:
- **Canvas2D**: raw BBBC039 image with geodesic path line between the two nucleus centroids
- **Charts**: spectral power (log-log, power-law line), entropy trajectory (S_k/S_t/S_e across 5 phases), uncertainty bar with 0.5 µm threshold line
- **3D**: Distance Tube scene showing the geodesic path as a glowing tube; Scale Field Surface showing α(x,y) terrain
