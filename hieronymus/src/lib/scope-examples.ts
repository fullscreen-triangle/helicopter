// SCOPE example programs — each script produces a rich set of visual outputs
// matching scope/playground/examples.md exactly.
//
// Every example includes enough visualise() steps to populate:
//   - Canvas 2D: raw image, scale field, segmentation, distance map, geodesic
//   - D3 charts:  spectral power, entropy trajectory, uncertainty bar, scale histogram
//   - Three.js:   distance tube or entropy sphere or scale field surface

export interface ScopeExample {
  id: string;
  title: string;
  description: string;
  source: string;
}

export const SCOPE_EXAMPLES: ScopeExample[] = [
  // ── Example 1 ────────────────────────────────────────────────────────────────
  // Canonical nuclear separation. PROPHASE fires nucleus_pair_measurement.
  // METAPHASE fires membrane_boundary (cell contour, no distance).
  // Five visualise steps produce: scale field, segmentation, geodesic, spectral
  // power, entropy trajectory.  Goal block checks δd, S-entropy, SNR.
  {
    id: 'ex1-nuclear-separation',
    title: 'Example 1 — Nuclear separation (BBBC039, HeLa)',
    description: 'Canonical HeLa nuclear distance. PROPHASE/ANAPHASE dispatch → nucleus_pair_measurement. Five visualise steps: scale_field, segmentation, spectral_power, entropy_trajectory, geodesic. Named rule, three-criterion goal block.',
    source: `scope nuclear_separation_dynamics {
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
    |> visualise(scale_field)
    |> catalyze(conservation(dna_mass))
    |> catalyze(phase_lock(chromatin), confidence = 0.9)
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(spectral_power)
    |> visualise(entropy_trajectory)
    |> visualise(geodesic)

  membrane_boundary = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_001.tif"), n = 10)
    |> catalyze(phase_lock(plasma_membrane))
    |> access(cell_boundary)
    |> visualise(scale_field)
    |> visualise(segmentation)

  dispatch {
    when PROPHASE  do execute(nucleus_pair_measurement)
    when METAPHASE do execute(membrane_boundary)
    when ANAPHASE  do execute(nucleus_pair_measurement)
  }
}`,
  },

  // ── Example 2 ────────────────────────────────────────────────────────────────
  // Goal warning: depth=6 makes δd_min ≈ 1.56 µm, impossible to satisfy < 0.1 µm.
  // The type checker emits GoalUnreachableAtDepth (warning, not error) and
  // execution continues — goal chip shows ✗ for the tight criterion, ✓ for 2.0 µm.
  // Visualise steps produce: scale field heatmap, segmentation, uncertainty bar,
  // spectral power, entropy trajectory.
  {
    id: 'ex2-goal-warning',
    title: 'Example 2 — Goal warning: depth too low (BBBC039)',
    description: 'depth=6 → δd_min ≈ 1.56 µm — type checker warns that distance_uncertainty < 0.1 µm is unreachable. Execution continues; goal chip shows ✗. Four visualise steps show scale field, segmentation, uncertainty bar, spectral power.',
    source: `scope goal_warning_demo {
  coordinate_space {
    field 100 x 100 µm
    depth 6
    lambda_s 0.10
    lambda_t 0.05
  }

  goal {
    distance_uncertainty < 0.1 µm
    distance_uncertainty < 2.0 µm
  }

  measure = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_001.tif"), n = 6)
    |> visualise(scale_field)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(spectral_power)
    |> visualise(uncertainty_bar)
}`,
  },

  // ── Example 3 ────────────────────────────────────────────────────────────────
  // Confidence-weighted catalysts + fuzzy access threshold on BBBC006 CHO cells.
  // Named rule for bilateral symmetry. Confidence 0.8 and 0.6 reduce ε_eff.
  // Threshold=0.7 gives tighter nucleus masks than default 0.5.
  // Six visualise steps: scale field, then segmentation after tight access,
  // then distance map, then spectral power, then scale histogram, then entropy trajectory.
  {
    id: 'ex3-confidence-threshold',
    title: 'Example 3 — Confidence-weighted catalysts + fuzzy threshold (BBBC006, CHO)',
    description: 'CHO mitotic spindle at 0.063 µm/px. Rule: symmetry(bilateral) ε=0.006. Confidence 0.8/0.6 on catalysts. Access threshold=0.7 → tighter masks. Six visualise steps: scale_field, segmentation, distance_map, spectral_power, scale_histogram, entropy_trajectory.',
    source: `scope spindle_with_confidence {
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

  rule conservation(dna_mass) {
    invariant: "total DAPI-stained area conserved"
    epsilon: 0.008
  }

  spindle_axis = observe(load(db="BBBC", dataset="BBBC006", image="v1_001.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(symmetry(bilateral), confidence = 0.8)
    |> catalyze(phase_lock(plasma_membrane), confidence = 0.6)
    |> access(nucleus_a, threshold = 0.7)
    |> access(nucleus_b, threshold = 0.7)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(distance_map)
    |> visualise(spectral_power)
    |> visualise(scale_histogram)
    |> visualise(entropy_trajectory)
}`,
  },

  // ── Example 4 ────────────────────────────────────────────────────────────────
  // Fused morphisms: membrane_estimate runs first, measures cell_boundary →
  // nucleus_centroid. nucleus_primary fuses with it at rho=0.6, producing a
  // combined d and δd lower than either alone.
  // Ends with entropy_sphere to show S-entropy redistribution after fusion.
  // Also includes scale field and segmentation for Canvas 2D.
  {
    id: 'ex4-fused',
    title: 'Example 4 — Fused morphisms: nucleus + membrane (BBBC039)',
    description: 'membrane_estimate measures cell_boundary→nucleus_centroid. nucleus_primary fuses at rho=0.6 → d_fused, δd_fused. Visualise steps: scale_field, segmentation, spectral_power, entropy_sphere.',
    source: `scope fused_nuclear_membrane {
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

  rule conservation(dna_mass) {
    invariant: "total DAPI-stained area is conserved ±5%"
    epsilon: 0.008
  }

  membrane_estimate = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_002.tif"), n = 10)
    |> catalyze(phase_lock(actin))
    |> access(cell_boundary)
    |> access(nucleus_centroid)
    |> measure_distance(cell_boundary, nucleus_centroid)

  nucleus_primary = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_002.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(conservation(dna_mass))
    |> catalyze(phase_lock(chromatin), confidence = 0.85)
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> fuse(membrane_estimate, rho = 0.6)
    |> visualise(spectral_power)
    |> visualise(entropy_sphere)
}`,
  },

  // ── Example 5 ────────────────────────────────────────────────────────────────
  // Three-cell dispatch. Synthetic timing drives ANAPHASE classification.
  // Three distinct morphisms — each with its own measurement strategy:
  //   early_nucleus    (PROPHASE)  — single centroid, segmentation only
  //   aligned_nucleus  (METAPHASE) — bilateral symmetry check, partition tree
  //   separating_nucleus (ANAPHASE) — two catalysts, full distance + entropy trajectory
  // Runtime dispatches to separating_nucleus.
  {
    id: 'ex5-anaphase',
    title: 'Example 5 — Three-cell dispatch, ANAPHASE fires (BBBC039)',
    description: 'PROPHASE → early_nucleus (centroid + segmentation). METAPHASE → aligned_nucleus (bilateral symmetry + partition_tree). ANAPHASE → separating_nucleus (two catalysts, distance, entropy_trajectory). Runtime dispatches to ANAPHASE.',
    source: `scope anaphase_tracking {
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

  rule conservation(dna_mass) {
    invariant: "total DAPI-stained area is conserved ±5%"
    epsilon: 0.008
  }

  rule symmetry(bilateral) {
    invariant: "nucleus pair has bilateral symmetry along division axis"
    epsilon: 0.006
  }

  early_nucleus = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_003.tif"), n = 10)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_centroid)
    |> visualise(scale_field)
    |> visualise(segmentation)

  aligned_nucleus = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_003.tif"), n = 10)
    |> catalyze(symmetry(bilateral))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(partition_tree)

  separating_nucleus = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_003.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(conservation(dna_mass))
    |> catalyze(phase_lock(chromatin))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(geodesic)
    |> visualise(entropy_trajectory)

  dispatch {
    when PROPHASE  do execute(early_nucleus)
    when METAPHASE do execute(aligned_nucleus)
    when ANAPHASE  do execute(separating_nucleus)
  }
}`,
  },

  // ── Example 6 ────────────────────────────────────────────────────────────────
  // Drosophila dual-channel BBBC008 at 0.08 µm/px.
  // gfp_morphology: actin channel, tighter threshold=0.6, point_cloud.
  // dapi_distance: DAPI channel, conservation rule, runs by default (last declared).
  // CRLB goal: Fisher information lower bound on localisation precision.
  // Visualise steps: scale_field, segmentation, distance_map, scale_histogram,
  // spectral_power, uncertainty_bar.
  {
    id: 'ex6-bbbc008',
    title: 'Example 6 — Drosophila dual-channel (BBBC008, GFP + DAPI)',
    description: 'BBBC008 at 0.08 µm/px. gfp_morphology: actin, threshold=0.6, point_cloud. dapi_distance (last declared, runs by default): conservation, full distance + CRLB goal. Six visualise steps.',
    source: `scope dual_channel_drosophila {
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

  rule conservation(dna_mass) {
    invariant: "total DAPI-stained area is conserved across channels"
    epsilon: 0.008
  }

  gfp_morphology = observe(load(db="BBBC", dataset="BBBC008", image="C57_01.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(phase_lock(actin), confidence = 0.75)
    |> access(nucleus_a, threshold = 0.6)
    |> access(nucleus_b, threshold = 0.6)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(point_cloud)

  dapi_distance = observe(load(db="BBBC", dataset="BBBC008", image="C57_01.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(distance_map)
    |> visualise(scale_histogram)
    |> visualise(spectral_power)
    |> visualise(uncertainty_bar)
}`,
  },
];

export function getScopeExample(id: string): ScopeExample | undefined {
  return SCOPE_EXAMPLES.find(e => e.id === id);
}
