// SCOPE example programs — each script produces a rich set of visual outputs.
// All examples use locally-available BBBC007 images so no network fetch is needed.

export interface ScopeExample {
  id: string;
  title: string;
  description: string;
  source: string;
}

export const SCOPE_EXAMPLES: ScopeExample[] = [
  // ── Example 1 ────────────────────────────────────────────────────────────────
  // Canonical nuclear separation. PROPHASE/ANAPHASE → nucleus_pair_measurement.
  // METAPHASE → membrane_boundary. Images: BBBC007 A9 DAPI (p10d) + fluorescence (p10f).
  {
    id: 'ex1-nuclear-separation',
    title: 'Example 1 — Nuclear separation (BBBC007, HeLa A9)',
    description: 'Canonical HeLa nuclear distance. PROPHASE/ANAPHASE dispatch → nucleus_pair_measurement. METAPHASE → membrane_boundary. Five visualise steps: scale_field, segmentation, spectral_power, entropy_trajectory, geodesic.',
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

  nucleus_pair_measurement = observe(load(db="BBBC", dataset="BBBC007", image="A9 p10d.tif"), n = 10)
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

  membrane_boundary = observe(load(db="BBBC", dataset="BBBC007", image="A9 p10f.tif"), n = 10)
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
  // Goal warning: depth=6 makes δd_min ≈ 1.56 µm → cannot satisfy < 0.1 µm.
  // Type-checker emits GoalUnreachableAtDepth warning; goal chip shows ✗.
  {
    id: 'ex2-goal-warning',
    title: 'Example 2 — Goal warning: depth too low (BBBC007, A9)',
    description: 'depth=6 → δd_min ≈ 1.56 µm. Type checker warns that distance_uncertainty < 0.1 µm is unreachable. Goal chip shows ✗. Four visualise steps: scale_field, segmentation, uncertainty_bar, spectral_power.',
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

  rule conservation(dna_mass) {
    invariant: "total DAPI-stained area is conserved ±5%"
    epsilon: 0.008
  }

  measure = observe(load(db="BBBC", dataset="BBBC007", image="A9 p9d.tif"), n = 6)
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
  // Confidence-weighted catalysts + fuzzy access threshold.
  // BBBC007 f96(17) HeLa cells. Symmetry rule bilateral, threshold=0.7 for tighter masks.
  {
    id: 'ex3-confidence-threshold',
    title: 'Example 3 — Confidence-weighted catalysts + fuzzy threshold (BBBC007, f96)',
    description: 'HeLa f96(17) images. Rule: symmetry(bilateral) ε=0.006. Confidence 0.8/0.6 on catalysts. Access threshold=0.7 → tighter masks. Six visualise steps: scale_field, segmentation, distance_map, spectral_power, scale_histogram, entropy_trajectory.',
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

  spindle_axis = observe(load(db="BBBC", dataset="BBBC007", image="17P1_POS0006_D_1UL.tif"), n = 10)
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
  // Fused morphisms: membrane_estimate (BBBC007 f9620 fluorescence) + nucleus_primary (DAPI).
  // nucleus_primary fuses with membrane_estimate at rho=0.6 → combined d, δd.
  {
    id: 'ex4-fused',
    title: 'Example 4 — Fused morphisms: nucleus + membrane (BBBC007, f9620)',
    description: 'membrane_estimate (fluorescence) measures cell_boundary→nucleus_centroid. nucleus_primary (DAPI) fuses at rho=0.6 → d_fused, δd_fused. Visualise: scale_field, segmentation, spectral_power, entropy_sphere.',
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

  membrane_estimate = observe(load(db="BBBC", dataset="BBBC007", image="20P1_POS0002_F_2UL.tif"), n = 10)
    |> catalyze(phase_lock(actin))
    |> access(cell_boundary)
    |> access(nucleus_centroid)
    |> measure_distance(cell_boundary, nucleus_centroid)

  nucleus_primary = observe(load(db="BBBC", dataset="BBBC007", image="20P1_POS0002_D_1UL.tif"), n = 10)
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
  // Three-cell dispatch, ANAPHASE fires. BBBC007 A9 p7d/p9d/p5d.
  // early_nucleus (PROPHASE): single centroid, segmentation only.
  // aligned_nucleus (METAPHASE): bilateral symmetry, partition_tree.
  // separating_nucleus (ANAPHASE): two catalysts, geodesic, entropy_trajectory.
  {
    id: 'ex5-anaphase',
    title: 'Example 5 — Three-cell dispatch, ANAPHASE fires (BBBC007, A9 series)',
    description: 'PROPHASE → early_nucleus. METAPHASE → aligned_nucleus (symmetry + partition_tree). ANAPHASE → separating_nucleus (two catalysts, distance, entropy_trajectory). Different A9 images per phase.',
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

  early_nucleus = observe(load(db="BBBC", dataset="BBBC007", image="A9 p7d.tif"), n = 10)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_centroid)
    |> visualise(scale_field)
    |> visualise(segmentation)

  aligned_nucleus = observe(load(db="BBBC", dataset="BBBC007", image="A9 p9d.tif"), n = 10)
    |> catalyze(symmetry(bilateral))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(partition_tree)

  separating_nucleus = observe(load(db="BBBC", dataset="BBBC007", image="A9 p5d.tif"), n = 10)
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
  // Dual-channel BBBC007: DAPI (d0) + fluorescence (f2UL).
  // gfp_morphology: actin channel, point_cloud.
  // dapi_distance: conservation, full distance + CRLB goal.
  {
    id: 'ex6-dual-channel',
    title: 'Example 6 — Dual-channel DAPI + fluorescence (BBBC007, f113)',
    description: 'f113 two-channel images. gfp_morphology: fluorescence d1 → actin, threshold=0.6, point_cloud. dapi_distance: DAPI d0, conservation, full distance + CRLB goal. Six visualise steps.',
    source: `scope dual_channel_hela {
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

  gfp_morphology = observe(load(db="BBBC", dataset="BBBC007", image="AS_09125_040701150004_A02f00d1.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(phase_lock(actin), confidence = 0.75)
    |> access(nucleus_a, threshold = 0.6)
    |> access(nucleus_b, threshold = 0.6)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(point_cloud)

  dapi_distance = observe(load(db="BBBC", dataset="BBBC007", image="AS_09125_040701150004_A02f00d0.tif"), n = 10)
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
