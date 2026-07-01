// SCOPE tutorial script library — organised as a progressive IDE file tree.
// Each folder is a lesson; scripts within a folder build on each other.
// Later scripts can reference results from earlier ones (same folder context).

export interface ScopeScript {
  id: string;
  filename: string;           // shown in file tree, e.g. "01_hello_scope.scope"
  title: string;
  description: string;
  source: string;
}

export interface ScopeFolder {
  id: string;
  label: string;              // e.g. "01_basics"
  icon: string;               // emoji or short text
  scripts: ScopeScript[];
}

// ─────────────────────────────────────────────────────────────────────────────
// 01 · BASICS
// ─────────────────────────────────────────────────────────────────────────────
const BASICS: ScopeFolder = {
  id: 'basics',
  label: '01_basics',
  icon: '📐',
  scripts: [
    {
      id: 'b01-hello',
      filename: '01_hello_scope.scope',
      title: 'Hello SCOPE',
      description: 'Minimal script: load one image, observe, visualise the raw image. No measurement, no goals.',
      source: `scope hello_scope {
  coordinate_space {
    field 100 x 100 µm
    depth 4
    lambda_s 0.10
    lambda_t 0.05
  }

  hello = observe(load(db="BBBC", dataset="BBBC007", image="A9 p10d.tif"), n = 4)
    |> visualise(raw_image)
    |> visualise(scale_field)
}`,
    },
    {
      id: 'b02-segments',
      filename: '02_first_segmentation.scope',
      title: 'First segmentation',
      description: 'Add access() to extract nucleus regions and visualise the segmentation mask.',
      source: `scope first_segmentation {
  coordinate_space {
    field 100 x 100 µm
    depth 6
    lambda_s 0.10
    lambda_t 0.05
  }

  segment = observe(load(db="BBBC", dataset="BBBC007", image="A9 p10d.tif"), n = 6)
    |> visualise(scale_field)
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
}`,
    },
    {
      id: 'b03-entropy',
      filename: '03_s_entropy.scope',
      title: 'S-entropy coordinates',
      description: 'Inspect the S-entropy triple (Sk, St, Se) via entropy_trajectory and scale_histogram. S-entropy sums to 1 — check the ✓ in the result bar.',
      source: `scope s_entropy_intro {
  coordinate_space {
    field 100 x 100 µm
    depth 8
    lambda_s 0.10
    lambda_t 0.05
  }

  entropy_demo = observe(load(db="BBBC", dataset="BBBC007", image="A9 p10f.tif"), n = 8)
    |> visualise(scale_field)
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> visualise(entropy_trajectory)
    |> visualise(scale_histogram)
    |> visualise(spectral_power)
}`,
    },
  ],
};

// ─────────────────────────────────────────────────────────────────────────────
// 02 · MEASUREMENT
// ─────────────────────────────────────────────────────────────────────────────
const MEASUREMENT: ScopeFolder = {
  id: 'measurement',
  label: '02_measurement',
  icon: '📏',
  scripts: [
    {
      id: 'm01-nuclear-sep',
      filename: '01_nuclear_separation.scope',
      title: 'Nuclear separation',
      description: 'Canonical HeLa nuclear distance with PROPHASE/ANAPHASE dispatch. Five visualise steps including geodesic and spectral_power.',
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
    {
      id: 'm02-goal-warning',
      filename: '02_goal_warning.scope',
      title: 'Goal warning: depth too low',
      description: 'depth=6 → δd_min ≈ 1.56 µm. Type checker warns distance_uncertainty < 0.1 µm is unreachable. Goal chip shows ✗.',
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
    {
      id: 'm03-uncertainty',
      filename: '03_uncertainty_analysis.scope',
      title: 'Uncertainty analysis',
      description: 'Compare depth=8 vs depth=12 qualitatively. Visualise uncertainty_bar and CRLB in result bar. Shows how more observation steps reduce δd.',
      source: `scope uncertainty_analysis {
  coordinate_space {
    field 100 x 100 µm
    depth 12
    lambda_s 0.10
    lambda_t 0.05
  }

  goal {
    distance_uncertainty < 0.3 µm
    crlb_pixels < 0.15
    snr > 10.0
  }

  rule conservation(dna_mass) {
    invariant: "DAPI mass conserved"
    epsilon: 0.008
  }

  deep_measure = observe(load(db="BBBC", dataset="BBBC007", image="A9 p10d.tif"), n = 12)
    |> visualise(scale_field)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(uncertainty_bar)
    |> visualise(scale_histogram)
    |> visualise(spectral_power)
}`,
    },
  ],
};

// ─────────────────────────────────────────────────────────────────────────────
// 03 · CATALYSTS
// ─────────────────────────────────────────────────────────────────────────────
const CATALYSTS: ScopeFolder = {
  id: 'catalysts',
  label: '03_catalysts',
  icon: '⚗️',
  scripts: [
    {
      id: 'c01-phase-lock',
      filename: '01_phase_lock.scope',
      title: 'Phase lock catalyst',
      description: 'phase_lock(chromatin) stabilises the observation channel. Run with and without to see mask quality change.',
      source: `scope phase_lock_demo {
  coordinate_space {
    field 100 x 100 µm
    depth 10
    lambda_s 0.10
    lambda_t 0.05
  }

  goal {
    distance_uncertainty < 0.4 µm
  }

  with_lock = observe(load(db="BBBC", dataset="BBBC007", image="A9 p7d.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(phase_lock(chromatin), confidence = 0.9)
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(entropy_trajectory)
    |> visualise(spectral_power)
}`,
    },
    {
      id: 'c02-confidence',
      filename: '02_confidence_weights.scope',
      title: 'Confidence-weighted catalysts',
      description: 'Two catalysts at confidence=0.8 and 0.6. Access with threshold=0.7 for tighter masks. Six visualise steps including distance_map.',
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
    {
      id: 'c03-symmetry',
      filename: '03_symmetry_rule.scope',
      title: 'Symmetry conservation rule',
      description: 'Rule symmetry(bilateral) ε=0.006 enforces mitotic axis symmetry. Compare with conservation(dna_mass) — two independent invariants.',
      source: `scope symmetry_conservation {
  coordinate_space {
    field 82 x 82 µm
    depth 10
    lambda_s 0.09
    lambda_t 0.04
  }

  goal {
    distance_uncertainty < 0.25 µm
    snr > 9.0
  }

  rule symmetry(bilateral) {
    invariant: "nucleus pair has bilateral symmetry along division axis"
    epsilon: 0.006
  }

  rule conservation(dna_mass) {
    invariant: "total DAPI area conserved ±5%"
    epsilon: 0.008
  }

  symmetric_pair = observe(load(db="BBBC", dataset="BBBC007", image="A9 p9d.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(symmetry(bilateral), confidence = 0.85)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(distance_map)
    |> visualise(entropy_trajectory)
    |> visualise(scale_histogram)
}`,
    },
  ],
};

// ─────────────────────────────────────────────────────────────────────────────
// 04 · MULTI-CHANNEL
// ─────────────────────────────────────────────────────────────────────────────
const MULTICHANNEL: ScopeFolder = {
  id: 'multichannel',
  label: '04_multichannel',
  icon: '🔀',
  scripts: [
    {
      id: 'mc01-dual',
      filename: '01_dual_channel.scope',
      title: 'Dual-channel DAPI + GFP',
      description: 'f113 two-channel images. gfp_morphology: actin channel → point_cloud. dapi_distance: DAPI → full distance + CRLB goal.',
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
    {
      id: 'mc02-fused',
      filename: '02_fused_morphisms.scope',
      title: 'Fused morphisms: nucleus + membrane',
      description: 'membrane_estimate (fluorescence) fuses into nucleus_primary at rho=0.6 → d_fused, δd_fused. Shows entropy_sphere after fusion.',
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
  ],
};

// ─────────────────────────────────────────────────────────────────────────────
// 05 · DISPATCH
// ─────────────────────────────────────────────────────────────────────────────
const DISPATCH: ScopeFolder = {
  id: 'dispatch',
  label: '05_dispatch',
  icon: '🔀',
  scripts: [
    {
      id: 'd01-two-phase',
      filename: '01_two_phase_dispatch.scope',
      title: 'Two-phase dispatch',
      description: 'PROPHASE → early_nucleus (scale_field only). ANAPHASE → separating_nucleus (full measurement). Simplest dispatch pattern.',
      source: `scope two_phase_dispatch {
  channels {
    sync dapi at 0.1 µm/pixel
    cell PROPHASE bounds (-2.0e-6, -0.8e-6) action early_nucleus
    cell ANAPHASE bounds ( 0.8e-6,  2.0e-6) action separating_nucleus
  }

  coordinate_space {
    field 100 x 100 µm
    depth 10
    lambda_s 0.10
    lambda_t 0.05
  }

  goal {
    distance_uncertainty < 0.5 µm
  }

  rule conservation(dna_mass) {
    invariant: "DAPI mass conserved"
    epsilon: 0.008
  }

  early_nucleus = observe(load(db="BBBC", dataset="BBBC007", image="A9 p7d.tif"), n = 10)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_centroid)
    |> visualise(scale_field)
    |> visualise(segmentation)

  separating_nucleus = observe(load(db="BBBC", dataset="BBBC007", image="A9 p5d.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(geodesic)
    |> visualise(spectral_power)

  dispatch {
    when PROPHASE do execute(early_nucleus)
    when ANAPHASE do execute(separating_nucleus)
  }
}`,
    },
    {
      id: 'd02-three-phase',
      filename: '02_three_phase_dispatch.scope',
      title: 'Three-phase dispatch: ANAPHASE fires',
      description: 'PROPHASE → early_nucleus. METAPHASE → aligned_nucleus (symmetry + partition_tree). ANAPHASE → separating_nucleus (two catalysts, entropy_trajectory).',
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
  ],
};

// ─────────────────────────────────────────────────────────────────────────────
// 06 · CONTACT MAPS  (new — from the bidirectional Dijkstra paper)
// ─────────────────────────────────────────────────────────────────────────────
const CONTACT_MAPS: ScopeFolder = {
  id: 'contact_maps',
  label: '06_contact_maps',
  icon: '🗺️',
  scripts: [
    {
      id: 'cm01-s-entropy',
      filename: '01_s_entropy_coords.scope',
      title: 'S-entropy contact coordinates',
      description: 'Compute (Sk, St, Se) per region. Visualise entropy_sphere and entropy_trajectory to see the 3-component structure of the contact space.',
      source: `scope s_entropy_contact_coords {
  coordinate_space {
    field 100 x 100 µm
    depth 10
    lambda_s 0.10
    lambda_t 0.05
  }

  goal {
    s_entropy_conservation < 1e-12
  }

  // Observe and extract the S-entropy triple for all regions.
  // The entropy_sphere visualises (Sk, St, Se) as a point in [0,1]^3.
  // entropy_trajectory shows how each component evolves through the
  // observation pipeline — the flat sum = 1 line is the conservation law.
  coords = observe(load(db="BBBC", dataset="BBBC007", image="A9 p10f.tif"), n = 10)
    |> visualise(scale_field)
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> visualise(entropy_trajectory)
    |> visualise(entropy_sphere)
    |> visualise(scale_histogram)
}`,
    },
    {
      id: 'cm02-contact-map',
      filename: '02_contact_map.scope',
      title: 'Contact map via SEBD',
      description: 'S-Entropy Bidirectional Dijkstra: cost = Euclidean distance in S-space. Visualise distance_map (the contact cost field) and point_cloud (regions as S-space nodes).',
      source: `scope contact_map_sebd {
  coordinate_space {
    field 100 x 100 µm
    depth 10
    lambda_s 0.10
    lambda_t 0.05
  }

  goal {
    distance_uncertainty < 0.4 µm
    s_entropy_conservation < 1e-12
  }

  rule conservation(dna_mass) {
    invariant: "DAPI mass conserved — resolution floor beta_* >= mu_min > 0"
    epsilon: 0.008
  }

  // The contact map assigns a cost CM(A,B) = d_S(A,B) to each adjacent
  // region pair. measure_distance here computes this cost via SEBD.
  // distance_map shows the cost field. point_cloud shows regions as
  // points in S-space coloured by SEBD cost from nucleus_a.
  contact_map = observe(load(db="BBBC", dataset="BBBC007", image="A9 p10f.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(distance_map)
    |> visualise(point_cloud)
    |> visualise(spectral_power)
    |> visualise(entropy_trajectory)
}`,
    },
    {
      id: 'cm03-slicing',
      filename: '03_holographic_slicing.scope',
      title: 'Contact-driven holographic slicing',
      description: 'Each contact resolution generates residue contacts. The geodesic path through S-space IS the holographic slice sequence. Visualise geodesic + scale_field + distance_tube.',
      source: `scope holographic_slicing {
  coordinate_space {
    field 100 x 100 µm
    depth 12
    lambda_s 0.10
    lambda_t 0.05
  }

  goal {
    distance_uncertainty < 0.3 µm
    s_entropy_conservation < 1e-12
    channel_capacity > 1.5 bits
  }

  rule conservation(dna_mass) {
    invariant: "DAPI mass conserved — non-instantaneity: beta_* >= mu_min > 0"
    epsilon: 0.008
  }

  rule symmetry(bilateral) {
    invariant: "contact is irreducible: contact ≠ centroid proximity"
    epsilon: 0.006
  }

  // Contact-driven slicing: the priority queue orders contacts by SEBD cost.
  // Each slice at depth z = CM(A,B) may spawn residue contacts.
  // geodesic shows the shortest path through S-space (the slice sequence).
  // distance_tube visualises the 3D tube swept by the geodesic.
  // scale_histogram shows the alpha distribution across all slices.
  slicing = observe(load(db="BBBC", dataset="BBBC007", image="A9 p10f.tif"), n = 12)
    |> visualise(scale_field)
    |> catalyze(conservation(dna_mass))
    |> catalyze(symmetry(bilateral), confidence = 0.85)
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(geodesic)
    |> visualise(distance_tube)
    |> visualise(scale_histogram)
    |> visualise(entropy_sphere)
}`,
    },
    {
      id: 'cm04-residue',
      filename: '04_residue_propagation.scope',
      title: 'Residue propagation across images',
      description: 'Multi-image: each image contributes contacts; residue from one propagates to the next. Uses A9 p5f (high contact density, mean residue 3.67) and A9 p7f together.',
      source: `scope residue_propagation {
  channels {
    sync dapi at 0.1 µm/pixel
    cell PROPHASE  bounds (-2.0e-6, -0.8e-6) action dense_contacts
    cell ANAPHASE  bounds ( 0.8e-6,  2.0e-6) action sparse_contacts
  }

  coordinate_space {
    field 100 x 100 µm
    depth 12
    lambda_s 0.10
    lambda_t 0.05
  }

  goal {
    distance_uncertainty < 0.4 µm
    s_entropy_conservation < 1e-12
  }

  rule conservation(dna_mass) {
    invariant: "DAPI mass conserved"
    epsilon: 0.008
  }

  // A9 p5f: 100 regions, 46 contacts — high residue (mean 11.56 per step).
  // Each contact resolution spawns ~11 new contacts (residue chain).
  dense_contacts = observe(load(db="BBBC", dataset="BBBC007", image="A9 p5f.tif"), n = 12)
    |> visualise(scale_field)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(distance_map)
    |> visualise(entropy_trajectory)
    |> visualise(scale_histogram)

  // A9 p7f: 145 regions, 26 contacts — lower residue (mean 1.93 per step).
  sparse_contacts = observe(load(db="BBBC", dataset="BBBC007", image="A9 p7d.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(geodesic)
    |> visualise(spectral_power)

  dispatch {
    when PROPHASE do execute(dense_contacts)
    when ANAPHASE do execute(sparse_contacts)
  }
}`,
    },
  ],
};

// ─────────────────────────────────────────────────────────────────────────────
// Flat list (for backwards-compat with old SCOPE_EXAMPLES consumers)
// ─────────────────────────────────────────────────────────────────────────────
export const SCOPE_FOLDERS: ScopeFolder[] = [
  BASICS,
  MEASUREMENT,
  CATALYSTS,
  MULTICHANNEL,
  DISPATCH,
  CONTACT_MAPS,
];

export const SCOPE_SCRIPTS: ScopeScript[] = SCOPE_FOLDERS.flatMap(f => f.scripts);

export function getScript(id: string): ScopeScript | undefined {
  return SCOPE_SCRIPTS.find(s => s.id === id);
}
