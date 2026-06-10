// SCOPE example programs — matches examples.md exactly

export interface ScopeExample {
  id: string;
  title: string;
  description: string;
  source: string;
}

export const SCOPE_EXAMPLES: ScopeExample[] = [
  {
    id: 'ex1-nuclear-separation',
    title: 'Example 1 — Nuclear separation (BBBC039)',
    description: 'Canonical HeLa cell nuclear separation with timing dispatch and goal block.',
    source: `scope nuclear_separation_dynamics {
  channels {
    sync dapi at 0.1 µm/pixel
    cell PROPHASE  bounds (-2.0e-6, -0.8e-6) action nucleus_pair_measurement
    cell METAPHASE bounds (-0.8e-6,  0.8e-6) action nucleus_pair_measurement
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
    invariant: "total DAPI-stained area is conserved"
    epsilon: 0.008
  }

  nucleus_pair_measurement = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_001.tif"), n = 10)
    |> catalyze(conservation(dna_mass))
    |> catalyze(phase_lock(chromatin), confidence = 0.9)
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(geodesic)

  dispatch {
    when PROPHASE  do execute(nucleus_pair_measurement)
    when METAPHASE do execute(nucleus_pair_measurement)
    when ANAPHASE  do execute(nucleus_pair_measurement)
  }
}`,
  },
  {
    id: 'ex2-goal-warning',
    title: 'Example 2 — Goal warning (depth too low)',
    description: 'Type checker warns that depth=6 cannot satisfy δd < 0.1 µm.',
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
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(uncertainty_bar)
}`,
  },
  {
    id: 'ex3-confidence-threshold',
    title: 'Example 3 — Confidence-weighted catalysts (BBBC006)',
    description: 'CHO cells with confidence on catalysts and tighter access threshold.',
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
    invariant: "mitotic spindle has bilateral symmetry"
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
}`,
  },
  {
    id: 'ex4-fused',
    title: 'Example 4 — Fused morphisms',
    description: 'Two morphisms combined with fuse(rho). Shows entropy sphere.',
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

  membrane_estimate = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_002.tif"), n = 10)
    |> catalyze(phase_lock(actin))
    |> access(cell_boundary)
    |> access(nucleus_a)
    |> measure_distance(cell_boundary, nucleus_a)

  nucleus_primary = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_002.tif"), n = 10)
    |> catalyze(conservation(dna_mass))
    |> catalyze(phase_lock(chromatin), confidence = 0.85)
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> measure_distance(nucleus_a, nucleus_b)
    |> fuse(membrane_estimate, rho = 0.6)
    |> visualise(entropy_sphere)
}`,
  },
  {
    id: 'ex5-anaphase',
    title: 'Example 5 — Three-cell dispatch (BBBC039)',
    description: 'Full dispatch block — anaphase fires separating_nucleus morphism.',
    source: `scope anaphase_tracking {
  channels {
    sync dapi at 0.1 µm/pixel
    cell PROPHASE  bounds (-2.0e-6, -0.8e-6) action early_nucleus
    cell METAPHASE bounds (-0.8e-6,  0.8e-6) action early_nucleus
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
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(segmentation)

  separating_nucleus = observe(load(db="BBBC", dataset="BBBC039", image="SiR_Actin_003.tif"), n = 10)
    |> catalyze(conservation(dna_mass))
    |> catalyze(phase_lock(chromatin))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(entropy_trajectory)

  dispatch {
    when PROPHASE  do execute(early_nucleus)
    when METAPHASE do execute(early_nucleus)
    when ANAPHASE  do execute(separating_nucleus)
  }
}`,
  },
  {
    id: 'ex6-bbbc008',
    title: 'Example 6 — Drosophila dual-channel (BBBC008)',
    description: 'BBBC008 at 0.08 µm/px with CRLB goal.',
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

  dapi_distance = observe(load(db="BBBC", dataset="BBBC008", image="C57_01.tif"), n = 10)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(uncertainty_bar)
}`,
  },
];

export function getScopeExample(id: string): ScopeExample | undefined {
  return SCOPE_EXAMPLES.find(e => e.id === id);
}
