# SCOPE Programming Examples

Progressive tutorial examples from simple to complex. Each compiles and runs on real BBBC datasets.

---

## Example 1: Hello World (Synthetic Data)

**Objective**: Verify compilation and execution of a minimal SCOPE program.

```scope
scope hello_world {
    channels {
        sync acquisition at 10.0e6 freq
    }

    coordinate_space {
        field 100.0 x 100.0 µm
        depth 1000
        lambda_s 0.10
        lambda_t 0.05
    }

    morphisms {
        hello =
            observe(synthetic_frame, n=1000)
    }

    dispatch {
        // No timing cells; empty dispatch
    }
}
```

**Execution**: Compiles successfully. Creates synthetic coordinate field, observes it, returns position (50.0, 50.0, 0.0) µm with S-entropy conservation.

---

## Example 2: Real Data — BBBC007 Drosophila

```scope
scope bbbc007_observation {
    channels {
        sync acquisition at 10.0e6 freq
        cell INTERPHASE bounds (-0.8e-6, 0.8e-6) action observe_cells
    }

    coordinate_space {
        field 100.0 x 100.0 µm
        depth 1024
        lambda_s 0.10
        lambda_t 0.05
    }

    morphisms {
        observe_cells =
            observe(bbbc007_image, n=1024)
    }

    dispatch {
        when INTERPHASE do execute(observe_cells)
    }
}
```

**Data**: Real BBBC007 image from `hieronymus/public/datasets/BBBC007_v1_images/A9/A9 p10d.tif`

**Phase 3 MEASURE**: FFT spectral decomposition on real image → dyadic scales → coherence enforcement → coordinate field Φ maps pixels to µm

**Result**: Position based on actual image features, S-entropy from real statistics.

---

## Example 3: Nuclear Separation (Distance Measurement)

```scope
scope nuclear_separation {
    channels {
        sync acquisition at 10.0e6 freq
        cell METAPHASE bounds (-0.8e-6, 0.8e-6) action measure_separation
    }

    coordinate_space {
        field 100.0 x 100.0 µm
        depth 1000
        lambda_s 0.10
        lambda_t 0.05
    }

    morphisms {
        measure_separation =
            observe(dapi_frame, n=1000)
            |> measure_distance(nucleus_a, nucleus_b)
    }

    dispatch {
        when METAPHASE do execute(measure_separation)
    }
}
```

**Key**: `measure_distance()` uses coordinate field Φ to return world-space distance in micrometers.

**Result**: Distance in µm with uncertainty propagated from spectral metric reconstruction.

---

## Example 4: Catalyzed Measurement

```scope
scope constrained_measurement {
    channels {
        sync acquisition at 10.0e6 freq
        cell METAPHASE bounds (-0.8e-6, 0.8e-6) action constrained_separation
    }

    coordinate_space {
        field 100.0 x 100.0 µm
        depth 1000
        lambda_s 0.10
        lambda_t 0.05
    }

    morphisms {
        constrained_separation =
            observe(dapi_frame, n=1000)
            |> catalyze(conservation(dna_mass))
            |> catalyze(phase_lock(chromatin))
            |> measure_distance(nucleus_a, nucleus_b)
    }

    dispatch {
        when METAPHASE do execute(constrained_separation)
    }
}
```

**Key**: `catalyze()` steps reduce categorical distance (increase S_k) by restricting partition to states matching constraints. Result: lower uncertainty, higher precision.

---

## Example 5: Multi-Channel Fusion

```scope
scope fused_measurement {
    channels {
        sync acquisition at 10.0e6 freq
        cell METAPHASE bounds (-0.8e-6, 0.8e-6) action fused_analysis
    }

    coordinate_space {
        field 100.0 x 100.0 µm
        depth 1000
        lambda_s 0.10
        lambda_t 0.05
    }

    morphisms {
        tubulin_observation =
            observe(tubulin_frame, n=1000)
            |> catalyze(phase_lock(microtubules))

        fused_analysis =
            observe(dapi_frame, n=1000)
            |> catalyze(conservation(dna_mass))
            |> fuse(tubulin_observation, rho=0.75)
            |> measure_distance(nucleus_center, spindle_pole)
    }

    dispatch {
        when METAPHASE do execute(fused_analysis)
    }
}
```

**Key**: `fuse(chain, rho)` blends two observations with threshold ρ ∈ [0,1]. Cross-modal measurement improves robustness.

---

## Example 6: Hierarchical Access

```scope
scope hierarchy_analysis {
    channels {
        sync acquisition at 10.0e6 freq
        cell METAPHASE bounds (-0.8e-6, 0.8e-6) action analyze_hierarchy
    }

    coordinate_space {
        field 100.0 x 100.0 µm
        depth 1000
        lambda_s 0.10
        lambda_t 0.05
    }

    morphisms {
        analyze_hierarchy =
            observe(dapi_frame, n=1000)
            |> catalyze(conservation(dna_mass))
            |> catalyze(phase_lock(chromatin))
            |> access(chromosome_pair)
            |> access(centromere_region)
    }

    dispatch {
        when METAPHASE do execute(analyze_hierarchy)
    }
}
```

**Key**: Each `access()` narrows partition space further, increasing S_k and S_e. Final result: position of deepest structure (centromere).

---

## Running in Browser IDE

1. Navigate to `/tools/analysis-studio`
2. Select tutorial from Examples folder
3. Click "Compile & Execute"
4. Watch execution log verify S-entropy conservation
5. See world-space measurements in Result section

---

## Compilation Expectations

✓ All examples compile without type errors  
✓ Partition depths match coordinate_space declaration  
✓ All dispatch rules reference existing morphisms  
✓ S-entropy conservation verified (sum ≈ 1.0 ± 10⁻⁶)  
✓ World-space measurements in micrometers with uncertainty
