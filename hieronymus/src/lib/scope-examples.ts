// SCOPE Program Examples — minimal and working

export const SCOPE_EXAMPLES = {
  phases: {
    PROPHASE: `scope prophase_analysis {
    channels {
        sync acquisition at 10000000
        cell PROPHASE bounds (-2.0, -0.8) action measure
    }

    coordinate_space {
        field 100.0 x 100.0 um
        depth 1000
        lambda_s 0.10
        lambda_t 0.05
    }

    morphisms {
        measure =
            observe(frame, n=1000)
            |> measure_distance(a, b)
    }

    dispatch {
        when PROPHASE do execute(measure)
    }
}`,

    METAPHASE: `scope metaphase_analysis {
    channels {
        sync acquisition at 10000000
        cell METAPHASE bounds (-0.8, 0.8) action measure
    }

    coordinate_space {
        field 100.0 x 100.0 um
        depth 1000
        lambda_s 0.10
        lambda_t 0.05
    }

    morphisms {
        measure =
            observe(frame, n=1000)
            |> measure_distance(a, b)
    }

    dispatch {
        when METAPHASE do execute(measure)
    }
}`,

    ANAPHASE: `scope anaphase_analysis {
    channels {
        sync acquisition at 10000000
        cell ANAPHASE bounds (0.8, 2.0) action measure
    }

    coordinate_space {
        field 100.0 x 100.0 um
        depth 1000
        lambda_s 0.10
        lambda_t 0.05
    }

    morphisms {
        measure =
            observe(frame, n=1000)
            |> measure_distance(a, b)
    }

    dispatch {
        when ANAPHASE do execute(measure)
    }
}`,
  },

  dataSources: {
    synthetic: `scope synthetic_analysis {
    channels {
        sync acquisition at 10000000
        cell METAPHASE bounds (-0.8, 0.8) action measure
    }

    coordinate_space {
        field 100.0 x 100.0 um
        depth 1000
        lambda_s 0.10
        lambda_t 0.05
    }

    morphisms {
        measure =
            observe(frame, n=1000)
            |> measure_distance(a, b)
    }

    dispatch {
        when METAPHASE do execute(measure)
    }
}`,

    microscopy: `scope microscopy_analysis {
    channels {
        sync acquisition at 10000000
        cell METAPHASE bounds (-0.8, 0.8) action measure
    }

    coordinate_space {
        field 100.0 x 100.0 um
        depth 1000
        lambda_s 0.10
        lambda_t 0.05
    }

    morphisms {
        measure =
            observe(frame, n=1000)
            |> measure_distance(a, b)
    }

    dispatch {
        when METAPHASE do execute(measure)
    }
}`,

    huggingface: `scope huggingface_analysis {
    channels {
        sync acquisition at 10000000
        cell METAPHASE bounds (-0.8, 0.8) action model
    }

    coordinate_space {
        field 100.0 x 100.0 um
        depth 1000
        lambda_s 0.10
        lambda_t 0.05
    }

    morphisms {
        model =
            observe(frame, n=1000)
            |> measure_distance(a, b)
    }

    dispatch {
        when METAPHASE do execute(model)
    }
}`,

    reactome: `scope reactome_analysis {
    channels {
        sync acquisition at 10000000
        cell METAPHASE bounds (-0.8, 0.8) action pathway
    }

    coordinate_space {
        field 100.0 x 100.0 um
        depth 1000
        lambda_s 0.10
        lambda_t 0.05
    }

    morphisms {
        pathway =
            observe(frame, n=1000)
            |> measure_distance(a, b)
    }

    dispatch {
        when METAPHASE do execute(pathway)
    }
}`,
  },
};

export function getScopeExample(
  phase: 'PROPHASE' | 'METAPHASE' | 'ANAPHASE',
  dataSource: 'synthetic' | 'microscopy' | 'huggingface' | 'reactome'
): string {
  return SCOPE_EXAMPLES.phases[phase];
}

export function getExampleDescription(
  phase: 'PROPHASE' | 'METAPHASE' | 'ANAPHASE',
  dataSource: 'synthetic' | 'microscopy' | 'huggingface' | 'reactome'
): string {
  const phaseDescriptions: Record<typeof phase, string> = {
    PROPHASE: 'Early cell cycle - nuclei condensing',
    METAPHASE: 'Middle cell cycle - nuclei aligned',
    ANAPHASE: 'Late cell cycle - chromatids separating',
  };

  return `Phase: ${phaseDescriptions[phase]}`;
}
