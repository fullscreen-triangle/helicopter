"""
Categorical completion operations.

Generates new BMD states through comparison with image regions.
Implements the BMD operation: β_{i+1} = Generate(β_i, R)
"""

import numpy as np
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..vision.bmd.bmd_state import BMDState, OscillatoryHole, PhaseStructure
    from ..regions.region import Region


class CategoricalCompletion:
    """
    Perform categorical completion operations.

    Completes oscillatory holes in BMD states by selecting one
    weak force configuration from ~10^6 equivalent possibilities,
    constrained by image region R.
    """

    def __init__(self, lambda_coupling: float = 1.0):
        """
        Initialize categorical completion engine.

        Args:
            lambda_coupling: Balance between energetic and informational costs
                           in configuration selection
        """
        self.lambda_coupling = lambda_coupling
        self.kB = 1.380649e-23  # Boltzmann constant
        self.T = 310.0  # Body temperature (K)

    def generate_bmd(
        self,
        current_bmd: 'BMDState',
        region: 'Region'
    ) -> 'BMDState':
        """
        Generate new BMD through categorical completion.

        β_{i+1} = Generate(β_i, R)

        Process:
        1. Extract oscillatory holes from β_i
        2. Extract constraints from region R
        3. Select completion configuration minimizing:
           E_fill(c_current → c) + λ · A(β_c, R)
        4. Construct new BMD state β_{i+1}

        Args:
            current_bmd: Current BMD state β_i
            region: Image region R providing constraints

        Returns:
            New BMD state β_{i+1}
        """
        from ..vision.bmd.bmd_state import BMDState, OscillatoryHole, PhaseStructure

        # Extract holes and constraints
        holes = current_bmd.holes
        region_constraints = self._extract_region_constraints(region)

        # Generate new holes for next step
        # (Completing one hole typically creates new ones)
        new_holes = self._propagate_holes(holes, region_constraints)

        # Select completion configuration
        config = self.select_completion_configuration(
            holes,
            region_constraints
        )

        # Update phase structure based on completion
        new_phase = self._update_phase_structure(
            current_bmd.phase,
            config,
            region
        )

        # New categorical state
        new_categorical_state = self._derive_categorical_state(
            current_bmd.c_current,
            region,
            config
        )

        # Construct new BMD
        new_bmd = BMDState(
            categorical_state=new_categorical_state,
            oscillatory_holes=new_holes,
            phase_structure=new_phase,
            metadata={
                'parent': current_bmd.c_current,
                'region': region.id,
                'config': config['id']
            }
        )

        return new_bmd

    def select_completion_configuration(
        self,
        oscillatory_holes: list,
        region_constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Select one weak force configuration from ~10^6 possibilities.

        c_new = argmin_{c ∈ C(R)} [E_fill(c_current → c) + λ · A(β_c, R)]

        This is THE fundamental BMD operation: selecting one categorical
        state from vast equivalence class.

        Args:
            oscillatory_holes: Holes requiring completion
            region_constraints: Constraints from image region

        Returns:
            Selected configuration containing:
            - Van der Waals angles
            - Dipole orientations
            - Vibrational phases
            - Rotational offsets
        """
        # Sample from ~10^6 possible configurations
        n_samples = min(1000, region_constraints.get('n_possible_configs', 1000))

        best_config = None
        best_cost = float('inf')

        for i in range(n_samples):
            # Generate candidate configuration
            config = self._sample_configuration(oscillatory_holes, region_constraints)

            # Compute cost
            E_fill = self._filling_energy(oscillatory_holes, config)
            A = self._config_ambiguity(config, region_constraints)

            cost = E_fill + self.lambda_coupling * A

            if cost < best_cost:
                best_cost = cost
                best_config = config

        return best_config

    def _extract_region_constraints(
        self,
        region: 'Region'
    ) -> Dict[str, Any]:
        """
        Extract constraints from image region.

        Constraints include:
        - Color distribution (constrains electron configurations)
        - Texture patterns (constrains vibrational modes)
        - Edge orientations (constrains dipole alignments)
        - Spatial structure (constrains Van der Waals networks)

        Returns:
            Dict of constraint specifications
        """
        if region.features is None:
            region.extract_features()

        features = region.features

        constraints = {
            'color': features.get('color_histogram', np.ones(8)),
            'texture': features.get('texture_features', np.ones(8)),
            'edges': features.get('edge_features', np.ones(8)),
            'moments': features.get('spatial_moments', np.ones(4)),
            'n_possible_configs': int(1e6)  # ~10^6 equivalence class size
        }

        return constraints

    def _propagate_holes(
        self,
        current_holes: list,
        region_constraints: Dict[str, Any]
    ) -> list:
        """
        Generate new oscillatory holes from completion.

        Completing one hole typically creates new holes downstream
        in the oscillatory cascade.

        Returns:
            List of new oscillatory holes
        """
        from ..vision.bmd.bmd_state import OscillatoryHole

        new_holes = []

        # Each current hole spawns ~2-3 new holes
        for hole in current_holes:
            n_spawn = np.random.randint(2, 4)

            for _ in range(n_spawn):
                # New hole at related frequency
                new_freq = hole.required_frequency * np.random.uniform(0.8, 1.2)
                new_phase = (hole.required_phase + np.random.uniform(0, 2*np.pi)) % (2*np.pi)
                new_amp = hole.required_amplitude * np.random.uniform(0.9, 1.1)

                new_hole = OscillatoryHole(
                    required_frequency=new_freq,
                    required_phase=new_phase,
                    required_amplitude=new_amp,
                    n_configurations=int(1e6)  # ~10^6 possibilities per hole
                )
                new_holes.append(new_hole)

        # Limit total holes
        if len(new_holes) > 100:
            # Keep highest amplitude holes
            new_holes.sort(key=lambda h: h.required_amplitude, reverse=True)
            new_holes = new_holes[:100]

        return new_holes

    def _update_phase_structure(
        self,
        current_phase: 'PhaseStructure',
        config: Dict[str, Any],
        region: 'Region'
    ) -> 'PhaseStructure':
        """
        Update phase structure based on completion configuration.

        Returns:
            Updated PhaseStructure
        """
        from ..vision.bmd.bmd_state import PhaseStructure

        # Update phases based on configuration
        new_phases = current_phase.phases.copy()
        new_phases += config['phase_shifts']
        new_phases = new_phases % (2*np.pi)

        # Update frequencies (typically stable)
        new_frequencies = current_phase.frequencies.copy()

        # Update coherence (region processing affects phase-locking)
        new_coherence = current_phase.coherence.copy()

        # Slightly increase coherence (processing adds constraints)
        coherence_boost = np.random.uniform(1.0, 1.05)
        new_coherence = np.minimum(new_coherence * coherence_boost, 1.0)

        return PhaseStructure(
            phases=new_phases,
            frequencies=new_frequencies,
            coherence=new_coherence
        )

    def _derive_categorical_state(
        self,
        current_state: Any,
        region: 'Region',
        config: Dict[str, Any]
    ) -> Any:
        """
        Derive new categorical state from completion.

        Returns:
            New categorical state identifier
        """
        # Combine current state, region, and configuration
        state_id = f"{current_state}_{region.id}_{config['id']}"
        return state_id

    def _sample_configuration(
        self,
        holes: list,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sample one configuration from equivalence class.

        Returns:
            Configuration dict
        """
        n_modes = len(holes) if holes else 10

        config = {
            'id': np.random.randint(0, int(1e6)),
            'van_der_waals_angles': np.random.uniform(0, 2*np.pi, n_modes),
            'dipole_orientations': np.random.uniform(0, 2*np.pi, n_modes),
            'vibrational_phases': np.random.uniform(0, 2*np.pi, n_modes),
            'rotational_offsets': np.random.uniform(0, 2*np.pi, n_modes),
            'phase_shifts': np.random.uniform(-np.pi/4, np.pi/4, n_modes)
        }

        return config

    def _filling_energy(
        self,
        holes: list,
        config: Dict[str, Any]
    ) -> float:
        """
        Compute energy cost of filling holes with configuration.

        E_fill = Σ |Ω_hole - Ω_config|²

        Returns:
            Energy cost (lower = better match)
        """
        if not holes:
            return 0.0

        energy = 0.0

        for i, hole in enumerate(holes):
            # Phase mismatch
            if i < len(config['vibrational_phases']):
                phase_diff = hole.required_phase - config['vibrational_phases'][i]
                phase_diff = np.minimum(
                    np.abs(phase_diff),
                    2*np.pi - np.abs(phase_diff)
                )
                energy += phase_diff ** 2

        return energy / len(holes)

    def _config_ambiguity(
        self,
        config: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> float:
        """
        Compute ambiguity of configuration given constraints.

        Returns:
            Ambiguity value (higher = more possibilities remain)
        """
        # Measure how well configuration matches constraints
        color_match = self._match_score(
            config['dipole_orientations'],
            constraints['color']
        )

        texture_match = self._match_score(
            config['vibrational_phases'],
            constraints['texture']
        )

        # Lower match = higher ambiguity = more possibilities
        ambiguity = 2.0 - (color_match + texture_match)

        return max(0.0, ambiguity)

    def _match_score(
        self,
        values: np.ndarray,
        constraint: np.ndarray
    ) -> float:
        """
        Compute match score between values and constraint distribution.

        Returns:
            Match score in [0, 1]
        """
        # Create histogram of values
        hist, _ = np.histogram(
            values,
            bins=len(constraint),
            range=(0, 2*np.pi),
            density=True
        )

        # Compute correlation
        correlation = np.corrcoef(hist, constraint)[0, 1]

        return max(0.0, correlation)
