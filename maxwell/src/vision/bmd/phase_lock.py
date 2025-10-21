"""
Phase-lock coupling operations for BMD composition.

Implements the ⊛ operator for hierarchical BMD composition:
β₁ ⊛ β₂ = Hierarchical composition through phase-lock coupling
"""

import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .bmd_state import BMDState, PhaseStructure, OscillatoryHole


class PhaseLockCoupling:
    """
    Phase-lock coupling for BMD composition.

    The ⊛ operator composes BMD states through phase-lock relationships,
    creating compound BMD that inherits structure from both parents.

    Key insight: Hierarchical composition (not tensor product)
    β₁ ⊛ β₂ ≠ β₁ ⊗ β₂

    Instead: β₁ ⊛ β₂ = compound BMD with nested structure
    """

    def __init__(self, coupling_strength: float = 1.0):
        """
        Initialize phase-lock coupling.

        Args:
            coupling_strength: Default coupling strength [0, 1]
        """
        self.coupling_strength = coupling_strength

    def compose(
        self,
        bmd1: 'BMDState',
        bmd2: 'BMDState',
        coupling: float = None
    ) -> 'BMDState':
        """
        Compose two BMD states: β₁ ⊛ β₂

        Creates compound BMD with:
        - Merged phase structures
        - Combined oscillatory holes
        - Increased categorical richness
        - New compound categorical state

        Args:
            bmd1: First BMD state
            bmd2: Second BMD state
            coupling: Override default coupling strength

        Returns:
            Compound BMD state
        """
        from .bmd_state import BMDState, PhaseStructure, OscillatoryHole

        if coupling is None:
            coupling = self.coupling_strength

        # Compose phase structures
        compound_phase = self._compose_phase_structures(
            bmd1.phase,
            bmd2.phase,
            coupling
        )

        # Merge oscillatory holes
        compound_holes = self._merge_holes(
            bmd1.holes,
            bmd2.holes
        )

        # Create compound categorical state
        compound_state = self._compose_categorical_states(
            bmd1.c_current,
            bmd2.c_current
        )

        # Create compound BMD
        compound_bmd = BMDState(
            categorical_state=compound_state,
            oscillatory_holes=compound_holes,
            phase_structure=compound_phase,
            metadata={
                'composition_type': 'hierarchical',
                'parent1': str(bmd1.c_current),
                'parent2': str(bmd2.c_current),
                'coupling': coupling
            }
        )

        return compound_bmd

    def compose_sequence(
        self,
        bmds: List['BMDState'],
        coupling: float = None
    ) -> 'BMDState':
        """
        Compose sequence of BMDs: β₁ ⊛ β₂ ⊛ ... ⊛ βₙ

        Hierarchical left-to-right composition.

        Args:
            bmds: List of BMD states
            coupling: Override coupling strength

        Returns:
            Final compound BMD
        """
        if not bmds:
            raise ValueError("Cannot compose empty BMD sequence")

        if len(bmds) == 1:
            return bmds[0]

        # Left-to-right composition
        result = bmds[0]
        for bmd in bmds[1:]:
            result = self.compose(result, bmd, coupling)

        return result

    def _compose_phase_structures(
        self,
        phase1: 'PhaseStructure',
        phase2: 'PhaseStructure',
        coupling: float
    ) -> 'PhaseStructure':
        """
        Compose two phase structures through coupling.

        Creates compound phase structure with:
        - Combined modes from both structures
        - Cross-coupling between structures
        - Enhanced coherence from coupling

        Returns:
            Compound PhaseStructure
        """
        from .bmd_state import PhaseStructure

        # Concatenate phases
        phases = np.concatenate([phase1.phases, phase2.phases])
        frequencies = np.concatenate([phase1.frequencies, phase2.frequencies])

        # Build compound coherence matrix
        n1 = len(phase1.phases)
        n2 = len(phase2.phases)
        n_total = n1 + n2

        coherence = np.zeros((n_total, n_total))

        # Block 1: phase1 internal coherence
        coherence[:n1, :n1] = phase1.coherence

        # Block 2: phase2 internal coherence
        coherence[n1:, n1:] = phase2.coherence

        # Cross-blocks: coupling between structures
        cross_coupling = coupling * 0.8  # Slightly weaker cross-coupling
        coherence[:n1, n1:] = cross_coupling
        coherence[n1:, :n1] = cross_coupling

        # Diagonal perfect coherence
        np.fill_diagonal(coherence, 1.0)

        return PhaseStructure(
            phases=phases,
            frequencies=frequencies,
            coherence=coherence
        )

    def _merge_holes(
        self,
        holes1: List['OscillatoryHole'],
        holes2: List['OscillatoryHole']
    ) -> List['OscillatoryHole']:
        """
        Merge oscillatory holes from two BMDs.

        Combines holes and creates interaction holes at boundaries.

        Returns:
            Merged list of oscillatory holes
        """
        from .bmd_state import OscillatoryHole

        # Start with all holes from both
        merged = holes1.copy() + holes2.copy()

        # Create interaction holes between structures
        if holes1 and holes2:
            # Sample a few interaction holes
            n_interaction = min(3, len(holes1), len(holes2))

            for i in range(n_interaction):
                h1 = holes1[np.random.randint(len(holes1))]
                h2 = holes2[np.random.randint(len(holes2))]

                # Interaction hole at combined frequency
                interaction_hole = OscillatoryHole(
                    required_frequency=(h1.required_frequency + h2.required_frequency) / 2,
                    required_phase=(h1.required_phase + h2.required_phase) % (2*np.pi),
                    required_amplitude=(h1.required_amplitude + h2.required_amplitude) / 2,
                    n_configurations=int(np.sqrt(h1.n_configurations * h2.n_configurations))
                )
                merged.append(interaction_hole)

        # Limit total holes
        if len(merged) > 100:
            # Keep highest amplitude
            merged.sort(key=lambda h: h.required_amplitude, reverse=True)
            merged = merged[:100]

        return merged

    def _compose_categorical_states(
        self,
        state1: any,
        state2: any
    ) -> any:
        """
        Compose categorical states hierarchically.

        Returns:
            Compound categorical state identifier
        """
        # Create compound identifier
        return f"({state1} ⊛ {state2})"

    def coupling_strength_from_coherence(
        self,
        bmd1: 'BMDState',
        bmd2: 'BMDState'
    ) -> float:
        """
        Estimate optimal coupling strength from phase coherences.

        Returns:
            Suggested coupling strength [0, 1]
        """
        # Average phase coherences
        q1 = bmd1.phase_lock_quality()
        q2 = bmd2.phase_lock_quality()

        # Coupling should be strong if both are coherent
        coupling = (q1 + q2) / 2

        return coupling

    def decompose(
        self,
        compound_bmd: 'BMDState'
    ) -> tuple:
        """
        Attempt to decompose compound BMD into components.

        This is non-trivial due to irreducibility of BMDs.

        Returns:
            Tuple of (success: bool, components: List[BMDState])
        """
        # Check metadata for composition info
        if 'parent1' not in compound_bmd.metadata:
            return (False, [compound_bmd])

        # BMDs are irreducible - cannot truly decompose
        # Can only identify parent structures
        return (True, [])  # Placeholder

    def hierarchical_distance(
        self,
        bmd1: 'BMDState',
        bmd2: 'BMDState'
    ) -> float:
        """
        Compute hierarchical distance between BMDs.

        Based on phase structure differences.

        Returns:
            Distance metric (lower = more similar)
        """
        # Phase angle distances
        n_min = min(len(bmd1.phase.phases), len(bmd2.phase.phases))

        if n_min == 0:
            return float('inf')

        phase_diff = np.mean(np.abs(
            bmd1.phase.phases[:n_min] - bmd2.phase.phases[:n_min]
        ))

        # Frequency distances
        freq_diff = np.mean(np.abs(
            np.log(bmd1.phase.frequencies[:n_min] + 1e-10) -
            np.log(bmd2.phase.frequencies[:n_min] + 1e-10)
        ))

        # Coherence difference
        coherence_diff = np.abs(
            bmd1.phase_lock_quality() - bmd2.phase_lock_quality()
        )

        # Combined distance
        distance = phase_diff + freq_diff + coherence_diff

        return distance
