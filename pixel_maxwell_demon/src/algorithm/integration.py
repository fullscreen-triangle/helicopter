"""
Hierarchical BMD integration operations.

Implements: β^(network)_{i+1} = IntegrateHierarchical(β^(network)_i, β_{i+1}, σ ∪ R)
"""

import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..vision.bmd import NetworkBMD, BMDState


class HierarchicalIntegration:
    """
    Hierarchical integration of new BMDs into network.

    Process:
    1. Add region BMD to network
    2. Generate pairwise compounds with recent regions
    3. Generate higher-order compounds up to max_order
    4. Propagate constraints hierarchically
    5. Update global network BMD
    """

    def __init__(self, max_compound_order: int = 5):
        """
        Initialize hierarchical integrator.

        Args:
            max_compound_order: Maximum order of compound BMDs
        """
        self.max_order = max_compound_order

    def integrate(
        self,
        network_bmd: 'NetworkBMD',
        new_region_bmd: 'BMDState',
        region_id: str,
        processing_sequence: List[str]
    ) -> 'NetworkBMD':
        """
        Hierarchically integrate new region BMD into network.

        This is the core operation that builds the network BMD structure.

        Args:
            network_bmd: Current network BMD
            new_region_bmd: New region BMD to integrate
            region_id: Region identifier
            processing_sequence: Current processing sequence

        Returns:
            Updated NetworkBMD (modified in place)
        """
        # Get current processing step
        processing_step = len(processing_sequence)

        # Integrate region BMD
        # (NetworkBMD.integrate_region_bmd handles all hierarchical logic)
        network_bmd.integrate_region_bmd(
            region_id=region_id,
            region_bmd=new_region_bmd,
            processing_step=processing_step
        )

        return network_bmd

    def hierarchical_composition(
        self,
        bmd_sequence: List['BMDState'],
        coupling_strength: float = 0.9
    ) -> 'BMDState':
        """
        Compose multiple BMDs hierarchically: β₁ ⊛ β₂ ⊛ ... ⊛ βₙ

        Uses phase-lock coupling to compose BMD states.

        Args:
            bmd_sequence: Sequence of BMDs to compose
            coupling_strength: Phase-lock coupling strength

        Returns:
            Composed BMD state
        """
        from ..vision.bmd import PhaseLockCoupling

        if not bmd_sequence:
            raise ValueError("Cannot compose empty BMD sequence")

        if len(bmd_sequence) == 1:
            return bmd_sequence[0]

        # Create phase-lock coupling engine
        phase_lock = PhaseLockCoupling(coupling_strength=coupling_strength)

        # Compose sequence
        result = phase_lock.compose_sequence(bmd_sequence)

        return result

    def compute_integration_cost(
        self,
        network_bmd: 'NetworkBMD',
        new_region_bmd: 'BMDState'
    ) -> float:
        """
        Compute cost of integrating new BMD into network.

        Cost = kT log(R_before / R_after)

        Higher cost = more information reduction = more specific completion

        Args:
            network_bmd: Current network BMD
            new_region_bmd: Candidate new BMD

        Returns:
            Integration cost (energy units)
        """
        kB = 1.380649e-23  # Boltzmann constant
        T = 310.0  # Body temperature

        # Current network richness
        R_before = network_bmd.network_categorical_richness()

        # Predicted richness after integration
        # (Approximate by adding new BMD richness)
        R_new = new_region_bmd.categorical_richness()
        R_after = R_before * (1 + np.log1p(R_new))

        # Cost
        if R_after > R_before:
            # Richness increased (typical case)
            cost = kB * T * np.log(R_after / R_before)
        else:
            # Richness decreased (unusual)
            cost = -kB * T * np.log(R_before / (R_after + 1e-10))

        return cost

    def integration_impact_analysis(
        self,
        network_bmd: 'NetworkBMD',
        new_region_bmd: 'BMDState',
        region_id: str
    ) -> dict:
        """
        Analyze impact of integrating new BMD.

        Returns:
            Dict with impact metrics:
            - richness_change: Change in network richness
            - phase_quality_change: Change in phase-lock quality
            - n_new_compounds: Number of new compounds generated
            - integration_cost: Energy cost
        """
        # Before state
        R_before = network_bmd.network_categorical_richness()
        Q_before = network_bmd.global_bmd.phase_lock_quality()
        n_compounds_before = len(network_bmd.compound_bmds)

        # After state (simulated)
        R_new = new_region_bmd.categorical_richness()
        Q_new = new_region_bmd.phase_lock_quality()

        # Predicted changes
        richness_change = np.log1p(R_new)  # Logarithmic growth
        phase_quality_change = (Q_before + Q_new) / 2 - Q_before  # Averaging effect

        # New compounds estimation
        recent_regions = len(network_bmd.get_recent_regions(n=10))
        n_new_compounds_est = recent_regions  # At least pairwise compounds

        # Integration cost
        cost = self.compute_integration_cost(network_bmd, new_region_bmd)

        return {
            'richness_change': richness_change,
            'phase_quality_change': phase_quality_change,
            'n_new_compounds': n_new_compounds_est,
            'integration_cost': cost,
            'R_before': R_before,
            'R_after_est': R_before * (1 + richness_change),
            'Q_before': Q_before,
            'Q_after_est': Q_before + phase_quality_change
        }
