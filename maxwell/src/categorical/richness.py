"""
Categorical richness calculations.

R(β) quantifies the number of distinct categorical completion pathways
available from BMD state β.
"""

import numpy as np
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..vision.bmd.bmd_state import BMDState
    from ..vision.bmd.network_bmd import NetworkBMD


class CategoricalRichnessCalculator:
    """
    Calculate categorical richness for BMD states.

    R(β) = |H(c_current)| × ∏_k N_k(Φ)

    Categorical richness measures how many distinct pathways exist
    for categorical completion from current state.
    """

    def __init__(self):
        """Initialize richness calculator."""
        pass

    def calculate_richness(self, bmd_state: 'BMDState') -> float:
        """
        Calculate R(β) for a BMD state.

        R(β) = |H(c)| × ∏_k N_k(Φ)

        Components:
        - |H(c)|: Number of oscillatory holes
        - N_k(Φ): Number of accessible configurations per mode k

        Args:
            bmd_state: BMD state to analyze

        Returns:
            Categorical richness value
        """
        # Use built-in method
        return bmd_state.categorical_richness()

    def calculate_network_richness(self, network_bmd: 'NetworkBMD') -> float:
        """
        Calculate total network categorical richness.

        R(β^(network)) grows as O(2^n) with processing steps due to
        compound BMD formation.

        Args:
            network_bmd: Network BMD structure

        Returns:
            Total network richness
        """
        return network_bmd.network_categorical_richness()

    def richness_growth_rate(self, richness_history: List[float]) -> float:
        """
        Calculate exponential growth rate of richness.

        Fit: R(n) = R_0 * exp(α * n)

        Returns growth rate α.

        Args:
            richness_history: Richness values over processing steps

        Returns:
            Growth rate α (should be positive for proper algorithm operation)
        """
        if len(richness_history) < 2:
            return 0.0

        # Log transform
        log_richness = np.log(np.array(richness_history) + 1e-10)

        # Linear fit in log space
        n = np.arange(len(log_richness))

        # α = slope of log(R) vs n
        alpha = np.polyfit(n, log_richness, deg=1)[0]

        return alpha

    def richness_contribution(
        self,
        bmd_state: 'BMDState',
        source: str = 'unknown'
    ) -> dict:
        """
        Break down richness contributions.

        Returns:
            Dict with:
            - hole_contribution: From oscillatory holes
            - phase_contribution: From phase structure
            - total: Overall richness
            - source: Origin of BMD
        """
        # Hole contribution
        hole_richness = np.prod([
            h.n_configurations for h in bmd_state.holes
        ]) if bmd_state.holes else 1.0

        # Phase contribution
        phase_richness = np.prod(
            1.0 / (bmd_state.phase.coherence.diagonal() + 1e-10)
        )

        return {
            'hole_contribution': hole_richness,
            'phase_contribution': phase_richness,
            'total': bmd_state.categorical_richness(),
            'source': source
        }

    def compare_richness(
        self,
        bmd1: 'BMDState',
        bmd2: 'BMDState'
    ) -> dict:
        """
        Compare richness between two BMD states.

        Returns:
            Dict with comparison metrics
        """
        R1 = bmd1.categorical_richness()
        R2 = bmd2.categorical_richness()

        return {
            'R1': R1,
            'R2': R2,
            'ratio': R2 / (R1 + 1e-10),
            'log_ratio': np.log(R2 + 1e-10) - np.log(R1 + 1e-10),
            'difference': R2 - R1,
            'richer': 'bmd2' if R2 > R1 else 'bmd1'
        }

    def richness_bounds(
        self,
        n_holes: int,
        n_modes: int,
        config_per_hole: int = int(1e6),
        coherence_mean: float = 0.8
    ) -> dict:
        """
        Calculate theoretical richness bounds.

        Args:
            n_holes: Number of oscillatory holes
            n_modes: Number of oscillatory modes
            config_per_hole: Configurations per hole (~10^6)
            coherence_mean: Average phase coherence

        Returns:
            Dict with lower and upper bounds
        """
        # Lower bound: minimum richness
        R_min = config_per_hole ** n_holes if n_holes > 0 else 1.0

        # Upper bound: maximum with phase contributions
        phase_factor = (1.0 / coherence_mean) ** n_modes
        R_max = R_min * phase_factor

        return {
            'R_min': R_min,
            'R_max': R_max,
            'log_R_min': np.log(R_min + 1e-10),
            'log_R_max': np.log(R_max + 1e-10)
        }
