"""
Ambiguity calculation for BMD-region comparisons.

Implements the dual-objective function:
- Ambiguity maximization: A(β^(network), R)
- Stream divergence minimization: D_stream(β^(network) ⊛ R, β^(stream))
"""

import numpy as np
from typing import TYPE_CHECKING
from scipy.stats import entropy as kl_divergence

if TYPE_CHECKING:
    from ..vision.bmd.bmd_state import BMDState
    from ..vision.bmd.network_bmd import NetworkBMD
    from ..regions.region import Region


class AmbiguityCalculator:
    """
    Calculate ambiguity measures for BMD-region comparisons.

    A(β, R) quantifies categorical uncertainty when comparing
    BMD state β with image region R.

    Higher ambiguity → more categorical possibilities
    → richer completion pathway
    """

    def __init__(self, temperature: float = 310.0):
        """
        Initialize ambiguity calculator.

        Args:
            temperature: System temperature in Kelvin (default: body temp)
        """
        self.kB = 1.380649e-23  # Boltzmann constant (J/K)
        self.T = temperature
        self.beta = 1.0 / (self.kB * self.T)  # Thermodynamic beta

    def compute_ambiguity(
        self,
        bmd_state: 'BMDState',
        region: 'Region'
    ) -> float:
        """
        Compute ambiguity A(β, R).

        A(β, R) = Σ_c P(c|R) · D_KL(P_complete(c|β) || P_image(c|R))

        Measures how many distinct categorical states are consistent
        with both the BMD state and the region.

        Args:
            bmd_state: Current BMD state β
            region: Image region R

        Returns:
            Ambiguity value (higher = more uncertainty = better)
        """
        # Get categorical state distributions
        P_complete = self._bmd_categorical_distribution(bmd_state)
        P_image = self._region_categorical_distribution(region)

        # Compute weighted KL divergence
        # High KL → high ambiguity → many possible completions
        ambiguity = 0.0

        for c, p_img in P_image.items():
            if c in P_complete and p_img > 0:
                p_comp = P_complete[c]
                if p_comp > 0:
                    # KL contribution weighted by region probability
                    kl_term = p_comp * np.log(p_comp / p_img)
                    ambiguity += p_img * kl_term

        # Scale by categorical richness (more pathways = more ambiguity)
        richness_factor = np.log1p(bmd_state.categorical_richness())
        ambiguity *= richness_factor

        return max(0.0, ambiguity)

    def compute_network_ambiguity(
        self,
        network_bmd: 'NetworkBMD',
        region: 'Region'
    ) -> float:
        """
        Compute ambiguity with respect to full network BMD.

        A(β^(network), R) considering all hierarchical structure:
        - Individual region BMDs
        - Compound BMDs from sequences
        - Global network state

        Args:
            network_bmd: Current network BMD state
            region: Image region R

        Returns:
            Network-level ambiguity value
        """
        # Get global network BMD
        global_bmd = network_bmd.get_global_bmd()

        # Base ambiguity from global state
        base_ambiguity = self.compute_ambiguity(global_bmd, region)

        # Hierarchical enhancement from compound BMDs
        # More compounds → more contextual possibilities → higher ambiguity
        hierarchical_factor = 1.0 + np.log1p(len(network_bmd.compound_bmds))

        return base_ambiguity * hierarchical_factor

    def stream_divergence(
        self,
        network_bmd: 'NetworkBMD',
        region: 'Region',
        hardware_stream: 'BMDState'
    ) -> float:
        """
        Compute divergence from hardware stream.

        D_stream(β^(network) ⊛ R, β^(stream))

        Measures how far the network would drift from physical reality
        if this region is processed.

        Lower divergence → stays grounded in hardware
        Higher divergence → risks absurd interpretations

        Args:
            network_bmd: Current network BMD
            region: Candidate region
            hardware_stream: Current hardware stream BMD

        Returns:
            Stream divergence (lower = better coherence with reality)
        """
        # Simulate composition: β^(network) ⊛ β_R
        # (Actual composition happens in completion phase)

        # Get phase distributions
        network_phase = self._extract_phase_distribution(
            network_bmd.get_global_bmd()
        )
        stream_phase = self._extract_phase_distribution(hardware_stream)
        region_phase = self._extract_region_phase_estimate(region)

        # Predicted network phase after processing region
        predicted_phase = self._combine_phase_distributions(
            network_phase,
            region_phase
        )

        # KL divergence from hardware stream
        divergence = self._phase_kl_divergence(predicted_phase, stream_phase)

        return divergence

    def dual_objective(
        self,
        network_bmd: 'NetworkBMD',
        region: 'Region',
        hardware_stream: 'BMDState',
        lambda_stream: float = 0.5
    ) -> float:
        """
        Compute dual objective for region selection.

        Score(R) = A(β^(network), R) - λ · D_stream(β^(network) ⊛ R, β^(stream))

        Balances:
        - Maximizing ambiguity (rich categorical exploration)
        - Minimizing stream divergence (hardware grounding)

        Args:
            network_bmd: Current network BMD
            region: Candidate region
            hardware_stream: Hardware stream BMD
            lambda_stream: Balance parameter (0 = pure ambiguity, 1 = pure stream)

        Returns:
            Combined score (higher = better candidate)
        """
        A = self.compute_network_ambiguity(network_bmd, region)
        D = self.stream_divergence(network_bmd, region, hardware_stream)

        score = A - lambda_stream * D

        return score

    # Helper methods

    def _bmd_categorical_distribution(
        self,
        bmd_state: 'BMDState'
    ) -> dict:
        """
        Extract categorical state probability distribution from BMD.

        P_complete(c|β) based on phase structure and holes.

        Returns:
            Dict mapping categorical state → probability
        """
        # Extract phase structure
        phases = bmd_state.phase.phases
        frequencies = bmd_state.phase.frequencies

        # Create categorical states from phase configurations
        # Each combination of phase bins is a categorical state
        n_bins = 8  # Discretize phase space

        P_dist = {}

        # Sample from phase structure
        for i in range(100):  # Monte Carlo sampling
            # Sample phase configuration
            phase_config = tuple(
                int(p / (2*np.pi) * n_bins) % n_bins
                for p in phases
            )

            # Convert to categorical state identifier
            c = hash(phase_config) % 1000  # Simple hash

            P_dist[c] = P_dist.get(c, 0) + 1

        # Normalize
        total = sum(P_dist.values())
        P_dist = {c: count/total for c, count in P_dist.items()}

        return P_dist

    def _region_categorical_distribution(
        self,
        region: 'Region'
    ) -> dict:
        """
        Extract categorical state possibilities from region.

        P_image(c|R) based on region features.

        Returns:
            Dict mapping categorical state → probability
        """
        # Extract features
        if region.features is None:
            region.extract_features()

        features = region.features

        # Create distribution based on feature space
        # (Simplified - in practice, would use learned models)

        P_dist = {}

        # Use feature histograms to generate categorical possibilities
        if 'color_histogram' in features:
            hist = features['color_histogram']
            # Each color bin suggests categorical states
            for i, count in enumerate(hist):
                if count > 0:
                    c = i % 1000
                    P_dist[c] = count

        if not P_dist:
            # Fallback: uniform over small set
            P_dist = {i: 1.0 for i in range(10)}

        # Normalize
        total = sum(P_dist.values())
        P_dist = {c: count/total for c, count in P_dist.items()}

        return P_dist

    def _extract_phase_distribution(
        self,
        bmd_state: 'BMDState'
    ) -> np.ndarray:
        """
        Extract phase angle distribution from BMD.

        Returns:
            Histogram of phase angles
        """
        phases = bmd_state.phase.phases
        hist, _ = np.histogram(phases, bins=32, range=(0, 2*np.pi), density=True)
        hist = hist + 1e-10  # Avoid zeros
        hist = hist / hist.sum()
        return hist

    def _extract_region_phase_estimate(
        self,
        region: 'Region'
    ) -> np.ndarray:
        """
        Estimate phase distribution from region features.

        Returns:
            Estimated phase histogram
        """
        # Extract edge orientations as phase proxies
        if region.features and 'edge_features' in region.features:
            edge_hist = region.features['edge_features']
            # Normalize
            edge_hist = edge_hist + 1e-10
            edge_hist = edge_hist / edge_hist.sum()
            return edge_hist

        # Fallback: uniform
        return np.ones(32) / 32

    def _combine_phase_distributions(
        self,
        phase1: np.ndarray,
        phase2: np.ndarray
    ) -> np.ndarray:
        """
        Combine two phase distributions (composition operation).

        Returns:
            Combined phase distribution
        """
        # Ensure same size
        size = min(len(phase1), len(phase2))
        phase1 = phase1[:size]
        phase2 = phase2[:size]

        # Convolution-like composition (phase addition)
        combined = np.convolve(phase1, phase2, mode='same')
        combined = combined / combined.sum()

        return combined

    def _phase_kl_divergence(
        self,
        P: np.ndarray,
        Q: np.ndarray
    ) -> float:
        """
        Compute KL divergence between phase distributions.

        D_KL(P || Q) = Σ P(i) log(P(i) / Q(i))

        Returns:
            KL divergence value
        """
        # Ensure same size
        size = min(len(P), len(Q))
        P = P[:size]
        Q = Q[:size]

        # Add small epsilon to avoid log(0)
        P = P + 1e-10
        Q = Q + 1e-10

        # Renormalize
        P = P / P.sum()
        Q = Q / Q.sum()

        # Compute KL divergence
        kl_div = np.sum(P * np.log(P / Q))

        return max(0.0, kl_div)
