"""
Region selection strategies for HCCC algorithm.

Implements dual-objective region selection:
Score(R) = A(β^(network), R) - λ · D_stream(β^(network) ⊛ R, β^(stream))
"""

import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..vision.bmd import NetworkBMD, BMDState
    from ..regions import Region
    from ..categorical import AmbiguityCalculator


class RegionSelector:
    """
    Select regions for processing based on dual objective.

    Balances:
    - Ambiguity maximization: Explore rich categorical structures
    - Stream coherence: Stay grounded in hardware reality
    """

    def __init__(self, ambiguity_calculator: 'AmbiguityCalculator'):
        """
        Initialize region selector.

        Args:
            ambiguity_calculator: Ambiguity computation engine
        """
        self.ambiguity_calc = ambiguity_calculator

    def select_next_region(
        self,
        network_bmd: 'NetworkBMD',
        available_regions: List['Region'],
        hardware_stream: 'BMDState',
        lambda_stream: float = 0.5
    ) -> Optional['Region']:
        """
        Select region maximizing dual objective.

        Score(R) = A(β^(network), R) - λ · D_stream(β^(network) ⊛ R, β^(stream))

        Args:
            network_bmd: Current network BMD
            available_regions: Candidate regions
            hardware_stream: Current hardware stream BMD
            lambda_stream: Balance parameter

        Returns:
            Selected region, or None if no suitable candidate
        """
        if not available_regions:
            return None

        best_region = None
        best_score = float('-inf')

        scores = []

        for region in available_regions:
            # Compute dual objective
            score = self.ambiguity_calc.dual_objective(
                network_bmd=network_bmd,
                region=region,
                hardware_stream=hardware_stream,
                lambda_stream=lambda_stream
            )

            scores.append((region, score))

            if score > best_score:
                best_score = score
                best_region = region

        return best_region

    def select_batch(
        self,
        network_bmd: 'NetworkBMD',
        available_regions: List['Region'],
        hardware_stream: 'BMDState',
        lambda_stream: float = 0.5,
        batch_size: int = 5
    ) -> List['Region']:
        """
        Select batch of top-scoring regions.

        Useful for parallel processing.

        Args:
            network_bmd: Current network BMD
            available_regions: Candidate regions
            hardware_stream: Hardware stream BMD
            lambda_stream: Balance parameter
            batch_size: Number of regions to select

        Returns:
            List of selected regions
        """
        if not available_regions:
            return []

        # Compute scores for all
        scores = []

        for region in available_regions:
            score = self.ambiguity_calc.dual_objective(
                network_bmd=network_bmd,
                region=region,
                hardware_stream=hardware_stream,
                lambda_stream=lambda_stream
            )
            scores.append((region, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top batch_size
        batch = [region for region, score in scores[:batch_size]]

        return batch

    def adaptive_lambda(
        self,
        iteration: int,
        total_regions: int,
        current_divergence: float,
        divergence_history: List[float]
    ) -> float:
        """
        Adaptively adjust λ_stream based on algorithm progress.

        Strategy:
        - Early: Lower λ (explore ambiguity)
        - Late: Higher λ (enforce coherence)
        - High divergence: Increase λ (pull back to hardware)

        Args:
            iteration: Current iteration
            total_regions: Total number of regions
            current_divergence: Current stream divergence
            divergence_history: Historical divergences

        Returns:
            Adaptive λ value
        """
        # Progress through algorithm
        progress = iteration / (total_regions + 1)

        # Base lambda increases with progress
        lambda_base = 0.3 + 0.7 * progress  # 0.3 → 1.0

        # Adjust for divergence
        if divergence_history:
            mean_divergence = np.mean(divergence_history)

            if current_divergence > 1.5 * mean_divergence:
                # Diverging too much, increase λ
                lambda_adjusted = lambda_base * 1.5
            elif current_divergence < 0.5 * mean_divergence:
                # Very coherent, can lower λ to explore
                lambda_adjusted = lambda_base * 0.7
            else:
                lambda_adjusted = lambda_base
        else:
            lambda_adjusted = lambda_base

        # Clamp to valid range
        return np.clip(lambda_adjusted, 0.1, 1.5)

    def rank_regions(
        self,
        network_bmd: 'NetworkBMD',
        regions: List['Region'],
        hardware_stream: 'BMDState',
        lambda_stream: float = 0.5
    ) -> List[tuple]:
        """
        Rank all regions by dual objective score.

        Args:
            network_bmd: Current network BMD
            regions: Regions to rank
            hardware_stream: Hardware stream BMD
            lambda_stream: Balance parameter

        Returns:
            List of (region, score) tuples sorted by score descending
        """
        scores = []

        for region in regions:
            score = self.ambiguity_calc.dual_objective(
                network_bmd=network_bmd,
                region=region,
                hardware_stream=hardware_stream,
                lambda_stream=lambda_stream
            )
            scores.append((region, score))

        # Sort descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores
