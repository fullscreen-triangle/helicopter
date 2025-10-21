"""
Validation metrics for HCCC algorithm.

Metrics for assessing algorithm performance:
- Energy dissipation
- Stream coherence
- Categorical richness growth
- Convergence quality
"""

import numpy as np
from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..vision.bmd import NetworkBMD, BMDState


class ValidationMetrics:
    """
    Compute validation metrics for algorithm performance.

    Metrics validate theoretical predictions:
    1. Energy dissipation: E_total = kT log(R₀ / R_final)
    2. Stream coherence: Maintains hardware grounding
    3. Richness growth: O(2^n) as predicted
    4. Convergence: Achieves network coherence
    """

    def __init__(self, kB: float = 1.380649e-23, T: float = 310.0):
        """
        Initialize metrics calculator.

        Args:
            kB: Boltzmann constant (J/K)
            T: Temperature (K)
        """
        self.kB = kB
        self.T = T

    def energy_dissipation(
        self,
        initial_bmd: 'BMDState',
        final_network_bmd: 'NetworkBMD'
    ) -> float:
        """
        Calculate total energy dissipated during processing.

        E_total = k_B T log(R(β₀) / R(β_final))

        From categorical completion theory: processing increases richness,
        requiring energy dissipation.

        Args:
            initial_bmd: Initial hardware stream BMD
            final_network_bmd: Final network BMD

        Returns:
            Energy dissipated (Joules)
        """
        R_initial = initial_bmd.categorical_richness()
        R_final = final_network_bmd.network_categorical_richness()

        if R_final > R_initial:
            # Richness increased (typical)
            E_total = self.kB * self.T * np.log(R_final / R_initial)
        else:
            # Richness decreased (unusual)
            E_total = -self.kB * self.T * np.log(R_initial / (R_final + 1e-10))

        return E_total

    def stream_coherence_score(
        self,
        network_bmd: 'NetworkBMD',
        hardware_stream: 'BMDState'
    ) -> float:
        """
        Measure final network coherence with hardware stream.

        Returns value in [0, 1], 1 = perfect coherence.

        Args:
            network_bmd: Final network BMD
            hardware_stream: Current hardware stream BMD

        Returns:
            Coherence score
        """
        # Phase-lock quality comparison
        Q_network = network_bmd.global_bmd.phase_lock_quality()
        Q_stream = hardware_stream.phase_lock_quality()

        # Phase structure similarity
        phase_similarity = self._phase_similarity(
            network_bmd.global_bmd,
            hardware_stream
        )

        # Combined coherence score
        coherence = 0.5 * (Q_network * Q_stream) + 0.5 * phase_similarity

        return coherence

    def categorical_richness_growth(
        self,
        richness_history: List[float]
    ) -> Dict[str, float]:
        """
        Analyze categorical richness growth.

        Should show O(2^n) exponential growth as predicted.

        Args:
            richness_history: Richness values over processing steps

        Returns:
            Dict with growth metrics:
            - growth_rate: Exponential growth rate α
            - doubling_time: Steps to double richness
            - r_squared: Fit quality
        """
        if len(richness_history) < 3:
            return {
                'growth_rate': 0.0,
                'doubling_time': float('inf'),
                'r_squared': 0.0
            }

        # Log-linear fit: log(R) = log(R₀) + α·n
        log_richness = np.log(np.array(richness_history) + 1e-10)
        steps = np.arange(len(log_richness))

        # Fit
        coeffs = np.polyfit(steps, log_richness, deg=1)
        alpha = coeffs[0]  # Growth rate
        log_R0 = coeffs[1]

        # Doubling time
        doubling_time = np.log(2) / alpha if alpha > 0 else float('inf')

        # R-squared
        fit_values = alpha * steps + log_R0
        ss_res = np.sum((log_richness - fit_values) ** 2)
        ss_tot = np.sum((log_richness - np.mean(log_richness)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))

        return {
            'growth_rate': alpha,
            'doubling_time': doubling_time,
            'r_squared': r_squared,
            'initial_richness': richness_history[0],
            'final_richness': richness_history[-1],
            'growth_factor': richness_history[-1] / (richness_history[0] + 1e-10)
        }

    def convergence_quality(
        self,
        ambiguity_history: List[float],
        divergence_history: List[float],
        coherence_threshold: float = 1.0
    ) -> Dict[str, Any]:
        """
        Assess convergence quality.

        Args:
            ambiguity_history: Ambiguity values
            divergence_history: Stream divergence values
            coherence_threshold: Target coherence

        Returns:
            Dict with convergence metrics
        """
        metrics = {}

        if ambiguity_history:
            # Final ambiguity
            metrics['final_ambiguity'] = ambiguity_history[-1]

            # Achieved coherence?
            metrics['achieved_coherence'] = (
                ambiguity_history[-1] < coherence_threshold
            )

            # Ambiguity reduction
            metrics['ambiguity_reduction'] = (
                ambiguity_history[0] - ambiguity_history[-1]
                if len(ambiguity_history) > 1 else 0.0
            )

            # Reduction rate
            if len(ambiguity_history) > 1:
                metrics['reduction_rate'] = (
                    metrics['ambiguity_reduction'] / len(ambiguity_history)
                )
            else:
                metrics['reduction_rate'] = 0.0

        if divergence_history:
            # Final divergence
            metrics['final_divergence'] = divergence_history[-1]

            # Mean divergence
            metrics['mean_divergence'] = np.mean(divergence_history)

            # Divergence bounded?
            metrics['divergence_bounded'] = (
                np.max(divergence_history) < 2.0 * np.mean(divergence_history)
            )

        # Overall quality score
        quality_score = 0.0
        if metrics.get('achieved_coherence', False):
            quality_score += 0.4
        if metrics.get('ambiguity_reduction', 0) > 0:
            quality_score += 0.3
        if metrics.get('divergence_bounded', False):
            quality_score += 0.3

        metrics['quality_score'] = quality_score

        return metrics

    def processing_efficiency(
        self,
        n_regions_processed: int,
        n_regions_total: int,
        iterations: int
    ) -> Dict[str, float]:
        """
        Measure processing efficiency.

        Args:
            n_regions_processed: Number of regions processed
            n_regions_total: Total regions available
            iterations: Total iterations

        Returns:
            Efficiency metrics
        """
        coverage = n_regions_processed / (n_regions_total + 1e-10)
        efficiency = n_regions_processed / (iterations + 1e-10)

        return {
            'coverage': coverage,
            'efficiency': efficiency,
            'regions_processed': n_regions_processed,
            'regions_total': n_regions_total,
            'iterations': iterations
        }

    def _phase_similarity(
        self,
        bmd1: 'BMDState',
        bmd2: 'BMDState'
    ) -> float:
        """
        Compute phase structure similarity between two BMDs.

        Returns:
            Similarity in [0, 1]
        """
        # Phase correlation
        n_min = min(len(bmd1.phase.phases), len(bmd2.phase.phases))

        if n_min == 0:
            return 0.0

        phases1 = bmd1.phase.phases[:n_min]
        phases2 = bmd2.phase.phases[:n_min]

        # Phase difference
        phase_diff = np.abs(phases1 - phases2)
        phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)  # Wrap around

        # Similarity (0 difference = 1 similarity)
        similarity = 1.0 - np.mean(phase_diff) / np.pi

        return max(0.0, similarity)

    def comprehensive_report(
        self,
        initial_bmd: 'BMDState',
        final_network_bmd: 'NetworkBMD',
        hardware_stream: 'BMDState',
        ambiguity_history: List[float],
        divergence_history: List[float],
        richness_history: List[float],
        n_regions_processed: int,
        n_regions_total: int,
        iterations: int,
        coherence_threshold: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate comprehensive metrics report.

        Returns:
            Complete metrics dictionary
        """
        report = {
            'energy_dissipation': self.energy_dissipation(
                initial_bmd,
                final_network_bmd
            ),
            'stream_coherence': self.stream_coherence_score(
                final_network_bmd,
                hardware_stream
            ),
            'richness_growth': self.categorical_richness_growth(
                richness_history
            ),
            'convergence': self.convergence_quality(
                ambiguity_history,
                divergence_history,
                coherence_threshold
            ),
            'efficiency': self.processing_efficiency(
                n_regions_processed,
                n_regions_total,
                iterations
            )
        }

        return report
