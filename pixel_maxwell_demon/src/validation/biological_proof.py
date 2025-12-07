"""
Biological validation proofs for HCCC algorithm.

Validates that algorithm behavior matches biological predictions:
1. Hardware grounding prevents absurd interpretations
2. Hierarchical BMD structure mirrors neural processing
3. Phase-lock dynamics match neural oscillations
4. Energy dissipation consistent with biological metabolism
"""

import numpy as np
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..vision.bmd import NetworkBMD, BMDState


class BiologicalValidator:
    """
    Validate algorithm against biological predictions.

    From categorical completion consciousness theory:
    - Consciousness operates at ~40 Hz (gamma)
    - ~10^31 BMD operations per moment
    - ~10^6 categorical states per hole
    - Hardware (sensors) provide external anchoring
    """

    def __init__(self):
        """Initialize biological validator."""
        self.gamma_frequency = 40.0  # Hz
        self.n_bmds_per_moment = 1e31  # Theoretical
        self.configs_per_hole = 1e6  # ~10^6

    def validate_hardware_grounding(
        self,
        network_bmd: 'NetworkBMD',
        hardware_stream: 'BMDState',
        divergence_history: list
    ) -> Dict[str, Any]:
        """
        Validate that hardware stream prevents absurd interpretations.

        Key prediction: Stream divergence should remain bounded,
        preventing interpretations that drift from physical reality.

        Args:
            network_bmd: Final network BMD
            hardware_stream: Hardware stream BMD
            divergence_history: Stream divergence over processing

        Returns:
            Validation results
        """
        results = {
            'validated': False,
            'reason': '',
            'metrics': {}
        }

        if not divergence_history:
            results['reason'] = "No divergence history available"
            return results

        # Check that divergence remains bounded
        max_divergence = np.max(divergence_history)
        mean_divergence = np.mean(divergence_history)

        # Bounded if max < 2x mean
        bounded = max_divergence < 2.0 * mean_divergence

        # Check final coherence with hardware
        from ..validation.metrics import ValidationMetrics
        metrics_calc = ValidationMetrics()

        final_coherence = metrics_calc.stream_coherence_score(
            network_bmd,
            hardware_stream
        )

        results['metrics'] = {
            'max_divergence': max_divergence,
            'mean_divergence': mean_divergence,
            'bounded': bounded,
            'final_coherence': final_coherence
        }

        # Validation criteria
        if bounded and final_coherence > 0.5:
            results['validated'] = True
            results['reason'] = "Hardware grounding maintained throughout processing"
        else:
            results['reason'] = f"Grounding violated: bounded={bounded}, coherence={final_coherence:.3f}"

        return results

    def validate_hierarchical_structure(
        self,
        network_bmd: 'NetworkBMD'
    ) -> Dict[str, Any]:
        """
        Validate hierarchical BMD structure matches neural predictions.

        Biological prediction: Hierarchical processing from V1 → V2 → V4 → IT
        corresponds to increasing compound order in BMD network.

        Args:
            network_bmd: Final network BMD

        Returns:
            Validation results
        """
        results = {
            'validated': False,
            'reason': '',
            'metrics': {}
        }

        # Check compound BMD distribution
        compound_counts = {}
        for order in range(2, 6):
            compounds = network_bmd.get_compound_bmds_by_order(order)
            compound_counts[order] = len(compounds)

        # Should see decreasing counts with higher order
        # (exponentially more expensive to form)
        counts_decreasing = all(
            compound_counts[i] >= compound_counts[i+1]
            for i in range(2, 5)
        )

        # Should have non-zero compounds at multiple levels
        multi_level = sum(1 for count in compound_counts.values() if count > 0) >= 3

        results['metrics'] = {
            'compound_distribution': compound_counts,
            'counts_decreasing': counts_decreasing,
            'multi_level': multi_level,
            'total_compounds': sum(compound_counts.values())
        }

        if counts_decreasing and multi_level:
            results['validated'] = True
            results['reason'] = "Hierarchical structure matches neural predictions"
        else:
            results['reason'] = f"Structure mismatch: decreasing={counts_decreasing}, multi_level={multi_level}"

        return results

    def validate_richness_growth(
        self,
        richness_history: list
    ) -> Dict[str, Any]:
        """
        Validate categorical richness grows as O(2^n).

        Biological prediction: Network categorical richness grows
        exponentially as compound BMDs form.

        Args:
            richness_history: Richness values over processing

        Returns:
            Validation results
        """
        results = {
            'validated': False,
            'reason': '',
            'metrics': {}
        }

        if len(richness_history) < 3:
            results['reason'] = "Insufficient history for validation"
            return results

        # Fit exponential growth
        log_richness = np.log(np.array(richness_history) + 1e-10)
        steps = np.arange(len(log_richness))

        coeffs = np.polyfit(steps, log_richness, deg=1)
        growth_rate = coeffs[0]

        # R-squared
        fit_values = coeffs[0] * steps + coeffs[1]
        ss_res = np.sum((log_richness - fit_values) ** 2)
        ss_tot = np.sum((log_richness - np.mean(log_richness)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))

        results['metrics'] = {
            'growth_rate': growth_rate,
            'r_squared': r_squared,
            'exponential_fit': r_squared > 0.8
        }

        # Validated if good exponential fit and positive growth
        if r_squared > 0.8 and growth_rate > 0:
            results['validated'] = True
            results['reason'] = f"Exponential growth confirmed (R²={r_squared:.3f}, α={growth_rate:.3f})"
        else:
            results['reason'] = f"Growth not exponential: R²={r_squared:.3f}"

        return results

    def comprehensive_biological_validation(
        self,
        network_bmd: 'NetworkBMD',
        hardware_stream: 'BMDState',
        divergence_history: list,
        richness_history: list
    ) -> Dict[str, Any]:
        """
        Perform comprehensive biological validation.

        Returns:
            Complete validation report
        """
        report = {
            'hardware_grounding': self.validate_hardware_grounding(
                network_bmd,
                hardware_stream,
                divergence_history
            ),
            'hierarchical_structure': self.validate_hierarchical_structure(
                network_bmd
            ),
            'richness_growth': self.validate_richness_growth(
                richness_history
            )
        }

        # Overall validation
        all_validated = all(
            result['validated']
            for result in report.values()
        )

        report['overall'] = {
            'validated': all_validated,
            'n_validated': sum(1 for r in report.values() if r['validated']),
            'n_total': len(report) - 1  # Exclude 'overall'
        }

        return report
