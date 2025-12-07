"""
Physical validation proofs for HCCC algorithm.

Validates physical/thermodynamic predictions:
1. Energy dissipation: E = kT log(R_final / R_initial)
2. Entropy increase through constraint accumulation
3. Phase-lock dynamics follow physical coupling
4. Hardware BMD measurements consistent with physical reality
"""

import numpy as np
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..vision.bmd import NetworkBMD, BMDState, HardwareBMDStream


class PhysicalValidator:
    """
    Validate algorithm against physical/thermodynamic predictions.

    Key physical principles:
    - Second law: Entropy increases
    - Energy dissipation proportional to information reduction
    - Phase-lock coupling follows physical constraints
    - Hardware measurements reflect real physical processes
    """

    def __init__(self, kB: float = 1.380649e-23, T: float = 310.0):
        """
        Initialize physical validator.

        Args:
            kB: Boltzmann constant (J/K)
            T: Temperature (K)
        """
        self.kB = kB
        self.T = T

    def validate_energy_dissipation(
        self,
        initial_bmd: 'BMDState',
        final_network_bmd: 'NetworkBMD'
    ) -> Dict[str, Any]:
        """
        Validate energy dissipation matches theoretical prediction.

        E_total = kT log(R_final / R_initial)

        Args:
            initial_bmd: Initial hardware stream BMD
            final_network_bmd: Final network BMD

        Returns:
            Validation results
        """
        results = {
            'validated': False,
            'reason': '',
            'metrics': {}
        }

        # Calculate richness change
        R_initial = initial_bmd.categorical_richness()
        R_final = final_network_bmd.network_categorical_richness()

        # Theoretical energy dissipation
        if R_final > R_initial:
            E_theory = self.kB * self.T * np.log(R_final / R_initial)
        else:
            E_theory = -self.kB * self.T * np.log(R_initial / (R_final + 1e-10))

        results['metrics'] = {
            'R_initial': R_initial,
            'R_final': R_final,
            'energy_dissipated': E_theory,
            'energy_per_bmd': E_theory / (len(final_network_bmd.processing_sequence) + 1)
        }

        # Validation: Energy should be positive (dissipated, not gained)
        if E_theory > 0 and R_final > R_initial:
            results['validated'] = True
            results['reason'] = f"Energy dissipated: {E_theory:.3e} J (consistent with richness growth)"
        else:
            results['reason'] = f"Energy violation: E={E_theory:.3e} J, R_ratio={R_final/R_initial:.2f}"

        return results

    def validate_entropy_increase(
        self,
        network_bmd: 'NetworkBMD'
    ) -> Dict[str, Any]:
        """
        Validate that entropy increases through processing.

        From phase-lock theory: Entropy ∝ |E(G)| (constraint graph edges)
        More constraints → higher entropy

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

        # Entropy proxies:
        # 1. Network richness growth (more accessible states)
        # 2. Compound BMD formation (more constraints)
        # 3. Phase-lock coherence (more phase relationships)

        richness_growth = (
            network_bmd.network_categorical_richness() /
            (network_bmd.hardware_stream.categorical_richness() + 1e-10)
        )

        n_compounds = len(network_bmd.compound_bmds)
        phase_quality = network_bmd.global_bmd.phase_lock_quality()

        results['metrics'] = {
            'richness_growth_factor': richness_growth,
            'n_compounds': n_compounds,
            'phase_quality': phase_quality,
            'entropy_increased': richness_growth > 1.0
        }

        # Validated if richness increased and compounds formed
        if richness_growth > 1.0 and n_compounds > 0:
            results['validated'] = True
            results['reason'] = f"Entropy increased: {richness_growth:.2f}x richness, {n_compounds} compounds"
        else:
            results['reason'] = f"Entropy violation: growth={richness_growth:.2f}, compounds={n_compounds}"

        return results

    def validate_phase_lock_dynamics(
        self,
        network_bmd: 'NetworkBMD'
    ) -> Dict[str, Any]:
        """
        Validate phase-lock coupling follows physical constraints.

        Physical constraints:
        - Phase coherence in [0, 1]
        - Coupling strength bounded
        - Phase relationships consistent

        Args:
            network_bmd: Network BMD to validate

        Returns:
            Validation results
        """
        results = {
            'validated': False,
            'reason': '',
            'metrics': {}
        }

        # Check phase coherence values
        coherence_matrix = network_bmd.global_bmd.phase.coherence

        # All coherence values should be in [0, 1]
        coherence_valid = np.all((coherence_matrix >= 0) & (coherence_matrix <= 1))

        # Diagonal should be 1 (perfect self-coherence)
        diagonal_correct = np.allclose(np.diag(coherence_matrix), 1.0)

        # Matrix should be symmetric
        symmetric = np.allclose(coherence_matrix, coherence_matrix.T)

        # Mean coherence should be reasonable
        mean_coherence = np.mean(
            coherence_matrix[np.triu_indices_from(coherence_matrix, k=1)]
        )

        results['metrics'] = {
            'coherence_valid_range': coherence_valid,
            'diagonal_correct': diagonal_correct,
            'symmetric': symmetric,
            'mean_coherence': mean_coherence
        }

        # Validated if all physical constraints satisfied
        if coherence_valid and diagonal_correct and symmetric:
            results['validated'] = True
            results['reason'] = f"Phase-lock dynamics physical (mean coherence={mean_coherence:.3f})"
        else:
            reasons = []
            if not coherence_valid:
                reasons.append("coherence out of range")
            if not diagonal_correct:
                reasons.append("diagonal invalid")
            if not symmetric:
                reasons.append("not symmetric")
            results['reason'] = f"Physical violations: {', '.join(reasons)}"

        return results

    def validate_hardware_measurements(
        self,
        hardware_stream: 'HardwareBMDStream'
    ) -> Dict[str, Any]:
        """
        Validate hardware BMD measurements are physically reasonable.

        Args:
            hardware_stream: Hardware BMD stream

        Returns:
            Validation results
        """
        results = {
            'validated': False,
            'reason': '',
            'metrics': {}
        }

        # Check that hardware stream was measured
        stream_bmd = hardware_stream.get_stream_state()

        if stream_bmd is None:
            results['reason'] = "No hardware measurements available"
            return results

        # Validate measurement properties
        n_devices = hardware_stream.device_count()
        richness = stream_bmd.categorical_richness()
        phase_quality = stream_bmd.phase_lock_quality()

        # Physical reasonableness checks
        has_devices = n_devices > 0
        positive_richness = richness > 0
        valid_phase_quality = 0 <= phase_quality <= 1

        results['metrics'] = {
            'n_devices': n_devices,
            'richness': richness,
            'phase_quality': phase_quality,
            'has_devices': has_devices,
            'positive_richness': positive_richness,
            'valid_phase_quality': valid_phase_quality
        }

        if has_devices and positive_richness and valid_phase_quality:
            results['validated'] = True
            results['reason'] = f"Hardware measurements physical ({n_devices} devices)"
        else:
            results['reason'] = "Hardware measurements unphysical"

        return results

    def comprehensive_physical_validation(
        self,
        initial_bmd: 'BMDState',
        final_network_bmd: 'NetworkBMD',
        hardware_stream: 'HardwareBMDStream'
    ) -> Dict[str, Any]:
        """
        Perform comprehensive physical validation.

        Returns:
            Complete validation report
        """
        report = {
            'energy_dissipation': self.validate_energy_dissipation(
                initial_bmd,
                final_network_bmd
            ),
            'entropy_increase': self.validate_entropy_increase(
                final_network_bmd
            ),
            'phase_lock_dynamics': self.validate_phase_lock_dynamics(
                final_network_bmd
            ),
            'hardware_measurements': self.validate_hardware_measurements(
                hardware_stream
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
            'n_total': len(report) - 1
        }

        return report
