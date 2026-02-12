"""
Sequential Exclusion Algorithm

Implements the core algorithm from Section 9 of the paper:
iteratively reducing structural ambiguity N_0 → N_1 → ... → N_12
through application of 12 measurement modalities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging


@dataclass
class ExclusionResult:
    """Result from applying a single modality."""
    modality_id: int
    modality_name: str
    exclusion_factor: float  # ε_i
    remaining_ambiguity: float  # N_i
    constraints_applied: Dict
    confidence: float


class SequentialExclusion:
    """
    Sequential exclusion algorithm for multi-modal constraint satisfaction.
    
    Algorithm from Section 9:
    1. Start with N_0 ~ 10^60 possible structures
    2. Apply Modality 1: N_1 = N_0 × ε_1
    3. Apply Modality 2: N_2 = N_1 × ε_2
    4. ... continue for all 12 modalities
    5. Final: N_12 = N_0 × ∏ε_i ~ 1 (unique determination)
    """
    
    # Exclusion factors for each modality (from paper)
    EXCLUSION_FACTORS = {
        1: 1e-15,  # Optical microscopy
        2: 1e-12,  # Spectral analysis
        3: 1e-10,  # Vibrational spectroscopy
        4: 1e-8,   # Metabolic GPS
        5: 1e-6,   # Temporal-causal consistency
        6: 1e-5,   # Harmonic network topology
        7: 1e-4,   # Thermodynamic consistency
        8: 1e-3,   # Fluid dynamics
        9: 1e-2,   # Current flow
        10: 0.1,   # Maxwell relations
        11: 0.5,   # Poincaré recurrence
        12: 0.8,   # Categorical transitions
    }
    
    MODALITY_NAMES = {
        1: "Optical Microscopy",
        2: "Spectral Analysis",
        3: "Vibrational Spectroscopy",
        4: "Metabolic GPS",
        5: "Temporal-Causal Consistency",
        6: "Harmonic Network Topology",
        7: "Thermodynamic Consistency",
        8: "Fluid Dynamics Constraints",
        9: "Current Flow Constraints",
        10: "Maxwell Relations",
        11: "Poincaré Recurrence",
        12: "Categorical Transition Limits",
    }
    
    def __init__(self, initial_ambiguity: float = 1e60):
        """
        Initialize sequential exclusion algorithm.
        
        Args:
            initial_ambiguity: N_0, initial number of possible structures
        """
        self.initial_ambiguity = initial_ambiguity
        self.current_ambiguity = initial_ambiguity
        self.exclusion_history: List[ExclusionResult] = []
        self.logger = logging.getLogger(__name__)
        
    def apply_modality(
        self, 
        modality_id: int, 
        measurements: Dict,
        custom_exclusion: Optional[float] = None
    ) -> ExclusionResult:
        """
        Apply a single modality and reduce ambiguity.
        
        Args:
            modality_id: Modality number (1-12)
            measurements: Dictionary with modality-specific measurements
            custom_exclusion: Override default exclusion factor
            
        Returns:
            ExclusionResult with updated ambiguity
        """
        if modality_id not in self.EXCLUSION_FACTORS:
            raise ValueError(f"Invalid modality_id: {modality_id} (must be 1-12)")
        
        # Get exclusion factor
        exclusion_factor = custom_exclusion or self.EXCLUSION_FACTORS[modality_id]
        
        # Apply exclusion
        previous_ambiguity = self.current_ambiguity
        self.current_ambiguity = previous_ambiguity * exclusion_factor
        
        # Estimate confidence based on measurement quality
        confidence = self._estimate_confidence(modality_id, measurements)
        
        result = ExclusionResult(
            modality_id=modality_id,
            modality_name=self.MODALITY_NAMES[modality_id],
            exclusion_factor=exclusion_factor,
            remaining_ambiguity=self.current_ambiguity,
            constraints_applied=measurements,
            confidence=confidence,
        )
        
        self.exclusion_history.append(result)
        
        self.logger.info(
            f"Modality {modality_id} ({result.modality_name}): "
            f"N_{modality_id-1} = {previous_ambiguity:.2e} → "
            f"N_{modality_id} = {self.current_ambiguity:.2e} "
            f"(ε = {exclusion_factor:.2e})"
        )
        
        return result
    
    def apply_all_modalities(self, measurements: Dict) -> List[ExclusionResult]:
        """
        Apply all available modalities sequentially.
        
        Args:
            measurements: Dictionary mapping modality_id → measurement data
            
        Returns:
            List of exclusion results
        """
        results = []
        
        for modality_id in sorted(measurements.keys()):
            if modality_id in self.EXCLUSION_FACTORS:
                result = self.apply_modality(modality_id, measurements[modality_id])
                results.append(result)
        
        return results
    
    def _estimate_confidence(self, modality_id: int, measurements: Dict) -> float:
        """
        Estimate confidence in modality measurement.
        
        Higher quality measurements → higher confidence → better exclusion.
        
        Args:
            modality_id: Modality number
            measurements: Measurement data
            
        Returns:
            Confidence value in [0, 1]
        """
        # Simple heuristics - can be improved
        if "signal_to_noise" in measurements:
            snr = measurements["signal_to_noise"]
            return min(1.0, snr / 20.0)  # Normalize to [0, 1]
        
        if "error" in measurements:
            error = measurements["error"]
            return max(0.0, 1.0 - error)  # Lower error → higher confidence
        
        # Default: moderate confidence
        return 0.7
    
    def get_ambiguity_reduction(self) -> Dict:
        """
        Get summary of ambiguity reduction.
        
        Returns:
            Dictionary with reduction statistics
        """
        if not self.exclusion_history:
            return {
                "initial": self.initial_ambiguity,
                "current": self.current_ambiguity,
                "reduction_factor": 1.0,
                "modalities_applied": 0,
            }
        
        reduction_factor = self.initial_ambiguity / self.current_ambiguity
        
        return {
            "initial": self.initial_ambiguity,
            "current": self.current_ambiguity,
            "reduction_factor": reduction_factor,
            "modalities_applied": len(self.exclusion_history),
            "unique_determination": self.current_ambiguity < 10.0,  # N < 10 → unique
            "exclusion_history": [
                {
                    "modality": r.modality_id,
                    "name": r.modality_name,
                    "exclusion_factor": r.exclusion_factor,
                    "remaining_ambiguity": r.remaining_ambiguity,
                }
                for r in self.exclusion_history
            ],
        }
    
    def reset(self):
        """Reset to initial state."""
        self.current_ambiguity = self.initial_ambiguity
        self.exclusion_history = []
    
    def plot_exclusion_cascade(self) -> Dict:
        """
        Generate data for plotting exclusion cascade.
        
        Returns:
            Dictionary with plotting data
        """
        if not self.exclusion_history:
            return {}
        
        modalities = [r.modality_id for r in self.exclusion_history]
        ambiguities = [r.remaining_ambiguity for r in self.exclusion_history]
        exclusion_factors = [r.exclusion_factor for r in self.exclusion_history]
        
        return {
            "modalities": modalities,
            "ambiguities": ambiguities,
            "exclusion_factors": exclusion_factors,
            "log_ambiguities": [np.log10(a) for a in ambiguities],
            "modality_names": [r.modality_name for r in self.exclusion_history],
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize
    exclusion = SequentialExclusion(initial_ambiguity=1e60)
    
    # Simulate applying modalities
    measurements = {
        1: {"signal_to_noise": 25, "error": 0.05},  # Optical
        2: {"signal_to_noise": 20, "error": 0.08},  # Spectral
        3: {"signal_to_noise": 15, "error": 0.12},  # Vibrational
    }
    
    results = exclusion.apply_all_modalities(measurements)
    
    # Summary
    summary = exclusion.get_ambiguity_reduction()
    print("\n" + "="*60)
    print("Ambiguity Reduction Summary:")
    print(f"  Initial: N_0 = {summary['initial']:.2e}")
    print(f"  Final:   N_{summary['modalities_applied']} = {summary['current']:.2e}")
    print(f"  Reduction: {summary['reduction_factor']:.2e}×")
    print(f"  Unique determination: {summary['unique_determination']}")
