"""
BMD State representation.

A BMD (Biological Maxwell Demon) state encodes:
- Current categorical state
- Oscillatory hole configuration
- Phase structure of coupled oscillatory modes
- Categorical richness
"""

import numpy as np
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, field


@dataclass
class OscillatoryHole:
    """
    Represents an oscillatory hole requiring completion.

    An oscillatory hole is a missing pattern in an oscillatory cascade
    that must be filled for propagation to continue.
    """
    required_frequency: float  # Required oscillatory frequency (Hz)
    required_phase: float  # Required phase relationship (radians)
    required_amplitude: float  # Required amplitude
    n_configurations: int  # Number of possible completions (~10^6)


@dataclass
class PhaseStructure:
    """
    Phase structure of coupled oscillatory modes.

    Φ = {φ_k} where φ_k is phase of mode k.
    """
    phases: np.ndarray  # Phase values for each mode
    frequencies: np.ndarray  # Frequency of each mode
    coherence: np.ndarray  # Coherence between modes (NxN matrix)

    def phase_lock_quality(self) -> float:
        """Calculate overall phase-lock quality [0, 1]."""
        return np.mean(self.coherence[np.triu_indices_from(self.coherence, k=1)])


class BMDState:
    """
    Base class for Biological Maxwell Demon state representation.

    β = ⟨c_current, H(c_current), Φ⟩

    Where:
    - c_current: Current categorical state
    - H(c_current): Set of oscillatory holes
    - Φ: Phase structure of coupled modes
    """

    def __init__(
        self,
        categorical_state: Any,
        oscillatory_holes: list[OscillatoryHole],
        phase_structure: PhaseStructure,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize BMD state.

        Args:
            categorical_state: Current categorical state identifier
            oscillatory_holes: List of oscillatory holes requiring completion
            phase_structure: Phase structure of coupled oscillatory modes
            metadata: Optional metadata (origin, timestamp, etc.)
        """
        self.c_current = categorical_state
        self.holes = oscillatory_holes
        self.phase = phase_structure
        self.metadata = metadata or {}
        self._richness_cache = None

    def categorical_richness(self) -> float:
        """
        Calculate categorical richness R(β).

        R(β) = |H(c_current)| × ∏_k N_k(Φ)

        Number of distinct completion pathways available.

        Returns:
            Categorical richness value
        """
        if self._richness_cache is not None:
            return self._richness_cache

        # Hole contribution: product of configurations per hole
        hole_richness = np.prod([h.n_configurations for h in self.holes]) if self.holes else 1.0

        # Phase structure contribution: accessible phase configurations per mode
        # Higher coherence → fewer accessible configurations
        phase_richness = np.prod(
            1.0 / (self.phase.coherence.diagonal() + 1e-10)
        )

        self._richness_cache = hole_richness * phase_richness
        return self._richness_cache

    def phase_lock_quality(self) -> float:
        """
        Measure phase coherence quality across oscillatory modes.

        Returns:
            Phase-lock quality in [0, 1], 1 = perfect phase-lock
        """
        return self.phase.phase_lock_quality()

    def invalidate_cache(self):
        """Invalidate cached computations (call after modifications)."""
        self._richness_cache = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize BMD state to dictionary.

        Returns:
            Dictionary representation for storage/transmission
        """
        return {
            'categorical_state': self.c_current,
            'n_holes': len(self.holes),
            'holes': [
                {
                    'freq': h.required_frequency,
                    'phase': h.required_phase,
                    'amp': h.required_amplitude,
                    'n_configs': h.n_configurations
                }
                for h in self.holes
            ],
            'phase_structure': {
                'phases': self.phase.phases.tolist(),
                'frequencies': self.phase.frequencies.tolist(),
                'coherence': self.phase.coherence.tolist()
            },
            'richness': self.categorical_richness(),
            'phase_quality': self.phase_lock_quality(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BMDState':
        """
        Deserialize BMD state from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            BMDState instance
        """
        holes = [
            OscillatoryHole(
                required_frequency=h['freq'],
                required_phase=h['phase'],
                required_amplitude=h['amp'],
                n_configurations=h['n_configs']
            )
            for h in data['holes']
        ]

        phase = PhaseStructure(
            phases=np.array(data['phase_structure']['phases']),
            frequencies=np.array(data['phase_structure']['frequencies']),
            coherence=np.array(data['phase_structure']['coherence'])
        )

        return cls(
            categorical_state=data['categorical_state'],
            oscillatory_holes=holes,
            phase_structure=phase,
            metadata=data.get('metadata', {})
        )

    @classmethod
    def from_hardware_measurement(
        cls,
        hardware_data: Dict[str, Any],
        device_name: str
    ) -> 'BMDState':
        """
        Create BMD state from hardware measurement.

        Args:
            hardware_data: Raw hardware measurement data
            device_name: Name of hardware device

        Returns:
            BMDState representing hardware dynamics
        """
        # Extract phase structure from hardware
        # This is device-specific and should be implemented per device type

        # For now, create a simple representation
        n_modes = hardware_data.get('n_modes', 10)

        phases = np.random.uniform(0, 2*np.pi, n_modes)
        frequencies = np.array(hardware_data.get('frequencies',
                                                 np.logspace(0, 15, n_modes)))
        coherence = np.eye(n_modes) * hardware_data.get('coherence', 0.8)

        phase_struct = PhaseStructure(phases, frequencies, coherence)

        # Hardware measurements typically have no explicit holes
        # (they're complete physical processes)
        holes = []

        return cls(
            categorical_state=f"hardware_{device_name}",
            oscillatory_holes=holes,
            phase_structure=phase_struct,
            metadata={
                'source': 'hardware',
                'device': device_name,
                'timestamp': hardware_data.get('timestamp')
            }
        )

    def __repr__(self) -> str:
        return (
            f"BMDState(c={self.c_current}, "
            f"holes={len(self.holes)}, "
            f"R={self.categorical_richness():.2e}, "
            f"Q={self.phase_lock_quality():.3f})"
        )

