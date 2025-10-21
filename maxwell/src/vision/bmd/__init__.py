"""
BMD (Biological Maxwell Demon) state representations for vision.

Core components:
- BMDState: Base BMD state with categorical and phase structure
- HardwareBMDStream: Unified hardware BMD measurement
- NetworkBMD: Hierarchical network integrating processing history
- PhaseLockCoupling: Phase-lock composition operations (⊛ operator)
"""

from .bmd_state import BMDState, OscillatoryHole, PhaseStructure
from .hardware_stream import HardwareBMDStream
from .network_bmd import NetworkBMD
from .phase_lock import PhaseLockCoupling

__all__ = [
    'BMDState',
    'OscillatoryHole',
    'PhaseStructure',
    'HardwareBMDStream',
    'NetworkBMD',
    'PhaseLockCoupling',
]
