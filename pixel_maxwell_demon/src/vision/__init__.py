"""
Vision processing using Biological Maxwell Demons (BMDs).

This package implements BMD-based image understanding through
hardware-constrained categorical completion.
"""

from .bmd import (
    BMDState,
    OscillatoryHole,
    PhaseStructure,
    HardwareBMDStream,
    NetworkBMD,
    PhaseLockCoupling
)

__all__ = [
    'BMDState',
    'OscillatoryHole',
    'PhaseStructure',
    'HardwareBMDStream',
    'NetworkBMD',
    'PhaseLockCoupling',
]
