"""
Helicopter Consciousness-Aware Computer Vision Framework

This module implements consciousness-aware visual processing through gas molecular
information dynamics, cross-modal BMD validation, and dual-mode processing architecture.

The framework operates on the principle that meaning emerges from gas molecular
configurations with minimal variance from undisturbed equilibrium, eliminating
the need for semantic dictionaries or computational lookup.
"""

from .gas_molecular import InformationGasMolecule, EquilibriumEngine, MolecularDynamics
from .bmd_validation import BMDValidator, CrossModalValidator, ConsciousnessCoordinator
from .moon_landing import MoonLandingController, AssistantMode, TurbulenceMode
from .integration import ConsciousnessFramework

__version__ = "1.0.0"
__author__ = "Kundai Farai Sachikonye"

# Consciousness processing constants
CONSCIOUSNESS_THRESHOLD = 0.61
EQUILIBRIUM_CONVERGENCE_TIME = 12e-9  # 12 nanoseconds target
VARIANCE_THRESHOLD = 1e-6
BMD_CONVERGENCE_RATE = 0.95

__all__ = [
    'InformationGasMolecule',
    'EquilibriumEngine', 
    'MolecularDynamics',
    'BMDValidator',
    'CrossModalValidator',
    'ConsciousnessCoordinator',
    'MoonLandingController',
    'AssistantMode',
    'TurbulenceMode',
    'ConsciousnessFramework',
    'CONSCIOUSNESS_THRESHOLD',
    'EQUILIBRIUM_CONVERGENCE_TIME',
    'VARIANCE_THRESHOLD',
    'BMD_CONVERGENCE_RATE'
]
