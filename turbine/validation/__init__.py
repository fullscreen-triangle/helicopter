"""
PTRM Validation Suite for BBBC039 Dataset

Validates theoretical predictions from:
- Oxygen-Mediated Categorical Microscopy
- Dodecapartite Virtual Microscopy
- Quintupartite Virtual Microscopy
- Hardware-Constrained Categorical CV

Tests partition coordinates, S-entropy conservation, sequential exclusion,
multimodal reaction localization, quintupartite uniqueness, and dual-membrane
conjugate states.
"""

from .data_loader import BBBC039DataLoader
from .partition_coordinates import PartitionCoordinateExtractor
from .s_entropy import SEntropyAnalyzer
from .sequential_exclusion import SequentialExclusionValidator
from .reaction_localization import MultimodalLocalizationValidator
from .quintupartite_validation import QuintupartiteValidator
from .dual_membrane_validation import DualMembraneValidator
from .oxygen_dynamics_validation import OxygenDynamicsValidator
from .visualization import ValidationPanelGenerator
from .run_validation import run_full_validation

__all__ = [
    'BBBC039DataLoader',
    'PartitionCoordinateExtractor',
    'SEntropyAnalyzer',
    'SequentialExclusionValidator',
    'MultimodalLocalizationValidator',
    'QuintupartiteValidator',
    'DualMembraneValidator',
    'OxygenDynamicsValidator',
    'ValidationPanelGenerator',
    'run_full_validation'
]
