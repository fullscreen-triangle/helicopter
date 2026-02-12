"""
PTRM Validation Suite for BBBC039 Dataset

Validates theoretical predictions from:
- Oxygen-Mediated Categorical Microscopy
- Dodecapartite Virtual Microscopy

Tests partition coordinates, S-entropy conservation, sequential exclusion,
and multimodal reaction localization.
"""

from .data_loader import BBBC039DataLoader
from .partition_coordinates import PartitionCoordinateExtractor
from .s_entropy import SEntropyAnalyzer
from .sequential_exclusion import SequentialExclusionValidator
from .reaction_localization import MultimodalLocalizationValidator
from .visualization import ValidationPanelGenerator
from .run_validation import run_full_validation

__all__ = [
    'BBBC039DataLoader',
    'PartitionCoordinateExtractor',
    'SEntropyAnalyzer',
    'SequentialExclusionValidator',
    'MultimodalLocalizationValidator',
    'ValidationPanelGenerator',
    'run_full_validation'
]
