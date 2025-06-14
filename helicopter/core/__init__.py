"""
Helicopter Core Module

Implements the revolutionary autonomous reconstruction approach:
"Best way to analyze an image is if AI can draw the image perfectly."

Now enhanced with Pakati-inspired regional control and HuggingFace API integration.
"""

from .autonomous_reconstruction_engine import AutonomousReconstructionEngine
from .pakati_inspired_reconstruction import PakatiInspiredReconstruction, ReconstructionStrategy
from .regional_reconstruction_engine import RegionalReconstructionEngine, MaskingStrategy
from .segment_aware_reconstruction import SegmentAwareReconstructionEngine, SegmentType
from .nicotine_context_validator import (
    NicotineContextValidator,
    NicotinePuzzle,
    ContextSnapshot,
    PuzzleType
)
from .hatata_mdp_engine import (
    HatataEngine,
    HatataMDPModel,
    UnderstandingState,
    HatataAction
)
from .zengeza_noise_detector import (
    ZengezaEngine,
    ZengezaNoiseAnalyzer,
    NoiseMetrics,
    NoiseType,
    NoiseLevel,
    ZengezaSegment
)

__all__ = [
    'AutonomousReconstructionEngine',
    'PakatiInspiredReconstruction', 
    'ReconstructionStrategy',
    'RegionalReconstructionEngine',
    'MaskingStrategy',
    'SegmentAwareReconstructionEngine',
    'SegmentType',
    'NicotineContextValidator',
    'NicotinePuzzle',
    'ContextSnapshot',
    'PuzzleType',
    'HatataEngine',
    'HatataMDPModel',
    'UnderstandingState',
    'HatataAction',
    'ZengezaEngine',
    'ZengezaNoiseAnalyzer',
    'NoiseMetrics',
    'NoiseType',
    'NoiseLevel',
    'ZengezaSegment'
] 