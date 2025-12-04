"""
Maxwell - Hardware-Constrained Categorical Completion

A BMD-based image understanding framework implementing the St-Stellas / S-Entropy theory.
"""

__version__ = '1.0.0'
__author__ = 'Kundai Farai Sachikonye'
__email__ = 'research@s-entropy.org'

from .vision.bmd import BMDState, HardwareBMDStream, NetworkBMD
from .algorithm import HCCCAlgorithm
from .categorical import AmbiguityCalculator, CategoricalCompletion
from .regions import Region, ImageSegmenter
from .validation import ValidationMetrics, ResultVisualizer

__all__ = [
    # Core algorithm
    'HCCCAlgorithm',

    # BMD components
    'BMDState',
    'HardwareBMDStream',
    'NetworkBMD',

    # Categorical operations
    'AmbiguityCalculator',
    'CategoricalCompletion',

    # Region processing
    'Region',
    'ImageSegmenter',

    # Validation
    'ValidationMetrics',
    'ResultVisualizer',
]
