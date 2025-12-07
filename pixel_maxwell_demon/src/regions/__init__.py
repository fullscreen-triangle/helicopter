"""
Image region processing for BMD-based vision.

Components:
- Region: Image region representation
- ImageSegmenter: Segmentation methods
- FeatureExtractor: Region feature extraction
"""

from .region import Region
from .segmentation import ImageSegmenter
from .features import FeatureExtractor

__all__ = [
    'Region',
    'ImageSegmenter',
    'FeatureExtractor',
]
