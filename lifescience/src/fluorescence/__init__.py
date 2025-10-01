# Fluorescence Microscopy Analysis Module
"""
Independent module applying the Helicopter framework to fluorescence microscopy images.
Specialized for analyzing fluorescent proteins, cellular structures, and dynamic processes
in live cell imaging with precise quantitative measurements.

Key Features:
- FluorescenceAnalyzer: Core analysis engine for fluorescence images
- FluorescenceMetrics: Comprehensive quantification (intensity, SNR, morphology)
- Multi-channel support: DAPI, GFP, RFP, FITC with colocalization analysis

Applications:
- Protein localization and trafficking studies
- Calcium imaging and signaling pathway analysis
- Cell division and migration quantification
- Drug response assessment in live cells
- High-content screening applications
"""

# Import core classes from submodules
from .metrics import FluorescenceChannel, FluorescenceMetrics
from .analyzer import FluorescenceAnalyzer

# Export main classes
__all__ = ['FluorescenceAnalyzer', 'FluorescenceChannel', 'FluorescenceMetrics'] 