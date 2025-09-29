# Electron Microscopy Analysis Module
"""
Self-contained module focused on applying the Helicopter framework to electron 
microscope images (SEM, TEM, cryo-EM). Specialized for high-resolution analysis
of cellular ultrastructure and molecular complexes.

Key Features:
- ElectronMicroscopyAnalyzer: Core analysis engine for EM images
- EMStructure: Quantified detection results with confidence metrics
- Multi-modal support: SEM, TEM, cryo-EM with optimized parameters

Applications:
- Cellular ultrastructure quantification (mitochondria, ER, vesicles)
- Membrane dynamics and morphology analysis
- Single particle analysis for cryo-EM
- Protein complex identification and characterization
- Correlative microscopy data integration
"""

# Import core classes from submodules
from .structures import EMType, UltrastructureType, EMStructure
from .analyzer import ElectronMicroscopyAnalyzer

# Export main classes
__all__ = ['ElectronMicroscopyAnalyzer', 'EMType', 'UltrastructureType','EMStructure'] 