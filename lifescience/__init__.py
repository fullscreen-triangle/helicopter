"""
Helicopter Life Science Modules

A comprehensive computational framework that applies the Helicopter metacognitive 
Bayesian computer vision framework to life science applications. Provides 
mathematically rigorous analysis capabilities specifically designed for biological 
and medical image processing.

Modules:
- gas: Gas molecular dynamics for biological information processing
- entropy: S-entropy coordinate transformation for biological images
- fluorescence: Fluorescence microscopy analysis
- electron: Electron microscopy analysis (SEM, TEM, cryo-EM)
- video: Video processing and cell tracking
- meta: Meta-information extraction and compression analysis
"""

__version__ = "0.1.0"
__author__ = "Kundai Mlilo"
__email__ = "kundai@fullscreen-triangle.com"
__description__ = "Helicopter Life Science Modules - Advanced Computer Vision for Biology"

# Import main modules
from . import src

# Make key classes available at package level
from .src.gas import InformationGasMolecule, GasMolecularSystem, BiologicalGasAnalyzer
from .src.entropy import SEntropyTransformer, SEntropyCoordinates, BiologicalContext
from .src.fluorescence import FluorescenceAnalyzer, FluorescenceChannel, FluorescenceMetrics
from .src.electron import ElectronMicroscopyAnalyzer, EMType, UltrastructureType
from .src.video import VideoAnalyzer, VideoType, CellTrack
from .src.meta import MetaInformationExtractor, InformationType, MetaInformation

__all__ = [
    # Gas Molecular Dynamics
    'InformationGasMolecule',
    'GasMolecularSystem', 
    'BiologicalGasAnalyzer',
    
    # S-Entropy Framework
    'SEntropyTransformer',
    'SEntropyCoordinates',
    'BiologicalContext',
    
    # Fluorescence Microscopy
    'FluorescenceAnalyzer',
    'FluorescenceChannel',
    'FluorescenceMetrics',
    
    # Electron Microscopy
    'ElectronMicroscopyAnalyzer',
    'EMType',
    'UltrastructureType',
    
    # Video Analysis
    'VideoAnalyzer',
    'VideoType',
    'CellTrack',
    
    # Meta-Information
    'MetaInformationExtractor',
    'InformationType',
    'MetaInformation',
]

