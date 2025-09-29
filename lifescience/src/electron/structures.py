"""
Electron Microscopy Structures - Data structures and enumerations
"""

from dataclasses import dataclass
from enum import Enum


class EMType(Enum):
    """Electron microscopy types"""
    SEM = "scanning_electron_microscopy"
    TEM = "transmission_electron_microscopy"
    CRYO_EM = "cryo_electron_microscopy"


class UltrastructureType(Enum):
    """Cellular ultrastructures visible in EM"""
    MITOCHONDRIA = "mitochondria"
    NUCLEUS = "nucleus"
    VESICLES = "vesicles"
    MEMBRANE = "membrane"


@dataclass
class EMStructure:
    """Electron microscopy structure detection result"""
    structure_type: UltrastructureType
    area: float
    circularity: float
    mean_intensity: float
    confidence: float
