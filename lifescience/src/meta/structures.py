"""
Meta-Information Structures - Data structures and enumerations
"""

from dataclasses import dataclass
from enum import Enum


class InformationType(Enum):
    """Types of information patterns"""
    STRUCTURED = "structured"
    RANDOM = "random"
    PERIODIC = "periodic"
    HIERARCHICAL = "hierarchical"


@dataclass
class MetaInformation:
    """Meta-information about data structure"""
    information_type: InformationType
    semantic_density: float
    compression_potential: float
    structural_complexity: float
    confidence: float
