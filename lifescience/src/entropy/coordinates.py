"""
S-Entropy Coordinates - Core coordinate system for biological analysis
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BiologicalContext(Enum):
    """Biological contexts for entropy analysis"""
    CELLULAR = "cellular"
    TISSUE = "tissue"
    MOLECULAR = "molecular"


@dataclass 
class SEntropyCoordinates:
    """S-entropy coordinates with biological interpretation"""
    structural: float  # Structural complexity (-1 to 1)
    functional: float  # Functional activity (-1 to 1)
    morphological: float  # Morphological diversity (-1 to 1)
    temporal: float  # Temporal dynamics (-1 to 1)
    
    biological_context: BiologicalContext = BiologicalContext.CELLULAR
    confidence: float = 1.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.structural, self.functional, self.morphological, self.temporal])
    
    def distance_to(self, other: 'SEntropyCoordinates') -> float:
        """Calculate distance to another coordinate"""
        return np.linalg.norm(self.to_array() - other.to_array())
    
    def magnitude(self) -> float:
        """Calculate magnitude in 4D space"""
        return np.linalg.norm(self.to_array())
    
    def normalize(self) -> 'SEntropyCoordinates':
        """Return normalized coordinates"""
        coords = self.to_array()
        norm = np.linalg.norm(coords)
        if norm > 0:
            coords = coords / norm
        
        return SEntropyCoordinates(
            structural=coords[0],
            functional=coords[1],
            morphological=coords[2],
            temporal=coords[3],
            biological_context=self.biological_context,
            confidence=self.confidence
        )
    
    def biological_interpretation(self) -> str:
        """Provide biological interpretation of coordinates"""
        coords = self.to_array()
        
        # Determine dominant characteristics
        dominant_dim = np.argmax(np.abs(coords))
        dominant_value = coords[dominant_dim]
        
        dimensions = ["structural", "functional", "morphological", "temporal"]
        dominant_name = dimensions[dominant_dim]
        
        if dominant_value > 0.5:
            interpretation = f"High {dominant_name.replace('_', ' ')} with "
        elif dominant_value < -0.5:
            interpretation = f"Low {dominant_name.replace('_', ' ')} with "
        else:
            interpretation = f"Moderate {dominant_name.replace('_', ' ')} with "
        
        # Add secondary characteristics
        secondary_chars = []
        for i, (dim, val) in enumerate(zip(dimensions, coords)):
            if i != dominant_dim and abs(val) > 0.3:
                if val > 0:
                    secondary_chars.append(f"elevated {dim.replace('_', ' ')}")
                else:
                    secondary_chars.append(f"reduced {dim.replace('_', ' ')}")
        
        if secondary_chars:
            interpretation += ", ".join(secondary_chars)
        else:
            interpretation += "balanced other dimensions"
        
        return interpretation + f" (confidence: {self.confidence:.2f})"
