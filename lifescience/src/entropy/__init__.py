# S-Entropy Framework for Life Science Applications
"""
Self-contained entropy framework implementing S-entropy coordinate transformation
specifically designed for biological and medical image analysis.

Key Features:
- SEntropyTransformer: Core transformation engine for biological images
- BiologicalSemanticAnalysis: Life science-specific semantic analysis  
- SEntropyCoordinates: 4D coordinate system for biological structures
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
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


class SEntropyTransformer:
    """Core S-entropy coordinate transformation engine"""
    
    def __init__(self, biological_context: BiologicalContext = BiologicalContext.CELLULAR):
        self.biological_context = biological_context
        logger.info(f"Initialized S-entropy transformer for {biological_context.value}")
    
    def transform(self, image: np.ndarray) -> SEntropyCoordinates:
        """Transform image to S-entropy coordinates"""
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.float32)
        
        # Normalize
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        
        # Analyze structural complexity
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        structural = edge_density * 2 - 1  # Normalize to [-1, 1]
        
        # Analyze functional activity (gradient magnitude)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        functional = np.clip(np.mean(gradient_magnitude) * 4 - 1, -1, 1)
        
        # Analyze morphological diversity (texture variation)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        morphological = np.clip(laplacian_var / 100.0 * 2 - 1, -1, 1)
        
        # Analyze temporal dynamics (single image - estimate potential)
        # Use asymmetry as proxy for dynamic potential
        left_half = gray[:, :gray.shape[1]//2]
        right_half = np.fliplr(gray[:, gray.shape[1]//2:])
        asymmetry = np.mean(np.abs(left_half - right_half[:, :left_half.shape[1]]))
        temporal = np.clip(asymmetry * 8 - 1, -1, 1)
        
        return SEntropyCoordinates(
            structural=structural,
            functional=functional,
            morphological=morphological,
            temporal=temporal,
            biological_context=self.biological_context,
            confidence=0.8  # Simplified confidence
        )
    
    def batch_transform(self, images: List[np.ndarray]) -> List[SEntropyCoordinates]:
        """Transform multiple images"""
        return [self.transform(img) for img in images]


# Export main classes
__all__ = ['SEntropyTransformer', 'SEntropyCoordinates', 'BiologicalContext']