# Electron Microscopy Analysis Module
"""
Self-contained module focused on applying the Helicopter framework to electron 
microscope images (SEM, TEM, cryo-EM). Specialized for high-resolution analysis
of cellular ultrastructure and molecular complexes.

Key Features:
- ElectronMicroscopyAnalyzer: Core analysis engine for EM images
- UltrastructureDetector: Organelle and membrane detection
- ParticleAnalyzer: Single particle analysis for cryo-EM
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import ndimage
from skimage import measure

logger = logging.getLogger(__name__)


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


class ElectronMicroscopyAnalyzer:
    """Core analysis engine for electron microscopy images"""
    
    def __init__(self, em_type: EMType = EMType.TEM):
        self.em_type = em_type
        self.analysis_results = []
        
    def analyze_image(self, image: np.ndarray,
                     target_structures: Optional[List[UltrastructureType]] = None) -> Dict[str, Any]:
        """Analyze electron microscopy image"""
        
        if target_structures is None:
            target_structures = [UltrastructureType.MITOCHONDRIA, UltrastructureType.VESICLES]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.float32)
        
        # Normalize
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        
        # Edge detection with parameters based on EM type
        if self.em_type == EMType.CRYO_EM:
            edges = cv2.Canny((gray * 255).astype(np.uint8), 20, 80)
            min_area = 50
        elif self.em_type == EMType.SEM:
            edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
            min_area = 500
        else:  # TEM
            edges = cv2.Canny((gray * 255).astype(np.uint8), 30, 100)
            min_area = 100
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_structures = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < min_area:
                continue
            
            # Calculate properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Calculate mean intensity in structure
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            mean_intensity = cv2.mean(gray, mask=mask)[0]
            
            # Simple structure classification
            structure_type, confidence = self._classify_structure(area, circularity, mean_intensity, target_structures)
            
            if confidence > 0.3:
                structure = EMStructure(
                    structure_type=structure_type,
                    area=area,
                    circularity=circularity,
                    mean_intensity=mean_intensity,
                    confidence=confidence
                )
                detected_structures.append(structure)
        
        results = {
            'em_type': self.em_type.value,
            'structures': detected_structures,
            'num_structures': len(detected_structures),
            'summary': self._create_summary(detected_structures)
        }
        
        self.analysis_results.append(results)
        logger.info(f"Analyzed {self.em_type.value} image: found {len(detected_structures)} structures")
        
        return results
    
    def _classify_structure(self, area: float, circularity: float, 
                          mean_intensity: float, target_structures: List[UltrastructureType]) -> Tuple[UltrastructureType, float]:
        """Simple structure classification"""
        
        scores = {}
        
        for structure_type in target_structures:
            score = 0.0
            
            if structure_type == UltrastructureType.MITOCHONDRIA:
                # Mitochondria: elongated, moderate size
                if 1000 < area < 10000:
                    score += 0.4
                if 0.2 < circularity < 0.6:
                    score += 0.6
                    
            elif structure_type == UltrastructureType.NUCLEUS:
                # Nucleus: large, circular
                if area > 5000:
                    score += 0.5
                if circularity > 0.6:
                    score += 0.5
                    
            elif structure_type == UltrastructureType.VESICLES:
                # Vesicles: small, circular
                if 100 < area < 2000:
                    score += 0.4
                if circularity > 0.6:
                    score += 0.6
                    
            elif structure_type == UltrastructureType.MEMBRANE:
                # Membrane: elongated, low circularity
                if circularity < 0.4:
                    score += 0.5
                if mean_intensity < 0.5:
                    score += 0.5
            
            scores[structure_type] = score
        
        if scores:
            best_structure = max(scores, key=scores.get)
            best_score = scores[best_structure]
            return best_structure, best_score
        else:
            return UltrastructureType.VESICLES, 0.1
    
    def _create_summary(self, structures: List[EMStructure]) -> Dict[str, Any]:
        """Create summary statistics"""
        
        if not structures:
            return {'total_count': 0}
        
        # Count by type
        type_counts = {}
        for structure in structures:
            struct_type = structure.structure_type.value
            type_counts[struct_type] = type_counts.get(struct_type, 0) + 1
        
        # Statistics
        areas = [s.area for s in structures]
        circularities = [s.circularity for s in structures]
        confidences = [s.confidence for s in structures]
        
        return {
            'total_count': len(structures),
            'type_distribution': type_counts,
            'mean_area': np.mean(areas),
            'mean_circularity': np.mean(circularities),
            'mean_confidence': np.mean(confidences)
        }
    
    def visualize_results(self, results: Dict[str, Any], 
                         original_image: np.ndarray) -> plt.Figure:
        """Visualize EM analysis results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"EM Analysis: {results['em_type']}")
        
        # Original image
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        structures = results['structures']
        
        # Structure count by type
        summary = results['summary']
        type_dist = summary.get('type_distribution', {})
        
        if type_dist:
            axes[0, 1].bar(type_dist.keys(), type_dist.values())
            axes[0, 1].set_title('Structure Types')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Area distribution
        if structures:
            areas = [s.area for s in structures]
            axes[1, 0].hist(areas, bins=15)
            axes[1, 0].set_title('Area Distribution')
            axes[1, 0].set_xlabel('Area (pixels)')
        
        # Summary text
        summary_text = f"Total structures: {summary.get('total_count', 0)}\n"
        summary_text += f"Mean area: {summary.get('mean_area', 0):.1f}\n"
        summary_text += f"Mean confidence: {summary.get('mean_confidence', 0):.2f}"
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, va='center')
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig


# Export main classes
__all__ = ['ElectronMicroscopyAnalyzer', 'EMType', 'UltrastructureType', 'EMStructure'] 