# Meta-Information Extraction and Ambiguous Compression Module
"""
Meta-information extraction and ambiguous compression implementation.
Focuses on analyzing structural patterns for problem space compression.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


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


class MetaInformationExtractor:
    """Extract meta-information from biological data"""
    
    def __init__(self):
        self.analysis_cache = {}
        
    def extract_meta_information(self, data: np.ndarray) -> MetaInformation:
        """Extract meta-information from data"""
        
        logger.info(f"Extracting meta-information from data shape: {data.shape}")
        
        # Analyze information type
        info_type = self._analyze_information_type(data)
        
        # Calculate semantic density
        semantic_density = self._calculate_semantic_density(data)
        
        # Estimate compression potential
        compression_potential = self._estimate_compression_potential(data)
        
        # Calculate structural complexity
        structural_complexity = self._calculate_structural_complexity(data)
        
        # Overall confidence
        confidence = 0.8  # Simplified confidence
        
        meta_info = MetaInformation(
            information_type=info_type,
            semantic_density=semantic_density,
            compression_potential=compression_potential,
            structural_complexity=structural_complexity,
            confidence=confidence
        )
        
        return meta_info
    
    def _analyze_information_type(self, data: np.ndarray) -> InformationType:
        """Analyze the type of information in the data"""
        
        if len(data.shape) >= 2:
            # Treat as image data
            if len(data.shape) == 3:
                gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            else:
                gray = data.astype(np.float32)
            
            # Normalize
            gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
            
            # Structural analysis (edge content)
            edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
            structure_score = np.sum(edges > 0) / edges.size
            
            # Randomness analysis (entropy)
            hist, _ = np.histogram(gray.flatten(), bins=32, density=True)
            entropy = -np.sum(hist * np.log(hist + 1e-8))
            
            # Simple classification
            if structure_score > 0.1:
                return InformationType.STRUCTURED
            elif entropy > 3.0:
                return InformationType.RANDOM
            else:
                return InformationType.PERIODIC
        
        else:
            # 1D data analysis
            if len(set(data.flat)) > len(data) * 0.8:
                return InformationType.RANDOM
            else:
                return InformationType.STRUCTURED
    
    def _calculate_semantic_density(self, data: np.ndarray) -> float:
        """Calculate semantic density of the data"""
        
        if len(data.shape) >= 2:
            if len(data.shape) == 3:
                gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            else:
                gray = data.astype(np.float32)
            
            # Use gradient magnitude as semantic content
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Semantic density as proportion of high-gradient pixels
            threshold = np.percentile(gradient_magnitude, 75)
            semantic_pixels = gradient_magnitude > threshold
            density = np.sum(semantic_pixels) / semantic_pixels.size
            
            return float(density)
        else:
            # For 1D data, use local variation
            variation = np.std(np.diff(data))
            max_variation = np.std(data)
            return float(variation / (max_variation + 1e-8))
    
    def _estimate_compression_potential(self, data: np.ndarray) -> float:
        """Estimate compression potential of the data"""
        
        # Simple compression potential based on entropy
        flat_data = data.flatten()
        
        # Calculate empirical entropy
        unique_values = len(np.unique(flat_data))
        total_values = len(flat_data)
        
        if unique_values == 1:
            return 1.0  # Perfect compression potential
        
        # Approximate compression ratio
        theoretical_compression = 1.0 - (unique_values / total_values)
        
        return float(min(1.0, max(0.1, theoretical_compression)))
    
    def _calculate_structural_complexity(self, data: np.ndarray) -> float:
        """Calculate structural complexity of the data"""
        
        if len(data.shape) >= 2:
            # For image data, use edge density
            if len(data.shape) == 3:
                gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            else:
                gray = data.astype(np.float32)
            
            # Edge-based complexity
            edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
            edge_complexity = np.sum(edges > 0) / edges.size
            
            return float(edge_complexity)
        else:
            # For 1D data, use signal variance
            complexity = np.var(data) / (np.mean(data)**2 + 1e-8)
            return float(min(1.0, complexity))
    
    def analyze_compression_ratio(self, data: np.ndarray) -> Dict[str, float]:
        """Analyze potential compression ratios"""
        
        meta_info = self.extract_meta_information(data)
        
        # Estimate different compression ratios
        ratios = {
            'lossless_ratio': 1.0 + meta_info.compression_potential * 2,
            'lossy_ratio': 1.0 + meta_info.compression_potential * 5,
            'semantic_ratio': 1.0 + meta_info.compression_potential * 10
        }
        
        return ratios
    
    def visualize_meta_information(self, data: np.ndarray, 
                                  meta_info: MetaInformation) -> plt.Figure:
        """Visualize meta-information analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Meta-Information Analysis')
        
        # Original data
        if len(data.shape) >= 2:
            if len(data.shape) == 3:
                axes[0, 0].imshow(data)
            else:
                axes[0, 0].imshow(data, cmap='gray')
        else:
            axes[0, 0].plot(data)
        axes[0, 0].set_title('Original Data')
        
        # Information type
        axes[0, 1].text(0.5, 0.5, f"Type: {meta_info.information_type.value}", 
                        ha='center', va='center', fontsize=14)
        axes[0, 1].set_title('Information Type')
        axes[0, 1].axis('off')
        
        # Metrics bar chart
        metrics = {
            'Semantic Density': meta_info.semantic_density,
            'Compression Potential': meta_info.compression_potential,
            'Structural Complexity': meta_info.structural_complexity,
            'Confidence': meta_info.confidence
        }
        
        axes[1, 0].bar(metrics.keys(), metrics.values())
        axes[1, 0].set_title('Meta-Information Metrics')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim(0, 1)
        
        # Summary text
        summary = f"Information Type: {meta_info.information_type.value}\n"
        summary += f"Semantic Density: {meta_info.semantic_density:.3f}\n"
        summary += f"Compression Potential: {meta_info.compression_potential:.3f}\n"
        summary += f"Structural Complexity: {meta_info.structural_complexity:.3f}\n"
        summary += f"Analysis Confidence: {meta_info.confidence:.3f}"
        
        axes[1, 1].text(0.1, 0.5, summary, fontsize=12, va='center')
        axes[1, 1].set_title('Analysis Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig


# Export main classes
__all__ = ['MetaInformationExtractor', 'InformationType', 'MetaInformation']