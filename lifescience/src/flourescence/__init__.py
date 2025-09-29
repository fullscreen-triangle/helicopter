# Fluorescence Microscopy Analysis Module
"""
Independent module applying the Helicopter framework to fluorescence microscopy images.
Specialized for analyzing fluorescent proteins, cellular structures, and dynamic processes.

Key Features:
- FluorescenceAnalyzer: Core analysis engine for fluorescence images
- ChannelProcessor: Multi-channel fluorescence processing
- IntensityQuantifier: Precise fluorescence intensity measurements
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import ndimage

logger = logging.getLogger(__name__)


class FluorescenceChannel(Enum):
    """Common fluorescence channels"""
    DAPI = "dapi"  # DNA/nucleus
    GFP = "gfp"    # Green fluorescent protein
    RFP = "rfp"    # Red fluorescent protein
    FITC = "fitc"  # Fluorescein


@dataclass
class FluorescenceMetrics:
    """Fluorescence analysis metrics"""
    mean_intensity: float
    max_intensity: float
    integrated_intensity: float
    area: float
    signal_to_noise: float


class FluorescenceAnalyzer:
    """Core analysis engine for fluorescence microscopy images"""
    
    def __init__(self):
        self.analysis_results = []
        
    def analyze_image(self, image: np.ndarray, channel: FluorescenceChannel) -> Dict[str, Any]:
        """Analyze fluorescence image"""
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.float32)
        
        # Background subtraction
        background = cv2.GaussianBlur(gray, (51, 51), 0)
        corrected = np.maximum(gray - background, 0)
        
        # Threshold to find structures
        threshold = np.percentile(corrected[corrected > 0], 75)
        binary = (corrected > threshold).astype(np.uint8)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary)
        
        # Analyze each structure
        structures = []
        for label_id in range(1, num_labels):
            mask = (labels == label_id)
            
            if np.sum(mask) > 50:  # Minimum size filter
                # Calculate metrics
                roi_intensities = corrected[mask]
                
                metrics = FluorescenceMetrics(
                    mean_intensity=float(np.mean(roi_intensities)),
                    max_intensity=float(np.max(roi_intensities)),
                    integrated_intensity=float(np.sum(roi_intensities)),
                    area=float(np.sum(mask)),
                    signal_to_noise=float(np.mean(roi_intensities) / np.std(roi_intensities))
                )
                
                structures.append({
                    'id': label_id,
                    'mask': mask,
                    'metrics': metrics
                })
        
        results = {
            'channel': channel.value,
            'num_structures': len(structures),
            'structures': structures,
            'summary': {
                'total_intensity': sum(s['metrics'].integrated_intensity for s in structures),
                'mean_area': np.mean([s['metrics'].area for s in structures]) if structures else 0,
                'mean_snr': np.mean([s['metrics'].signal_to_noise for s in structures]) if structures else 0
            }
        }
        
        self.analysis_results.append(results)
        logger.info(f"Analyzed {channel.value}: found {len(structures)} structures")
        
        return results
    
    def visualize_results(self, results: Dict[str, Any]) -> plt.Figure:
        """Visualize analysis results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Fluorescence Analysis: {results['channel']}")
        
        structures = results['structures']
        
        if not structures:
            axes[0, 0].text(0.5, 0.5, 'No structures detected', ha='center', va='center')
            return fig
        
        # Intensity distribution
        intensities = [s['metrics'].mean_intensity for s in structures]
        axes[0, 0].hist(intensities, bins=15)
        axes[0, 0].set_title('Mean Intensity Distribution')
        
        # Area distribution
        areas = [s['metrics'].area for s in structures]
        axes[0, 1].hist(areas, bins=15)
        axes[0, 1].set_title('Area Distribution')
        
        # Intensity vs Area
        axes[1, 0].scatter(areas, intensities)
        axes[1, 0].set_xlabel('Area')
        axes[1, 0].set_ylabel('Mean Intensity')
        axes[1, 0].set_title('Area vs Intensity')
        
        # Summary stats
        summary = results['summary']
        summary_text = f"Structures: {results['num_structures']}\n"
        summary_text += f"Total Intensity: {summary['total_intensity']:.0f}\n"
        summary_text += f"Mean Area: {summary['mean_area']:.0f}\n"
        summary_text += f"Mean SNR: {summary['mean_snr']:.2f}"
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, va='center')
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig


# Export main classes
__all__ = ['FluorescenceAnalyzer', 'FluorescenceChannel', 'FluorescenceMetrics'] 