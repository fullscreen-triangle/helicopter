"""
Fluorescence Microscopy Analysis - Core analyzer implementation
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

from .metrics import FluorescenceChannel, FluorescenceMetrics

logger = logging.getLogger(__name__)


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
    
    def analyze_multi_channel(self, images: Dict[FluorescenceChannel, np.ndarray]) -> Dict[str, Any]:
        """Analyze multiple fluorescence channels"""
        
        logger.info(f"Analyzing {len(images)} channels: {[ch.value for ch in images.keys()]}")
        
        # Analyze each channel
        channel_results = {}
        for channel, image in images.items():
            result = self.analyze_image(image, channel)
            channel_results[channel.value] = result
        
        # Cross-channel analysis
        colocalization_results = self._analyze_colocalization(images, channel_results)
        
        # Combined results
        multi_channel_results = {
            'channels': channel_results,
            'colocalization': colocalization_results,
            'summary': {
                'total_channels': len(images),
                'total_structures': sum(len(r['structures']) for r in channel_results.values())
            }
        }
        
        return multi_channel_results
    
    def _analyze_colocalization(self, images: Dict[FluorescenceChannel, np.ndarray],
                               channel_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze colocalization between channels"""
        
        channels = list(images.keys())
        if len(channels) < 2:
            return {'error': 'Need at least 2 channels for colocalization analysis'}
        
        colocalization_results = {}
        
        # Pairwise colocalization analysis
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                ch1, ch2 = channels[i], channels[j]
                img1, img2 = images[ch1], images[ch2]
                
                # Calculate colocalization metrics
                colocalization = self._calculate_colocalization_metrics(img1, img2)
                
                pair_key = f"{ch1.value}_{ch2.value}"
                colocalization_results[pair_key] = colocalization
        
        return colocalization_results
    
    def _calculate_colocalization_metrics(self, image1: np.ndarray, 
                                        image2: np.ndarray) -> Dict[str, float]:
        """Calculate colocalization metrics between two images"""
        
        # Ensure same shape
        if image1.shape != image2.shape:
            logger.warning("Images have different shapes for colocalization analysis")
            return {'error': 'Shape mismatch'}
        
        # Flatten images
        img1_flat = image1.flatten().astype(np.float64)
        img2_flat = image2.flatten().astype(np.float64)
        
        # Remove zero pixels (background)
        nonzero_mask = (img1_flat > 0) & (img2_flat > 0)
        if np.sum(nonzero_mask) == 0:
            return {'error': 'No overlapping signal'}
        
        img1_nonzero = img1_flat[nonzero_mask]
        img2_nonzero = img2_flat[nonzero_mask]
        
        # Pearson correlation coefficient
        pearson_r = np.corrcoef(img1_nonzero, img2_nonzero)[0, 1]
        if np.isnan(pearson_r):
            pearson_r = 0.0
        
        # Simple colocalization metrics
        img1_threshold = np.percentile(img1_nonzero, 50)
        img2_threshold = np.percentile(img2_nonzero, 50)
        
        img1_above_threshold = img1_nonzero > img1_threshold
        img2_above_threshold = img2_nonzero > img2_threshold
        
        colocalized_pixels = img1_above_threshold & img2_above_threshold
        
        if np.sum(img1_above_threshold) > 0:
            manders_m1 = np.sum(colocalized_pixels) / np.sum(img1_above_threshold)
        else:
            manders_m1 = 0.0
        
        if np.sum(img2_above_threshold) > 0:
            manders_m2 = np.sum(colocalized_pixels) / np.sum(img2_above_threshold)
        else:
            manders_m2 = 0.0
        
        return {
            'pearson_correlation': float(pearson_r),
            'manders_m1': float(manders_m1),
            'manders_m2': float(manders_m2),
            'colocalized_pixels': int(np.sum(colocalized_pixels)),
            'total_pixels': int(len(img1_nonzero))
        }
    
    def visualize_results(self, results: Dict[str, Any]) -> plt.Figure:
        """Visualize analysis results"""
        
        if 'channels' in results:  # Multi-channel results
            return self._visualize_multi_channel_results(results)
        else:  # Single channel results
            return self._visualize_single_channel_results(results)
    
    def _visualize_single_channel_results(self, results: Dict[str, Any]) -> plt.Figure:
        """Visualize single channel analysis results"""
        
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
    
    def _visualize_multi_channel_results(self, results: Dict[str, Any]) -> plt.Figure:
        """Visualize multi-channel analysis results"""
        
        channels = results['channels']
        num_channels = len(channels)
        
        fig, axes = plt.subplots(2, num_channels, figsize=(4 * num_channels, 8))
        if num_channels == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Multi-Channel Fluorescence Analysis', fontsize=16)
        
        # Plot each channel
        for idx, (channel_name, channel_data) in enumerate(channels.items()):
            structures = channel_data['structures']
            
            if structures:
                intensities = [s['metrics'].mean_intensity for s in structures]
                areas = [s['metrics'].area for s in structures]
                
                # Intensity histogram
                axes[0, idx].hist(intensities, bins=15, alpha=0.7)
                axes[0, idx].set_title(f'{channel_name} - Intensity')
                axes[0, idx].set_xlabel('Mean Intensity')
                axes[0, idx].set_ylabel('Count')
                
                # Area histogram  
                axes[1, idx].hist(areas, bins=15, alpha=0.7, color='orange')
                axes[1, idx].set_title(f'{channel_name} - Area')
                axes[1, idx].set_xlabel('Area (pixels)')
                axes[1, idx].set_ylabel('Count')
            else:
                axes[0, idx].text(0.5, 0.5, f'No data\n{channel_name}', ha='center', va='center')
                axes[1, idx].text(0.5, 0.5, f'No data\n{channel_name}', ha='center', va='center')
        
        plt.tight_layout()
        return fig
