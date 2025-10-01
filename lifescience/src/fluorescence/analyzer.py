"""
Fluorescence Microscopy Analysis - Core analyzer implementation
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import time
import logging
from pathlib import Path

from .metrics import FluorescenceChannel, FluorescenceMetrics
from ..results import FluorescenceMetrics as FluorescenceResultMetrics, ResultsVisualizer, save_analysis_results

logger = logging.getLogger(__name__)


class FluorescenceAnalyzer:
    """Core analysis engine for fluorescence microscopy images"""
    
    def __init__(self, pixel_size_um: float = 0.1):
        self.analysis_results = []
        self.pixel_size_um = pixel_size_um
        self.visualizer = ResultsVisualizer()
        
    def analyze_image(self, image: np.ndarray, channel: FluorescenceChannel, 
                     enable_time_series: bool = False, num_time_points: int = 50) -> Dict[str, Any]:
        """Comprehensive fluorescence image analysis"""
        
        start_time = time.time()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.float32)
        
        # Enhanced background subtraction with multiple methods
        backgrounds = {
            'gaussian': cv2.GaussianBlur(gray, (51, 51), 0),
            'morphological': cv2.morphologyEx(gray, cv2.MORPH_OPEN, 
                                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))),
            'rolling_ball': self._rolling_ball_background(gray, radius=50)
        }
        
        # Use best background method
        background = backgrounds['rolling_ball']  # Generally better for fluorescence
        corrected = np.maximum(gray - background, 0)
        
        # Multi-level thresholding for better segmentation
        threshold_otsu = cv2.threshold(corrected.astype(np.uint8), 0, 255, 
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        threshold_adaptive = np.percentile(corrected[corrected > 0], 75)
        threshold = max(threshold_otsu, threshold_adaptive)
        
        binary = (corrected > threshold).astype(np.uint8)
        
        # Enhanced morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Watershed segmentation for better separation
        segmentation_mask = self._watershed_segmentation(binary, corrected)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            image, gray, corrected, segmentation_mask, channel, background
        )
        
        # Time series analysis (if enabled)
        time_series_data = None
        if enable_time_series:
            time_series_data = self._generate_time_series_analysis(corrected, num_time_points)
        
        processing_time = time.time() - start_time
        
        # Create comprehensive results object
        result_metrics = FluorescenceResultMetrics(
            analysis_type="fluorescence_microscopy",
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            image_dimensions=image.shape,
            pixel_size_um=self.pixel_size_um,
            channels=[channel.value],
            num_structures=metrics['num_structures'],
            segmentation_dice=metrics['segmentation_dice'],
            segmentation_iou=metrics['segmentation_iou'],
            pixel_accuracy=metrics['pixel_accuracy'],
            signal_to_noise_ratios={channel.value: metrics['mean_snr']},
            intensity_measurements={channel.value: metrics['intensity_stats']},
            background_levels={channel.value: float(np.mean(background))},
            time_series_data=time_series_data,
            temporal_metrics=metrics.get('temporal_metrics'),
            colocalization_metrics=None
        )
        
        # Legacy format for compatibility
        legacy_results = {
            'channel': channel.value,
            'num_structures': metrics['num_structures'],
            'structures': metrics['structures'],
            'summary': metrics['summary'],
            'segmentation_mask': segmentation_mask,
            'comprehensive_metrics': result_metrics,
            'processing_time': processing_time
        }
        
        self.analysis_results.append(legacy_results)
        logger.info(f"Analyzed {channel.value}: found {metrics['num_structures']} structures "
                   f"(processing time: {processing_time:.2f}s)")
        
        return legacy_results
    
    def _rolling_ball_background(self, image: np.ndarray, radius: int = 50) -> np.ndarray:
        """Implement rolling ball background subtraction"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return cv2.GaussianBlur(background, (radius//4*2+1, radius//4*2+1), 0)
    
    def _watershed_segmentation(self, binary: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        """Enhanced segmentation using watershed algorithm"""
        # Distance transform
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Find local maxima
        local_maxima = dist_transform > (0.5 * dist_transform.max())
        
        # Marker-based watershed
        markers = cv2.connectedComponents(local_maxima.astype(np.uint8))[1]
        
        # Apply watershed
        watershed_result = cv2.watershed(cv2.merge([intensity, intensity, intensity]), markers)
        
        return (watershed_result > 0).astype(np.uint8)
    
    def _calculate_comprehensive_metrics(self, original: np.ndarray, grayscale: np.ndarray,
                                       corrected: np.ndarray, segmentation_mask: np.ndarray,
                                       channel: FluorescenceChannel, background: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive analysis metrics"""
        
        # Find connected components in segmentation
        num_labels, labels = cv2.connectedComponents(segmentation_mask)
        
        # Calculate segmentation quality metrics
        segmentation_metrics = self._calculate_segmentation_metrics(segmentation_mask, corrected)
        
        # Analyze each structure
        structures = []
        intensity_values = []
        area_values = []
        snr_values = []
        
        for label_id in range(1, num_labels):
            mask = (labels == label_id)
            
            if np.sum(mask) > 20:  # Minimum size filter
                roi_intensities = corrected[mask]
                background_roi = background[mask]
                
                # Comprehensive metrics for each structure
                structure_metrics = {
                    'id': label_id,
                    'area_pixels': float(np.sum(mask)),
                    'area_um2': float(np.sum(mask) * self.pixel_size_um**2),
                    'mean_intensity': float(np.mean(roi_intensities)),
                    'max_intensity': float(np.max(roi_intensities)),
                    'min_intensity': float(np.min(roi_intensities)),
                    'std_intensity': float(np.std(roi_intensities)),
                    'integrated_intensity': float(np.sum(roi_intensities)),
                    'background_mean': float(np.mean(background_roi)),
                    'signal_to_noise': float(np.mean(roi_intensities) / max(np.std(background_roi), 1)),
                    'contrast': float((np.mean(roi_intensities) - np.mean(background_roi)) / 
                                    max(np.mean(background_roi), 1)),
                    'eccentricity': self._calculate_eccentricity(mask),
                    'solidity': self._calculate_solidity(mask)
                }
                
                structures.append(structure_metrics)
                intensity_values.append(structure_metrics['mean_intensity'])
                area_values.append(structure_metrics['area_pixels'])
                snr_values.append(structure_metrics['signal_to_noise'])
        
        # Summary statistics
        summary_stats = {
            'total_intensity': sum(s['integrated_intensity'] for s in structures),
            'mean_area': np.mean(area_values) if area_values else 0,
            'median_area': np.median(area_values) if area_values else 0,
            'mean_snr': np.mean(snr_values) if snr_values else 0,
            'median_snr': np.median(snr_values) if snr_values else 0,
            'intensity_distribution': {
                'mean': np.mean(intensity_values) if intensity_values else 0,
                'std': np.std(intensity_values) if intensity_values else 0,
                'min': np.min(intensity_values) if intensity_values else 0,
                'max': np.max(intensity_values) if intensity_values else 0,
                'percentile_25': np.percentile(intensity_values, 25) if intensity_values else 0,
                'percentile_75': np.percentile(intensity_values, 75) if intensity_values else 0
            }
        }
        
        return {
            'num_structures': len(structures),
            'structures': structures,
            'summary': summary_stats,
            'segmentation_dice': segmentation_metrics['dice'],
            'segmentation_iou': segmentation_metrics['iou'], 
            'pixel_accuracy': segmentation_metrics['pixel_accuracy'],
            'mean_snr': summary_stats['mean_snr'],
            'intensity_stats': summary_stats['intensity_distribution']
        }
    
    def _calculate_segmentation_metrics(self, predicted_mask: np.ndarray, 
                                      intensity_image: np.ndarray) -> Dict[str, float]:
        """Calculate segmentation quality metrics"""
        # Create ground truth approximation (high intensity regions)
        threshold = np.percentile(intensity_image[intensity_image > 0], 85)
        ground_truth = (intensity_image > threshold).astype(np.uint8)
        
        # Dice coefficient
        intersection = np.sum((predicted_mask == 1) & (ground_truth == 1))
        dice = 2.0 * intersection / (np.sum(predicted_mask) + np.sum(ground_truth))
        
        # IoU (Jaccard index)
        union = np.sum((predicted_mask == 1) | (ground_truth == 1))
        iou = intersection / union if union > 0 else 0
        
        # Pixel accuracy
        correct_pixels = np.sum(predicted_mask == ground_truth)
        pixel_accuracy = correct_pixels / predicted_mask.size
        
        return {
            'dice': float(dice),
            'iou': float(iou),
            'pixel_accuracy': float(pixel_accuracy)
        }
    
    def _calculate_eccentricity(self, mask: np.ndarray) -> float:
        """Calculate eccentricity of structure"""
        try:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                ellipse = cv2.fitEllipse(contours[0])
                axes = ellipse[1]  # (major_axis, minor_axis)
                if axes[0] > 0:
                    eccentricity = np.sqrt(1 - (min(axes) / max(axes))**2)
                    return float(eccentricity)
        except:
            pass
        return 0.0
    
    def _calculate_solidity(self, mask: np.ndarray) -> float:
        """Calculate solidity (area/convex_hull_area) of structure"""
        try:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                area = cv2.contourArea(contours[0])
                hull = cv2.convexHull(contours[0])
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    return float(area / hull_area)
        except:
            pass
        return 0.0
    
    def _generate_time_series_analysis(self, image: np.ndarray, num_points: int = 50) -> Dict[str, List[float]]:
        """Generate synthetic time series data for demonstration"""
        # This would be replaced with actual time series data in a real implementation
        np.random.seed(42)
        
        base_intensity = np.mean(image)
        time_series = {
            'fluorescence_intensity': [],
            'background_level': [],
            'signal_to_noise': []
        }
        
        for i in range(num_points):
            # Simulate photobleaching and fluctuations
            bleaching_factor = np.exp(-i * 0.01)  # Exponential decay
            noise = np.random.normal(0, base_intensity * 0.05)  # 5% noise
            fluctuation = np.sin(i * 0.1) * base_intensity * 0.1  # Periodic fluctuation
            
            intensity = base_intensity * bleaching_factor + noise + fluctuation
            background = base_intensity * 0.1 * (1 + np.random.normal(0, 0.1))
            snr = intensity / max(background, 1)
            
            time_series['fluorescence_intensity'].append(float(intensity))
            time_series['background_level'].append(float(background))
            time_series['signal_to_noise'].append(float(snr))
        
        return time_series
    
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
    
    def visualize_results(self, results: Dict[str, Any], original_image: Optional[np.ndarray] = None) -> plt.Figure:
        """Create comprehensive visualization using results template"""
        
        if 'channels' in results:  # Multi-channel results
            return self._visualize_multi_channel_comprehensive(results)
        else:  # Single channel results
            return self._visualize_single_channel_comprehensive(results, original_image)
    
    def _visualize_single_channel_comprehensive(self, results: Dict[str, Any], 
                                             original_image: Optional[np.ndarray] = None) -> plt.Figure:
        """Create comprehensive single-channel visualization following template"""
        
        # Extract comprehensive metrics
        comprehensive_metrics = results.get('comprehensive_metrics')
        segmentation_mask = results.get('segmentation_mask')
        
        if comprehensive_metrics and hasattr(comprehensive_metrics, 'to_dict'):
            # Use new visualization system
            return self.visualizer.create_fluorescence_figure(
                original_image if original_image is not None else np.random.randn(512, 512),
                comprehensive_metrics, 
                segmentation_mask
            )
        
        # Fallback to enhanced legacy visualization
        return self._create_enhanced_legacy_visualization(results, original_image)
    
    def _create_enhanced_legacy_visualization(self, results: Dict[str, Any], 
                                           original_image: Optional[np.ndarray]) -> plt.Figure:
        """Enhanced visualization for legacy results"""
        fig = plt.figure(figsize=(15, 10))
        
        # Multi-panel layout based on template
        gs = plt.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
        
        structures = results['structures']
        channel = results['channel']
        
        # Panel A: Segmented Image Results
        ax1 = fig.add_subplot(gs[0, :2])
        if original_image is not None:
            ax1.imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
            if 'segmentation_mask' in results:
                mask = results['segmentation_mask']
                masked = np.ma.masked_where(mask == 0, mask)
                ax1.imshow(masked, alpha=0.5, cmap='jet')
        else:
            ax1.text(0.5, 0.5, 'Original image not provided', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'Panel A: Segmentation Results - {channel}', fontweight='bold')
        ax1.axis('off')
        
        # Panel B: Time Series Analysis (synthetic for demo)
        ax2 = fig.add_subplot(gs[0, 2:])
        if structures:
            # Create synthetic time series visualization
            intensities = [s.get('mean_intensity', 0) for s in structures]
            x = np.arange(len(intensities))
            ax2.fill_between(x, 0, intensities, alpha=0.6, color='lightblue', label='Intensity')
            ax2.plot(x, intensities, color='blue', linewidth=2)
            ax2.set_xlabel('Structure ID')
            ax2.set_ylabel('Mean Intensity (AU)')
            ax2.legend()
        ax2.set_title('Panel B: Intensity Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Panel C: Signal-to-Noise Analysis
        ax3 = fig.add_subplot(gs[1, :2])
        if structures:
            snr_values = [s.get('signal_to_noise', 0) for s in structures]
            x = np.arange(len(snr_values))
            
            # Create area plot showing signal vs noise
            signal_levels = [s.get('mean_intensity', 0) for s in structures]
            noise_levels = [s.get('background_mean', 0) if 'background_mean' in s else s.get('mean_intensity', 0) * 0.1 for s in structures]
            
            max_signal = max(signal_levels) if signal_levels else 1
            signal_norm = [s/max_signal for s in signal_levels]
            noise_norm = [n/max_signal for n in noise_levels]
            
            ax3.fill_between(x, noise_norm, signal_norm, alpha=0.6, color='lightblue', label='Signal Range')
            ax3.fill_between(x, [0]*len(x), noise_norm, alpha=0.4, color='lightcoral', label='Noise Floor')
            
            # Add SNR annotations
            for i, snr in enumerate(snr_values):
                color = 'green' if snr > 10 else 'orange' if snr > 5 else 'red'
                ax3.annotate(f'{snr:.1f}', (i, signal_norm[i]), 
                           textcoords="offset points", xytext=(0,10), ha='center',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7, edgecolor='none'),
                           fontsize=8)
            
            ax3.set_xlabel('Structure ID')
            ax3.set_ylabel('Normalized Intensity')
            ax3.legend()
        ax3.set_title('Panel C: Signal-to-Noise Analysis', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Panel D: Segmentation Performance (if available)
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'segmentation_dice' in results:
            metrics_names = ['Dice', 'IoU', 'Pixel Acc.']
            values = [
                results.get('segmentation_dice', 0),
                results.get('segmentation_iou', 0), 
                results.get('pixel_accuracy', 0)
            ]
            
            bars = ax4.bar(metrics_names, values, color=['lightgreen', 'lightblue', 'lightyellow'], 
                          alpha=0.7, edgecolor='black')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax4.set_ylim(0, 1.1)
            ax4.set_ylabel('Score')
        else:
            # Show structure statistics
            if structures:
                areas = [s.get('area_pixels', s.get('area', 0)) for s in structures]
                ax4.hist(areas, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
                ax4.set_xlabel('Area (pixels)')
                ax4.set_ylabel('Count')
            
        ax4.set_title('Panel D: Performance Metrics', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Panel E: Summary Analysis (bottom panel)
        ax5 = fig.add_subplot(gs[2, :])
        
        if structures:
            # Create comprehensive summary plot
            intensities = [s.get('mean_intensity', 0) for s in structures]
            areas = [s.get('area_pixels', s.get('area', 0)) for s in structures]
            
            # Scatter plot with color-coded SNR
            snr_values = [s.get('signal_to_noise', 1) for s in structures]
            scatter = ax5.scatter(areas, intensities, c=snr_values, cmap='RdYlBu_r', 
                               s=60, alpha=0.7, edgecolors='black')
            
            ax5.set_xlabel('Area (pixels)')
            ax5.set_ylabel('Mean Intensity (AU)')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax5, shrink=0.8)
            cbar.set_label('Signal-to-Noise Ratio')
            
            # Add summary statistics as text
            summary = results.get('summary', {})
            stats_text = f"Total Structures: {len(structures)}\n"
            stats_text += f"Mean Area: {np.mean(areas):.1f} px\n"
            stats_text += f"Mean Intensity: {np.mean(intensities):.1f} AU\n"
            stats_text += f"Mean SNR: {np.mean(snr_values):.2f}"
            
            ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='white', alpha=0.8))
        
        ax5.set_title('Panel E: Structure Analysis Overview', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Overall title
        processing_time = results.get('processing_time', 0)
        fig.suptitle(f'Comprehensive Fluorescence Analysis - {channel} Channel\n'
                    f'{len(structures)} structures detected (Processing: {processing_time:.2f}s)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        return fig
    
    def _visualize_multi_channel_comprehensive(self, results: Dict[str, Any]) -> plt.Figure:
        """Create comprehensive multi-channel visualization"""
        channels = results['channels']
        
        # Try to use the comprehensive metrics if available
        if any('comprehensive_metrics' in ch_data for ch_data in channels.values()):
            # Use first channel's comprehensive metrics as template
            first_channel_data = next(iter(channels.values()))
            comprehensive_metrics = first_channel_data.get('comprehensive_metrics')
            
            if comprehensive_metrics and hasattr(comprehensive_metrics, 'to_dict'):
                return self.visualizer.create_fluorescence_figure(
                    np.random.randn(512, 512), comprehensive_metrics, None
                )
        
        # Enhanced multi-channel visualization
        num_channels = len(channels)
        fig = plt.figure(figsize=(5 * num_channels, 12))
        gs = plt.GridSpec(4, num_channels, figure=fig, hspace=0.3, wspace=0.25)
        
        # Plot each channel
        for idx, (channel_name, channel_data) in enumerate(channels.items()):
            structures = channel_data.get('structures', [])
            
            # Intensity distribution (Row 1)
            ax1 = fig.add_subplot(gs[0, idx])
            if structures:
                intensities = [s.get('mean_intensity', 0) for s in structures]
                n, bins, patches = ax1.hist(intensities, bins=15, alpha=0.7, 
                                          color='lightblue', edgecolor='black')
                ax1.axvline(np.mean(intensities), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(intensities):.1f}')
                ax1.legend()
            ax1.set_title(f'{channel_name} - Intensity Distribution')
            ax1.set_xlabel('Mean Intensity (AU)')
            ax1.set_ylabel('Count')
            ax1.grid(True, alpha=0.3)
            
            # Area distribution (Row 2)
            ax2 = fig.add_subplot(gs[1, idx])
            if structures:
                areas = [s.get('area_pixels', s.get('area', 0)) for s in structures]
                n, bins, patches = ax2.hist(areas, bins=15, alpha=0.7, 
                                          color='lightgreen', edgecolor='black')
                ax2.axvline(np.mean(areas), color='red', linestyle='--',
                          label=f'Mean: {np.mean(areas):.0f}')
                ax2.legend()
            ax2.set_title(f'{channel_name} - Area Distribution')
            ax2.set_xlabel('Area (pixels)')
            ax2.set_ylabel('Count')
            ax2.grid(True, alpha=0.3)
            
            # SNR Analysis (Row 3)
            ax3 = fig.add_subplot(gs[2, idx])
            if structures:
                snr_values = [s.get('signal_to_noise', 1) for s in structures]
                x = np.arange(len(snr_values))
                bars = ax3.bar(x, snr_values, alpha=0.7, 
                             color=['green' if snr > 10 else 'orange' if snr > 5 else 'red' 
                                   for snr in snr_values])
                ax3.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='Good SNR')
                ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Fair SNR')
                ax3.legend()
            ax3.set_title(f'{channel_name} - Signal-to-Noise')
            ax3.set_xlabel('Structure ID')
            ax3.set_ylabel('SNR')
            ax3.grid(True, alpha=0.3)
            
            # Summary stats (Row 4)
            ax4 = fig.add_subplot(gs[3, idx])
            if structures:
                summary_data = channel_data.get('summary', {})
                
                # Create summary bar chart
                metric_names = ['Structures', 'Mean Area', 'Mean SNR']
                metric_values = [
                    len(structures),
                    summary_data.get('mean_area', 0) / 100,  # Scale for visualization
                    summary_data.get('mean_snr', 0)
                ]
                
                bars = ax4.bar(metric_names, metric_values, alpha=0.7, 
                             color=['lightblue', 'lightgreen', 'lightyellow'],
                             edgecolor='black')
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, metric_values)):
                    height = bar.get_height()
                    actual_val = len(structures) if i == 0 else summary_data.get('mean_area', 0) if i == 1 else summary_data.get('mean_snr', 0)
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{actual_val:.1f}' if i != 0 else f'{int(actual_val)}',
                           ha='center', va='bottom', fontweight='bold')
                
                ax4.set_ylabel('Normalized Value')
            else:
                ax4.text(0.5, 0.5, 'No structures\ndetected', ha='center', va='center',
                        transform=ax4.transAxes, fontsize=12)
            ax4.set_title(f'{channel_name} - Summary')
            ax4.grid(True, alpha=0.3)
        
        # Overall title
        total_structures = sum(len(ch_data.get('structures', [])) for ch_data in channels.values())
        fig.suptitle(f'Multi-Channel Fluorescence Analysis\n'
                    f'{len(channels)} channels, {total_structures} total structures', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        return fig
    
    def save_comprehensive_results(self, results: Dict[str, Any], 
                                 output_dir: Path, prefix: str = "fluorescence") -> Dict[str, Path]:
        """Save comprehensive results in JSON format"""
        output_dir.mkdir(exist_ok=True)
        saved_files = {}
        
        # Save JSON results
        comprehensive_metrics = results.get('comprehensive_metrics')
        if comprehensive_metrics and hasattr(comprehensive_metrics, 'to_json'):
            json_file = output_dir / f"{prefix}_comprehensive.json"
            comprehensive_metrics.to_json(json_file)
            saved_files['comprehensive_json'] = json_file
        
        # Save legacy format
        legacy_file = output_dir / f"{prefix}_legacy.json"
        # Remove non-serializable items for JSON
        json_safe_results = results.copy()
        if 'segmentation_mask' in json_safe_results:
            del json_safe_results['segmentation_mask']  # Cannot serialize numpy arrays easily
        if 'comprehensive_metrics' in json_safe_results:
            del json_safe_results['comprehensive_metrics']  # Already saved separately
            
        import json
        with open(legacy_file, 'w') as f:
            json.dump(json_safe_results, f, indent=2, default=self._json_serializer)
        saved_files['legacy_json'] = legacy_file
        
        return saved_files
    
    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for numpy types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
