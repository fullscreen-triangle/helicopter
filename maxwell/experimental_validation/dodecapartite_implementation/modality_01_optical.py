"""
Modality 1: Optical Microscopy Processing

Implements optical image analysis for brightfield and phase contrast microscopy.
Extracts structural information, intensity distributions, and prepares data
for categorical framework.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
from skimage import filters, morphology, segmentation, measure
from skimage.feature import peak_local_maxima
from typing import Dict, Tuple, Optional
import cv2


class OpticalModality:
    """
    Process optical microscopy images (brightfield, phase contrast).
    
    Extracts:
    - Intensity distributions
    - Edge detection and boundaries
    - Texture features
    - Initial structure estimates
    """
    
    def __init__(self, pixel_size_nm: float = 100.0):
        """
        Initialize optical modality processor.
        
        Args:
            pixel_size_nm: Physical pixel size in nanometers
        """
        self.pixel_size_nm = pixel_size_nm
        
    def process(self, image: np.ndarray, image_type: str = "brightfield") -> Dict:
        """
        Process optical microscopy image.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            image_type: "brightfield" or "phase_contrast"
            
        Returns:
            Dictionary with extracted features and structure estimates
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image_type == "brightfield":
                # Brightfield: use average or specific channel
                image = np.mean(image, axis=2)
            else:
                # Phase contrast: typically single channel
                image = image[:, :, 0] if image.shape[2] > 0 else image
        
        # Normalize
        image = self._normalize(image)
        
        # Extract features
        features = {
            "intensity_distribution": self._intensity_distribution(image),
            "edges": self._edge_detection(image),
            "texture": self._texture_features(image),
            "initial_segmentation": self._initial_segmentation(image, image_type),
            "spatial_frequency": self._spatial_frequency_analysis(image),
            "pixel_size_nm": self.pixel_size_nm,
        }
        
        return features
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        return image
    
    def _intensity_distribution(self, image: np.ndarray) -> Dict:
        """
        Extract intensity distribution statistics.
        
        Returns:
            Dictionary with mean, std, histogram, etc.
        """
        return {
            "mean": float(np.mean(image)),
            "std": float(np.std(image)),
            "min": float(np.min(image)),
            "max": float(np.max(image)),
            "histogram": np.histogram(image, bins=256)[0].astype(float),
            "percentiles": {
                "p10": float(np.percentile(image, 10)),
                "p25": float(np.percentile(image, 25)),
                "p50": float(np.percentile(image, 50)),
                "p75": float(np.percentile(image, 75)),
                "p90": float(np.percentile(image, 90)),
            }
        }
    
    def _edge_detection(self, image: np.ndarray) -> Dict:
        """
        Detect edges and boundaries.
        
        Returns:
            Dictionary with edge maps and boundary information
        """
        # Sobel edge detection
        sobel_x = sobel(image, axis=1)
        sobel_y = sobel(image, axis=0)
        edge_magnitude = np.hypot(sobel_x, sobel_y)
        edge_magnitude = edge_magnitude / edge_magnitude.max() if edge_magnitude.max() > 0 else edge_magnitude
        
        # Canny edges
        canny_edges = filters.canny(image, sigma=1.0)
        
        return {
            "sobel_magnitude": edge_magnitude,
            "canny_edges": canny_edges.astype(float),
            "edge_density": float(np.mean(edge_magnitude)),
            "boundary_length": float(np.sum(canny_edges)),
        }
    
    def _texture_features(self, image: np.ndarray) -> Dict:
        """
        Extract texture features using local binary patterns and gradients.
        
        Returns:
            Dictionary with texture statistics
        """
        # Local standard deviation (texture measure)
        kernel_size = 5
        local_std = ndimage.generic_filter(
            image, np.std, size=kernel_size, mode='reflect'
        )
        
        # Gradient magnitude (texture variation)
        grad_mag = np.hypot(
            ndimage.sobel(image, axis=1),
            ndimage.sobel(image, axis=0)
        )
        
        return {
            "local_std_mean": float(np.mean(local_std)),
            "local_std_std": float(np.std(local_std)),
            "gradient_mean": float(np.mean(grad_mag)),
            "gradient_std": float(np.std(grad_mag)),
        }
    
    def _initial_segmentation(self, image: np.ndarray, image_type: str) -> Dict:
        """
        Perform initial structure segmentation.
        
        For brightfield: threshold-based
        For phase contrast: watershed or active contours
        
        Returns:
            Dictionary with segmentation masks and regions
        """
        # Adaptive thresholding
        if image_type == "brightfield":
            # Otsu thresholding
            threshold = filters.threshold_otsu(image)
            binary = image > threshold
            
            # Morphological operations
            binary = morphology.binary_opening(binary, morphology.disk(3))
            binary = morphology.binary_closing(binary, morphology.disk(5))
            
        else:  # phase_contrast
            # Phase contrast: cells appear as dark rings
            # Invert and threshold
            inverted = 1.0 - image
            threshold = filters.threshold_otsu(inverted)
            binary = inverted > threshold
            
            # Fill holes (cells are dark centers)
            binary = morphology.binary_fill_holes(binary)
        
        # Label connected components
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)
        
        # Extract region properties
        region_data = []
        for region in regions:
            if region.area > 50:  # Filter small noise
                region_data.append({
                    "area": region.area,
                    "centroid": region.centroid,
                    "bbox": region.bbox,
                    "eccentricity": region.eccentricity,
                    "solidity": region.solidity,
                    "perimeter": region.perimeter,
                })
        
        return {
            "binary_mask": binary.astype(float),
            "labeled_regions": labeled.astype(int),
            "num_regions": len(region_data),
            "regions": region_data,
        }
    
    def _spatial_frequency_analysis(self, image: np.ndarray) -> Dict:
        """
        Analyze spatial frequency content (Fourier domain).
        
        Returns:
            Dictionary with frequency domain features
        """
        # 2D FFT
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Frequency coordinates
        h, w = image.shape
        freq_y = np.fft.fftshift(np.fft.fftfreq(h))
        freq_x = np.fft.fftshift(np.fft.fftfreq(w))
        
        # Dominant frequencies
        # Find peaks in frequency domain
        peaks = peak_local_maxima(magnitude, min_distance=10, threshold_abs=magnitude.max() * 0.1)
        
        # Power spectral density
        psd = magnitude ** 2
        
        return {
            "dominant_frequencies": [
                (float(freq_x[x]), float(freq_y[y])) 
                for y, x in zip(peaks[0], peaks[1])
            ],
            "total_power": float(np.sum(psd)),
            "high_freq_power": float(np.sum(psd[magnitude > np.percentile(magnitude, 90)])),
        }
    
    def estimate_structure(self, features: Dict) -> Dict:
        """
        Estimate initial structure from optical features.
        
        This provides starting point for sequential exclusion algorithm.
        
        Returns:
            Dictionary with structure estimates
        """
        segmentation = features["initial_segmentation"]
        
        structure = {
            "num_cells": segmentation["num_regions"],
            "cell_positions": [r["centroid"] for r in segmentation["regions"]],
            "cell_sizes": [r["area"] for r in segmentation["regions"]],
            "cell_shapes": [r["eccentricity"] for r in segmentation["regions"]],
            "confidence": 0.5,  # Initial low confidence
            "modality": "optical",
        }
        
        return structure


if __name__ == "__main__":
    # Test with synthetic image
    test_image = np.random.rand(512, 512)
    test_image[200:300, 200:300] = 0.8  # Simulate cell
    
    processor = OpticalModality(pixel_size_nm=100.0)
    features = processor.process(test_image, image_type="brightfield")
    
    print("Optical features extracted:")
    print(f"  Intensity mean: {features['intensity_distribution']['mean']:.3f}")
    print(f"  Edge density: {features['edges']['edge_density']:.3f}")
    print(f"  Regions detected: {features['initial_segmentation']['num_regions']}")
