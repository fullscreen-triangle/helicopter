"""
Feature extraction for image regions.

Extracts features that constrain categorical state selection:
- Color histograms
- Texture descriptors
- Edge orientations
- Spatial moments
"""

import numpy as np
from typing import Dict, Any, TYPE_CHECKING
from skimage import feature, color, filters
from scipy import ndimage

if TYPE_CHECKING:
    from .region import Region


class FeatureExtractor:
    """
    Extract features from image regions.

    Features constrain categorical completion by providing
    information about:
    - Electron configurations (color)
    - Vibrational modes (texture)
    - Dipole alignments (edges)
    - Spatial structure (moments)
    """

    def __init__(
        self,
        color_bins: int = 8,
        texture_scales: int = 4,
        edge_bins: int = 8
    ):
        """
        Initialize feature extractor.

        Args:
            color_bins: Number of bins for color histograms
            texture_scales: Number of scales for texture analysis
            edge_bins: Number of bins for edge orientation
        """
        self.color_bins = color_bins
        self.texture_scales = texture_scales
        self.edge_bins = edge_bins

    def extract(self, region: 'Region') -> Dict[str, Any]:
        """
        Extract all features from region.

        Returns:
            Dict containing:
            - color_histogram: RGB distribution
            - texture_features: Multi-scale texture
            - edge_features: Edge orientation histogram
            - spatial_moments: Shape descriptors
        """
        pixels = region.get_pixels()
        mask = region.mask

        features = {}

        # Color features
        features['color_histogram'] = self._extract_color_histogram(pixels)
        features['color_moments'] = self._extract_color_moments(pixels)

        # Texture features
        features['texture_features'] = self._extract_texture(region.image_data, mask)

        # Edge features
        features['edge_features'] = self._extract_edges(region.image_data, mask)

        # Spatial features
        features['spatial_moments'] = self._extract_spatial_moments(mask)

        return features

    def _extract_color_histogram(self, pixels: np.ndarray) -> np.ndarray:
        """
        Extract color histogram.

        Represents electron configuration constraints.

        Args:
            pixels: Region pixels (N x C)

        Returns:
            Normalized histogram
        """
        if pixels.size == 0:
            return np.zeros(self.color_bins * 3)

        # Separate RGB channels
        histograms = []

        for channel in range(min(3, pixels.shape[-1])):
            hist, _ = np.histogram(
                pixels[:, channel],
                bins=self.color_bins,
                range=(0, 256),
                density=True
            )
            histograms.append(hist)

        # Concatenate channels
        color_hist = np.concatenate(histograms)

        return color_hist

    def _extract_color_moments(self, pixels: np.ndarray) -> np.ndarray:
        """
        Extract color moments (mean, std, skewness).

        Args:
            pixels: Region pixels

        Returns:
            Array of moments
        """
        if pixels.size == 0:
            return np.zeros(9)

        moments = []

        for channel in range(min(3, pixels.shape[-1])):
            channel_data = pixels[:, channel]

            mean = np.mean(channel_data)
            std = np.std(channel_data)
            skew = np.mean(((channel_data - mean) / (std + 1e-10)) ** 3)

            moments.extend([mean, std, skew])

        return np.array(moments)

    def _extract_texture(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract texture features using Local Binary Patterns.

        Represents vibrational mode constraints.

        Args:
            image: Full image
            mask: Region mask

        Returns:
            Texture feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image

        # Compute LBP
        radius = 1
        n_points = 8 * radius

        try:
            lbp = feature.local_binary_pattern(
                gray,
                n_points,
                radius,
                method='uniform'
            )

            # Extract histogram from masked region
            lbp_masked = lbp[mask]

            if lbp_masked.size == 0:
                return np.zeros(self.texture_scales * 8)

            hist, _ = np.histogram(
                lbp_masked,
                bins=10,
                density=True
            )

            # Pad to expected size
            texture_vec = np.pad(
                hist,
                (0, max(0, self.texture_scales * 8 - len(hist))),
                mode='constant'
            )[:self.texture_scales * 8]

        except Exception:
            # Fallback
            texture_vec = np.zeros(self.texture_scales * 8)

        return texture_vec

    def _extract_edges(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract edge orientation histogram.

        Represents dipole alignment constraints.

        Args:
            image: Full image
            mask: Region mask

        Returns:
            Edge orientation histogram
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image

        # Compute gradients
        grad_y = filters.sobel_v(gray)
        grad_x = filters.sobel_h(gray)

        # Compute orientations
        orientations = np.arctan2(grad_y, grad_x)

        # Extract from masked region
        orientations_masked = orientations[mask]

        if orientations_masked.size == 0:
            return np.zeros(self.edge_bins)

        # Histogram of orientations
        hist, _ = np.histogram(
            orientations_masked,
            bins=self.edge_bins,
            range=(-np.pi, np.pi),
            density=True
        )

        return hist

    def _extract_spatial_moments(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract spatial moments.

        Represents spatial structure constraints.

        Args:
            mask: Region mask

        Returns:
            Array of spatial moments
        """
        # Get coordinates
        y_coords, x_coords = np.where(mask)

        if len(y_coords) == 0:
            return np.zeros(7)

        # Centroid
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)

        # Second moments (variance)
        var_y = np.var(y_coords)
        var_x = np.var(x_coords)

        # Covariance
        cov_xy = np.mean((x_coords - centroid_x) * (y_coords - centroid_y))

        # Eccentricity
        lambda1 = (var_x + var_y) / 2 + np.sqrt(((var_x - var_y) / 2)**2 + cov_xy**2)
        lambda2 = (var_x + var_y) / 2 - np.sqrt(((var_x - var_y) / 2)**2 + cov_xy**2)
        eccentricity = np.sqrt(1 - lambda2 / (lambda1 + 1e-10))

        # Orientation
        orientation = 0.5 * np.arctan2(2 * cov_xy, var_x - var_y)

        moments = np.array([
            centroid_x,
            centroid_y,
            var_x,
            var_y,
            cov_xy,
            eccentricity,
            orientation
        ])

        return moments

    def extract_selective(
        self,
        region: 'Region',
        feature_types: list
    ) -> Dict[str, Any]:
        """
        Extract only specified feature types.

        Args:
            region: Region to process
            feature_types: List of feature names
                ['color', 'texture', 'edge', 'spatial']

        Returns:
            Dict of requested features
        """
        features = {}
        pixels = region.get_pixels()
        mask = region.mask

        if 'color' in feature_types:
            features['color_histogram'] = self._extract_color_histogram(pixels)
            features['color_moments'] = self._extract_color_moments(pixels)

        if 'texture' in feature_types:
            features['texture_features'] = self._extract_texture(
                region.image_data,
                mask
            )

        if 'edge' in feature_types:
            features['edge_features'] = self._extract_edges(
                region.image_data,
                mask
            )

        if 'spatial' in feature_types:
            features['spatial_moments'] = self._extract_spatial_moments(mask)

        return features
