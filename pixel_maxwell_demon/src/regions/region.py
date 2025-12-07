"""
Image region representation.

A Region contains:
- Pixel data and mask
- Feature descriptors
- Categorical state possibilities
- Processing metadata
"""

import numpy as np
from typing import Dict, Any, Optional, Set


class Region:
    """
    Image region representation for BMD processing.

    Each region is a potential categorical state that can be
    compared with BMD states to generate completions.
    """

    def __init__(
        self,
        region_id: str,
        mask: np.ndarray,
        image_data: np.ndarray,
        bbox: Optional[tuple] = None
    ):
        """
        Initialize region.

        Args:
            region_id: Unique identifier for region
            mask: Binary mask indicating region pixels (H x W)
            image_data: Full image data (H x W x C)
            bbox: Optional bounding box (x_min, y_min, x_max, y_max)
        """
        self.id = region_id
        self.mask = mask.astype(bool)
        self.image_data = image_data

        # Compute bounding box if not provided
        if bbox is None:
            self.bbox = self._compute_bbox()
        else:
            self.bbox = bbox

        # To be filled by processing
        self.features: Optional[Dict[str, Any]] = None
        self.categorical_states: Optional[Set[str]] = None
        self.processing_history: list = []

        # Cache
        self._pixels_cache: Optional[np.ndarray] = None
        self._area_cache: Optional[int] = None

    def extract_features(self, extractor=None) -> Dict[str, Any]:
        """
        Extract features from region.

        Args:
            extractor: Optional FeatureExtractor instance

        Returns:
            Dict of features
        """
        if extractor is None:
            from .features import FeatureExtractor
            extractor = FeatureExtractor()

        self.features = extractor.extract(self)
        return self.features

    def estimate_categorical_states(self, n_states: int = 100) -> Set[str]:
        """
        Estimate set of categorical states C(R) compatible with region.

        Based on features, estimates which categorical states
        this region could occupy.

        Args:
            n_states: Approximate number of categorical states to generate

        Returns:
            Set of categorical state identifiers
        """
        if self.features is None:
            self.extract_features()

        # Generate categorical states from features
        states = set()

        # Use color histogram as basis
        if 'color_histogram' in self.features:
            hist = self.features['color_histogram']
            # Each significant bin suggests categorical states
            for i, count in enumerate(hist):
                if count > 0.01:  # Threshold
                    states.add(f"color_{i}")

        # Use texture features
        if 'texture_features' in self.features:
            texture = self.features['texture_features']
            for i, val in enumerate(texture):
                if val > 0.01:
                    states.add(f"texture_{i}")

        # Use edge orientations
        if 'edge_features' in self.features:
            edges = self.features['edge_features']
            for i, val in enumerate(edges):
                if val > 0.01:
                    states.add(f"edge_{i}")

        # Fallback: generic states
        if not states:
            states = {f"state_{i}" for i in range(10)}

        self.categorical_states = states
        return states

    def get_pixels(self) -> np.ndarray:
        """
        Get masked pixel values from region.

        Returns:
            Array of pixels (N x C) where N = number of pixels
        """
        if self._pixels_cache is not None:
            return self._pixels_cache

        # Extract pixels using mask
        pixels = self.image_data[self.mask]
        self._pixels_cache = pixels

        return pixels

    def get_area(self) -> int:
        """
        Get region area (number of pixels).

        Returns:
            Pixel count
        """
        if self._area_cache is not None:
            return self._area_cache

        self._area_cache = np.sum(self.mask)
        return self._area_cache

    def get_centroid(self) -> tuple:
        """
        Get region centroid.

        Returns:
            (y, x) centroid coordinates
        """
        y_coords, x_coords = np.where(self.mask)

        if len(y_coords) == 0:
            return (0, 0)

        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)

        return (centroid_y, centroid_x)

    def _compute_bbox(self) -> tuple:
        """
        Compute bounding box from mask.

        Returns:
            (x_min, y_min, x_max, y_max)
        """
        y_coords, x_coords = np.where(self.mask)

        if len(y_coords) == 0:
            return (0, 0, 0, 0)

        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)

        return (x_min, y_min, x_max, y_max)

    def overlaps_with(self, other: 'Region', threshold: float = 0.1) -> bool:
        """
        Check if this region overlaps with another.

        Args:
            other: Another region
            threshold: Minimum IoU to consider overlap

        Returns:
            True if overlap exceeds threshold
        """
        intersection = np.logical_and(self.mask, other.mask)
        union = np.logical_or(self.mask, other.mask)

        iou = np.sum(intersection) / (np.sum(union) + 1e-10)

        return iou > threshold

    def mark_processed(self, step: int, bmd_state: Any):
        """
        Mark region as processed at given step.

        Args:
            step: Processing step number
            bmd_state: BMD state generated from this region
        """
        self.processing_history.append({
            'step': step,
            'bmd': str(bmd_state),
            'timestamp': np.datetime64('now')
        })

    def was_processed(self) -> bool:
        """Check if region has been processed."""
        return len(self.processing_history) > 0

    def last_processed_step(self) -> int:
        """
        Get last processing step for this region.

        Returns:
            Step number, or -1 if never processed
        """
        if not self.processing_history:
            return -1
        return self.processing_history[-1]['step']

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize region to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'id': self.id,
            'bbox': self.bbox,
            'area': self.get_area(),
            'centroid': self.get_centroid(),
            'features': self.features,
            'n_categorical_states': len(self.categorical_states) if self.categorical_states else 0,
            'processed': self.was_processed(),
            'processing_history': self.processing_history
        }

    def __repr__(self) -> str:
        return (
            f"Region(id={self.id}, area={self.get_area()}, "
            f"bbox={self.bbox}, processed={self.was_processed()})"
        )
