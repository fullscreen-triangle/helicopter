"""
Image segmentation methods for region extraction.

Provides multiple segmentation algorithms:
- SLIC superpixels
- Felzenszwalb's graph-based segmentation
- Watershed segmentation
- Hierarchical segmentation
"""

import numpy as np
from typing import List
from skimage import segmentation, color
from skimage.filters import sobel
from skimage.measure import regionprops
from .region import Region


class ImageSegmenter:
    """
    Image segmentation for BMD-based vision.

    Extracts regions that will be compared with BMD states.
    """

    def __init__(self):
        """Initialize segmenter."""
        pass

    def segment(
        self,
        image: np.ndarray,
        method: str = 'slic',
        **kwargs
    ) -> List[Region]:
        """
        Segment image into regions.

        Args:
            image: Input image (H x W x C) or (H x W)
            method: Segmentation method
                - 'slic': SLIC superpixels
                - 'felzenszwalb': Graph-based segmentation
                - 'watershed': Watershed segmentation
                - 'hierarchical': Hierarchical segmentation
            **kwargs: Method-specific parameters

        Returns:
            List of Region objects
        """
        # Ensure RGB
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # Select method
        if method == 'slic':
            return self.segment_slic(image, **kwargs)
        elif method == 'felzenszwalb':
            return self.segment_felzenszwalb(image, **kwargs)
        elif method == 'watershed':
            return self.segment_watershed(image, **kwargs)
        elif method == 'hierarchical':
            return self.segment_hierarchical(image, **kwargs)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

    def segment_slic(
        self,
        image: np.ndarray,
        n_segments: int = 100,
        compactness: float = 10.0,
        sigma: float = 1.0
    ) -> List[Region]:
        """
        SLIC superpixel segmentation.

        Args:
            image: Input image
            n_segments: Target number of segments
            compactness: Balance color-proximity vs spatial-proximity
            sigma: Gaussian smoothing before segmentation

        Returns:
            List of Region objects
        """
        # SLIC segmentation
        labels = segmentation.slic(
            image,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            start_label=0
        )

        # Convert labels to regions
        regions = self._labels_to_regions(labels, image)

        return regions

    def segment_felzenszwalb(
        self,
        image: np.ndarray,
        scale: float = 100.0,
        sigma: float = 0.5,
        min_size: int = 50
    ) -> List[Region]:
        """
        Felzenszwalb's efficient graph-based segmentation.

        Args:
            image: Input image
            scale: Free parameter (larger = larger segments)
            sigma: Gaussian kernel smoothing
            min_size: Minimum component size

        Returns:
            List of Region objects
        """
        labels = segmentation.felzenszwalb(
            image,
            scale=scale,
            sigma=sigma,
            min_size=min_size
        )

        regions = self._labels_to_regions(labels, image)

        return regions

    def segment_watershed(
        self,
        image: np.ndarray,
        compactness: float = 0.001
    ) -> List[Region]:
        """
        Watershed segmentation.

        Args:
            image: Input image
            compactness: Higher values = more compact regions

        Returns:
            List of Region objects
        """
        # Convert to grayscale for gradient
        if image.shape[-1] == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image

        # Compute gradient
        elevation_map = sobel(gray)

        # Watershed
        labels = segmentation.watershed(
            elevation_map,
            compactness=compactness
        )

        regions = self._labels_to_regions(labels, image)

        return regions

    def segment_hierarchical(
        self,
        image: np.ndarray,
        n_levels: int = 3
    ) -> List[Region]:
        """
        Hierarchical segmentation matching BMD hierarchy.

        Creates regions at multiple scales, mimicking the
        hierarchical BMD structure.

        Args:
            image: Input image
            n_levels: Number of hierarchical levels

        Returns:
            List of Region objects from all levels
        """
        all_regions = []

        # Segment at each level with different granularity
        for level in range(n_levels):
            n_segments = int(100 / (2 ** level))  # Coarser at higher levels
            n_segments = max(10, n_segments)  # At least 10 segments

            level_labels = segmentation.slic(
                image,
                n_segments=n_segments,
                compactness=10.0,
                start_label=len(all_regions)
            )

            level_regions = self._labels_to_regions(
                level_labels,
                image,
                prefix=f"L{level}_"
            )

            all_regions.extend(level_regions)

        return all_regions

    def _labels_to_regions(
        self,
        labels: np.ndarray,
        image: np.ndarray,
        prefix: str = ""
    ) -> List[Region]:
        """
        Convert label map to Region objects.

        Args:
            labels: Label map (H x W)
            image: Original image
            prefix: Prefix for region IDs

        Returns:
            List of Region objects
        """
        regions = []

        # Get unique labels
        unique_labels = np.unique(labels)

        for label_id in unique_labels:
            # Create mask
            mask = (labels == label_id)

            # Skip very small regions
            if np.sum(mask) < 10:
                continue

            # Compute bounding box
            props = regionprops(mask.astype(int))[0]
            bbox = (
                props.bbox[1],  # x_min
                props.bbox[0],  # y_min
                props.bbox[3],  # x_max
                props.bbox[2]   # y_max
            )

            # Create region
            region_id = f"{prefix}region_{label_id}"
            region = Region(
                region_id=region_id,
                mask=mask,
                image_data=image,
                bbox=bbox
            )

            regions.append(region)

        return regions

    def refine_regions(
        self,
        regions: List[Region],
        min_area: int = 50,
        max_overlap: float = 0.5
    ) -> List[Region]:
        """
        Refine region set by removing small/overlapping regions.

        Args:
            regions: Initial regions
            min_area: Minimum region area
            max_overlap: Maximum allowed overlap ratio

        Returns:
            Refined region list
        """
        refined = []

        for region in regions:
            # Filter by area
            if region.get_area() < min_area:
                continue

            # Check overlap with existing refined regions
            overlaps = False
            for existing in refined:
                if region.overlaps_with(existing, threshold=max_overlap):
                    overlaps = True
                    break

            if not overlaps:
                refined.append(region)

        return refined

    def merge_small_regions(
        self,
        regions: List[Region],
        min_area: int = 100
    ) -> List[Region]:
        """
        Merge small regions with adjacent larger regions.

        Args:
            regions: Region list
            min_area: Threshold for "small" regions

        Returns:
            Merged region list
        """
        # Separate small and large
        small = [r for r in regions if r.get_area() < min_area]
        large = [r for r in regions if r.get_area() >= min_area]

        # Merge small into adjacent large
        for small_region in small:
            # Find best adjacent region
            best_neighbor = None
            max_overlap = 0.0

            for large_region in large:
                # Check adjacency (simple proximity check)
                centroid_s = small_region.get_centroid()
                centroid_l = large_region.get_centroid()

                dist = np.sqrt(
                    (centroid_s[0] - centroid_l[0])**2 +
                    (centroid_s[1] - centroid_l[1])**2
                )

                if dist < 50:  # Close enough
                    best_neighbor = large_region
                    break

            # Merge if found (in practice, would update mask)
            # For now, just skip small regions

        return large
