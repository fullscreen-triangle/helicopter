"""
Data loader for BBBC039 nuclei segmentation dataset.

Handles loading of fluorescence microscopy images and ground truth masks.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import warnings


@dataclass
class NucleusData:
    """Container for a single nucleus's data."""
    label: int
    centroid: Tuple[float, float]
    area: int
    eccentricity: float
    orientation: float
    mean_intensity: float
    bbox: Tuple[int, int, int, int]
    solidity: float
    perimeter: float


@dataclass
class ImageData:
    """Container for image and mask pair with extracted nuclei."""
    image: np.ndarray
    mask: np.ndarray
    filename: str
    nuclei: List[NucleusData]

    @property
    def n_nuclei(self) -> int:
        return len(self.nuclei)


class BBBC039DataLoader:
    """
    Loader for BBBC039 nuclei segmentation dataset.

    Dataset structure:
    - public/images/images/*.tif (16-bit fluorescence)
    - public/masks/masks/*.png (instance segmentation)
    - public/metadata/ (experimental metadata)
    """

    def __init__(self, data_root: str):
        """
        Initialize data loader.

        Args:
            data_root: Path to turbine/public directory
        """
        self.data_root = Path(data_root)
        self.image_dir = self.data_root / "images" / "images"
        self.mask_dir = self.data_root / "masks" / "masks"

        # Verify directories exist
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # Index available files
        self._index_files()

    def _index_files(self):
        """Index available image-mask pairs."""
        image_files = {p.stem: p for p in self.image_dir.glob("*.tif")}
        mask_files = {p.stem: p for p in self.mask_dir.glob("*.png")}

        # Find matching pairs
        self.matched_pairs = []
        for stem in sorted(image_files.keys()):
            if stem in mask_files:
                self.matched_pairs.append({
                    'stem': stem,
                    'image': image_files[stem],
                    'mask': mask_files[stem]
                })

        print(f"Indexed {len(self.matched_pairs)} image-mask pairs")

    def __len__(self) -> int:
        return len(self.matched_pairs)

    def load_image(self, path: Path) -> np.ndarray:
        """Load a single image file."""
        try:
            from PIL import Image
            img = Image.open(path)
            return np.array(img)
        except ImportError:
            # Fallback to tifffile if available
            try:
                import tifffile
                return tifffile.imread(path)
            except ImportError:
                raise ImportError("Install PIL or tifffile: pip install pillow tifffile")

    def load_mask(self, path: Path) -> np.ndarray:
        """Load a single mask file and convert to instance segmentation."""
        from PIL import Image
        from scipy import ndimage

        img = Image.open(path)
        mask = np.array(img)

        # Handle RGBA/RGB masks - convert to single channel
        if mask.ndim == 3:
            # Use first channel (which contains class/label info)
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        # Create binary mask (foreground vs background)
        binary = mask_2d > 0

        # Use connected component labeling to get instance segmentation
        # This separates touching nuclei
        labeled, n_features = ndimage.label(binary)

        return labeled.astype(np.int32)

    def extract_nuclei_properties(self, image: np.ndarray, mask: np.ndarray) -> List[NucleusData]:
        """
        Extract properties for each nucleus in the image.

        Uses skimage.measure.regionprops for morphological analysis.
        """
        try:
            from skimage.measure import regionprops, label as label_image
        except ImportError:
            raise ImportError("Install scikit-image: pip install scikit-image")

        # Ensure mask is labeled (instance segmentation)
        if mask.max() <= 1:
            mask = label_image(mask > 0)

        nuclei = []
        for region in regionprops(mask, intensity_image=image):
            if region.area < 50:  # Skip tiny artifacts
                continue

            nuclei.append(NucleusData(
                label=region.label,
                centroid=region.centroid,
                area=region.area,
                eccentricity=region.eccentricity,
                orientation=region.orientation,
                mean_intensity=region.mean_intensity,
                bbox=region.bbox,
                solidity=region.solidity,
                perimeter=region.perimeter
            ))

        return nuclei

    def load_sample(self, index: int) -> ImageData:
        """
        Load a single image-mask pair with extracted nuclei.

        Args:
            index: Index into matched pairs

        Returns:
            ImageData with image, mask, and nucleus properties
        """
        if index >= len(self.matched_pairs):
            raise IndexError(f"Index {index} out of range (max: {len(self)-1})")

        pair = self.matched_pairs[index]
        image = self.load_image(pair['image'])
        mask = self.load_mask(pair['mask'])

        # Normalize image to float
        if image.dtype == np.uint16:
            image_norm = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            image_norm = image.astype(np.float32) / 255.0
        else:
            image_norm = image.astype(np.float32)
            if image_norm.max() > 1:
                image_norm = image_norm / image_norm.max()

        nuclei = self.extract_nuclei_properties(image_norm, mask)

        return ImageData(
            image=image_norm,
            mask=mask,
            filename=pair['stem'],
            nuclei=nuclei
        )

    def load_batch(self, indices: List[int]) -> List[ImageData]:
        """Load multiple image-mask pairs."""
        return [self.load_sample(i) for i in indices]

    def load_all(self, max_samples: Optional[int] = None) -> List[ImageData]:
        """
        Load all available samples.

        Args:
            max_samples: Maximum number of samples to load (None for all)
        """
        n = min(len(self), max_samples) if max_samples else len(self)
        return self.load_batch(range(n))

    def get_dataset_statistics(self, samples: List[ImageData]) -> Dict:
        """Compute dataset-wide statistics."""
        total_nuclei = sum(s.n_nuclei for s in samples)
        areas = [n.area for s in samples for n in s.nuclei]
        eccentricities = [n.eccentricity for s in samples for n in s.nuclei]
        intensities = [n.mean_intensity for s in samples for n in s.nuclei]

        return {
            'n_images': len(samples),
            'total_nuclei': total_nuclei,
            'nuclei_per_image': total_nuclei / len(samples) if samples else 0,
            'area_mean': np.mean(areas),
            'area_std': np.std(areas),
            'eccentricity_mean': np.mean(eccentricities),
            'eccentricity_std': np.std(eccentricities),
            'intensity_mean': np.mean(intensities),
            'intensity_std': np.std(intensities),
        }
