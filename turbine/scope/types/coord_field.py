"""
Coordinate field types: mapping pixels to world-space via spectral metric reconstruction.
"""

from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np


@dataclass
class CoordField:
    """
    Coordinate field Φ: (u, v) → (x, y, z) in world-space.

    Constructed via 3-stage spectral pipeline:
    1. FFT decomposition → dyadic scale coefficients
    2. Scale field estimation → α(u, v) in meters/pixel
    3. Coherence enforcement → smooth while preserving edges
    """
    field_size_x: float  # Field size in micrometers (e.g., 100.0)
    field_size_y: float
    resolution: float  # Resolution in micrometers (e.g., 0.1)
    depth: int  # n = log₂(field_size / resolution)

    scale_field: np.ndarray  # α(u, v): shape (height, width)
    phase_field: np.ndarray  # Φ phase estimate: shape (height, width)

    lambda_s: float  # Spatial coherence wavelength (micrometers)
    lambda_t: float  # Temporal coherence wavelength (time units)

    def __post_init__(self):
        expected_depth = int(np.log2(self.field_size_x / self.resolution))
        if self.depth != expected_depth:
            raise ValueError(
                f"Depth mismatch: expected {expected_depth} from field_size/resolution, "
                f"got {self.depth}"
            )

        if self.scale_field.shape != self.phase_field.shape:
            raise ValueError("scale_field and phase_field must have same shape")

        height, width = self.scale_field.shape
        expected_height = 2 ** self.depth
        if height != expected_height or width != expected_height:
            raise ValueError(
                f"Field shape must be ({expected_height}, {expected_height}), "
                f"got {(height, width)}"
            )

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float, float]:
        """
        Map pixel coordinates (u, v) to world-space (x, y, z).

        Args:
            u: Pixel column (0 to width-1)
            v: Pixel row (0 to height-1)

        Returns:
            (x, y, z) in world-space (meters)
        """
        height, width = self.scale_field.shape

        # Bounds check
        if not (0 <= u < width and 0 <= v < height):
            raise ValueError(f"Pixel ({u}, {v}) out of bounds [{width}x{height}]")

        # Get scale factor at pixel location (in meters/pixel)
        alpha = self.scale_field[int(v), int(u)]

        # Get phase estimate (in radians or normalized units)
        phase = self.phase_field[int(v), int(u)]

        # Map to world-space: scale by α, include phase
        x = u * alpha
        y = v * alpha
        z = phase * alpha  # Phase contributes to depth

        return (x, y, z)

    def distance(self, u1: float, v1: float, u2: float, v2: float) -> float:
        """
        Compute world-space distance between two pixels.

        Args:
            u1, v1: First pixel location
            u2, v2: Second pixel location

        Returns:
            Distance in meters
        """
        x1, y1, z1 = self.pixel_to_world(u1, v1)
        x2, y2, z2 = self.pixel_to_world(u2, v2)

        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return float(distance)

    def uncertainty_at(self, u: float, v: float) -> float:
        """
        Estimate measurement uncertainty at a pixel location.

        From Theorem 2: uncertainty is proportional to scale factor α(u, v).

        Args:
            u, v: Pixel location

        Returns:
            Uncertainty in meters
        """
        height, width = self.scale_field.shape
        u_int, v_int = int(u) % width, int(v) % height

        alpha = self.scale_field[v_int, u_int]
        # Uncertainty ~1% of the scale factor per partition level
        uncertainty = alpha / (2 ** self.depth) * 1.01
        return float(uncertainty)

    def to_dict(self) -> dict:
        """Serialize for storage"""
        return {
            'field_size_x': self.field_size_x,
            'field_size_y': self.field_size_y,
            'resolution': self.resolution,
            'depth': self.depth,
            'lambda_s': self.lambda_s,
            'lambda_t': self.lambda_t,
            'scale_field': self.scale_field.tolist(),
            'phase_field': self.phase_field.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CoordField':
        """Reconstruct from serialized form"""
        return cls(
            field_size_x=d['field_size_x'],
            field_size_y=d['field_size_y'],
            resolution=d['resolution'],
            depth=d['depth'],
            lambda_s=d['lambda_s'],
            lambda_t=d['lambda_t'],
            scale_field=np.array(d['scale_field']),
            phase_field=np.array(d['phase_field']),
        )


@dataclass
class ScaleFieldEstimate:
    """
    Estimate of the metric scale field α(u, v) from spectral decomposition.

    Two common estimation methods:
    1. Ratio method: α = v / ν (velocity / frequency)
    2. Distance method: α = d / (f * ω) (distance / focal_number / frequency)
    """
    method: str  # "ratio" or "distance"
    scale_map: np.ndarray  # α(u, v): shape (height, width), units: m/pixel
    confidence: np.ndarray  # Confidence map: shape (height, width), [0, 1]

    coherence_spatial: float  # Spatial coherence score [0, 1]
    coherence_temporal: float  # Temporal coherence score [0, 1]

    def mean_scale(self) -> float:
        """Mean scale factor across field"""
        return float(np.mean(self.scale_map))

    def std_scale(self) -> float:
        """Standard deviation of scale factor"""
        return float(np.std(self.scale_map))

    def effective_depth(self) -> int:
        """Estimate effective depth from scale variation"""
        scale_range = np.max(self.scale_map) - np.min(self.scale_map)
        mean_scale = self.mean_scale()
        if mean_scale > 0:
            relative_variation = scale_range / mean_scale
            # Effective depth is log2(1 / relative_variation)
            if relative_variation > 1e-6:
                return int(np.log2(1.0 / relative_variation))
        return 0

    def to_dict(self) -> dict:
        return {
            'method': self.method,
            'scale_map': self.scale_map.tolist(),
            'confidence': self.confidence.tolist(),
            'coherence_spatial': self.coherence_spatial,
            'coherence_temporal': self.coherence_temporal,
        }
