"""
Phase 3: MEASURE

Run 3-stage spectral pipeline to estimate coordinate field and scale factor.
Output: Φ mapping pixels to world-space.
Entropy: No change (coordinate field is deterministic bijection).
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import logging

from ..types.coord_field import CoordField, ScaleFieldEstimate
from ..types.partition_state import SEntropy


logger = logging.getLogger(__name__)


@dataclass
class MeasurePhaseInput:
    """
    Input to MEASURE phase.

    Attributes:
        frame: 2D image array (height, width)
        field_size_x: Field size in micrometers
        field_size_y: Field size in micrometers
        resolution: Pixel resolution in micrometers
        depth: Partition depth n
        lambda_s: Spatial coherence wavelength
        lambda_t: Temporal coherence wavelength
    """
    frame: np.ndarray  # (height, width)
    field_size_x: float
    field_size_y: float
    resolution: float
    depth: int
    lambda_s: float = 0.10
    lambda_t: float = 0.05


@dataclass
class MeasurePhaseOutput:
    """
    Output from MEASURE phase.

    Attributes:
        coord_field: Coordinate field Φ: (u,v) → (x,y,z)
        scale_field_estimate: Scale factor estimate α(u,v)
        fft_magnitude: FFT magnitude spectrum (for visualization)
        s_entropy_before: Entropy before measurement
        s_entropy_after: Entropy after measurement (unchanged)
    """
    coord_field: CoordField
    scale_field_estimate: ScaleFieldEstimate
    fft_magnitude: Optional[np.ndarray]
    s_entropy_before: SEntropy
    s_entropy_after: SEntropy

    def to_dict(self) -> dict:
        return {
            'coord_field': self.coord_field.to_dict(),
            'scale_field_estimate': self.scale_field_estimate.to_dict(),
            'fft_magnitude': self.fft_magnitude.tolist() if self.fft_magnitude is not None else None,
            's_entropy_before': self.s_entropy_before.to_dict(),
            's_entropy_after': self.s_entropy_after.to_dict(),
        }


def _fft_decomposition(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 1: FFT spectral decomposition.

    Args:
        frame: Input image (height, width)

    Returns:
        (magnitude_spectrum, phase_spectrum)
    """
    # Apply Hann window to reduce edge artifacts
    h, w = frame.shape
    window = np.hanning(h)[:, np.newaxis] * np.hanning(w)[np.newaxis, :]
    windowed = frame * window

    # Compute 2D FFT
    fft_result = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft_result)

    magnitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)

    # Avoid log(0)
    magnitude_log = np.log1p(magnitude)

    logger.debug(
        f"FFT decomposition: frame shape {frame.shape}, "
        f"magnitude range [{magnitude.min():.2e}, {magnitude.max():.2e}]"
    )

    return magnitude_log, phase


def _dyadic_decomposition(magnitude: np.ndarray, depth: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 2: Dyadic scale decomposition via simple scale pyramid.

    Uses Gaussian pyramid approximation of wavelet decomposition.

    Args:
        magnitude: Log FFT magnitude
        depth: Partition depth (controls scale levels)

    Returns:
        (scale_field, confidence_field)
    """
    h, w = magnitude.shape

    # Create scale field: estimate of meter/pixel at each location
    # Use Gaussian pyramid: coarser scales → lower frequencies → smaller scale
    scale_field = np.ones((h, w), dtype=np.float32)

    # Apply Gaussian blur at multiple scales
    # Deeper levels (higher n) → finer scales → larger meter/pixel
    from scipy.ndimage import gaussian_filter
    blurred = magnitude
    for level in range(min(depth, 8)):
        sigma = 2 ** level
        blurred = gaussian_filter(blurred, sigma=sigma)

    # Estimate scale as inverse of frequency magnitude
    # Regions with strong high-frequency content → small scale (fine detail)
    # Regions with weak high-frequency content → large scale (coarse detail)
    min_mag = np.percentile(magnitude, 10)
    max_mag = np.percentile(magnitude, 90)

    if max_mag > min_mag:
        normalized = (magnitude - min_mag) / (max_mag - min_mag)
    else:
        normalized = np.ones_like(magnitude)

    # Scale field: 1% to 10% of field height per pixel
    scale_field = 0.01 + normalized * 0.09  # meters/pixel (normalized)

    # Confidence: inverse of variation (smooth regions → high confidence)
    confidence = 1.0 - (normalized ** 2)
    confidence = np.clip(confidence, 0, 1)

    logger.debug(
        f"Dyadic decomposition (depth={depth}): "
        f"scale_field range [{scale_field.min():.3f}, {scale_field.max():.3f}], "
        f"mean confidence {confidence.mean():.3f}"
    )

    return scale_field, confidence


def _coherence_enforcement(
    scale_field: np.ndarray,
    lambda_s: float,
    lambda_t: float
) -> np.ndarray:
    """
    Stage 3: Coherence enforcement via bilateral filtering.

    Smooth the scale field while preserving edges.

    Args:
        scale_field: Scale factor map (height, width)
        lambda_s: Spatial coherence wavelength
        lambda_t: Temporal coherence wavelength

    Returns:
        Smoothed scale field
    """
    from scipy.ndimage import gaussian_filter

    # Bilateral filter approximation: Gaussian blur with edge-aware weighting
    sigma_spatial = lambda_s * 10  # Scale up for pixel units
    blurred = gaussian_filter(scale_field, sigma=sigma_spatial)

    # Edge-aware weighting: reduce blur near edges
    # Compute gradients
    gy, gx = np.gradient(scale_field)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Weight for bilateral: 0 at edges, 1 in flat regions
    edge_threshold = np.percentile(grad_mag, 75)
    edge_weight = np.exp(-(grad_mag / (edge_threshold + 1e-6))**2)

    # Blend: more blurring in smooth regions
    smoothed = scale_field * (1 - edge_weight * 0.5) + blurred * edge_weight * 0.5

    logger.debug(
        f"Coherence enforcement (λ_s={lambda_s}, λ_t={lambda_t}): "
        f"smoothed field range [{smoothed.min():.3f}, {smoothed.max():.3f}]"
    )

    return smoothed


def measure_phase(inputs: MeasurePhaseInput) -> MeasurePhaseOutput:
    """
    Execute MEASURE phase: spectral pipeline for coordinate field estimation.

    3-stage pipeline:
    1. FFT decomposition → dyadic scale coefficients
    2. Scale field estimation → α(u,v) in meters/pixel
    3. Coherence enforcement → smooth while preserving edges

    Args:
        inputs: MeasurePhaseInput with frame and parameters

    Returns:
        MeasurePhaseOutput with coordinate field and scale estimates
    """
    logger.info(f"MEASURE phase starting: frame shape {inputs.frame.shape}, depth={inputs.depth}")

    # Validate frame
    if inputs.frame.ndim != 2:
        raise ValueError(f"Frame must be 2D, got shape {inputs.frame.shape}")

    frame_float = inputs.frame.astype(np.float32)

    # Normalize frame to [0, 1]
    fmin, fmax = frame_float.min(), frame_float.max()
    if fmax > fmin:
        frame_norm = (frame_float - fmin) / (fmax - fmin)
    else:
        frame_norm = np.ones_like(frame_float)

    # Stage 1: FFT
    magnitude_spectrum, phase_spectrum = _fft_decomposition(frame_norm)

    # Stage 2: Dyadic decomposition
    scale_field_raw, confidence_field = _dyadic_decomposition(magnitude_spectrum, inputs.depth)

    # Scale to actual micrometers/pixel
    # Raw scale_field is in [0.01, 0.10]; we need to map to actual resolution
    scale_field_meters = scale_field_raw * (inputs.field_size_x / 1e6)  # Convert μm to m

    # Stage 3: Coherence enforcement
    scale_field_smooth = _coherence_enforcement(
        scale_field_meters,
        inputs.lambda_s / 1e6,  # Convert μm to m
        inputs.lambda_t
    )

    # Build coordinate field
    # Phase field: derived from FFT phase, normalized
    phase_norm = (phase_spectrum - phase_spectrum.min()) / (phase_spectrum.max() - phase_spectrum.min() + 1e-6)

    coord_field = CoordField(
        field_size_x=inputs.field_size_x,
        field_size_y=inputs.field_size_y,
        resolution=inputs.resolution,
        depth=inputs.depth,
        scale_field=scale_field_smooth,
        phase_field=phase_norm,
        lambda_s=inputs.lambda_s,
        lambda_t=inputs.lambda_t,
    )

    # Build scale field estimate
    scale_estimate = ScaleFieldEstimate(
        method="ratio",
        scale_map=scale_field_smooth,
        confidence=confidence_field,
        coherence_spatial=float(1.0 - np.std(scale_field_raw)),
        coherence_temporal=inputs.lambda_t,
    )

    # Entropy: no change (coordinate field is deterministic bijection)
    s_entropy = SEntropy.uniform()  # Placeholder

    logger.info(
        f"MEASURE phase complete: "
        f"scale_field mean={scale_estimate.mean_scale():.3e}, "
        f"coherence_spatial={scale_estimate.coherence_spatial:.3f}"
    )

    return MeasurePhaseOutput(
        coord_field=coord_field,
        scale_field_estimate=scale_estimate,
        fft_magnitude=magnitude_spectrum,
        s_entropy_before=s_entropy,
        s_entropy_after=s_entropy,
    )
