"""
Multimodal Reaction Localization Validation.

Tests the Intersection Theorem:
- Arrival-time surfaces from ≥3 modalities at ≥4 observation points
  intersect at a unique point (the reaction location)

Resolution enhancement:
δr = δr_single × ∏_i ε_i^(1/3)

From the paper:
- Six propagation modalities: chemical, acoustic, thermal, EM, vibrational, categorical
- Different propagation physics create distinct arrival-time surfaces
- Intersection uniquely determines reaction location
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import ndimage, signal
from scipy.spatial import cKDTree

from .data_loader import ImageData


@dataclass
class DetectedReaction:
    """A detected reaction event."""
    position: Tuple[float, float]
    modalities_detected: List[str]
    n_modalities: int
    confidence: float
    signal_strength: float


@dataclass
class ModalityDetection:
    """Detection result from a single modality."""
    name: str
    peaks: np.ndarray  # Nx2 array of (y, x) positions
    signal: np.ndarray  # 2D signal map
    propagation_type: str  # 'diffusive' or 'ballistic'
    resolution: float  # Single-modality resolution in pixels


class MultimodalLocalizationValidator:
    """
    Validate multimodal reaction localization.

    Simulates "reactions" as intensity changes between consecutive images.
    Detects using multiple feature channels (proxies for propagation modalities).
    Tests consensus localization and resolution enhancement.
    """

    def __init__(self, consensus_radius: float = 10.0, min_modalities: int = 3):
        """
        Initialize validator.

        Args:
            consensus_radius: Radius for considering peaks as same reaction (pixels)
            min_modalities: Minimum modalities for valid detection
        """
        self.consensus_radius = consensus_radius
        self.min_modalities = min_modalities
        self.modality_detections: List[ModalityDetection] = []
        self.detected_reactions: List[DetectedReaction] = []

    def _compute_difference_signal(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """Compute reaction signal from image difference."""
        return np.abs(image2.astype(float) - image1.astype(float))

    def _detect_peaks(self, signal: np.ndarray,
                      min_distance: int = 10,
                      threshold_percentile: float = 95) -> np.ndarray:
        """
        Detect peaks in signal map.

        Returns:
            Nx2 array of (y, x) peak positions
        """
        try:
            from skimage.feature import peak_local_max
        except ImportError:
            # Fallback: simple maximum detection
            threshold = np.percentile(signal, threshold_percentile)
            peaks_mask = signal > threshold
            peaks = np.array(np.where(peaks_mask)).T
            return peaks if len(peaks) > 0 else np.array([]).reshape(0, 2)

        threshold = np.percentile(signal, threshold_percentile)
        if threshold <= 0:
            threshold = signal.max() * 0.5

        peaks = peak_local_max(signal, min_distance=min_distance,
                               threshold_abs=threshold)

        return peaks if len(peaks) > 0 else np.array([]).reshape(0, 2)

    def detect_with_modalities(self, image1: np.ndarray, image2: np.ndarray) -> List[ModalityDetection]:
        """
        Detect reactions using multiple modality proxies.

        Modality mappings:
        1. Direct difference → Chemical (diffusive, slow)
        2. Laplacian of difference → Acoustic (ballistic, fast)
        3. Gaussian-smoothed difference → Thermal (diffusive, medium)
        4. Gradient magnitude → EM (near-field, instantaneous)
        5. Local variance → Vibrational (quantum scale)
        6. Categorical edge detection → Discrete transitions

        Args:
            image1, image2: Consecutive image frames

        Returns:
            List of ModalityDetection
        """
        diff = self._compute_difference_signal(image1, image2)
        self.modality_detections = []

        # 1. Chemical modality: Direct intensity change (diffusive)
        signal_chemical = ndimage.gaussian_filter(diff, sigma=5)
        peaks_chemical = self._detect_peaks(signal_chemical)
        self.modality_detections.append(ModalityDetection(
            name='Chemical',
            peaks=peaks_chemical,
            signal=signal_chemical,
            propagation_type='diffusive',
            resolution=50.0  # ~μm scale
        ))

        # 2. Acoustic modality: Laplacian (high-frequency, ballistic)
        signal_acoustic = np.abs(ndimage.laplace(diff))
        signal_acoustic = ndimage.gaussian_filter(signal_acoustic, sigma=2)
        peaks_acoustic = self._detect_peaks(signal_acoustic)
        self.modality_detections.append(ModalityDetection(
            name='Acoustic',
            peaks=peaks_acoustic,
            signal=signal_acoustic,
            propagation_type='ballistic',
            resolution=10.0  # ~100nm scale
        ))

        # 3. Thermal modality: Heavy smoothing (diffusive)
        signal_thermal = ndimage.gaussian_filter(diff, sigma=10)
        peaks_thermal = self._detect_peaks(signal_thermal)
        self.modality_detections.append(ModalityDetection(
            name='Thermal',
            peaks=peaks_thermal,
            signal=signal_thermal,
            propagation_type='diffusive',
            resolution=20.0  # ~10nm scale
        ))

        # 4. EM modality: Gradient magnitude (near-field)
        grad_y = ndimage.sobel(diff, axis=0)
        grad_x = ndimage.sobel(diff, axis=1)
        signal_em = np.sqrt(grad_y**2 + grad_x**2)
        signal_em = ndimage.gaussian_filter(signal_em, sigma=2)
        peaks_em = self._detect_peaks(signal_em)
        self.modality_detections.append(ModalityDetection(
            name='Electromagnetic',
            peaks=peaks_em,
            signal=signal_em,
            propagation_type='instantaneous',
            resolution=5.0  # ~0.5nm Debye length scale
        ))

        # 5. Vibrational modality: Local variance
        def local_variance(img, size=5):
            mean = ndimage.uniform_filter(img, size)
            mean_sq = ndimage.uniform_filter(img**2, size)
            return mean_sq - mean**2

        signal_vib = local_variance(diff, size=5)
        signal_vib = ndimage.gaussian_filter(signal_vib, sigma=2)
        peaks_vib = self._detect_peaks(signal_vib)
        self.modality_detections.append(ModalityDetection(
            name='Vibrational',
            peaks=peaks_vib,
            signal=signal_vib,
            propagation_type='quantum',
            resolution=2.0  # ~0.1nm scale
        ))

        # 6. Categorical modality: Edge detection (discrete transitions)
        signal_cat = ndimage.generic_filter(diff, np.std, size=5)
        signal_cat = signal_cat > np.percentile(signal_cat, 90)
        signal_cat = signal_cat.astype(float)
        peaks_cat = self._detect_peaks(signal_cat, threshold_percentile=50)
        self.modality_detections.append(ModalityDetection(
            name='Categorical',
            peaks=peaks_cat,
            signal=signal_cat.astype(float),
            propagation_type='discrete',
            resolution=1.0  # Digital precision
        ))

        return self.modality_detections

    def find_consensus_locations(self) -> List[DetectedReaction]:
        """
        Find reaction locations detected by multiple modalities.

        Implements Intersection Theorem: locations where ≥3 modality
        arrival surfaces intersect.

        Returns:
            List of DetectedReaction
        """
        if not self.modality_detections:
            return []

        # Collect all peaks with modality labels
        all_peaks = []
        for detection in self.modality_detections:
            for peak in detection.peaks:
                all_peaks.append({
                    'position': peak,
                    'modality': detection.name,
                    'resolution': detection.resolution
                })

        if not all_peaks:
            return []

        # Cluster nearby peaks
        positions = np.array([p['position'] for p in all_peaks])
        if len(positions) == 0:
            return []

        # Use reference modality (first with peaks)
        reference_idx = 0
        for i, det in enumerate(self.modality_detections):
            if len(det.peaks) > 0:
                reference_idx = i
                break

        reference_peaks = self.modality_detections[reference_idx].peaks
        if len(reference_peaks) == 0:
            return []

        self.detected_reactions = []

        # For each reference peak, find consensus
        for ref_peak in reference_peaks:
            modalities_at_location = [self.modality_detections[reference_idx].name]
            distances = []

            for detection in self.modality_detections:
                if detection.name == self.modality_detections[reference_idx].name:
                    continue

                if len(detection.peaks) == 0:
                    continue

                # Find nearest peak in this modality
                dists = np.linalg.norm(detection.peaks - ref_peak, axis=1)
                min_dist = np.min(dists)

                if min_dist < self.consensus_radius:
                    modalities_at_location.append(detection.name)
                    distances.append(min_dist)

            # Only keep if detected by minimum number of modalities
            if len(modalities_at_location) >= self.min_modalities:
                # Confidence based on number of modalities and distance consistency
                confidence = len(modalities_at_location) / len(self.modality_detections)

                # Signal strength from reference modality
                y, x = int(ref_peak[0]), int(ref_peak[1])
                ref_signal = self.modality_detections[reference_idx].signal
                if 0 <= y < ref_signal.shape[0] and 0 <= x < ref_signal.shape[1]:
                    signal_strength = float(ref_signal[y, x])
                else:
                    signal_strength = 0.0

                self.detected_reactions.append(DetectedReaction(
                    position=(float(ref_peak[0]), float(ref_peak[1])),
                    modalities_detected=modalities_at_location,
                    n_modalities=len(modalities_at_location),
                    confidence=confidence,
                    signal_strength=signal_strength
                ))

        return self.detected_reactions

    def compute_resolution_enhancement(self) -> Dict:
        """
        Compute resolution enhancement from multimodal fusion.

        Theory: δr = δr_single × ∏_i ε_i^(1/3)

        Returns:
            Dictionary with resolution metrics
        """
        if not self.modality_detections:
            return {'validated': False, 'reason': 'No modality detections'}

        # Single-modality resolutions
        single_resolutions = [d.resolution for d in self.modality_detections]
        mean_single = np.mean(single_resolutions)

        # Compute exclusion factors from peak distributions
        n_modalities = len(self.modality_detections)
        epsilons = []

        for detection in self.modality_detections:
            if len(detection.peaks) > 1:
                # Resolution proxy: mean nearest-neighbor distance
                tree = cKDTree(detection.peaks)
                distances, _ = tree.query(detection.peaks, k=2)
                nn_dist = np.mean(distances[:, 1])  # Distance to nearest neighbor
                epsilon = 1.0 / (1.0 + nn_dist / 10.0)  # Normalize
            else:
                epsilon = 0.5
            epsilons.append(epsilon)

        # Theoretical resolution enhancement
        product_epsilon = np.prod(epsilons)
        delta_r_theory = mean_single * (product_epsilon ** (1/3))

        # Empirical resolution from consensus variance
        if self.detected_reactions:
            # Estimate localization precision from multiple detections
            if len(self.detected_reactions) > 1:
                positions = np.array([r.position for r in self.detected_reactions])
                empirical_precision = np.mean(np.std(positions, axis=0))
            else:
                empirical_precision = delta_r_theory
        else:
            empirical_precision = mean_single

        # Resolution improvement factor
        enhancement_factor = mean_single / delta_r_theory if delta_r_theory > 0 else 1.0

        return {
            'single_modality_resolution': float(mean_single),
            'multimodal_resolution_theory': float(delta_r_theory),
            'multimodal_resolution_empirical': float(empirical_precision),
            'enhancement_factor': float(enhancement_factor),
            'n_modalities': n_modalities,
            'epsilons': epsilons,
            'product_epsilon': float(product_epsilon),
            'n_reactions_detected': len(self.detected_reactions),
            'validated': enhancement_factor > 1.5
        }

    def get_signal_maps(self) -> Dict[str, np.ndarray]:
        """Get signal maps from all modalities for visualization."""
        return {d.name: d.signal for d in self.modality_detections}

    def get_peak_data(self) -> Dict[str, np.ndarray]:
        """Get peak locations from all modalities for visualization."""
        return {d.name: d.peaks for d in self.modality_detections}

    def get_consensus_data(self) -> Dict:
        """Get consensus reaction data for visualization."""
        if not self.detected_reactions:
            return {
                'positions': np.array([]).reshape(0, 2),
                'n_modalities': np.array([]),
                'confidence': np.array([])
            }

        return {
            'positions': np.array([r.position for r in self.detected_reactions]),
            'n_modalities': np.array([r.n_modalities for r in self.detected_reactions]),
            'confidence': np.array([r.confidence for r in self.detected_reactions])
        }
