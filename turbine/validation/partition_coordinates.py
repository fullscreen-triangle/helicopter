"""
Partition Coordinate Extraction and Validation.

Maps nuclear properties to partition coordinates (n, ℓ, m, s) and validates
that the observed distribution follows C(n) = 2n² capacity theorem.

From the paper:
- n (principal): Quantized from volume via C(n) = 2n²
- ℓ (angular): Shape eccentricity class, ℓ ∈ {0, ..., n-1}
- m (magnetic): Orientation quantum number, m ∈ {-ℓ, ..., +ℓ}
- s (spin): Chromatin state (condensed/decondensed), s ∈ {-1/2, +1/2}
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
from scipy import stats

from .data_loader import ImageData, NucleusData


@dataclass
class PartitionCoordinate:
    """Partition coordinate for a single nucleus."""
    n: int          # Principal quantum number
    ell: int        # Angular momentum quantum number
    m: int          # Magnetic quantum number
    s: float        # Spin quantum number (+/- 0.5)

    # Original nucleus properties for reference
    label: int
    area: int
    eccentricity: float
    orientation: float
    mean_intensity: float

    def to_tuple(self) -> Tuple[int, int, int, float]:
        """Return (n, ℓ, m, s) tuple."""
        return (self.n, self.ell, self.m, self.s)

    def state_index(self) -> int:
        """
        Compute unique state index within the partition structure.

        Total states up to depth n: Σ(2k²) for k=1..n-1, then position within n.
        """
        # States in lower shells
        lower_states = sum(2 * k**2 for k in range(1, self.n))

        # Position within current shell
        # ℓ contributes: Σ(2ℓ'+1) for ℓ'=0..ℓ-1 = ℓ²
        ell_offset = self.ell ** 2 if self.ell > 0 else 0

        # m contributes: m + ℓ (shift to positive)
        m_offset = self.m + self.ell

        # s contributes: 0 for -1/2, 1 for +1/2
        s_offset = 0 if self.s < 0 else 1

        return lower_states + ell_offset + 2 * m_offset + s_offset


class PartitionCoordinateExtractor:
    """
    Extract partition coordinates from nuclear properties.

    Validates the partition capacity theorem: C(n) = 2n²
    """

    def __init__(self, n_max: int = 10, area_scale: float = 100.0):
        """
        Initialize extractor.

        Args:
            n_max: Maximum principal quantum number
            area_scale: Scale factor for area → n mapping
        """
        self.n_max = n_max
        self.area_scale = area_scale

        # Precompute capacity at each n
        self.capacities = {n: 2 * n**2 for n in range(1, n_max + 1)}

        # Total states up to each n
        self.cumulative_states = {}
        total = 0
        for n in range(1, n_max + 1):
            total += self.capacities[n]
            self.cumulative_states[n] = total

    def extract_single(self, nucleus: NucleusData,
                       median_intensity: float) -> PartitionCoordinate:
        """
        Extract partition coordinates for a single nucleus.

        Mapping:
        - n: From area via √(area / π) quantized to integer
        - ℓ: From eccentricity (0=circular → low ℓ, 1=elongated → high ℓ)
        - m: From orientation angle
        - s: From intensity relative to median (chromatin state)
        """
        # Principal quantum number from area
        # Area ~ πr², so r ~ √(area/π)
        # Map to n with C(n) = 2n² → area scales as n²
        effective_radius = np.sqrt(nucleus.area / np.pi)
        n = max(1, min(self.n_max, int(np.ceil(effective_radius / np.sqrt(self.area_scale)))))

        # Angular momentum from eccentricity
        # ℓ ∈ {0, 1, ..., n-1}
        max_ell = max(0, n - 1)
        ell = int(nucleus.eccentricity * max_ell)
        ell = max(0, min(max_ell, ell))

        # Magnetic quantum number from orientation
        # m ∈ {-ℓ, ..., 0, ..., +ℓ}
        # orientation ∈ [-π/2, π/2]
        if ell > 0:
            normalized_orientation = nucleus.orientation / (np.pi / 2)  # [-1, 1]
            m = int(normalized_orientation * ell)
            m = max(-ell, min(ell, m))
        else:
            m = 0

        # Spin from chromatin state (intensity relative to median)
        # High intensity = condensed chromatin = s = +1/2
        # Low intensity = decondensed chromatin = s = -1/2
        s = 0.5 if nucleus.mean_intensity > median_intensity else -0.5

        return PartitionCoordinate(
            n=n, ell=ell, m=m, s=s,
            label=nucleus.label,
            area=nucleus.area,
            eccentricity=nucleus.eccentricity,
            orientation=nucleus.orientation,
            mean_intensity=nucleus.mean_intensity
        )

    def extract_from_image(self, image_data: ImageData) -> List[PartitionCoordinate]:
        """Extract partition coordinates for all nuclei in an image."""
        if not image_data.nuclei:
            return []

        # Compute median intensity for spin assignment
        intensities = [n.mean_intensity for n in image_data.nuclei]
        median_intensity = np.median(intensities)

        return [self.extract_single(nucleus, median_intensity)
                for nucleus in image_data.nuclei]

    def extract_from_dataset(self, samples: List[ImageData]) -> List[PartitionCoordinate]:
        """Extract partition coordinates from entire dataset."""
        all_coords = []
        for sample in samples:
            all_coords.extend(self.extract_from_image(sample))
        return all_coords

    def compute_n_distribution(self, coords: List[PartitionCoordinate]) -> Dict[int, int]:
        """Compute distribution of principal quantum numbers."""
        n_values = [c.n for c in coords]
        return dict(Counter(n_values))

    def compute_state_distribution(self, coords: List[PartitionCoordinate]) -> Dict[Tuple, int]:
        """Compute distribution over (n, ℓ, m, s) states."""
        states = [c.to_tuple() for c in coords]
        return dict(Counter(states))

    def test_capacity_theorem(self, coords: List[PartitionCoordinate]) -> Dict:
        """
        Test whether observed distribution follows C(n) = 2n².

        If theory is correct, nuclei should populate partition space
        with probability proportional to available states at each n.

        Returns:
            Dictionary with chi-squared test results and validation status
        """
        # Observed counts at each n
        n_distribution = self.compute_n_distribution(coords)

        # Theoretical prediction: P(n) ∝ C(n) = 2n²
        n_values = sorted(set(n_distribution.keys()))
        total_capacity = sum(self.capacities.get(n, 2*n**2) for n in n_values)

        # Expected counts under capacity theorem
        total_observed = len(coords)
        expected_counts = {
            n: (self.capacities.get(n, 2*n**2) / total_capacity) * total_observed
            for n in n_values
        }

        # Chi-squared goodness of fit
        observed = np.array([n_distribution.get(n, 0) for n in n_values])
        expected = np.array([expected_counts[n] for n in n_values])

        # Filter out bins with very low expected counts
        valid_bins = expected >= 1  # Lowered threshold for small datasets
        if valid_bins.sum() < 2:
            # Not enough bins for chi-squared test
            return {
                'chi2': np.nan,
                'p_value': np.nan,
                'df': 0,
                'observed_distribution': n_distribution,
                'expected_distribution': expected_counts,
                'capacity_theorem': self.capacities,
                'n_total': total_observed,
                'validated': False,
                'reason': 'Insufficient bins for chi-squared test'
            }

        # Rescale expected to match observed sum for valid comparison
        obs_valid = observed[valid_bins]
        exp_valid = expected[valid_bins]
        exp_valid = exp_valid * (obs_valid.sum() / exp_valid.sum())  # Normalize

        chi2, p_value = stats.chisquare(obs_valid, exp_valid)
        df = valid_bins.sum() - 1

        # Validation: fail to reject null (p > 0.01) means data consistent with theory
        validated = p_value > 0.01

        # Compute effect size (Cramér's V)
        n_samples = observed[valid_bins].sum()
        cramers_v = np.sqrt(chi2 / (n_samples * df)) if df > 0 and n_samples > 0 else 0

        return {
            'chi2': float(chi2),
            'p_value': float(p_value),
            'df': int(df),
            'cramers_v': float(cramers_v),
            'observed_distribution': n_distribution,
            'expected_distribution': {int(k): float(v) for k, v in expected_counts.items()},
            'capacity_theorem': self.capacities,
            'n_total': total_observed,
            'validated': validated,
            'reason': 'Consistent with C(n) = 2n²' if validated else 'Deviates from C(n) = 2n²'
        }

    def compute_partition_entropy(self, coords: List[PartitionCoordinate]) -> Dict:
        """
        Compute entropy of partition coordinate distribution.

        High entropy = nuclei spread across many states (high disorder)
        Low entropy = nuclei concentrated in few states (high order)
        """
        state_dist = self.compute_state_distribution(coords)
        total = len(coords)

        # Shannon entropy
        probs = np.array([count / total for count in state_dist.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Maximum possible entropy (uniform distribution)
        n_states = len(state_dist)
        max_entropy = np.log2(n_states) if n_states > 0 else 0

        # Normalized entropy
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return {
            'entropy': float(entropy),
            'max_entropy': float(max_entropy),
            'normalized_entropy': float(normalized_entropy),
            'n_occupied_states': n_states,
            'n_possible_states': self.cumulative_states.get(self.n_max, 0)
        }

    def get_3d_distribution(self, coords: List[PartitionCoordinate]) -> Dict:
        """
        Get data for 3D visualization of partition coordinate distribution.

        Returns arrays suitable for 3D scatter plot of (n, ℓ, m) with size ∝ count.
        """
        # Aggregate by (n, ℓ, m), ignoring spin
        nlm_counts = Counter((c.n, c.ell, c.m) for c in coords)

        n_vals = []
        ell_vals = []
        m_vals = []
        counts = []

        for (n, ell, m), count in nlm_counts.items():
            n_vals.append(n)
            ell_vals.append(ell)
            m_vals.append(m)
            counts.append(count)

        return {
            'n': np.array(n_vals),
            'ell': np.array(ell_vals),
            'm': np.array(m_vals),
            'counts': np.array(counts),
            'total': len(coords)
        }
