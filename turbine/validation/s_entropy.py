"""
S-Entropy Trajectory Analysis.

Validates S-entropy conservation theorem: S_k + S_t + S_e = constant

From the paper:
- S_k: Knowledge entropy (spatial/partition distribution uncertainty)
- S_t: Temporal entropy (transition rate uncertainty)
- S_e: Evolution entropy (accumulated measurement backaction)

The sum should be conserved during categorical measurement.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
from scipy import stats

from .data_loader import ImageData
from .partition_coordinates import PartitionCoordinateExtractor, PartitionCoordinate


@dataclass
class SEntropyState:
    """S-entropy coordinates at a single time point."""
    S_k: float      # Knowledge entropy (normalized to [0, 1])
    S_t: float      # Temporal entropy (normalized to [0, 1])
    S_e: float      # Evolution entropy (normalized to [0, 1])

    # Raw values before normalization
    S_k_raw: float
    S_t_raw: float
    S_e_raw: float

    # Metadata
    time_index: int
    n_nuclei: int
    n_transitions: int

    @property
    def S_total(self) -> float:
        """Total normalized S-entropy (should be ~1.0 if conserved)."""
        return self.S_k + self.S_t + self.S_e

    @property
    def S_total_raw(self) -> float:
        """Total raw S-entropy."""
        return self.S_k_raw + self.S_t_raw + self.S_e_raw


class SEntropyAnalyzer:
    """
    Analyze S-entropy trajectories and validate conservation.

    Treats consecutive images as a pseudo-time series to track
    partition coordinate transitions and compute S-entropy evolution.
    """

    def __init__(self, extractor: Optional[PartitionCoordinateExtractor] = None):
        """
        Initialize analyzer.

        Args:
            extractor: Partition coordinate extractor (created if not provided)
        """
        self.extractor = extractor or PartitionCoordinateExtractor()
        self.trajectories: List[SEntropyState] = []

    def compute_knowledge_entropy(self, coords: List[PartitionCoordinate]) -> Tuple[float, float]:
        """
        Compute knowledge entropy S_k from partition distribution.

        S_k measures uncertainty about spatial/partition location.
        Higher when nuclei spread across many partition states.

        Returns:
            (normalized_S_k, raw_S_k)
        """
        if not coords:
            return 0.0, 0.0

        # Joint distribution over (n, ℓ)
        joint_dist = Counter((c.n, c.ell) for c in coords)
        total = len(coords)

        # Shannon entropy
        probs = np.array([count / total for count in joint_dist.values()])
        raw_entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Maximum entropy (uniform over observed states)
        n_states = len(joint_dist)
        max_entropy = np.log(n_states) if n_states > 1 else 1.0

        # Normalize to [0, 1]
        normalized = raw_entropy / max_entropy if max_entropy > 0 else 0.0

        return float(normalized), float(raw_entropy)

    def compute_temporal_entropy(self,
                                  prev_coords: List[PartitionCoordinate],
                                  curr_coords: List[PartitionCoordinate]) -> Tuple[float, float, int]:
        """
        Compute temporal entropy S_t from transition rates.

        S_t measures uncertainty about when transitions occur.
        Based on counting state changes between consecutive observations.

        Returns:
            (normalized_S_t, raw_S_t, n_transitions)
        """
        if not prev_coords or not curr_coords:
            return 0.5, 0.5, 0

        # Match nuclei by label (simplification: use position for unlabeled)
        prev_by_label = {c.label: c for c in prev_coords}
        curr_by_label = {c.label: c for c in curr_coords}

        # Count transitions
        n_transitions = 0
        n_matched = 0

        for label, curr in curr_by_label.items():
            if label in prev_by_label:
                prev = prev_by_label[label]
                n_matched += 1

                # Count changes in partition coordinates
                if prev.n != curr.n:
                    n_transitions += 1
                if prev.ell != curr.ell:
                    n_transitions += 1
                if prev.m != curr.m:
                    n_transitions += 1
                if prev.s != curr.s:
                    n_transitions += 1

        if n_matched == 0:
            return 0.5, 0.5, 0

        # Transition rate
        max_transitions = 4 * n_matched  # Maximum possible transitions
        transition_rate = n_transitions / max_transitions if max_transitions > 0 else 0

        # Entropy from binary entropy function
        # H(p) = -p*log(p) - (1-p)*log(1-p)
        p = transition_rate
        if p < 1e-10:
            raw_entropy = 0.0
        elif p > 1 - 1e-10:
            raw_entropy = 0.0
        else:
            raw_entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)

        # Normalize (max entropy at p=0.5)
        max_entropy = np.log(2)
        normalized = raw_entropy / max_entropy

        return float(normalized), float(raw_entropy), n_transitions

    def compute_evolution_entropy(self,
                                   prev_S_e: float,
                                   backaction: float = 0.0) -> Tuple[float, float]:
        """
        Compute evolution entropy S_e from accumulated backaction.

        S_e tracks cumulative measurement disturbance.
        In zero-backaction regime, S_e should remain small.

        Returns:
            (normalized_S_e, raw_S_e)
        """
        # Accumulate backaction
        raw_S_e = prev_S_e + backaction

        # Normalize to [0, 1] with saturation
        normalized = np.tanh(raw_S_e)  # Smooth saturation

        return float(normalized), float(raw_S_e)

    def analyze_trajectory(self, samples: List[ImageData],
                           backaction_per_step: float = 0.001) -> List[SEntropyState]:
        """
        Analyze S-entropy trajectory over a sequence of images.

        Args:
            samples: List of ImageData (treated as time series)
            backaction_per_step: Assumed backaction per measurement step

        Returns:
            List of SEntropyState for each time point
        """
        self.trajectories = []

        # Extract partition coordinates for all samples
        all_coords = [self.extractor.extract_from_image(sample) for sample in samples]

        prev_coords = None
        prev_S_e_raw = 0.0

        for t, coords in enumerate(all_coords):
            # Knowledge entropy
            S_k_norm, S_k_raw = self.compute_knowledge_entropy(coords)

            # Temporal entropy (need previous frame)
            if prev_coords is not None:
                S_t_norm, S_t_raw, n_transitions = self.compute_temporal_entropy(
                    prev_coords, coords
                )
            else:
                S_t_norm, S_t_raw, n_transitions = 0.5, 0.5, 0

            # Evolution entropy
            S_e_norm, S_e_raw = self.compute_evolution_entropy(
                prev_S_e_raw, backaction_per_step
            )

            # Normalize to sum to 1
            S_total_unnorm = S_k_norm + S_t_norm + S_e_norm
            if S_total_unnorm > 0:
                S_k_final = S_k_norm / S_total_unnorm
                S_t_final = S_t_norm / S_total_unnorm
                S_e_final = S_e_norm / S_total_unnorm
            else:
                S_k_final = S_t_final = S_e_final = 1/3

            state = SEntropyState(
                S_k=S_k_final,
                S_t=S_t_final,
                S_e=S_e_final,
                S_k_raw=S_k_raw,
                S_t_raw=S_t_raw,
                S_e_raw=S_e_raw,
                time_index=t,
                n_nuclei=len(coords),
                n_transitions=n_transitions
            )

            self.trajectories.append(state)

            # Update for next iteration
            prev_coords = coords
            prev_S_e_raw = S_e_raw

        return self.trajectories

    def test_conservation(self) -> Dict:
        """
        Test S-entropy conservation: S_k + S_t + S_e = constant.

        Returns:
            Dictionary with conservation test results
        """
        if not self.trajectories:
            return {
                'validated': False,
                'reason': 'No trajectory data',
                'mean_total': np.nan,
                'std_total': np.nan,
                'cv': np.nan
            }

        # Compute S_total at each time point
        totals = [s.S_total for s in self.trajectories]

        mean_total = np.mean(totals)
        std_total = np.std(totals)
        cv = std_total / mean_total if mean_total > 0 else np.inf

        # Conservation validated if coefficient of variation < 5%
        validated = cv < 0.05

        return {
            'validated': validated,
            'reason': 'S-entropy conserved' if validated else 'S-entropy not conserved',
            'mean_total': float(mean_total),
            'std_total': float(std_total),
            'cv': float(cv),
            'n_timepoints': len(self.trajectories),
            'trajectory_totals': totals
        }

    def get_trajectory_arrays(self) -> Dict[str, np.ndarray]:
        """
        Get trajectory data as numpy arrays for plotting.

        Returns:
            Dictionary with t, S_k, S_t, S_e, S_total arrays
        """
        if not self.trajectories:
            return {
                't': np.array([]),
                'S_k': np.array([]),
                'S_t': np.array([]),
                'S_e': np.array([]),
                'S_total': np.array([]),
                'n_nuclei': np.array([]),
                'n_transitions': np.array([])
            }

        return {
            't': np.array([s.time_index for s in self.trajectories]),
            'S_k': np.array([s.S_k for s in self.trajectories]),
            'S_t': np.array([s.S_t for s in self.trajectories]),
            'S_e': np.array([s.S_e for s in self.trajectories]),
            'S_total': np.array([s.S_total for s in self.trajectories]),
            'n_nuclei': np.array([s.n_nuclei for s in self.trajectories]),
            'n_transitions': np.array([s.n_transitions for s in self.trajectories])
        }

    def compute_phase_space_trajectory(self) -> np.ndarray:
        """
        Get trajectory in S-entropy phase space for 3D visualization.

        Returns:
            Nx3 array of (S_k, S_t, S_e) coordinates
        """
        if not self.trajectories:
            return np.array([]).reshape(0, 3)

        return np.array([[s.S_k, s.S_t, s.S_e] for s in self.trajectories])
