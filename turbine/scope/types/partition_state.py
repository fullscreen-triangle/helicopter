"""
Unified PartitionState type: the core type that bridges Partition Calculus,
Context-Dependent Coordinates, and Temporal Programming.
"""

from dataclasses import dataclass
from typing import Union, Literal
from enum import Enum


class Spin(Enum):
    """Spin/polarity: encodes handedness (partition), quadrature (FFT), timing phase (temporal)"""
    NEG_HALF = -0.5
    POS_HALF = +0.5

    def to_int(self) -> int:
        return -1 if self.value == -0.5 else 1


@dataclass(frozen=True)
class PartitionState:
    """
    Universal record encoding observable state in all three frameworks.

    Attributes:
        n: Depth (partition resolution, dyadic scale, oscillator depth)
        ℓ: Angular mode (partition label, Fourier orientation, channel index)
        m: Phase/offset (sub-index, Fourier sub-band, trajectory index)
        s: Spin/polarity (handedness, quadrature, early/late)
    """
    n: int  # depth: log₂(field_size / resolution)
    ℓ: int  # ell: mode (orientation in Fourier, channel in timing)
    m: int  # em: offset/index within mode
    s: Spin  # spin: binary polarity

    def __post_init__(self):
        if self.n < 0:
            raise ValueError(f"Depth n must be non-negative, got {self.n}")
        if self.ℓ < 0:
            raise ValueError(f"Mode ℓ must be non-negative, got {self.ℓ}")

    def __str__(self) -> str:
        spin_str = "↑" if self.s == Spin.POS_HALF else "↓"
        return f"Σ({self.n},{self.ℓ},{self.m},{spin_str})"

    def __repr__(self) -> str:
        return f"PartitionState(n={self.n}, ℓ={self.ℓ}, m={self.m}, s={self.s.name})"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'n': self.n,
            'ℓ': self.ℓ,
            'm': self.m,
            's': self.s.value
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'PartitionState':
        """Reconstruct from dictionary"""
        s_val = d['s']
        s = Spin.POS_HALF if s_val > 0 else Spin.NEG_HALF
        return cls(n=d['n'], ℓ=d['ℓ'], m=d['m'], s=s)

    def categorical_distance(self, other: 'PartitionState', epsilon_catalysts: float = 0.0) -> float:
        """
        Compute categorical distance to another partition state.

        From Partition Calculus: distance is the effort required to transform
        one state into another via morphism chains.

        Args:
            other: Target partition state
            epsilon_catalysts: Sum of catalyst epsilon values applied

        Returns:
            Categorical distance (float)
        """
        depth_diff = abs(self.n - other.n)
        mode_diff = abs(self.ℓ - other.ℓ)
        phase_diff = abs(self.m - other.m)
        spin_diff = 0.5 if self.s != other.s else 0.0

        base_distance = depth_diff + mode_diff + phase_diff + spin_diff
        return base_distance * (1.0 + epsilon_catalysts)

    def refinement_error(self, depth_delta_n: int) -> float:
        """
        Estimate partition refinement error at a new depth.

        From Theorem 2: δ_partition(n, ε) = L_field / 2^n * (1 + Σε_i)

        This method assumes L_field = 1 (normalized), returns fractional error.

        Args:
            depth_delta_n: Change in depth (Δn)

        Returns:
            Fractional error bound
        """
        if depth_delta_n < 0:
            raise ValueError(f"Depth delta must be >= 0, got {depth_delta_n}")
        # Base error at current depth
        base_error = 1.0 / (2 ** self.n) if self.n > 0 else 1.0
        # Error after refinement
        refined_error = 1.0 / (2 ** (self.n + depth_delta_n)) if (self.n + depth_delta_n) > 0 else 1.0
        return refined_error


@dataclass
class SEntropy:
    """
    Shannon entropy binding: joint conservation of S_k + S_t + S_e = 1

    Attributes:
        S_k: Knowledge entropy (partition state uncertainty)
        S_t: Timing entropy (temporal classification uncertainty)
        S_e: Environmental entropy (measurement backaction)
    """
    S_k: float
    S_t: float
    S_e: float

    def __post_init__(self):
        total = self.S_k + self.S_t + self.S_e
        if not (-1e-10 < total - 1.0 < 1e-10):
            raise ValueError(f"S_entropy must sum to 1, got {total} (S_k={self.S_k}, S_t={self.S_t}, S_e={self.S_e})")

    def __add__(self, other: 'SEntropy') -> 'SEntropy':
        """Combine entropies (for multi-phase analysis)"""
        s_k = self.S_k + other.S_k
        s_t = self.S_t + other.S_t
        s_e = self.S_e + other.S_e

        # Normalize to maintain conservation
        total = s_k + s_t + s_e
        if total > 1e-10:
            s_k /= total
            s_t /= total
            s_e /= total
        return SEntropy(S_k=s_k, S_t=s_t, S_e=s_e)

    def to_dict(self) -> dict:
        return {'S_k': self.S_k, 'S_t': self.S_t, 'S_e': self.S_e}

    @classmethod
    def initial(cls) -> 'SEntropy':
        """Initial entropy state: all uncertainty in timing"""
        return cls(S_k=0.0, S_t=1.0, S_e=0.0)

    @classmethod
    def uniform(cls) -> 'SEntropy':
        """Uniform entropy distribution"""
        return cls(S_k=1/3, S_t=1/3, S_e=1/3)


# Type aliases for clarity
PartitionStateId = str  # e.g., "nucleus_a", "membrane_boundary"
CoordFieldId = str  # e.g., "dapi_field", "ph_field"
MorphismChainId = str  # e.g., "nucleus_pair_measurement"
