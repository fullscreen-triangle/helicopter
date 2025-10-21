"""
Hierarchical network BMD for integrating all processing history.

The network BMD β^(network) is irreducible and nested:
- Individual region BMDs at lowest level
- Compound BMDs from sequences
- Global BMD encoding complete history
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from .bmd_state import BMDState
from .phase_lock import PhaseLockCoupling


class NetworkBMD:
    """
    Hierarchical network BMD integrating all processing history.

    Structure:
    - region_bmds: Individual region BMDs {region_id: BMD}
    - compound_bmds: Compound BMDs {(region_ids): BMD}
    - processing_sequence: Order of processing σ
    - global_bmd: Current unified network state

    All network changes reference the global network BMD.
    """

    def __init__(
        self,
        initial_hardware_stream: BMDState,
        max_compound_order: int = 5
    ):
        """
        Initialize network BMD with hardware stream.

        Args:
            initial_hardware_stream: Hardware BMD stream as foundation
            max_compound_order: Maximum order of compound BMDs
        """
        self.hardware_stream = initial_hardware_stream
        self.max_order = max_compound_order

        # Storage structures
        self.region_bmds: Dict[str, BMDState] = {}
        self.compound_bmds: Dict[Tuple[str, ...], BMDState] = {}
        self.processing_sequence: List[str] = []

        # Global network BMD starts as hardware stream
        self.global_bmd = initial_hardware_stream

        # Phase-lock coupling engine
        self.phase_lock = PhaseLockCoupling(coupling_strength=0.9)

        # History tracking
        self.richness_history: List[float] = []
        self.coherence_history: List[float] = []

    def integrate_region_bmd(
        self,
        region_id: str,
        region_bmd: BMDState,
        processing_step: int
    ):
        """
        Hierarchically integrate new region BMD into network.

        β^(network)_{i+1} = IntegrateHierarchical(β^(network}_i, β_{i+1}, σ ∪ R)

        Process:
        1. Add region BMD to collection
        2. Generate pairwise compounds with recent regions
        3. Generate higher-order compounds up to max_order
        4. Update global network BMD
        5. Record processing sequence

        Args:
            region_id: Identifier for region
            region_bmd: BMD state for this region
            processing_step: Step number in processing sequence
        """
        # Store individual region BMD
        self.region_bmds[region_id] = region_bmd
        self.processing_sequence.append(region_id)

        # Generate compound BMDs
        self._generate_compound_bmds(region_id)

        # Update global network BMD
        self._update_global_bmd(region_bmd)

        # Track metrics
        self.richness_history.append(self.network_categorical_richness())
        self.coherence_history.append(self.global_bmd.phase_lock_quality())

    def _generate_compound_bmds(self, new_region_id: str):
        """
        Generate compound BMDs involving new region.

        Creates:
        - Order 2: Pairwise (new_region, other_region)
        - Order 3: Triplets (new_region, r1, r2)
        - ...
        - Order max_order: Higher-order compounds
        """
        # Generate pairwise compounds with recent regions
        recent_regions = self.processing_sequence[-10:]  # Last 10 regions

        for other_id in recent_regions:
            if other_id != new_region_id and other_id in self.region_bmds:
                compound_id = tuple(sorted([new_region_id, other_id]))

                if compound_id not in self.compound_bmds:
                    # Create pairwise compound
                    bmd1 = self.region_bmds[new_region_id]
                    bmd2 = self.region_bmds[other_id]

                    compound = self.phase_lock.compose(bmd1, bmd2)
                    self.compound_bmds[compound_id] = compound

        # Generate higher-order compounds
        for order in range(3, self.max_order + 1):
            self._generate_order_k_compounds(new_region_id, order)

    def _generate_order_k_compounds(self, new_region_id: str, k: int):
        """
        Generate order-k compound BMDs involving new region.

        Args:
            new_region_id: New region to include
            k: Order of compound
        """
        if k > len(self.region_bmds):
            return

        # Sample k-1 other recent regions
        recent = [r for r in self.processing_sequence[-20:] if r != new_region_id]

        if len(recent) < k - 1:
            return

        # Create a few order-k compounds (not all, too expensive)
        n_samples = min(3, len(recent) - k + 2)

        for _ in range(n_samples):
            # Sample k-1 regions
            sampled = np.random.choice(recent, size=k-1, replace=False).tolist()
            compound_id = tuple(sorted([new_region_id] + sampled))

            if compound_id not in self.compound_bmds:
                # Compose BMDs
                bmds = [self.region_bmds[rid] for rid in compound_id
                       if rid in self.region_bmds]

                if len(bmds) == k:
                    compound = self.phase_lock.compose_sequence(bmds)
                    self.compound_bmds[compound_id] = compound

    def _update_global_bmd(self, new_region_bmd: BMDState):
        """
        Update global network BMD with new region.

        β^(network)_new = β^(network)_old ⊛ β_region
        """
        self.global_bmd = self.phase_lock.compose(
            self.global_bmd,
            new_region_bmd,
            coupling=0.95  # Strong coupling for global integration
        )

    def get_global_bmd(self) -> BMDState:
        """
        Get current global network BMD state.

        Returns:
            Global BMD representing entire processing history
        """
        return self.global_bmd

    def network_categorical_richness(self) -> float:
        """
        Calculate total network categorical richness.

        R(β^(network)) = R(β^(global)) × (1 + Σ R(β^(compound)))

        Grows as O(2^n) with processing steps due to compound formation.

        Returns:
            Total network richness
        """
        # Base richness from global BMD
        R_global = self.global_bmd.categorical_richness()

        # Compound BMD contributions
        R_compounds = sum(
            bmd.categorical_richness()
            for bmd in self.compound_bmds.values()
        )

        # Hierarchical scaling
        R_network = R_global * (1.0 + np.log1p(R_compounds))

        return R_network

    def get_compound_bmds_by_order(self, order: int) -> Dict[Tuple[str, ...], BMDState]:
        """
        Get all compound BMDs of specific order.

        Args:
            order: Compound order (2 = pairs, 3 = triplets, etc.)

        Returns:
            Dict of compound BMDs of requested order
        """
        return {
            compound_id: bmd
            for compound_id, bmd in self.compound_bmds.items()
            if len(compound_id) == order
        }

    def get_region_processing_order(self, region_id: str) -> int:
        """
        Get processing order (step) for region.

        Args:
            region_id: Region identifier

        Returns:
            Processing step number (0-indexed), or -1 if not processed
        """
        try:
            return self.processing_sequence.index(region_id)
        except ValueError:
            return -1

    def has_region(self, region_id: str) -> bool:
        """Check if region has been processed."""
        return region_id in self.region_bmds

    def get_recent_regions(self, n: int = 5) -> List[str]:
        """
        Get n most recently processed regions.

        Args:
            n: Number of recent regions

        Returns:
            List of region IDs in processing order
        """
        return self.processing_sequence[-n:]

    def count_compounds_involving(self, region_id: str) -> int:
        """
        Count compound BMDs involving specific region.

        Args:
            region_id: Region to check

        Returns:
            Number of compounds containing this region
        """
        count = 0
        for compound_id in self.compound_bmds.keys():
            if region_id in compound_id:
                count += 1
        return count

    def prune_low_richness_compounds(self, threshold_quantile: float = 0.25):
        """
        Prune compound BMDs with low categorical richness.

        Keeps network size manageable by removing least-informative compounds.

        Args:
            threshold_quantile: Keep only compounds above this richness quantile
        """
        if not self.compound_bmds:
            return

        # Calculate richness for all compounds
        richnesses = {
            cid: bmd.categorical_richness()
            for cid, bmd in self.compound_bmds.items()
        }

        # Find threshold
        threshold = np.quantile(list(richnesses.values()), threshold_quantile)

        # Keep only high-richness compounds
        self.compound_bmds = {
            cid: bmd
            for cid, bmd in self.compound_bmds.items()
            if richnesses[cid] >= threshold
        }

    def get_richness_growth_rate(self) -> float:
        """
        Calculate exponential growth rate of network richness.

        R(n) ~ R_0 * exp(α * n)

        Returns:
            Growth rate α (should be positive)
        """
        if len(self.richness_history) < 3:
            return 0.0

        # Log-linear fit
        log_richness = np.log(np.array(self.richness_history) + 1e-10)
        steps = np.arange(len(log_richness))

        alpha = np.polyfit(steps, log_richness, deg=1)[0]

        return alpha

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize network BMD to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'n_regions': len(self.region_bmds),
            'n_compounds': len(self.compound_bmds),
            'processing_sequence': self.processing_sequence,
            'global_bmd': self.global_bmd.to_dict(),
            'network_richness': self.network_categorical_richness(),
            'richness_history': self.richness_history,
            'coherence_history': self.coherence_history,
            'growth_rate': self.get_richness_growth_rate()
        }

    def __repr__(self) -> str:
        return (
            f"NetworkBMD(regions={len(self.region_bmds)}, "
            f"compounds={len(self.compound_bmds)}, "
            f"steps={len(self.processing_sequence)}, "
            f"R={self.network_categorical_richness():.2e})"
        )
