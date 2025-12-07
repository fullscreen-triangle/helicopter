"""
Dual-Membrane Ambiguity Calculator: Extended ambiguity for conjugate structure
=============================================================================

Extends HCCC ambiguity calculation to account for dual-membrane structure,
conjugate faces, and hardware stream coherence.

Author: Kundai Sachikonye
"""

import numpy as np
from typing import Optional

# Import integration modules
from .dual_bmd_state import DualMembraneBMDState
from .dual_region import DualMembraneRegion
from .dual_network_bmd import DualMembraneNetworkBMD
from .pixel_hardware_stream import PixelDemonHardwareStream

# Import from HCCC framework
from categorical import AmbiguityCalculator


class DualMembraneAmbiguityCalculator(AmbiguityCalculator):
    """
    Extended ambiguity calculator for dual-membrane BMDs.
    
    Ambiguity definition for dual structure:
    A(β^(network), R) = A_front + A_back - λ_conjugate · C(front, back)
    
    Where:
    - A_front: Ambiguity on front face
    - A_back: Ambiguity on back face
    - C(front, back): Conjugate constraint satisfaction
    - λ_conjugate: Conjugate coupling strength
    """
    
    def __init__(
        self,
        conjugate_weight: float = 0.5,
        stream_weight: float = 0.5
    ):
        """
        Initialize dual-membrane ambiguity calculator.
        
        Args:
            conjugate_weight: Weight for conjugate constraint (λ_conjugate)
            stream_weight: Weight for stream coherence (λ_stream)
        """
        super().__init__()
        self.conjugate_weight = conjugate_weight
        self.stream_weight = stream_weight
    
    def calculate_dual_ambiguity(
        self,
        network_dual_bmd: DualMembraneBMDState,
        region: DualMembraneRegion
    ) -> float:
        """
        Calculate ambiguity for dual-membrane structure.
        
        A(β^(network), R) = A_front + A_back - λ_c · C(front, back)
        
        Args:
            network_dual_bmd: Current network dual BMD
            region: Candidate dual-membrane region
        
        Returns:
            Total ambiguity value
        """
        # Get regional dual BMD
        region_dual_bmd = region.get_regional_bmd_state()
        
        # Calculate front face ambiguity
        A_front = self.calculate(
            network_dual_bmd.front_bmd,
            region.image
        )
        
        # Calculate back face ambiguity
        A_back = self.calculate(
            network_dual_bmd.back_bmd,
            region.image
        )
        
        # Calculate conjugate constraint satisfaction
        C_conjugate = self._calculate_conjugate_constraint(
            network_dual_bmd,
            region_dual_bmd
        )
        
        # Total ambiguity
        A_dual = A_front + A_back - self.conjugate_weight * C_conjugate
        
        return A_dual
    
    def calculate_stream_coherent_ambiguity(
        self,
        network_dual_bmd: DualMembraneBMDState,
        region: DualMembraneRegion,
        hardware_stream: PixelDemonHardwareStream
    ) -> float:
        """
        Calculate stream-coherent ambiguity (dual objective).
        
        A_sc(β^(network), R) = A(β^(network), R) - λ · D_stream(β^(network) ⊛ R, β^(stream))
        
        Args:
            network_dual_bmd: Current network dual BMD
            region: Candidate region
            hardware_stream: Pixel demon hardware stream
        
        Returns:
            Stream-coherent ambiguity
        """
        # Base dual ambiguity
        A_dual = self.calculate_dual_ambiguity(network_dual_bmd, region)
        
        # Get compound BMD (network ⊛ region)
        region_dual_bmd = region.get_regional_bmd_state()
        from dual_network_bmd import DualMembraneNetworkBMD
        
        temp_network = DualMembraneNetworkBMD()
        compound_bmd = temp_network.compose_dual_bmds(
            network_dual_bmd,
            region_dual_bmd
        )
        
        # Calculate stream divergence
        D_stream = hardware_stream.calculate_stream_divergence(compound_bmd)
        
        # Stream-coherent ambiguity
        A_sc = A_dual - self.stream_weight * D_stream
        
        return A_sc
    
    def _calculate_conjugate_constraint(
        self,
        network_dual_bmd: DualMembraneBMDState,
        region_dual_bmd: DualMembraneBMDState
    ) -> float:
        """
        Calculate conjugate constraint satisfaction.
        
        C(front, back) = |T(β_front) - β_back|²
        
        Where T is the conjugate transform.
        
        Args:
            network_dual_bmd: Network dual BMD
            region_dual_bmd: Region dual BMD
        
        Returns:
            Constraint violation (0 = perfect conjugate relationship)
        """
        # Extract S-coordinates from BMD metadata
        # Network
        S_k_net_front = network_dual_bmd.front_bmd.metadata.get('S_k', 0.0)
        S_k_net_back = network_dual_bmd.back_bmd.metadata.get('S_k', 0.0)
        
        # Region
        S_k_reg_front = region_dual_bmd.front_bmd.metadata.get('S_k', 0.0)
        S_k_reg_back = region_dual_bmd.back_bmd.metadata.get('S_k', 0.0)
        
        # For phase conjugate: S_k_back = -S_k_front
        # Constraint violation
        violation_net = abs(S_k_net_back + S_k_net_front)  # Should be ~0
        violation_reg = abs(S_k_reg_back + S_k_reg_front)  # Should be ~0
        
        total_violation = violation_net + violation_reg
        
        return total_violation
    
    def calculate_depth_weighted_ambiguity(
        self,
        network_dual_bmd: DualMembraneBMDState,
        region: DualMembraneRegion,
        depth_weight: float = 0.1
    ) -> float:
        """
        Calculate ambiguity weighted by categorical depth.
        
        A_depth(β, R) = A(β, R) · (1 + α · d(R))
        
        Where d(R) is average membrane thickness of region.
        
        Args:
            network_dual_bmd: Network dual BMD
            region: Region with depth information
            depth_weight: Weight for depth contribution (α)
        
        Returns:
            Depth-weighted ambiguity
        """
        # Base ambiguity
        A_base = self.calculate_dual_ambiguity(network_dual_bmd, region)
        
        # Average depth
        depth = region.average_depth if region.average_depth else 0.0
        
        # Depth-weighted ambiguity
        A_depth = A_base * (1.0 + depth_weight * depth)
        
        return A_depth
    
    def calculate_cascade_enhanced_ambiguity(
        self,
        network_dual_bmd: DualMembraneBMDState,
        region: DualMembraneRegion,
        cascade_depth: int = 10
    ) -> float:
        """
        Calculate ambiguity enhanced by reflectance cascade.
        
        Cascade provides O(N³) information gain, enhancing ambiguity precision.
        
        Args:
            network_dual_bmd: Network dual BMD
            region: Region
            cascade_depth: Cascade levels (N)
        
        Returns:
            Cascade-enhanced ambiguity
        """
        # Base ambiguity
        A_base = self.calculate_dual_ambiguity(network_dual_bmd, region)
        
        # Cascade information enhancement factor
        # I_N = N(N+1)(2N+1)/6 ≈ N³/3
        N = cascade_depth
        cascade_factor = N * (N + 1) * (2 * N + 1) / 6
        
        # Normalize to reasonable range
        cascade_factor = min(cascade_factor, 1000)  # Cap at 1000
        cascade_enhancement = np.log1p(cascade_factor)  # log(1 + factor) for stability
        
        # Enhanced ambiguity
        A_cascade = A_base * cascade_enhancement
        
        return A_cascade
    
    def select_best_region(
        self,
        network_dual_bmd: DualMembraneBMDState,
        regions: list,  # List[DualMembraneRegion]
        hardware_stream: PixelDemonHardwareStream,
        use_depth_weighting: bool = True,
        use_cascade_enhancement: bool = True
    ) -> int:
        """
        Select region with maximum stream-coherent ambiguity.
        
        r* = argmax_r [A_sc(β^(network), r) + bonuses]
        
        Args:
            network_dual_bmd: Current network dual BMD
            regions: List of candidate regions
            hardware_stream: Pixel demon hardware stream
            use_depth_weighting: Include depth weighting
            use_cascade_enhancement: Include cascade enhancement
        
        Returns:
            Index of selected region
        """
        if not regions:
            raise ValueError("Empty region list")
        
        ambiguities = []
        
        for region in regions:
            # Base stream-coherent ambiguity
            A = self.calculate_stream_coherent_ambiguity(
                network_dual_bmd,
                region,
                hardware_stream
            )
            
            # Add depth weighting if enabled
            if use_depth_weighting and region.average_depth:
                depth_bonus = 0.1 * region.average_depth
                A += depth_bonus
            
            # Add cascade enhancement if enabled
            if use_cascade_enhancement:
                cascade_bonus = 0.01 * region.metadata.get('cascade_depth', 10)
                A += cascade_bonus
            
            ambiguities.append(A)
        
        # Select region with maximum ambiguity
        best_idx = int(np.argmax(ambiguities))
        
        return best_idx

