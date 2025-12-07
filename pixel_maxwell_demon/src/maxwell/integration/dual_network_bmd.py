"""
Dual-Membrane Network BMD: Hierarchical network with conjugate structure
========================================================================

Extends HCCC NetworkBMD to maintain dual-membrane relationships
throughout the hierarchical network.

Author: Kundai Sachikonye
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field

# Import integration modules
from .dual_bmd_state import DualMembraneBMDState
from .dual_region import DualMembraneRegion

# Import from HCCC framework
from vision.bmd import NetworkBMD, BMDState, PhaseLockCoupling


@dataclass
class DualMembraneNetworkBMD:
    """
    Hierarchical network BMD with dual-membrane structure.
    
    Maintains:
    - Individual region dual BMDs
    - Compound dual BMDs (from processing sequences)
    - Global network dual BMD (irreducible network state)
    - Conjugate relationships throughout hierarchy
    """
    
    # Individual region dual BMDs {region_id: DualMembraneBMDState}
    region_bmds: Dict[int, DualMembraneBMDState] = field(default_factory=dict)
    
    # Compound dual BMDs {frozenset(region_ids): DualMembraneBMDState}
    compound_bmds: Dict[frozenset, DualMembraneBMDState] = field(default_factory=dict)
    
    # Global network dual BMD (combines all processing)
    global_network_bmd: Optional[DualMembraneBMDState] = None
    
    # Processing order (for path-dependence tracking)
    processing_sequence: List[int] = field(default_factory=list)
    
    # Coupling operator for composition
    coupling: PhaseLockCoupling = field(default_factory=PhaseLockCoupling)
    
    def add_region_bmd(self, region_id: int, dual_bmd: DualMembraneBMDState):
        """Add individual region dual BMD to network."""
        self.region_bmds[region_id] = dual_bmd
        self.processing_sequence.append(region_id)
        self._update_global_network()
    
    def add_compound_bmd(
        self,
        region_ids: Set[int],
        dual_bmd: DualMembraneBMDState
    ):
        """Add compound dual BMD from processing sequence."""
        self.compound_bmds[frozenset(region_ids)] = dual_bmd
        self._update_global_network()
    
    def compose_dual_bmds(
        self,
        dual_bmd_1: DualMembraneBMDState,
        dual_bmd_2: DualMembraneBMDState
    ) -> DualMembraneBMDState:
        """
        Compose two dual-membrane BMDs using phase-lock coupling.
        
        β^(12) = β^(1) ⊛ β^(2)
        
        Both front and back faces are composed separately to maintain
        conjugate relationship.
        
        Args:
            dual_bmd_1: First dual BMD
            dual_bmd_2: Second dual BMD
        
        Returns:
            Composed dual BMD
        """
        # Compose front faces
        front_composed = self.coupling.compose(
            dual_bmd_1.front_bmd,
            dual_bmd_2.front_bmd
        )
        
        # Compose back faces
        back_composed = self.coupling.compose(
            dual_bmd_1.back_bmd,
            dual_bmd_2.back_bmd
        )
        
        # Create composed dual BMD
        composed_dual = DualMembraneBMDState(
            front_bmd=front_composed,
            back_bmd=back_composed,
            observable_face=dual_bmd_1.observable_face,  # Inherit from first
            transform=dual_bmd_1.transform,
            source_pixel_demon=None
        )
        
        return composed_dual
    
    def compose_sequence(
        self,
        region_ids: List[int]
    ) -> DualMembraneBMDState:
        """
        Compose sequence of regions into compound dual BMD.
        
        β^(123...n) = (...((β^(1) ⊛ β^(2)) ⊛ β^(3))... ⊛ β^(n))
        
        Args:
            region_ids: Ordered list of region IDs
        
        Returns:
            Compound dual BMD
        """
        if not region_ids:
            raise ValueError("Empty region sequence")
        
        # Start with first region
        compound = self.region_bmds[region_ids[0]]
        
        # Compose with remaining regions
        for rid in region_ids[1:]:
            compound = self.compose_dual_bmds(compound, self.region_bmds[rid])
        
        # Store compound
        self.add_compound_bmd(set(region_ids), compound)
        
        return compound
    
    def calculate_network_richness(self) -> float:
        """
        Calculate total categorical richness of entire network.
        
        R_network = R(β^(global))
        
        Returns:
            Total richness including all faces
        """
        if self.global_network_bmd is None:
            return 0.0
        
        return self.global_network_bmd.categorical_richness_dual()
    
    def calculate_depth_statistics(self) -> Dict[str, float]:
        """
        Calculate depth statistics across entire network.
        
        Returns:
            Dictionary with mean, std, min, max depth
        """
        if not self.region_bmds:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        depths = [bmd.membrane_thickness() for bmd in self.region_bmds.values()]
        
        return {
            'mean': np.mean(depths),
            'std': np.std(depths),
            'min': np.min(depths),
            'max': np.max(depths)
        }
    
    def extract_global_depth_map(
        self,
        regions: List[DualMembraneRegion],
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Extract global depth map from all regions.
        
        Args:
            regions: List of dual-membrane regions
            image_shape: (height, width) of original image
        
        Returns:
            Depth map with shape (height, width)
        """
        h, w = image_shape
        depth_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.int32)
        
        for region_id, region in enumerate(regions):
            # Get region depth map
            region_depth = region.extract_depth_map()
            
            # Get region mask
            mask = region.mask
            
            # Add to global map (with blending for overlaps)
            y1, x1, y2, x2 = region.bbox
            depth_map[y1:y2, x1:x2][mask] += region_depth[mask]
            count_map[y1:y2, x1:x2][mask] += 1
        
        # Average where regions overlap
        depth_map = np.divide(
            depth_map,
            count_map,
            out=np.zeros_like(depth_map),
            where=count_map > 0
        )
        
        return depth_map
    
    def get_hierarchical_structure(self) -> Dict[str, any]:
        """
        Get complete hierarchical network structure.
        
        Returns:
            Dictionary describing the hierarchy
        """
        structure = {
            'n_regions': len(self.region_bmds),
            'n_compounds': len(self.compound_bmds),
            'processing_sequence': self.processing_sequence,
            'total_richness': self.calculate_network_richness(),
            'depth_statistics': self.calculate_depth_statistics(),
            'has_global_bmd': self.global_network_bmd is not None
        }
        
        # Add compound structure
        compound_structure = []
        for region_set, bmd in self.compound_bmds.items():
            compound_structure.append({
                'regions': sorted(list(region_set)),
                'size': len(region_set),
                'richness': bmd.categorical_richness_dual(),
                'depth': bmd.membrane_thickness()
            })
        
        structure['compounds'] = compound_structure
        
        return structure
    
    def _update_global_network(self):
        """Update global network BMD from all processing."""
        if not self.region_bmds:
            return
        
        # Compose all regions in processing order
        self.global_network_bmd = self.compose_sequence(self.processing_sequence)
    
    def to_dict(self) -> Dict[str, any]:
        """Serialize network to dictionary."""
        return {
            'structure': self.get_hierarchical_structure(),
            'region_bmds': {
                rid: bmd.to_dict()
                for rid, bmd in self.region_bmds.items()
            },
            'global_bmd': self.global_network_bmd.to_dict() if self.global_network_bmd else None
        }

