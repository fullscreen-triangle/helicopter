"""
Dual-Membrane Region: Image regions with pixel demon grids
==========================================================

Extends HCCC Region with pixel Maxwell demon capabilities.

Author: Kundai Sachikonye
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import sys
from pathlib import Path

# Import from parent maxwell module
sys.path.insert(0, str(Path(__file__).parent.parent))
from dual_membrane_pixel_demon import DualMembranePixelDemon
from pixel_maxwell_demon import PixelDemonGrid

# Import from HCCC framework
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from regions import Region
from regions.segmentation import ImageSegmenter

# Import integration modules
from dual_bmd_state import DualMembraneBMDState, pixel_demon_to_bmd_state


@dataclass
class DualMembraneRegion(Region):
    """
    Image region with pixel Maxwell demon grid for dual-membrane processing.
    
    Extends HCCC Region with:
    - Pixel demon grid for molecular-level information
    - Dual-membrane BMD states
    - Categorical depth from membrane thickness
    - Zero-backaction queries
    """
    
    # Pixel demon grid for this region
    pixel_grid: Optional[PixelDemonGrid] = None
    
    # Dual-membrane BMD states (one per pixel)
    dual_bmd_states: Optional[List[DualMembraneBMDState]] = None
    
    # Average membrane thickness (depth) for region
    average_depth: Optional[float] = None
    
    def initialize_pixel_demons(
        self,
        atmospheric_conditions: Optional[Dict] = None,
        use_cascade: bool = True,
        cascade_depth: int = 10
    ):
        """
        Initialize pixel Maxwell demons for this region.
        
        Args:
            atmospheric_conditions: Temperature, pressure, humidity
            use_cascade: Enable reflectance cascade
            cascade_depth: Cascade levels for O(N³) information gain
        """
        # Get region dimensions
        h, w = self.image.shape[:2]
        
        # Default atmospheric conditions
        if atmospheric_conditions is None:
            atmospheric_conditions = {
                'temperature': 298.15,  # 25°C in Kelvin
                'pressure': 101325,     # 1 atm in Pa
                'humidity': 0.5         # 50% relative humidity
            }
        
        # Create pixel demon grid
        self.pixel_grid = PixelDemonGrid(
            width=w,
            height=h,
            atmospheric_conditions=atmospheric_conditions
        )
        
        # Initialize grid from region image
        self.pixel_grid.initialize_from_image(self.image)
        
        # Convert each pixel demon to dual-membrane BMD state
        self.dual_bmd_states = []
        depth_values = []
        
        for y in range(h):
            for x in range(w):
                # Get pixel demon at (x, y)
                pixel_demon = self.pixel_grid.grid[y, x]
                
                # Convert to dual BMD state
                dual_bmd = pixel_demon_to_bmd_state(
                    pixel_demon,
                    use_cascade=use_cascade,
                    cascade_depth=cascade_depth
                )
                
                self.dual_bmd_states.append(dual_bmd)
                depth_values.append(dual_bmd.membrane_thickness())
        
        # Calculate average depth for region
        self.average_depth = np.mean(depth_values) if depth_values else 0.0
    
    def get_regional_bmd_state(self) -> DualMembraneBMDState:
        """
        Aggregate all pixel BMDs into single regional BMD.
        
        Uses phase-lock coupling to compose the network.
        
        Returns:
            Composite dual-membrane BMD for entire region
        """
        if not self.dual_bmd_states:
            raise ValueError("Pixel demons not initialized. Call initialize_pixel_demons() first.")
        
        # Import coupling operator
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from vision.bmd import PhaseLockCoupling
        
        coupling = PhaseLockCoupling()
        
        # Compose all front faces
        front_bmds = [state.front_bmd for state in self.dual_bmd_states]
        regional_front = coupling.compose_sequence(front_bmds)
        
        # Compose all back faces
        back_bmds = [state.back_bmd for state in self.dual_bmd_states]
        regional_back = coupling.compose_sequence(back_bmds)
        
        # Create regional dual-membrane BMD
        regional_dual_bmd = DualMembraneBMDState(
            front_bmd=regional_front,
            back_bmd=regional_back,
            observable_face=self.dual_bmd_states[0].observable_face,  # Use first pixel's face
            transform=self.dual_bmd_states[0].transform,
            source_pixel_demon=None  # This is a composite
        )
        
        return regional_dual_bmd
    
    def extract_depth_map(self) -> np.ndarray:
        """
        Extract depth map from membrane thickness values.
        
        Returns:
            Depth map with shape (height, width)
        """
        if not self.dual_bmd_states:
            raise ValueError("Pixel demons not initialized.")
        
        h, w = self.image.shape[:2]
        depth_map = np.zeros((h, w), dtype=np.float32)
        
        idx = 0
        for y in range(h):
            for x in range(w):
                depth_map[y, x] = self.dual_bmd_states[idx].membrane_thickness()
                idx += 1
        
        return depth_map
    
    def query_categorical_state(
        self,
        query_type: str = 'S_k'
    ) -> np.ndarray:
        """
        Perform zero-backaction categorical query on all pixels.
        
        Args:
            query_type: 'S_k', 'S_t', or 'S_e'
        
        Returns:
            Query result map with shape (height, width)
        """
        if not self.dual_bmd_states:
            raise ValueError("Pixel demons not initialized.")
        
        h, w = self.image.shape[:2]
        query_map = np.zeros((h, w), dtype=np.float32)
        
        idx = 0
        for y in range(h):
            for x in range(w):
                # Get observable BMD
                bmd = self.dual_bmd_states[idx].get_observable_bmd()
                
                # Extract query value from categorical state
                query_map[y, x] = bmd.c_current.get(query_type, 0.0)
                idx += 1
        
        return query_map


def create_dual_regions_from_image(
    image: np.ndarray,
    segmentation_method: str = 'slic',
    n_segments: int = 100,
    atmospheric_conditions: Optional[Dict] = None,
    use_cascade: bool = True,
    cascade_depth: int = 10
) -> List[DualMembraneRegion]:
    """
    Segment image into dual-membrane regions with pixel demon grids.
    
    Args:
        image: Input image (H, W, 3)
        segmentation_method: 'slic', 'felzenszwalb', or 'watershed'
        n_segments: Approximate number of segments
        atmospheric_conditions: Temperature, pressure, humidity
        use_cascade: Enable reflectance cascade
        cascade_depth: Cascade levels
    
    Returns:
        List of DualMembraneRegion objects
    """
    # Create segmenter
    segmenter = ImageSegmenter()
    
    # Segment image
    if segmentation_method == 'slic':
        regions = segmenter.slic_segmentation(image, n_segments=n_segments)
    elif segmentation_method == 'felzenszwalb':
        regions = segmenter.felzenszwalb_segmentation(image, scale=100, min_size=50)
    elif segmentation_method == 'watershed':
        regions = segmenter.watershed_segmentation(image)
    else:
        raise ValueError(f"Unknown segmentation method: {segmentation_method}")
    
    # Convert to dual-membrane regions
    dual_regions = []
    
    for region in regions:
        # Create dual-membrane region
        dual_region = DualMembraneRegion(
            image=region.image,
            mask=region.mask,
            bbox=region.bbox,
            features=region.features,
            categorical_states=region.categorical_states,
            pixel_grid=None,
            dual_bmd_states=None,
            average_depth=None
        )
        
        # Initialize pixel demons
        dual_region.initialize_pixel_demons(
            atmospheric_conditions=atmospheric_conditions,
            use_cascade=use_cascade,
            cascade_depth=cascade_depth
        )
        
        dual_regions.append(dual_region)
    
    return dual_regions

