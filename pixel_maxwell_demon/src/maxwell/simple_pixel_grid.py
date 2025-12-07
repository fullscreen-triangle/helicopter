"""
Simple Pixel Demon Grid: Easy-to-Use Wrapper for Image Processing
================================================================

This provides a simplified API for creating pixel demon grids from images,
wrapping the more complex DualMembraneGrid class.

Author: Kundai Sachikonye
Date: 2024
"""

import numpy as np
from typing import Dict, Optional
import logging

from .dual_membrane_pixel_demon import DualMembraneGrid, DualMembranePixelDemon
from .pixel_maxwell_demon import SEntropyCoordinates

logger = logging.getLogger(__name__)


class PixelDemonGrid:
    """
    Simple wrapper for DualMembraneGrid that provides easy image-based initialization.
    
    Usage:
        grid = PixelDemonGrid(width=512, height=512)
        grid.initialize_from_image(image)
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        atmospheric_conditions: Optional[Dict] = None
    ):
        """
        Initialize pixel demon grid.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            atmospheric_conditions: Dict with 'temperature', 'pressure', 'humidity'
        """
        self.width = width
        self.height = height
        
        # Default atmospheric conditions
        if atmospheric_conditions is None:
            atmospheric_conditions = {
                'temperature': 298.15,  # 25°C
                'pressure': 101325,
                'humidity': 0.5
            }
        
        self.atmospheric_conditions = atmospheric_conditions
        
        # Create underlying dual membrane grid
        # shape is (height, width), physical_extent in meters
        self.dual_grid = DualMembraneGrid(
            shape=(height, width),
            physical_extent=(0.1, 0.1),  # 10cm x 10cm
            transform_type='phase_conjugate',
            synchronized_switching=True,
            switching_frequency=5.0,  # 5 Hz
            name="image_grid"
        )
        
        # Initialize atmospheric lattices
        self.dual_grid.initialize_all_atmospheric(
            temperature_k=atmospheric_conditions['temperature'],
            pressure_pa=atmospheric_conditions['pressure'],
            humidity_fraction=atmospheric_conditions['humidity']
        )
        
        # Grid of demons (convenient access)
        self.grid = self.dual_grid.demons
        
        logger.info(f"Created PixelDemonGrid: {width}×{height}")
    
    def initialize_from_image(self, image: np.ndarray):
        """
        Initialize pixel demons from image data.
        
        Args:
            image: RGB image array (H, W, 3)
        """
        h, w = image.shape[:2]
        
        if h != self.height or w != self.width:
            raise ValueError(
                f"Image shape ({h}, {w}) doesn't match grid shape ({self.height}, {self.width})"
            )
        
        # Convert to grayscale for S_k initialization
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Normalize to [0, 1]
        gray_normalized = gray / 255.0 if gray.max() > 1.0 else gray
        
        # Initialize each pixel demon's S-state from image intensity
        for y in range(h):
            for x in range(w):
                demon = self.grid[y, x]
                intensity = gray_normalized[y, x]
                
                # Map intensity to S_k coordinate
                # Higher intensity → higher knowledge (more information)
                S_k = intensity * 2.0 - 1.0  # Map [0,1] → [-1, 1]
                
                # S_t from spatial gradient (local variation)
                if x > 0 and x < w-1 and y > 0 and y < h-1:
                    dx = float(gray_normalized[y, x+1] - gray_normalized[y, x-1])
                    dy = float(gray_normalized[y+1, x] - gray_normalized[y-1, x])
                    gradient_mag = np.sqrt(dx**2 + dy**2)
                    S_t = gradient_mag
                else:
                    S_t = 0.0
                
                # S_e from local entropy (texture)
                if x > 1 and x < w-2 and y > 1 and y < h-2:
                    local_patch = gray_normalized[y-1:y+2, x-1:x+2]
                    S_e = float(np.std(local_patch))
                else:
                    S_e = 0.0
                
                # Set front S-state
                demon.dual_state.front_s = SEntropyCoordinates(
                    S_k=S_k,
                    S_t=S_t,
                    S_e=S_e
                )
                
                # Back state is phase conjugate
                demon.dual_state.back_s = SEntropyCoordinates(
                    S_k=-S_k,  # Phase conjugate
                    S_t=S_t,   # Same temporal
                    S_e=S_e    # Same entropy
                )
        
        logger.info(f"Initialized {h}×{w} pixel demons from image")
    
    def get_depth_map(self) -> np.ndarray:
        """
        Extract depth from membrane thickness.
        
        Returns:
            2D depth map
        """
        depth_map = np.zeros((self.height, self.width))
        
        for y in range(self.height):
            for x in range(self.width):
                demon = self.grid[y, x]
                s_front = demon.dual_state.front_s
                s_back = demon.dual_state.back_s
                
                # Depth = membrane thickness = |S_k^(front) - S_k^(back)|
                depth = abs(s_front.S_k - s_back.S_k)
                depth_map[y, x] = depth
        
        return depth_map
    
    def switch_all_faces(self):
        """Switch all pixel faces simultaneously."""
        self.dual_grid.switch_all_faces()
    
    def measure_observable_grid(self) -> np.ndarray:
        """Get observable face measurements."""
        return self.dual_grid.measure_observable_grid()

