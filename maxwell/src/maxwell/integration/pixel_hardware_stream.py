"""
Pixel Demon Hardware Stream: Hardware BMD stream from pixel demon measurements
=============================================================================

Integrates pixel Maxwell demons as hardware BMD references,
creating a unified phase-locked reality stream.

Author: Kundai Sachikonye
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import time
import sys
from pathlib import Path

# Import from parent maxwell module
sys.path.insert(0, str(Path(__file__).parent.parent))
from pixel_maxwell_demon import PixelMaxwellDemon, PixelDemonGrid
from dual_membrane_pixel_demon import DualMembranePixelDemon

# Import integration modules
from dual_bmd_state import pixel_demon_to_bmd_state, DualMembraneBMDState

# Import from HCCC framework
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from vision.bmd import HardwareBMDStream, BMDState, PhaseLockCoupling


@dataclass
class PixelDemonHardwareStream(HardwareBMDStream):
    """
    Hardware BMD stream based on pixel Maxwell demon measurements.
    
    Extends HCCC HardwareBMDStream with:
    - Pixel demon grid as hardware sensor
    - Atmospheric molecular measurements
    - Zero-backaction queries
    - O(1) harmonic coincidence network access
    """
    
    # Pixel demon grid (main sensor)
    pixel_grid: Optional[PixelDemonGrid] = None
    
    # Atmospheric conditions
    atmospheric_conditions: Dict[str, float] = field(default_factory=lambda: {
        'temperature': 298.15,  # K
        'pressure': 101325,     # Pa
        'humidity': 0.5,        # relative
        'CO2_ppm': 400,         # ppm
        'O2_fraction': 0.21     # mole fraction
    })
    
    # Molecular demon measurements
    molecular_measurements: Dict[str, any] = field(default_factory=dict)
    
    def initialize_pixel_sensors(
        self,
        width: int,
        height: int,
        atmospheric_conditions: Optional[Dict] = None
    ):
        """
        Initialize pixel demon grid as hardware sensor.
        
        Args:
            width: Grid width
            height: Grid height
            atmospheric_conditions: Override default atmospheric conditions
        """
        if atmospheric_conditions:
            self.atmospheric_conditions.update(atmospheric_conditions)
        
        # Create pixel demon grid
        self.pixel_grid = PixelDemonGrid(
            width=width,
            height=height,
            atmospheric_conditions=self.atmospheric_conditions
        )
        
        # Initialize from atmospheric state
        self.pixel_grid.initialize_from_atmospheric_state()
        
        # Measure molecular demons
        self._measure_molecular_demons()
        
        # Add to device BMDs
        self.device_bmds['pixel_grid'] = self._create_pixel_grid_bmd()
        
        # Update stream
        self.update_stream()
    
    def _measure_molecular_demons(self):
        """
        Measure molecular demon lattice from atmospheric conditions.
        
        Stores S-entropy coordinates, vibrational modes, and collision networks.
        """
        # Sample a representative pixel demon
        if self.pixel_grid and self.pixel_grid.grid.size > 0:
            sample_demon = self.pixel_grid.grid[0, 0]
            
            # Extract molecular measurements
            self.molecular_measurements = {
                'species': list(sample_demon.molecular_demons.keys()),
                'number_densities': {
                    mol: demon.number_density
                    for mol, demon in sample_demon.molecular_demons.items()
                },
                'vibrational_modes': {
                    mol: demon.vibrational_modes
                    for mol, demon in sample_demon.molecular_demons.items()
                },
                's_coordinates': {
                    mol: {
                        'S_k': demon.s_state.S_k,
                        'S_t': demon.s_state.S_t,
                        'S_e': demon.s_state.S_e
                    }
                    for mol, demon in sample_demon.molecular_demons.items()
                }
            }
    
    def _create_pixel_grid_bmd(self) -> BMDState:
        """
        Create hardware BMD from pixel demon grid.
        
        Returns:
            BMDState representing entire grid
        """
        if not self.pixel_grid:
            raise ValueError("Pixel grid not initialized")
        
        # Sample multiple pixels and compose
        h, w = self.pixel_grid.grid.shape
        sample_points = [
            (0, 0),                # Top-left
            (h//2, w//2),          # Center
            (h-1, w-1),            # Bottom-right
            (0, w-1),              # Top-right
            (h-1, 0)               # Bottom-left
        ]
        
        coupling = PhaseLockCoupling()
        bmd_states = []
        
        for y, x in sample_points:
            if 0 <= y < h and 0 <= x < w:
                pixel_demon = self.pixel_grid.grid[y, x]
                
                # Convert to BMD state
                dual_bmd = pixel_demon_to_bmd_state(
                    pixel_demon,
                    use_cascade=True,
                    cascade_depth=10
                )
                
                # Use front face (observable)
                bmd_states.append(dual_bmd.front_bmd)
        
        # Compose all sample BMDs
        grid_bmd = coupling.compose_sequence(bmd_states)
        
        # Add metadata
        grid_bmd.metadata.update({
            'device_type': 'pixel_demon_grid',
            'grid_shape': (h, w),
            'atmospheric_conditions': self.atmospheric_conditions,
            'timestamp': time.time()
        })
        
        return grid_bmd
    
    def update_from_image(self, image: np.ndarray):
        """
        Update pixel grid and hardware stream from new image.
        
        Args:
            image: Input image (H, W, 3)
        """
        if not self.pixel_grid:
            raise ValueError("Pixel grid not initialized")
        
        # Update pixel grid from image
        self.pixel_grid.initialize_from_image(image)
        
        # Remeasure molecular demons
        self._measure_molecular_demons()
        
        # Update pixel grid BMD
        self.device_bmds['pixel_grid'] = self._create_pixel_grid_bmd()
        
        # Update stream
        self.update_stream()
    
    def measure_stream_coherence_with_region(
        self,
        region_dual_bmd: DualMembraneBMDState
    ) -> float:
        """
        Calculate phase coherence between hardware stream and region BMD.
        
        Uses observable face of region BMD.
        
        Args:
            region_dual_bmd: Dual-membrane BMD from image region
        
        Returns:
            Coherence value ∈ [0, 1]
        """
        if not self.stream_bmd:
            return 0.0
        
        # Get observable BMD from region
        region_bmd = region_dual_bmd.get_observable_bmd()
        
        # Calculate coherence using phase-lock coupling
        coupling = PhaseLockCoupling()
        coherence = coupling.calculate_coherence(
            self.stream_bmd,
            region_bmd
        )
        
        return coherence
    
    def calculate_stream_divergence(
        self,
        compound_bmd: DualMembraneBMDState
    ) -> float:
        """
        Calculate divergence between hardware stream and compound BMD.
        
        D_stream = 1 - Φ(β^(stream), β^(compound))
        
        Args:
            compound_bmd: Compound dual BMD from region processing
        
        Returns:
            Divergence value ∈ [0, 1]
        """
        coherence = self.measure_stream_coherence_with_region(compound_bmd)
        divergence = 1.0 - coherence
        
        return divergence
    
    def get_molecular_demon_statistics(self) -> Dict[str, any]:
        """
        Get statistics on molecular demon measurements.
        
        Returns:
            Dictionary with species counts, densities, modes
        """
        if not self.molecular_measurements:
            return {}
        
        stats = {
            'n_species': len(self.molecular_measurements.get('species', [])),
            'species_list': self.molecular_measurements.get('species', []),
            'total_density': sum(
                self.molecular_measurements.get('number_densities', {}).values()
            ),
            'average_s_k': np.mean([
                coords['S_k']
                for coords in self.molecular_measurements.get('s_coordinates', {}).values()
            ]),
            'average_s_t': np.mean([
                coords['S_t']
                for coords in self.molecular_measurements.get('s_coordinates', {}).values()
            ]),
            'average_s_e': np.mean([
                coords['S_e']
                for coords in self.molecular_measurements.get('s_coordinates', {}).values()
            ])
        }
        
        return stats
    
    def to_dict(self) -> Dict[str, any]:
        """Serialize hardware stream to dictionary."""
        base_dict = super().to_dict()
        
        # Add pixel demon specific info
        base_dict.update({
            'pixel_grid_shape': self.pixel_grid.grid.shape if self.pixel_grid else None,
            'atmospheric_conditions': self.atmospheric_conditions,
            'molecular_statistics': self.get_molecular_demon_statistics()
        })
        
        return base_dict

