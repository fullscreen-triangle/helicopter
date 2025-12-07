"""
Dual-Membrane HCCC Algorithm: Modified algorithm with pixel demon integration
============================================================================

Complete HCCC algorithm extended for dual-membrane pixel Maxwell demons.

Key features:
- Dual-membrane BMD states throughout hierarchy
- Zero-backaction categorical queries
- O(N³) cascade information gain
- Hardware stream coherence via pixel demons
- Depth extraction from membrane thickness

Author: Kundai Sachikonye
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time
import sys
from pathlib import Path

# Import integration modules
from dual_bmd_state import DualMembraneBMDState, pixel_demon_to_bmd_state
from dual_region import DualMembraneRegion, create_dual_regions_from_image
from dual_network_bmd import DualMembraneNetworkBMD
from pixel_hardware_stream import PixelDemonHardwareStream
from dual_ambiguity import DualMembraneAmbiguityCalculator


@dataclass
class DualHCCCResult:
    """Results from dual-membrane HCCC processing."""
    
    # Final network BMD
    network_bmd: DualMembraneNetworkBMD
    
    # Extracted depth map
    depth_map: np.ndarray
    
    # Processing sequence
    processing_order: List[int]
    
    # Iteration history
    iteration_history: List[Dict]
    
    # Hardware stream state
    hardware_stream: PixelDemonHardwareStream
    
    # Computational metrics
    total_iterations: int
    total_time: float
    energy_dissipation: float
    
    # Convergence metrics
    final_richness: float
    final_stream_coherence: float
    converged: bool


class DualMembraneHCCCAlgorithm:
    """
    Hardware-Constrained Categorical Completion with Dual-Membrane Pixel Demons.
    
    Algorithm:
    1. Initialize pixel demon hardware stream
    2. Segment image into dual-membrane regions
    3. While not converged:
        a. Select region with max stream-coherent ambiguity
        b. Process region via zero-backaction query
        c. Integrate into network BMD
        d. Update hardware stream
        e. Check convergence
    4. Extract depth map from membrane thickness
    5. Return results
    """
    
    def __init__(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        lambda_stream: float = 0.5,
        lambda_conjugate: float = 0.5,
        use_cascade: bool = True,
        cascade_depth: int = 10,
        atmospheric_conditions: Optional[Dict] = None
    ):
        """
        Initialize dual-membrane HCCC algorithm.
        
        Args:
            max_iterations: Maximum processing iterations
            convergence_threshold: Convergence criterion (Δ richness)
            lambda_stream: Stream coherence weight
            lambda_conjugate: Conjugate constraint weight
            use_cascade: Enable reflectance cascade
            cascade_depth: Cascade levels (N for O(N³) gain)
            atmospheric_conditions: Temperature, pressure, humidity
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.lambda_stream = lambda_stream
        self.lambda_conjugate = lambda_conjugate
        self.use_cascade = use_cascade
        self.cascade_depth = cascade_depth
        self.atmospheric_conditions = atmospheric_conditions or {
            'temperature': 298.15,
            'pressure': 101325,
            'humidity': 0.5
        }
        
        # Components
        self.hardware_stream: Optional[PixelDemonHardwareStream] = None
        self.network_bmd: Optional[DualMembraneNetworkBMD] = None
        self.ambiguity_calc: Optional[DualMembraneAmbiguityCalculator] = None
        self.regions: List[DualMembraneRegion] = []
        
        # Processing state
        self.processed_regions: set = set()
        self.iteration_history: List[Dict] = []
    
    def process_image(
        self,
        image: np.ndarray,
        n_segments: int = 100,
        segmentation_method: str = 'slic'
    ) -> DualHCCCResult:
        """
        Process image with dual-membrane HCCC algorithm.
        
        Args:
            image: Input image (H, W, 3)
            n_segments: Number of regions to create
            segmentation_method: 'slic', 'felzenszwalb', or 'watershed'
        
        Returns:
            DualHCCCResult with depth map and network BMD
        """
        start_time = time.time()
        
        # Step 1: Initialize hardware stream
        print("Step 1: Initializing pixel demon hardware stream...")
        self._initialize_hardware_stream(image.shape[:2])
        
        # Step 2: Segment image into dual-membrane regions
        print(f"Step 2: Segmenting image into {n_segments} dual-membrane regions...")
        self._segment_image(
            image,
            n_segments=n_segments,
            method=segmentation_method
        )
        
        # Step 3: Initialize network BMD
        print("Step 3: Initializing network BMD...")
        self.network_bmd = DualMembraneNetworkBMD()
        
        # Step 4: Initialize ambiguity calculator
        self.ambiguity_calc = DualMembraneAmbiguityCalculator(
            conjugate_weight=self.lambda_conjugate,
            stream_weight=self.lambda_stream
        )
        
        # Step 5: Main processing loop
        print(f"Step 5: Processing regions (max {self.max_iterations} iterations)...")
        converged = False
        iteration = 0
        prev_richness = 0.0
        
        while iteration < self.max_iterations and not converged:
            # Select best region
            region_idx = self._select_best_region()
            
            if region_idx is None:
                print("All regions processed.")
                break
            
            # Process region
            self._process_region(region_idx)
            
            # Mark as processed
            self.processed_regions.add(region_idx)
            
            # Check convergence
            current_richness = self.network_bmd.calculate_network_richness()
            delta_richness = abs(current_richness - prev_richness)
            
            converged = delta_richness < self.convergence_threshold
            prev_richness = current_richness
            
            # Record iteration
            self.iteration_history.append({
                'iteration': iteration,
                'region_idx': region_idx,
                'richness': current_richness,
                'delta_richness': delta_richness,
                'stream_coherence': self._measure_stream_coherence()
            })
            
            iteration += 1
            
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: richness = {current_richness:.6f}, "
                      f"Δ = {delta_richness:.8f}")
        
        # Step 6: Extract depth map
        print("Step 6: Extracting depth map from membrane thickness...")
        depth_map = self._extract_depth_map(image.shape[:2])
        
        # Calculate total time and energy
        total_time = time.time() - start_time
        energy_dissipation = self._calculate_energy_dissipation()
        
        # Create result
        result = DualHCCCResult(
            network_bmd=self.network_bmd,
            depth_map=depth_map,
            processing_order=self.network_bmd.processing_sequence,
            iteration_history=self.iteration_history,
            hardware_stream=self.hardware_stream,
            total_iterations=iteration,
            total_time=total_time,
            energy_dissipation=energy_dissipation,
            final_richness=prev_richness,
            final_stream_coherence=self._measure_stream_coherence(),
            converged=converged
        )
        
        print(f"\nProcessing complete!")
        print(f"  Total iterations: {iteration}")
        print(f"  Final richness: {result.final_richness:.6f}")
        print(f"  Stream coherence: {result.final_stream_coherence:.4f}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Energy dissipation: {energy_dissipation:.3e} J")
        print(f"  Converged: {converged}")
        
        return result
    
    def _initialize_hardware_stream(self, image_shape: Tuple[int, int]):
        """Initialize pixel demon hardware stream."""
        h, w = image_shape
        
        self.hardware_stream = PixelDemonHardwareStream()
        self.hardware_stream.initialize_pixel_sensors(
            width=w,
            height=h,
            atmospheric_conditions=self.atmospheric_conditions
        )
    
    def _segment_image(
        self,
        image: np.ndarray,
        n_segments: int,
        method: str
    ):
        """Segment image into dual-membrane regions."""
        self.regions = create_dual_regions_from_image(
            image,
            segmentation_method=method,
            n_segments=n_segments,
            atmospheric_conditions=self.atmospheric_conditions,
            use_cascade=self.use_cascade,
            cascade_depth=self.cascade_depth
        )
        
        print(f"  Created {len(self.regions)} dual-membrane regions")
    
    def _select_best_region(self) -> Optional[int]:
        """
        Select region with maximum stream-coherent ambiguity.
        
        Returns:
            Region index, or None if all processed
        """
        unprocessed_indices = [
            i for i in range(len(self.regions))
            if i not in self.processed_regions
        ]
        
        if not unprocessed_indices:
            return None
        
        # Get current network BMD (or create initial one)
        if len(self.processed_regions) == 0:
            # First region: select one with maximum richness
            richnesses = [
                r.get_regional_bmd_state().categorical_richness_dual()
                for r in self.regions
            ]
            return int(np.argmax(richnesses))
        
        # Use ambiguity calculator
        network_dual_bmd = self.network_bmd.global_network_bmd
        
        unprocessed_regions = [self.regions[i] for i in unprocessed_indices]
        
        best_local_idx = self.ambiguity_calc.select_best_region(
            network_dual_bmd,
            unprocessed_regions,
            self.hardware_stream,
            use_depth_weighting=True,
            use_cascade_enhancement=self.use_cascade
        )
        
        # Convert local index to global index
        best_global_idx = unprocessed_indices[best_local_idx]
        
        return best_global_idx
    
    def _process_region(self, region_idx: int):
        """Process region and integrate into network."""
        region = self.regions[region_idx]
        
        # Get regional dual BMD
        regional_dual_bmd = region.get_regional_bmd_state()
        
        # Add to network
        self.network_bmd.add_region_bmd(region_idx, regional_dual_bmd)
        
        # Update hardware stream
        self.hardware_stream.update_from_image(region.image)
    
    def _extract_depth_map(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Extract global depth map from all regions."""
        return self.network_bmd.extract_global_depth_map(
            self.regions,
            image_shape
        )
    
    def _measure_stream_coherence(self) -> float:
        """Measure coherence between network and hardware stream."""
        if not self.network_bmd or not self.network_bmd.global_network_bmd:
            return 0.0
        
        return self.hardware_stream.measure_stream_coherence_with_region(
            self.network_bmd.global_network_bmd
        )
    
    def _calculate_energy_dissipation(self) -> float:
        """
        Calculate total energy dissipation (Landauer's principle).
        
        E_total = k_B T ln(2) × (# bit erasures)
        
        Returns:
            Energy in Joules
        """
        k_B = 1.380649e-23  # Boltzmann constant (J/K)
        T = self.atmospheric_conditions.get('temperature', 298.15)
        
        # Each categorical completion erases ~1 bit of uncertainty
        n_completions = len(self.processed_regions)
        
        # Each pixel demon has multiple oscillatory holes
        # Estimate ~1000 holes per region
        n_holes_per_region = 1000
        
        total_bits = n_completions * n_holes_per_region
        
        E_total = k_B * T * np.log(2) * total_bits
        
        return E_total

