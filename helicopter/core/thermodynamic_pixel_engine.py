"""
Thermodynamic Pixel Processing Engine

Implements the thermodynamic approach to pixel processing where each pixel is treated
as a thermodynamic entity with entropy, temperature, and equilibrium properties.

Key Features:
- Pixel-level entropy modeling
- Temperature-controlled resource allocation
- Equilibrium-based optimization
- Adaptive processing based on local complexity
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from scipy.optimize import minimize
import math
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessingState(Enum):
    """Thermodynamic processing states"""
    COLD = "cold"          # Low entropy, minimal processing
    WARM = "warm"          # Medium entropy, standard processing  
    HOT = "hot"            # High entropy, intensive processing
    CRITICAL = "critical"  # Very high entropy, maximum processing


@dataclass
class PixelState:
    """Thermodynamic state of a single pixel"""
    entropy: float                    # Information entropy
    temperature: float               # Computational temperature
    energy: float                   # Internal energy
    free_energy: float              # Helmholtz free energy (E - T*S)
    processing_resources: int       # Allocated computational resources
    equilibrium_reached: bool       # Whether pixel has reached equilibrium
    class_probabilities: np.ndarray # Probability distribution over classes
    neighbors_influence: float      # Influence from neighboring pixels
    

@dataclass
class ThermodynamicMetrics:
    """Global thermodynamic metrics for the image"""
    total_entropy: float
    average_temperature: float
    total_free_energy: float
    equilibrium_percentage: float
    processing_efficiency: float
    resource_allocation: Dict[str, int]


class ThermodynamicPixelEngine:
    """
    Core thermodynamic pixel processing engine.
    
    Models each pixel as a thermodynamic entity and optimizes processing
    resources based on local entropy and temperature.
    """
    
    def __init__(
        self,
        base_temperature: float = 1.0,
        max_temperature: float = 10.0,
        equilibrium_threshold: float = 1e-6,
        max_iterations: int = 100,
        num_classes: int = 256,
        neighborhood_size: int = 3
    ):
        self.base_temperature = base_temperature
        self.max_temperature = max_temperature
        self.equilibrium_threshold = equilibrium_threshold
        self.max_iterations = max_iterations
        self.num_classes = num_classes
        self.neighborhood_size = neighborhood_size
        
        # Processing states thresholds
        self.state_thresholds = {
            ProcessingState.COLD: 0.5,
            ProcessingState.WARM: 2.0,
            ProcessingState.HOT: 5.0,
            ProcessingState.CRITICAL: 8.0
        }
        
        # Resource allocation per state
        self.state_resources = {
            ProcessingState.COLD: 1,
            ProcessingState.WARM: 4,
            ProcessingState.HOT: 16,
            ProcessingState.CRITICAL: 64
        }
        
        logger.info("Initialized Thermodynamic Pixel Engine")
    
    def calculate_pixel_entropy(
        self, 
        pixel_value: Union[int, np.ndarray],
        neighborhood: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate entropy for a single pixel based on its value and neighborhood.
        
        Args:
            pixel_value: Pixel intensity or RGB values
            neighborhood: Neighboring pixel values for context
            
        Returns:
            Entropy value for the pixel
        """
        if isinstance(pixel_value, (int, float)):
            # Grayscale pixel
            probabilities = self._get_class_probabilities_grayscale(pixel_value, neighborhood)
        else:
            # RGB pixel
            probabilities = self._get_class_probabilities_rgb(pixel_value, neighborhood)
        
        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _get_class_probabilities_grayscale(
        self, 
        pixel_value: float, 
        neighborhood: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate class probabilities for grayscale pixel"""
        # Simple model: probability decreases with distance from pixel value
        classes = np.arange(self.num_classes)
        distances = np.abs(classes - pixel_value)
        
        # Add neighborhood influence
        if neighborhood is not None:
            neighborhood_mean = np.mean(neighborhood)
            neighborhood_distances = np.abs(classes - neighborhood_mean)
            distances = 0.7 * distances + 0.3 * neighborhood_distances
        
        # Convert to probabilities using softmax with temperature
        probabilities = np.exp(-distances / self.base_temperature)
        probabilities = probabilities / np.sum(probabilities)
        
        return probabilities
    
    def _get_class_probabilities_rgb(
        self, 
        pixel_value: np.ndarray, 
        neighborhood: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate class probabilities for RGB pixel"""
        # For RGB, we calculate entropy based on color space uncertainty
        r, g, b = pixel_value
        
        # Create probability distribution in RGB space
        # This is a simplified model - in practice, would use learned distributions
        color_variance = np.var([r, g, b])
        
        # Higher variance = higher entropy
        base_entropy = color_variance / (255 * 255)
        
        # Create uniform distribution with entropy-dependent spread
        num_probable_classes = max(1, int(base_entropy * self.num_classes))
        probabilities = np.zeros(self.num_classes)
        
        # Assign probabilities to most likely classes
        center_class = int(np.mean([r, g, b]))
        half_spread = num_probable_classes // 2
        
        start_idx = max(0, center_class - half_spread)
        end_idx = min(self.num_classes, center_class + half_spread + 1)
        
        probabilities[start_idx:end_idx] = 1.0 / (end_idx - start_idx)
        
        return probabilities
    
    def calculate_temperature(
        self, 
        entropy: float, 
        min_entropy: float, 
        max_entropy: float
    ) -> float:
        """
        Calculate computational temperature based on entropy.
        
        Higher entropy pixels get higher temperature (more resources).
        """
        if max_entropy <= min_entropy:
            return self.base_temperature
        
        # Normalize entropy to [0, 1]
        normalized_entropy = (entropy - min_entropy) / (max_entropy - min_entropy)
        
        # Exponential scaling for temperature
        temperature = self.base_temperature * np.exp(
            normalized_entropy * np.log(self.max_temperature / self.base_temperature)
        )
        
        return min(temperature, self.max_temperature)
    
    def calculate_internal_energy(
        self, 
        pixel_state: PixelState,
        local_context: np.ndarray
    ) -> float:
        """
        Calculate internal energy of a pixel based on local feature consistency.
        """
        # Energy based on local feature consistency
        pixel_value = np.argmax(pixel_state.class_probabilities)
        local_variance = np.var(local_context)
        
        # Higher variance = higher energy (less stable)
        consistency_energy = local_variance / (255 * 255)
        
        # Add entropy contribution
        entropy_energy = pixel_state.entropy / np.log2(self.num_classes)
        
        # Combine energies
        total_energy = 0.6 * consistency_energy + 0.4 * entropy_energy
        
        return total_energy
    
    def calculate_free_energy(self, energy: float, temperature: float, entropy: float) -> float:
        """Calculate Helmholtz free energy: F = E - T*S"""
        return energy - temperature * entropy
    
    def get_processing_state(self, temperature: float) -> ProcessingState:
        """Determine processing state based on temperature"""
        if temperature <= self.state_thresholds[ProcessingState.COLD]:
            return ProcessingState.COLD
        elif temperature <= self.state_thresholds[ProcessingState.WARM]:
            return ProcessingState.WARM
        elif temperature <= self.state_thresholds[ProcessingState.HOT]:
            return ProcessingState.HOT
        else:
            return ProcessingState.CRITICAL
    
    def allocate_resources(self, processing_state: ProcessingState) -> int:
        """Allocate computational resources based on processing state"""
        return self.state_resources[processing_state]
    
    def process_image_thermodynamically(
        self, 
        image: np.ndarray,
        return_metrics: bool = True
    ) -> Tuple[np.ndarray, Optional[ThermodynamicMetrics]]:
        """
        Process entire image using thermodynamic principles.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            return_metrics: Whether to return thermodynamic metrics
            
        Returns:
            Processed image and optional metrics
        """
        logger.info("Starting thermodynamic image processing")
        
        if len(image.shape) == 2:
            height, width = image.shape
            channels = 1
        else:
            height, width, channels = image.shape
        
        # Initialize pixel states
        pixel_states = np.empty((height, width), dtype=object)
        
        # First pass: Calculate entropy for all pixels
        entropies = np.zeros((height, width))
        
        for i in range(height):
            for j in range(width):
                # Get neighborhood
                neighborhood = self._get_neighborhood(image, i, j)
                
                if channels == 1:
                    pixel_value = image[i, j]
                else:
                    pixel_value = image[i, j, :]
                
                entropy = self.calculate_pixel_entropy(pixel_value, neighborhood)
                entropies[i, j] = entropy
        
        min_entropy = np.min(entropies)
        max_entropy = np.max(entropies)
        
        # Second pass: Calculate temperatures and initialize states
        total_resources = 0
        state_counts = {state: 0 for state in ProcessingState}
        
        for i in range(height):
            for j in range(width):
                entropy = entropies[i, j]
                temperature = self.calculate_temperature(entropy, min_entropy, max_entropy)
                
                # Get local context for energy calculation
                local_context = self._get_neighborhood(image, i, j)
                
                # Initialize pixel state
                if channels == 1:
                    probabilities = self._get_class_probabilities_grayscale(
                        image[i, j], local_context
                    )
                else:
                    probabilities = self._get_class_probabilities_rgb(
                        image[i, j, :], local_context
                    )
                
                pixel_state = PixelState(
                    entropy=entropy,
                    temperature=temperature,
                    energy=0.0,  # Will be calculated
                    free_energy=0.0,  # Will be calculated
                    processing_resources=0,  # Will be allocated
                    equilibrium_reached=False,
                    class_probabilities=probabilities,
                    neighbors_influence=0.0
                )
                
                # Calculate energy
                pixel_state.energy = self.calculate_internal_energy(pixel_state, local_context)
                pixel_state.free_energy = self.calculate_free_energy(
                    pixel_state.energy, pixel_state.temperature, pixel_state.entropy
                )
                
                # Determine processing state and allocate resources
                processing_state = self.get_processing_state(temperature)
                pixel_state.processing_resources = self.allocate_resources(processing_state)
                
                pixel_states[i, j] = pixel_state
                total_resources += pixel_state.processing_resources
                state_counts[processing_state] += 1
        
        logger.info(f"Resource allocation: {dict(state_counts)}")
        logger.info(f"Total resources allocated: {total_resources}")
        
        # Third pass: Iterative equilibrium seeking
        processed_image = image.copy()
        equilibrium_iterations = 0
        
        for iteration in range(self.max_iterations):
            energy_changes = []
            equilibrium_count = 0
            
            for i in range(height):
                for j in range(width):
                    pixel_state = pixel_states[i, j]
                    
                    if pixel_state.equilibrium_reached:
                        equilibrium_count += 1
                        continue
                    
                    # Process pixel with allocated resources
                    new_state = self._process_pixel_with_resources(
                        pixel_state, processed_image, i, j
                    )
                    
                    # Check for equilibrium
                    energy_change = abs(new_state.free_energy - pixel_state.free_energy)
                    energy_changes.append(energy_change)
                    
                    if energy_change < self.equilibrium_threshold:
                        new_state.equilibrium_reached = True
                        equilibrium_count += 1
                    
                    pixel_states[i, j] = new_state
                    
                    # Update processed image
                    if channels == 1:
                        processed_image[i, j] = np.argmax(new_state.class_probabilities)
                    else:
                        # For RGB, update based on probability distribution
                        rgb_values = self._probabilities_to_rgb(new_state.class_probabilities)
                        processed_image[i, j, :] = rgb_values
            
            equilibrium_iterations = iteration + 1
            equilibrium_percentage = equilibrium_count / (height * width) * 100
            
            logger.debug(f"Iteration {iteration}: {equilibrium_percentage:.1f}% at equilibrium")
            
            # Check global equilibrium
            if equilibrium_count == height * width:
                logger.info(f"Global equilibrium reached after {iteration + 1} iterations")
                break
            
            if len(energy_changes) > 0 and np.mean(energy_changes) < self.equilibrium_threshold:
                logger.info(f"System converged after {iteration + 1} iterations")
                break
        
        # Calculate metrics if requested
        metrics = None
        if return_metrics:
            metrics = self._calculate_thermodynamic_metrics(
                pixel_states, state_counts, total_resources, equilibrium_iterations
            )
        
        logger.info("Thermodynamic processing completed")
        return processed_image, metrics
    
    def _get_neighborhood(self, image: np.ndarray, i: int, j: int) -> np.ndarray:
        """Get neighborhood around pixel (i, j)"""
        half_size = self.neighborhood_size // 2
        height, width = image.shape[:2]
        
        i_start = max(0, i - half_size)
        i_end = min(height, i + half_size + 1)
        j_start = max(0, j - half_size)
        j_end = min(width, j + half_size + 1)
        
        if len(image.shape) == 2:
            neighborhood = image[i_start:i_end, j_start:j_end]
        else:
            neighborhood = image[i_start:i_end, j_start:j_end, :]
        
        return neighborhood
    
    def _process_pixel_with_resources(
        self,
        pixel_state: PixelState,
        image: np.ndarray,
        i: int,
        j: int
    ) -> PixelState:
        """Process pixel using allocated computational resources"""
        # More resources = more sophisticated processing
        resources = pixel_state.processing_resources
        
        # Get current neighborhood
        neighborhood = self._get_neighborhood(image, i, j)
        
        # Update class probabilities based on resources
        if resources >= 16:  # HOT/CRITICAL processing
            # Sophisticated neighborhood analysis
            updated_probabilities = self._sophisticated_probability_update(
                pixel_state.class_probabilities, neighborhood, pixel_state.temperature
            )
        elif resources >= 4:  # WARM processing
            # Standard neighborhood analysis
            updated_probabilities = self._standard_probability_update(
                pixel_state.class_probabilities, neighborhood
            )
        else:  # COLD processing
            # Minimal processing
            updated_probabilities = pixel_state.class_probabilities
        
        # Create new state
        new_state = PixelState(
            entropy=pixel_state.entropy,  # Entropy doesn't change during processing
            temperature=pixel_state.temperature,
            energy=0.0,  # Will be recalculated
            free_energy=0.0,  # Will be recalculated
            processing_resources=pixel_state.processing_resources,
            equilibrium_reached=pixel_state.equilibrium_reached,
            class_probabilities=updated_probabilities,
            neighbors_influence=np.mean(neighborhood)
        )
        
        # Recalculate energy and free energy
        new_state.energy = self.calculate_internal_energy(new_state, neighborhood)
        new_state.free_energy = self.calculate_free_energy(
            new_state.energy, new_state.temperature, new_state.entropy
        )
        
        return new_state
    
    def _sophisticated_probability_update(
        self,
        current_probs: np.ndarray,
        neighborhood: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """Sophisticated probability update using high computational resources"""
        # Analyze neighborhood patterns
        neighborhood_flat = neighborhood.flatten()
        neighborhood_mean = np.mean(neighborhood_flat)
        neighborhood_std = np.std(neighborhood_flat)
        
        # Update probabilities based on neighborhood statistics
        classes = np.arange(len(current_probs))
        
        # Gaussian influence from neighborhood
        gaussian_influence = np.exp(-0.5 * ((classes - neighborhood_mean) / (neighborhood_std + 1e-6))**2)
        gaussian_influence = gaussian_influence / np.sum(gaussian_influence)
        
        # Combine current probabilities with neighborhood influence
        # Higher temperature = more influence from neighborhood
        influence_weight = min(0.8, temperature / self.max_temperature)
        updated_probs = (1 - influence_weight) * current_probs + influence_weight * gaussian_influence
        
        # Renormalize
        updated_probs = updated_probs / np.sum(updated_probs)
        
        return updated_probs
    
    def _standard_probability_update(
        self,
        current_probs: np.ndarray,
        neighborhood: np.ndarray
    ) -> np.ndarray:
        """Standard probability update using moderate computational resources"""
        neighborhood_mean = np.mean(neighborhood)
        
        # Simple influence based on neighborhood mean
        classes = np.arange(len(current_probs))
        distances = np.abs(classes - neighborhood_mean)
        
        neighborhood_influence = np.exp(-distances / self.base_temperature)
        neighborhood_influence = neighborhood_influence / np.sum(neighborhood_influence)
        
        # Mix with current probabilities
        updated_probs = 0.7 * current_probs + 0.3 * neighborhood_influence
        updated_probs = updated_probs / np.sum(updated_probs)
        
        return updated_probs
    
    def _probabilities_to_rgb(self, probabilities: np.ndarray) -> np.ndarray:
        """Convert class probabilities back to RGB values"""
        # Simple mapping: use expected value
        expected_value = np.sum(probabilities * np.arange(len(probabilities)))
        
        # Map to RGB (simplified - in practice would use proper color space mapping)
        r = int(expected_value) % 256
        g = int(expected_value * 1.1) % 256
        b = int(expected_value * 1.2) % 256
        
        return np.array([r, g, b])
    
    def _calculate_thermodynamic_metrics(
        self,
        pixel_states: np.ndarray,
        state_counts: Dict[ProcessingState, int],
        total_resources: int,
        equilibrium_iterations: int
    ) -> ThermodynamicMetrics:
        """Calculate global thermodynamic metrics"""
        height, width = pixel_states.shape
        total_pixels = height * width
        
        # Calculate global metrics
        total_entropy = 0.0
        total_temperature = 0.0
        total_free_energy = 0.0
        equilibrium_count = 0
        
        for i in range(height):
            for j in range(width):
                state = pixel_states[i, j]
                total_entropy += state.entropy
                total_temperature += state.temperature
                total_free_energy += state.free_energy
                if state.equilibrium_reached:
                    equilibrium_count += 1
        
        # Processing efficiency based on resource allocation
        max_possible_resources = total_pixels * self.state_resources[ProcessingState.CRITICAL]
        processing_efficiency = 1.0 - (total_resources / max_possible_resources)
        
        return ThermodynamicMetrics(
            total_entropy=total_entropy,
            average_temperature=total_temperature / total_pixels,
            total_free_energy=total_free_energy,
            equilibrium_percentage=equilibrium_count / total_pixels * 100,
            processing_efficiency=processing_efficiency,
            resource_allocation={state.value: count for state, count in state_counts.items()}
        )
    
    def get_efficiency_report(self, metrics: ThermodynamicMetrics) -> str:
        """Generate human-readable efficiency report"""
        report = f"""
Thermodynamic Processing Report:
================================
Total Entropy: {metrics.total_entropy:.2f}
Average Temperature: {metrics.average_temperature:.2f}
Total Free Energy: {metrics.total_free_energy:.2f}
Equilibrium Achieved: {metrics.equilibrium_percentage:.1f}%
Processing Efficiency: {metrics.processing_efficiency:.1%}

Resource Allocation:
{'-' * 20}
"""
        for state, count in metrics.resource_allocation.items():
            percentage = count / sum(metrics.resource_allocation.values()) * 100
            report += f"{state.capitalize()}: {count} pixels ({percentage:.1f}%)\n"
        
        return report 