#!/usr/bin/env python3
"""
Constrained Stochastic Sampling (Pogo Stick Jumps)
==================================================

Implements constrained random walks with semantic gravity for sampling
in compressed coordinate spaces. Based on st-stellas-moon-landing.tex.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from scipy.stats import multivariate_normal, truncnorm
from scipy.spatial.distance import cdist
import cv2

class SemanticGravityField:
    """
    Implements semantic gravity field g_s(r) = -∇U_s(r) 
    where U_s(r) is semantic potential energy.
    """
    
    def __init__(self, system_size=(10.0, 10.0)):
        self.system_size = system_size
        self.gravity_zones = []
        
    def add_gravity_zone(self, center, strength, zone_type='attractive'):
        """
        Add semantic gravity zone.
        
        Args:
            center: Zone center coordinates
            strength: Gravity strength (higher = stronger)
            zone_type: 'attractive' or 'repulsive'
        """
        self.gravity_zones.append({
            'center': np.array(center),
            'strength': strength,
            'type': zone_type
        })
    
    def calculate_potential_energy(self, position):
        """
        Calculate semantic potential energy U_s(r) at position.
        
        Args:
            position: Coordinate position
            
        Returns:
            float: Potential energy
        """
        position = np.array(position)
        total_potential = 0.0
        
        for zone in self.gravity_zones:
            # Distance to zone center
            r = np.linalg.norm(position - zone['center'])
            
            # Avoid singularity at center
            r = max(r, 0.1)
            
            # Potential energy (proportional to 1/r for gravity-like field)
            if zone['type'] == 'attractive':
                potential = -zone['strength'] / r
            else:  # repulsive
                potential = zone['strength'] / r
            
            total_potential += potential
        
        return total_potential
    
    def calculate_gravity_force(self, position):
        """
        Calculate semantic gravity force g_s(r) = -∇U_s(r).
        
        Args:
            position: Coordinate position
            
        Returns:
            numpy.ndarray: Gravity force vector
        """
        position = np.array(position)
        total_force = np.zeros_like(position)
        
        for zone in self.gravity_zones:
            # Vector from position to zone center
            r_vec = zone['center'] - position
            r_magnitude = np.linalg.norm(r_vec)
            
            # Avoid singularity
            if r_magnitude < 0.1:
                continue
            
            # Unit vector
            r_hat = r_vec / r_magnitude
            
            # Force magnitude (derivative of potential)
            if zone['type'] == 'attractive':
                force_magnitude = zone['strength'] / (r_magnitude ** 2)
                force_direction = r_hat  # Toward center
            else:  # repulsive
                force_magnitude = zone['strength'] / (r_magnitude ** 2)
                force_direction = -r_hat  # Away from center
            
            total_force += force_magnitude * force_direction
        
        return total_force
    
    def get_gravity_magnitude(self, position):
        """Get magnitude of gravity field at position."""
        force = self.calculate_gravity_force(position)
        return np.linalg.norm(force)

class FuzzyWindow:
    """
    Implements fuzzy window aperture function ψ_j(x) = exp(-(x-c_j)²/2σ_j²).
    """
    
    def __init__(self, center, sigma, dimension_name):
        self.center = center
        self.sigma = sigma
        self.dimension_name = dimension_name
    
    def aperture_function(self, x):
        """Calculate fuzzy window aperture at position x."""
        return np.exp(-0.5 * ((x - self.center) / self.sigma) ** 2)
    
    def sample_weight(self, position, dimension_index):
        """Calculate sample weight for position in this dimension."""
        return self.aperture_function(position[dimension_index])

class ConstrainedStochasticSampler:
    """
    Implements constrained random walks with semantic gravity constraints.
    Performs "pogo stick jumps" with step size limited by local gravity.
    """
    
    def __init__(self, semantic_gravity_field, fuzzy_windows, processing_velocity=1.0):
        self.gravity_field = semantic_gravity_field
        self.fuzzy_windows = fuzzy_windows
        self.processing_velocity = processing_velocity
        self.samples = []
        self.trajectory = []
        
    def calculate_max_step_size(self, position):
        """
        Calculate maximum step size: Δr_max = v_0 / |g_s(r)|.
        
        Args:
            position: Current position
            
        Returns:
            float: Maximum allowed step size
        """
        gravity_magnitude = self.gravity_field.get_gravity_magnitude(position)
        
        # Avoid division by zero
        gravity_magnitude = max(gravity_magnitude, 0.01)
        
        max_step = self.processing_velocity / gravity_magnitude
        
        # Reasonable bounds
        max_step = max(0.1, min(2.0, max_step))
        
        return max_step
    
    def sample_next_position(self, current_position, max_step_size):
        """
        Sample next position from truncated multivariate normal distribution.
        
        Args:
            current_position: Current position
            max_step_size: Maximum step size constraint
            
        Returns:
            numpy.ndarray: Next position
        """
        current_position = np.array(current_position)
        
        # Covariance matrix (isotropic)
        sigma = max_step_size / 3.0  # 3-sigma rule
        covariance = sigma ** 2 * np.eye(len(current_position))
        
        # Sample from multivariate normal
        proposed_position = multivariate_normal.rvs(
            mean=current_position, 
            cov=covariance
        )
        
        # Apply system boundaries
        for i in range(len(proposed_position)):
            if i < len(self.gravity_field.system_size):
                proposed_position[i] = max(0, min(self.gravity_field.system_size[i], 
                                                proposed_position[i]))
        
        return proposed_position
    
    def calculate_sample_weight(self, position):
        """
        Calculate total sample weight w(r) = ψ_t(r_t) * ψ_i(r_i) * ψ_e(r_e).
        
        Args:
            position: Sample position
            
        Returns:
            float: Combined sample weight
        """
        total_weight = 1.0
        
        for window in self.fuzzy_windows:
            # Assume position dimensions correspond to window dimensions
            if hasattr(window, 'dimension_index'):
                dim_idx = window.dimension_index
            else:
                # Default mapping
                if window.dimension_name == 'temporal':
                    dim_idx = 0
                elif window.dimension_name == 'informational':
                    dim_idx = 1 if len(position) > 1 else 0
                elif window.dimension_name == 'entropic':
                    dim_idx = 2 if len(position) > 2 else 0
                else:
                    dim_idx = 0
            
            if dim_idx < len(position):
                weight = window.sample_weight(position, dim_idx)
                total_weight *= weight
        
        return total_weight
    
    def perform_constrained_sampling(self, initial_position, n_samples=100):
        """
        Perform constrained stochastic sampling (pogo stick jumps).
        
        Args:
            initial_position: Starting position
            n_samples: Number of samples to collect
            
        Returns:
            dict: Sampling results
        """
        print(f"Starting constrained stochastic sampling from {initial_position}")
        print(f"Target samples: {n_samples}")
        
        self.samples = []
        self.trajectory = []
        
        current_position = np.array(initial_position)
        self.trajectory.append(current_position.copy())
        
        for sample_idx in range(n_samples):
            # Calculate maximum step size based on local gravity
            max_step = self.calculate_max_step_size(current_position)
            
            # Sample next position
            next_position = self.sample_next_position(current_position, max_step)
            
            # Calculate sample weight from fuzzy windows
            sample_weight = self.calculate_sample_weight(next_position)
            
            # Store sample
            sample_info = {
                'position': next_position.copy(),
                'weight': sample_weight,
                'step_size': np.linalg.norm(next_position - current_position),
                'max_step_allowed': max_step,
                'gravity_magnitude': self.gravity_field.get_gravity_magnitude(current_position),
                'sample_index': sample_idx
            }
            
            self.samples.append(sample_info)
            self.trajectory.append(next_position.copy())
            
            # Update current position
            current_position = next_position
            
            # Progress report
            if (sample_idx + 1) % (n_samples // 10) == 0:
                print(f"  Progress: {sample_idx + 1}/{n_samples} samples")
        
        # Calculate sampling statistics
        step_sizes = [s['step_size'] for s in self.samples]
        weights = [s['weight'] for s in self.samples]
        gravity_magnitudes = [s['gravity_magnitude'] for s in self.samples]
        
        results = {
            'samples': self.samples,
            'trajectory': self.trajectory,
            'n_samples': len(self.samples),
            'mean_step_size': np.mean(step_sizes),
            'std_step_size': np.std(step_sizes),
            'mean_weight': np.mean(weights),
            'std_weight': np.std(weights),
            'mean_gravity': np.mean(gravity_magnitudes),
            'effective_sample_size': np.sum(weights) ** 2 / np.sum(np.array(weights) ** 2)
        }
        
        print(f"Sampling completed!")
        print(f"  Mean step size: {results['mean_step_size']:.3f}")
        print(f"  Mean sample weight: {results['mean_weight']:.3f}")
        print(f"  Effective sample size: {results['effective_sample_size']:.1f}")
        
        return results

def visualize_constrained_sampling(sampler, sampling_results, save_path=None):
    """
    Visualize constrained stochastic sampling results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    trajectory = np.array(sampling_results['trajectory'])
    samples = sampling_results['samples']
    
    # 1. Sampling trajectory with gravity field
    x_range = np.linspace(0, sampler.gravity_field.system_size[0], 50)
    y_range = np.linspace(0, sampler.gravity_field.system_size[1], 50)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate gravity magnitude at each point
    gravity_magnitudes = np.zeros_like(X)
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            gravity_magnitudes[j, i] = sampler.gravity_field.get_gravity_magnitude([X[j, i], Y[j, i]])
    
    # Plot gravity field as contour
    contour = axes[0,0].contourf(X, Y, gravity_magnitudes, levels=20, cmap='Reds', alpha=0.6)
    plt.colorbar(contour, ax=axes[0,0], label='Gravity Magnitude')
    
    # Plot trajectory
    if len(trajectory) > 1:
        axes[0,0].plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        axes[0,0].scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start')
        axes[0,0].scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='x', label='End')
    
    # Plot gravity zones
    for zone in sampler.gravity_field.gravity_zones:
        circle = Circle(zone['center'], 0.5, fill=False, 
                       color='black' if zone['type'] == 'attractive' else 'purple',
                       linewidth=2, linestyle='--')
        axes[0,0].add_patch(circle)
    
    axes[0,0].set_xlim(0, sampler.gravity_field.system_size[0])
    axes[0,0].set_ylim(0, sampler.gravity_field.system_size[1])
    axes[0,0].set_title('Sampling Trajectory with Gravity Field')
    axes[0,0].set_xlabel('X Coordinate')
    axes[0,0].set_ylabel('Y Coordinate')
    axes[0,0].legend()
    
    # 2. Step size distribution
    step_sizes = [s['step_size'] for s in samples]
    axes[0,1].hist(step_sizes, bins=20, alpha=0.7, color='blue')
    axes[0,1].set_title('Step Size Distribution')
    axes[0,1].set_xlabel('Step Size')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].axvline(np.mean(step_sizes), color='red', linestyle='--',
                      label=f'Mean: {np.mean(step_sizes):.3f}')
    axes[0,1].legend()
    
    # 3. Sample weight distribution
    weights = [s['weight'] for s in samples]
    axes[0,2].hist(weights, bins=20, alpha=0.7, color='green')
    axes[0,2].set_title('Sample Weight Distribution')
    axes[0,2].set_xlabel('Sample Weight')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].axvline(np.mean(weights), color='red', linestyle='--',
                      label=f'Mean: {np.mean(weights):.3f}')
    axes[0,2].legend()
    
    # 4. Step size vs gravity magnitude
    gravity_mags = [s['gravity_magnitude'] for s in samples]
    axes[1,0].scatter(gravity_mags, step_sizes, alpha=0.6)
    axes[1,0].set_title('Step Size vs Local Gravity')
    axes[1,0].set_xlabel('Gravity Magnitude')
    axes[1,0].set_ylabel('Step Size')
    
    # Add theoretical relationship line
    if len(gravity_mags) > 0:
        g_range = np.linspace(min(gravity_mags), max(gravity_mags), 100)
        theoretical_steps = sampler.processing_velocity / g_range
        theoretical_steps = np.clip(theoretical_steps, 0.1, 2.0)  # Apply bounds
        axes[1,0].plot(g_range, theoretical_steps, 'r--', linewidth=2, 
                       label='Theoretical: v₀/|g|')
        axes[1,0].legend()
    
    # 5. Sampling efficiency over time
    cumulative_weights = np.cumsum(weights)
    sample_indices = range(1, len(weights) + 1)
    axes[1,1].plot(sample_indices, cumulative_weights, 'b-', linewidth=2)
    axes[1,1].set_title('Cumulative Sample Weight')
    axes[1,1].set_xlabel('Sample Index')
    axes[1,1].set_ylabel('Cumulative Weight')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Sampling statistics summary
    stats_text = f"""Sampling Statistics:
    
Total Samples: {sampling_results['n_samples']}
Effective Sample Size: {sampling_results['effective_sample_size']:.1f}

Step Size Statistics:
  Mean: {sampling_results['mean_step_size']:.3f}
  Std: {sampling_results['std_step_size']:.3f}
  
Sample Weight Statistics:
  Mean: {sampling_results['mean_weight']:.3f}
  Std: {sampling_results['std_weight']:.3f}
  
Gravity Field Statistics:
  Mean Gravity: {sampling_results['mean_gravity']:.3f}
  
Sampling Efficiency:
  {sampling_results['effective_sample_size']/sampling_results['n_samples']*100:.1f}%
"""
    
    axes[1,2].text(0.05, 0.95, stats_text, transform=axes[1,2].transAxes,
                   verticalalignment='top', fontsize=10, fontfamily='monospace')
    axes[1,2].set_xlim(0, 1)
    axes[1,2].set_ylim(0, 1)
    axes[1,2].axis('off')
    axes[1,2].set_title('Summary Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def demonstrate_constrained_sampling():
    """
    Demonstrate constrained stochastic sampling with semantic gravity.
    """
    print("Constrained Stochastic Sampling Demonstration")
    print("=" * 50)
    
    # Create semantic gravity field
    print("Setting up semantic gravity field...")
    gravity_field = SemanticGravityField(system_size=(10.0, 10.0))
    
    # Add gravity zones representing different semantic regions
    gravity_field.add_gravity_zone(center=[2, 2], strength=3.0, zone_type='attractive')  # High semantic density
    gravity_field.add_gravity_zone(center=[8, 8], strength=2.0, zone_type='attractive')  # Medium semantic density  
    gravity_field.add_gravity_zone(center=[5, 5], strength=1.5, zone_type='repulsive')   # Semantic barrier
    gravity_field.add_gravity_zone(center=[2, 8], strength=1.0, zone_type='attractive')  # Small semantic cluster
    
    print("Created gravity field with 4 semantic zones")
    
    # Create fuzzy windows
    print("Setting up fuzzy windows...")
    fuzzy_windows = [
        FuzzyWindow(center=2.5, sigma=1.5, dimension_name='temporal'),
        FuzzyWindow(center=3.0, sigma=2.0, dimension_name='informational'),
        FuzzyWindow(center=4.0, sigma=1.0, dimension_name='entropic')
    ]
    
    # Assign dimension indices
    for i, window in enumerate(fuzzy_windows):
        window.dimension_index = i % 2  # Map to 2D coordinate system
    
    print(f"Created {len(fuzzy_windows)} fuzzy windows")
    
    # Initialize sampler
    sampler = ConstrainedStochasticSampler(
        semantic_gravity_field=gravity_field,
        fuzzy_windows=fuzzy_windows,
        processing_velocity=2.0
    )
    
    # Perform sampling
    print("Performing constrained stochastic sampling...")
    initial_position = [1.0, 1.0]  # Start in bottom-left
    sampling_results = sampler.perform_constrained_sampling(
        initial_position=initial_position,
        n_samples=200
    )
    
    # Visualize results
    print("Visualizing sampling results...")
    visualize_constrained_sampling(
        sampler, sampling_results,
        save_path="constrained_sampling_demo.png"
    )
    
    print("Constrained sampling demonstration completed!")
    
    return sampler, sampling_results

if __name__ == "__main__":
    demonstrate_constrained_sampling()
