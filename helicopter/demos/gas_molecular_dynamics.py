#!/usr/bin/env python3
"""
Gas Molecular Dynamics for Information Processing
================================================

Implements gas molecular information processing where data elements behave as
thermodynamic gas molecules seeking equilibrium states.
Based on the mathematical framework from st-stellas-neural-networks.tex.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import cv2

class InformationGasMolecule:
    """
    Represents an information gas molecule with thermodynamic properties.
    
    Properties:
    - E_i: semantic energy
    - S_i: information entropy  
    - T_i: processing temperature
    - P_i: semantic pressure
    - V_i: conceptual volume
    - p_i: semantic position
    - v_i: information velocity
    - sigma_i: meaning cross-section
    """
    
    def __init__(self, position, semantic_energy, info_entropy, initial_velocity=None):
        self.position = np.array(position, dtype=float)  # p_i
        self.velocity = np.array(initial_velocity if initial_velocity is not None 
                                else np.random.normal(0, 0.1, len(position)), dtype=float)  # v_i
        self.semantic_energy = semantic_energy  # E_i
        self.info_entropy = info_entropy  # S_i
        self.temperature = 1.0  # T_i (processing temperature)
        self.pressure = 1.0  # P_i (semantic pressure)
        self.volume = 1.0  # V_i (conceptual volume)
        self.cross_section = 0.1  # sigma_i (meaning cross-section)
        self.mass = 1.0  # molecular mass
        
    def update_thermodynamic_properties(self):
        """Update thermodynamic properties based on current state."""
        # Temperature proportional to kinetic energy
        kinetic_energy = 0.5 * self.mass * np.sum(self.velocity**2)
        self.temperature = max(0.1, kinetic_energy)
        
        # Pressure relates to velocity and temperature
        self.pressure = self.temperature * self.info_entropy
        
        # Volume inversely related to semantic density
        self.volume = 1.0 / max(0.1, self.semantic_energy)

class GasMolecularSystem:
    """
    System of information gas molecules with thermodynamic interactions.
    Implements equilibrium-seeking dynamics for understanding emergence.
    """
    
    def __init__(self, molecules, system_size=(10.0, 10.0)):
        self.molecules = molecules
        self.system_size = system_size
        self.time = 0.0
        self.dt = 0.001  # Smaller time step for numerical stability
        self.k_boltzmann = 1.0  # Boltzmann constant (normalized)
        self.equilibrium_threshold = 1e-4
        self.force_cutoff = 2.0  # Maximum interaction distance
        self.max_force = 50.0  # Maximum allowed force magnitude
        self.min_distance = 0.1  # Minimum allowed distance between molecules
        
    def calculate_semantic_forces(self):
        """
        Calculate semantic forces between molecules.
        F_ij = semantic interaction force between molecules i and j
        """
        forces = [np.zeros_like(mol.position) for mol in self.molecules]
        
        for i, mol_i in enumerate(self.molecules):
            for j, mol_j in enumerate(self.molecules):
                if i != j:
                    # Distance vector
                    r_ij = mol_j.position - mol_i.position
                    distance = np.linalg.norm(r_ij)
                    
                    if distance < self.force_cutoff and distance > self.min_distance:
                        # Unit vector
                        r_hat = r_ij / distance
                        
                        # Semantic attraction/repulsion based on information similarity
                        semantic_similarity = self.calculate_semantic_similarity(mol_i, mol_j)
                        
                        # Lennard-Jones-like potential with semantic modulation
                        epsilon = 0.1 * semantic_similarity  # Reduced well depth for stability
                        sigma = 0.3  # Characteristic distance
                        
                        # Force magnitude (derivative of LJ potential) with stability check
                        r_ratio = sigma / distance
                        if r_ratio < 3.0:  # Prevent extreme forces
                            force_magnitude = 24 * epsilon * (2 * r_ratio**12 - r_ratio**6) / distance
                            
                            # Cap the force magnitude for numerical stability
                            force_magnitude = np.clip(force_magnitude, -self.max_force, self.max_force)
                            
                            # Apply force
                            force = force_magnitude * r_hat
                            forces[i] += force
        
        return forces
    
    def calculate_semantic_similarity(self, mol_i, mol_j):
        """Calculate semantic similarity between two molecules."""
        energy_similarity = np.exp(-abs(mol_i.semantic_energy - mol_j.semantic_energy))
        entropy_similarity = np.exp(-abs(mol_i.info_entropy - mol_j.info_entropy))
        return (energy_similarity + entropy_similarity) / 2
    
    def update_system(self):
        """Update system state using molecular dynamics."""
        # Calculate forces
        forces = self.calculate_semantic_forces()
        
        # Update positions and velocities (Velocity Verlet integration)
        for i, mol in enumerate(self.molecules):
            # Update velocity (half step)
            acceleration = forces[i] / mol.mass
            mol.velocity += 0.5 * self.dt * acceleration
            
            # Apply velocity damping for numerical stability
            damping_factor = 0.99
            mol.velocity *= damping_factor
            
            # Cap velocity magnitude
            max_velocity = 10.0
            velocity_magnitude = np.linalg.norm(mol.velocity)
            if velocity_magnitude > max_velocity:
                mol.velocity = mol.velocity * (max_velocity / velocity_magnitude)
            
            # Update position
            mol.position += self.dt * mol.velocity
            
            # Apply boundary conditions (reflective walls)
            for dim in range(len(mol.position)):
                if mol.position[dim] < 0:
                    mol.position[dim] = 0
                    mol.velocity[dim] *= -0.8  # Inelastic collision
                elif mol.position[dim] > self.system_size[dim]:
                    mol.position[dim] = self.system_size[dim]
                    mol.velocity[dim] *= -0.8
            
            # Update velocity (remaining half step)
            mol.velocity += 0.5 * self.dt * acceleration
            
            # Update thermodynamic properties
            mol.update_thermodynamic_properties()
        
        self.time += self.dt
    
    def calculate_system_variance(self):
        """
        Calculate system variance from equilibrium state.
        Variance minimization drives understanding emergence.
        """
        positions = np.array([mol.position for mol in self.molecules])
        velocities = np.array([mol.velocity for mol in self.molecules])
        
        # Position variance (spatial distribution)
        position_variance = np.var(positions, axis=0).mean()
        
        # Velocity variance (kinetic energy distribution)  
        velocity_variance = np.var(velocities, axis=0).mean()
        
        # Temperature variance (thermodynamic equilibrium)
        temperatures = [mol.temperature for mol in self.molecules]
        temperature_variance = np.var(temperatures)
        
        # Total system variance
        total_variance = position_variance + velocity_variance + temperature_variance
        return total_variance
    
    def calculate_system_energy(self):
        """Calculate total system energy."""
        kinetic_energy = sum(0.5 * mol.mass * np.sum(mol.velocity**2) for mol in self.molecules)
        
        # Potential energy (pairwise interactions)
        potential_energy = 0.0
        for i, mol_i in enumerate(self.molecules):
            for j, mol_j in enumerate(self.molecules[i+1:], i+1):
                r = np.linalg.norm(mol_j.position - mol_i.position)
                if r > 1e-6 and r < self.force_cutoff:
                    similarity = self.calculate_semantic_similarity(mol_i, mol_j)
                    epsilon = 1.0 * similarity
                    sigma = 0.5
                    potential_energy += 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
        
        return kinetic_energy + potential_energy
    
    def seek_equilibrium(self, max_steps=1000, variance_threshold=1e-4):
        """
        Run simulation until equilibrium is achieved (variance minimized).
        
        Returns:
            dict: Equilibrium analysis results
        """
        variance_history = []
        energy_history = []
        
        for step in range(max_steps):
            # Update system
            self.update_system()
            
            # Calculate metrics
            variance = self.calculate_system_variance()
            energy = self.calculate_system_energy()
            
            # Check for numerical overflow and early stopping
            if not np.isfinite(variance) or not np.isfinite(energy):
                print(f"Numerical overflow detected at step {step}. Stopping simulation.")
                break
                
            if variance > 1e6 or abs(energy) > 1e6:
                print(f"Values too large at step {step}. Stopping simulation.")
                break
            
            variance_history.append(variance)
            energy_history.append(energy)
            
            # Check for equilibrium
            if variance < variance_threshold:
                print(f"Equilibrium reached at step {step}, variance: {variance:.6f}")
                break
                
            # Progress reporting
            if step % 1000 == 0 and step > 0:
                print(f"  Step {step}: variance = {variance:.6f}, energy = {energy:.6f}")
        
        # Calculate final equilibrium state
        final_positions = np.array([mol.position for mol in self.molecules])
        final_velocities = np.array([mol.velocity for mol in self.molecules])
        final_temperatures = [mol.temperature for mol in self.molecules]
        
        return {
            'equilibrium_reached': variance < variance_threshold,
            'final_variance': variance,
            'final_energy': energy,
            'steps_to_equilibrium': step + 1,
            'variance_history': variance_history,
            'energy_history': energy_history,
            'final_positions': final_positions,
            'final_velocities': final_velocities,
            'final_temperatures': final_temperatures
        }

def create_molecules_from_image(image, n_molecules=50):
    """
    Create information gas molecules from image data.
    
    Args:
        image: Input image as numpy array
        n_molecules: Number of molecules to create
        
    Returns:
        list: Information gas molecules
    """
    molecules = []
    
    # Convert image to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Normalize image
    gray_normalized = gray.astype(float) / 255.0
    
    # Sample points from image
    h, w = gray.shape
    
    for i in range(n_molecules):
        # Random position in image space
        x = np.random.uniform(0, 10)  # Scaled to system coordinates
        y = np.random.uniform(0, 10)
        
        # Sample corresponding image pixel
        img_x = int((x / 10.0) * w)
        img_y = int((y / 10.0) * h)
        img_x = max(0, min(w-1, img_x))
        img_y = max(0, min(h-1, img_y))
        
        # Calculate semantic properties from image
        pixel_intensity = gray_normalized[img_y, img_x]
        
        # Semantic energy based on local variance (texture)
        local_region = gray_normalized[max(0, img_y-2):min(h, img_y+3), 
                                      max(0, img_x-2):min(w, img_x+3)]
        semantic_energy = np.var(local_region) + 0.1  # Add baseline
        
        # Information entropy based on local complexity
        info_entropy = pixel_intensity * semantic_energy + 0.1
        
        # Create molecule
        molecule = InformationGasMolecule(
            position=[x, y],
            semantic_energy=semantic_energy,
            info_entropy=info_entropy,
            initial_velocity=np.random.normal(0, 0.1, 2)
        )
        
        molecules.append(molecule)
    
    return molecules

def visualize_gas_molecular_dynamics(system, equilibrium_results, save_path=None):
    """
    Visualize gas molecular dynamics and equilibrium seeking.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Final molecular positions
    positions = equilibrium_results['final_positions']
    temperatures = equilibrium_results['final_temperatures']
    
    scatter = axes[0, 0].scatter(positions[:, 0], positions[:, 1], 
                               c=temperatures, s=50, cmap='plasma', alpha=0.7)
    axes[0, 0].set_xlim(0, 10)
    axes[0, 0].set_ylim(0, 10)
    axes[0, 0].set_title('Final Molecular Positions (colored by temperature)')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    plt.colorbar(scatter, ax=axes[0, 0], label='Temperature')
    
    # 2. Variance history (equilibrium seeking) - with overflow protection
    variance_history = np.array(equilibrium_results['variance_history'])
    # Cap extreme values to prevent plotting issues
    variance_history = np.clip(variance_history, 1e-10, 1e10)
    
    axes[0, 1].plot(variance_history, 'b-', linewidth=2)
    axes[0, 1].set_title('System Variance vs Time (Equilibrium Seeking)')
    axes[0, 1].set_xlabel('Simulation Steps')
    axes[0, 1].set_ylabel('System Variance (capped)')
    try:
        axes[0, 1].set_yscale('log')
    except Exception:
        pass  # Skip if log scale causes issues
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Energy history - with overflow protection
    energy_history = np.array(equilibrium_results['energy_history'])
    # Cap extreme values to prevent plotting issues
    energy_history = np.clip(energy_history, -1e10, 1e10)
    
    axes[0, 2].plot(energy_history, 'r-', linewidth=2)
    axes[0, 2].set_title('System Energy vs Time')
    axes[0, 2].set_xlabel('Simulation Steps')
    axes[0, 2].set_ylabel('Total Energy (capped)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Temperature distribution
    axes[1, 0].hist(temperatures, bins=20, alpha=0.7, color='orange')
    axes[1, 0].set_title('Temperature Distribution at Equilibrium')
    axes[1, 0].set_xlabel('Temperature')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].axvline(np.mean(temperatures), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(temperatures):.3f}')
    axes[1, 0].legend()
    
    # 5. Velocity magnitude distribution
    velocities = equilibrium_results['final_velocities']
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
    axes[1, 1].hist(velocity_magnitudes, bins=20, alpha=0.7, color='green')
    axes[1, 1].set_title('Velocity Magnitude Distribution')
    axes[1, 1].set_xlabel('Velocity Magnitude')
    axes[1, 1].set_ylabel('Count')
    
    # 6. Equilibrium metrics - with overflow protection
    final_variance = equilibrium_results['final_variance']
    final_energy = equilibrium_results['final_energy']
    steps = equilibrium_results['steps_to_equilibrium']
    
    # Handle extreme values that cause matplotlib overflow
    if not np.isfinite(final_variance) or final_variance > 1e8:
        final_variance = 1e8
    if not np.isfinite(final_energy) or abs(final_energy) > 1e8:
        final_energy = np.sign(final_energy) * 1e8 if final_energy != 0 else 0
        
    metrics = {
        'Final Variance': final_variance,
        'Final Energy': final_energy, 
        'Steps to Equilibrium': steps
    }
    
    labels = []
    values = []
    for key, value in metrics.items():
        labels.append(key)
        # Ensure all values are finite and reasonable for plotting
        safe_value = float(value) if np.isfinite(float(value)) else 0
        values.append(safe_value)
    
    bars = axes[1, 2].barh(range(len(values)), values)
    axes[1, 2].set_yticks(range(len(values)))
    axes[1, 2].set_yticklabels(labels)
    axes[1, 2].set_title('Equilibrium Metrics (overflow protected)')
    axes[1, 2].set_xlabel('Value')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        label_text = f'{value:.2e}' if abs(value) > 1000 else f'{value:.2f}'
        axes[1, 2].text(bar.get_width() * 0.5, bar.get_y() + bar.get_height()/2, 
                       label_text, va='center', ha='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def demonstrate_gas_molecular_processing():
    """
    Demonstrate gas molecular information processing on image data.
    """
    print("Gas Molecular Dynamics Information Processing")
    print("=" * 50)
    
    # Create test image (similar to S-entropy demo)
    print("Creating test image...")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), 2)
    cv2.circle(img, (50, 50), 20, (128, 128, 128), -1)
    cv2.line(img, (10, 50), (90, 50), (200, 200, 200), 1)
    
    # Create molecules from image
    print("Creating information gas molecules from image...")
    molecules = create_molecules_from_image(img, n_molecules=30)
    
    print(f"Created {len(molecules)} information gas molecules")
    
    # Initialize gas molecular system
    system = GasMolecularSystem(molecules)
    
    print("Initial system variance:", system.calculate_system_variance())
    print("Initial system energy:", system.calculate_system_energy())
    
    # Run equilibrium seeking (more steps due to smaller time step)
    print("\nRunning gas molecular dynamics to seek equilibrium...")
    equilibrium_results = system.seek_equilibrium(max_steps=10000, variance_threshold=1e-2)
    
    # Print results
    print("\nEquilibrium Results:")
    print(f"  Equilibrium reached: {equilibrium_results['equilibrium_reached']}")
    print(f"  Final variance: {equilibrium_results['final_variance']:.6f}")
    print(f"  Final energy: {equilibrium_results['final_energy']:.6f}")
    print(f"  Steps to equilibrium: {equilibrium_results['steps_to_equilibrium']}")
    
    # Visualize results
    print("\nVisualizing gas molecular dynamics...")
    visualize_gas_molecular_dynamics(
        system, equilibrium_results,
        save_path="gas_molecular_dynamics_demo.png"
    )
    
    print("\nGas molecular dynamics demonstration completed!")
    
    return system, equilibrium_results

if __name__ == "__main__":
    demonstrate_gas_molecular_processing()
