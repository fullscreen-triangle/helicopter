"""
Molecular Dynamics Implementation

Simulates gas molecular interactions and dynamics for Information Gas Molecules,
implementing the physical laws governing semantic interactions and equilibrium seeking.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .information_gas_molecule import InformationGasMolecule


@dataclass 
class SimulationState:
    """State of molecular dynamics simulation"""
    time_step: float
    total_time: float
    total_energy: float
    system_temperature: float
    pressure: float
    volume: float
    consciousness_level: float


class MolecularDynamics:
    """
    Molecular dynamics simulator for Information Gas Molecules.
    
    Implements Newtonian mechanics adapted for semantic space interactions,
    enabling gas molecules to naturally evolve toward equilibrium configurations.
    """
    
    def __init__(self,
                 time_step: float = 1e-12,  # femtosecond timesteps
                 temperature: float = 300.0,  # Semantic temperature
                 pressure: float = 1.0,      # Semantic pressure
                 volume: float = 1000.0,     # Semantic volume
                 thermostat_enabled: bool = True,
                 max_force: float = 1e6):
        """
        Initialize molecular dynamics simulator.
        
        Args:
            time_step: Simulation time step (femtoseconds)
            temperature: System temperature for thermostat
            pressure: System pressure
            volume: System volume
            thermostat_enabled: Whether to use temperature control
            max_force: Maximum allowed force magnitude
        """
        self.time_step = time_step
        self.target_temperature = temperature
        self.pressure = pressure
        self.volume = volume
        self.thermostat_enabled = thermostat_enabled
        self.max_force = max_force
        
        # Simulation state tracking
        self.current_state = SimulationState(
            time_step=0.0,
            total_time=0.0,
            total_energy=0.0,
            system_temperature=temperature,
            pressure=pressure,
            volume=volume,
            consciousness_level=0.0
        )
        
        # Performance optimization
        self.force_cutoff = 10.0  # Maximum interaction distance
        self.neighbor_list_update_frequency = 10
        self.step_count = 0
        
        # Trajectory recording
        self.trajectory_history: List[Dict[str, Any]] = []
        self.energy_history: List[float] = []
        self.consciousness_history: List[float] = []
        
    def simulate_gas_molecular_evolution(self,
                                       gas_molecules: List[InformationGasMolecule],
                                       num_steps: int,
                                       equilibrium_target: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Simulate evolution of gas molecular system toward equilibrium.
        
        Args:
            gas_molecules: List of information gas molecules
            num_steps: Number of simulation steps
            equilibrium_target: Target equilibrium configuration
            
        Returns:
            Simulation results including final state and trajectories
        """
        start_time = time.time_ns()
        
        # Initialize simulation
        self._initialize_simulation(gas_molecules)
        
        # Main simulation loop
        for step in range(num_steps):
            self.step_count = step
            
            # Calculate forces between all molecules
            forces = self._calculate_all_forces(gas_molecules)
            
            # Update molecular positions and velocities
            self._update_molecular_states(gas_molecules, forces)
            
            # Apply thermostat if enabled
            if self.thermostat_enabled:
                self._apply_thermostat(gas_molecules)
                
            # Update system properties
            self._update_system_properties(gas_molecules)
            
            # Record trajectory
            if step % 10 == 0:  # Record every 10 steps
                self._record_trajectory_point(gas_molecules, step)
                
            # Check for equilibrium convergence
            if equilibrium_target is not None and self._check_equilibrium_convergence(gas_molecules, equilibrium_target):
                break
                
        end_time = time.time_ns()
        simulation_time_ns = end_time - start_time
        
        # Compile simulation results
        results = {
            'final_state': self.current_state,
            'gas_molecules': gas_molecules,
            'trajectory_history': self.trajectory_history,
            'energy_history': self.energy_history,
            'consciousness_history': self.consciousness_history,
            'simulation_time_ns': simulation_time_ns,
            'steps_completed': step + 1,
            'equilibrium_achieved': equilibrium_target is not None and self._check_equilibrium_convergence(gas_molecules, equilibrium_target)
        }
        
        return results
        
    def _initialize_simulation(self, gas_molecules: List[InformationGasMolecule]) -> None:
        """Initialize simulation state and molecular properties."""
        # Set initial random velocities according to Maxwell-Boltzmann distribution
        boltzmann_constant = 1.380649e-23
        
        for molecule in gas_molecules:
            # Temperature-dependent velocity distribution
            velocity_scale = np.sqrt(boltzmann_constant * self.target_temperature / molecule.effective_mass)
            
            # Random velocity from Maxwell-Boltzmann distribution  
            velocity = np.random.normal(0, velocity_scale, size=3)
            molecule.kinetic_state.info_velocity = velocity
            
        # Update initial system properties
        self._update_system_properties(gas_molecules)
        
    def _calculate_all_forces(self, gas_molecules: List[InformationGasMolecule]) -> List[np.ndarray]:
        """
        Calculate forces between all pairs of gas molecules.
        
        Uses efficient neighbor listing to reduce O(N²) complexity.
        """
        forces = [np.zeros(3) for _ in gas_molecules]
        
        # Reset force accumulators
        for molecule in gas_molecules:
            molecule.kinetic_state.force_accumulator = np.zeros(3)
        
        # Calculate pairwise forces
        for i, mol1 in enumerate(gas_molecules):
            for j, mol2 in enumerate(gas_molecules[i+1:], i+1):
                # Calculate distance
                distance_vector = mol2.kinetic_state.semantic_position - mol1.kinetic_state.semantic_position
                distance = np.linalg.norm(distance_vector)
                
                # Skip if beyond cutoff
                if distance > self.force_cutoff:
                    continue
                    
                # Calculate force magnitude
                force_magnitude = mol1._calculate_semantic_interaction(mol2, distance)
                
                # Limit maximum force to prevent instability
                force_magnitude = np.clip(force_magnitude, -self.max_force, self.max_force)
                
                # Force direction
                if distance > 1e-10:
                    force_direction = distance_vector / distance
                    force = force_magnitude * force_direction
                    
                    # Apply Newton's third law
                    forces[i] += force
                    forces[j] -= force
                    
                    # Update molecule force accumulators
                    mol1.kinetic_state.force_accumulator += force
                    mol2.kinetic_state.force_accumulator -= force
                    
        return forces
        
    def _update_molecular_states(self, 
                               gas_molecules: List[InformationGasMolecule],
                               forces: List[np.ndarray]) -> None:
        """
        Update molecular positions and velocities using Verlet integration.
        
        Verlet integration provides better energy conservation than simple Euler method.
        """
        for molecule, force in zip(gas_molecules, forces):
            # Verlet integration: x(t+dt) = x(t) + v(t)dt + 0.5*a(t)*dt²
            acceleration = force / molecule.effective_mass
            
            # Update position
            new_position = (molecule.kinetic_state.semantic_position + 
                          molecule.kinetic_state.info_velocity * self.time_step +
                          0.5 * acceleration * self.time_step**2)
            
            # Update velocity: v(t+dt) = v(t) + a(t)*dt  
            new_velocity = (molecule.kinetic_state.info_velocity + 
                          acceleration * self.time_step)
            
            # Apply periodic boundary conditions if needed
            new_position = self._apply_boundary_conditions(new_position)
            
            # Update molecule state
            molecule.kinetic_state.semantic_position = new_position
            molecule.kinetic_state.info_velocity = new_velocity
            
            # Update molecular dynamics
            molecule.update_dynamics(self.time_step, force)
            
    def _apply_boundary_conditions(self, position: np.ndarray) -> np.ndarray:
        """Apply periodic boundary conditions to position."""
        # Simple cubic box with periodic boundaries
        box_size = np.cbrt(self.volume)
        
        # Wrap positions that exceed boundaries
        wrapped_position = position % box_size
        
        return wrapped_position
        
    def _apply_thermostat(self, gas_molecules: List[InformationGasMolecule]) -> None:
        """
        Apply velocity scaling thermostat to maintain target temperature.
        
        Uses simple velocity scaling method for temperature control.
        """
        current_temp = self._calculate_system_temperature(gas_molecules)
        
        if current_temp > 0:
            # Scaling factor to adjust temperature
            scaling_factor = np.sqrt(self.target_temperature / current_temp)
            
            # Apply gentle scaling to avoid sudden velocity changes
            gentle_scaling = 0.1 * (scaling_factor - 1.0) + 1.0
            
            # Scale velocities
            for molecule in gas_molecules:
                molecule.kinetic_state.info_velocity *= gentle_scaling
                
    def _calculate_system_temperature(self, gas_molecules: List[InformationGasMolecule]) -> float:
        """Calculate instantaneous system temperature from kinetic energy."""
        if not gas_molecules:
            return 0.0
            
        total_kinetic_energy = 0.0
        degrees_of_freedom = 0
        
        for molecule in gas_molecules:
            kinetic_energy = 0.5 * molecule.effective_mass * np.sum(molecule.kinetic_state.info_velocity**2)
            total_kinetic_energy += kinetic_energy
            degrees_of_freedom += 3  # 3 translational degrees of freedom
            
        # Temperature from equipartition theorem: <E_kinetic> = (f/2) * k_B * T
        if degrees_of_freedom > 0:
            boltzmann_constant = 1.380649e-23
            temperature = (2.0 * total_kinetic_energy) / (degrees_of_freedom * boltzmann_constant)
        else:
            temperature = 0.0
            
        return temperature
        
    def _update_system_properties(self, gas_molecules: List[InformationGasMolecule]) -> None:
        """Update system-level thermodynamic properties."""
        # Update simulation state
        self.current_state.total_time += self.time_step
        self.current_state.system_temperature = self._calculate_system_temperature(gas_molecules)
        self.current_state.total_energy = self._calculate_total_energy(gas_molecules)
        self.current_state.consciousness_level = self._calculate_system_consciousness(gas_molecules)
        
        # Record energy history
        self.energy_history.append(self.current_state.total_energy)
        self.consciousness_history.append(self.current_state.consciousness_level)
        
    def _calculate_total_energy(self, gas_molecules: List[InformationGasMolecule]) -> float:
        """Calculate total system energy (kinetic + potential)."""
        total_energy = 0.0
        
        for molecule in gas_molecules:
            # Kinetic energy
            kinetic_energy = 0.5 * molecule.effective_mass * np.sum(molecule.kinetic_state.info_velocity**2)
            
            # Potential energy (simplified harmonic oscillator)
            potential_energy = 0.5 * np.sum(molecule.kinetic_state.semantic_position**2)
            
            total_energy += kinetic_energy + potential_energy
            
        return total_energy
        
    def _calculate_system_consciousness(self, gas_molecules: List[InformationGasMolecule]) -> float:
        """Calculate overall system consciousness from individual molecules."""
        if not gas_molecules:
            return 0.0
            
        # Average individual consciousness levels
        consciousness_levels = [mol.consciousness_level for mol in gas_molecules]
        avg_consciousness = np.mean(consciousness_levels)
        
        # System coherence factor
        consciousness_std = np.std(consciousness_levels)
        coherence_factor = 1.0 / (1.0 + consciousness_std)
        
        # Energy stability factor
        if len(self.energy_history) > 1:
            energy_stability = 1.0 / (1.0 + abs(self.energy_history[-1] - self.energy_history[-2]))
        else:
            energy_stability = 1.0
            
        system_consciousness = avg_consciousness * coherence_factor * energy_stability
        
        return np.clip(system_consciousness, 0.0, 1.0)
        
    def _record_trajectory_point(self, gas_molecules: List[InformationGasMolecule], step: int) -> None:
        """Record current state for trajectory analysis."""
        trajectory_point = {
            'step': step,
            'time': self.current_state.total_time,
            'positions': [mol.kinetic_state.semantic_position.copy() for mol in gas_molecules],
            'velocities': [mol.kinetic_state.info_velocity.copy() for mol in gas_molecules],
            'energies': [mol.thermodynamic_state.semantic_energy for mol in gas_molecules],
            'consciousness_levels': [mol.consciousness_level for mol in gas_molecules],
            'total_energy': self.current_state.total_energy,
            'system_temperature': self.current_state.system_temperature,
            'system_consciousness': self.current_state.consciousness_level
        }
        
        self.trajectory_history.append(trajectory_point)
        
    def _check_equilibrium_convergence(self, 
                                     gas_molecules: List[InformationGasMolecule],
                                     equilibrium_target: np.ndarray) -> bool:
        """Check if system has converged to equilibrium configuration."""
        # Calculate current system variance from target
        total_variance = 0.0
        
        for i, molecule in enumerate(gas_molecules):
            if i * 3 + 2 < len(equilibrium_target):
                target_position = equilibrium_target[i*3:(i+1)*3]
                position_variance = np.sum((molecule.kinetic_state.semantic_position - target_position)**2)
                velocity_variance = np.sum(molecule.kinetic_state.info_velocity**2)
                
                total_variance += position_variance + velocity_variance
                
        # Convergence threshold
        convergence_threshold = 1e-6
        
        return total_variance < convergence_threshold
        
    def get_equilibrium_metrics(self, gas_molecules: List[InformationGasMolecule]) -> Dict[str, Any]:
        """Get metrics describing equilibrium quality."""
        if not gas_molecules:
            return {}
            
        # Position clustering (molecules near equilibrium positions)
        equilibrium_positions = [mol.get_equilibrium_target() for mol in gas_molecules]
        current_positions = [mol.kinetic_state.semantic_position for mol in gas_molecules]
        
        position_deviations = [
            np.linalg.norm(current - equilibrium) 
            for current, equilibrium in zip(current_positions, equilibrium_positions)
        ]
        
        # Velocity magnitudes (lower = more equilibrium-like)
        velocity_magnitudes = [
            np.linalg.norm(mol.kinetic_state.info_velocity) 
            for mol in gas_molecules
        ]
        
        # Energy stability
        if len(self.energy_history) > 10:
            recent_energies = self.energy_history[-10:]
            energy_stability = 1.0 / (1.0 + np.std(recent_energies))
        else:
            energy_stability = 0.0
            
        return {
            'average_position_deviation': np.mean(position_deviations),
            'max_position_deviation': np.max(position_deviations),
            'average_velocity_magnitude': np.mean(velocity_magnitudes),
            'max_velocity_magnitude': np.max(velocity_magnitudes),
            'energy_stability': energy_stability,
            'system_consciousness': self.current_state.consciousness_level,
            'equilibrium_quality_score': self._calculate_equilibrium_quality_score(gas_molecules)
        }
        
    def _calculate_equilibrium_quality_score(self, gas_molecules: List[InformationGasMolecule]) -> float:
        """Calculate overall equilibrium quality score (0-1)."""
        metrics = self.get_equilibrium_metrics(gas_molecules)
        
        if not metrics:
            return 0.0
            
        # Combine metrics into quality score
        position_score = 1.0 / (1.0 + metrics.get('average_position_deviation', 1.0))
        velocity_score = 1.0 / (1.0 + metrics.get('average_velocity_magnitude', 1.0))
        energy_score = metrics.get('energy_stability', 0.0)
        consciousness_score = metrics.get('system_consciousness', 0.0)
        
        # Weighted average
        quality_score = (0.3 * position_score + 
                        0.3 * velocity_score + 
                        0.2 * energy_score + 
                        0.2 * consciousness_score)
        
        return np.clip(quality_score, 0.0, 1.0)
