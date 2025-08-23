"""
Information Gas Molecule Implementation

An Information Gas Molecule (IGM) represents a visual element as a thermodynamic
gas molecule with semantic energy, information entropy, processing temperature,
and other consciousness-aware properties.
"""

import numpy as np
from typing import Optional, Dict, Any, List
import time
from dataclasses import dataclass, field


@dataclass
class ThermodynamicState:
    """Thermodynamic state of an information gas molecule"""
    semantic_energy: float
    info_entropy: float
    processing_temperature: float
    semantic_pressure: float
    conceptual_volume: float
    equilibrium_state: Optional[np.ndarray] = None


@dataclass
class KineticState:
    """Kinetic state of an information gas molecule"""
    semantic_position: np.ndarray
    info_velocity: np.ndarray
    meaning_cross_section: float
    force_accumulator: np.ndarray = field(default_factory=lambda: np.zeros(3))


class InformationGasMolecule:
    """
    Information Gas Molecule (IGM) representing a visual element as a 
    thermodynamic entity seeking equilibrium configurations.
    
    The IGM operates through genuine gas molecular dynamics where meaning
    emerges from variance minimization rather than computational processing.
    """
    
    def __init__(self, 
                 semantic_energy: float,
                 info_entropy: float, 
                 processing_temperature: float,
                 semantic_position: np.ndarray,
                 info_velocity: np.ndarray,
                 meaning_cross_section: float,
                 semantic_pressure: float = 1.0,
                 conceptual_volume: float = 1.0,
                 molecule_id: Optional[str] = None):
        """
        Initialize Information Gas Molecule with thermodynamic and kinetic properties.
        
        Args:
            semantic_energy: Internal semantic energy of the molecule
            info_entropy: Information entropy content
            processing_temperature: Temperature controlling processing resources
            semantic_position: Position in semantic space (3D)
            info_velocity: Velocity in information space (3D)  
            meaning_cross_section: Cross-section for meaning interactions
            semantic_pressure: Pressure in semantic space
            conceptual_volume: Volume in conceptual space
            molecule_id: Unique identifier for the molecule
        """
        self.thermodynamic_state = ThermodynamicState(
            semantic_energy=semantic_energy,
            info_entropy=info_entropy,
            processing_temperature=processing_temperature,
            semantic_pressure=semantic_pressure,
            conceptual_volume=conceptual_volume
        )
        
        self.kinetic_state = KineticState(
            semantic_position=np.array(semantic_position, dtype=np.float64),
            info_velocity=np.array(info_velocity, dtype=np.float64),
            meaning_cross_section=meaning_cross_section
        )
        
        self.molecule_id = molecule_id or f"igm_{time.time_ns()}"
        self.interaction_history: List[Dict[str, Any]] = []
        self.consciousness_level: float = 0.0
        
        # Gas molecular constants
        self.boltzmann_constant = 1.380649e-23  # Semantic Boltzmann constant
        self.gas_constant = 8.314462618  # Universal gas constant for information
        self.effective_mass = self._calculate_effective_mass()
        
    def _calculate_effective_mass(self) -> float:
        """Calculate effective mass based on semantic energy and entropy."""
        # Mass emerges from semantic energy and information entropy relationship
        # E = mc² adapted for information: E_semantic = m_effective * c_info²
        c_info_squared = 2.998e8 ** 2  # Information speed constant squared
        return max(self.thermodynamic_state.semantic_energy / c_info_squared, 1e-30)
        
    def calculate_semantic_forces(self, other_molecules: List['InformationGasMolecule']) -> np.ndarray:
        """
        Calculate semantic forces with other gas molecules for equilibrium seeking.
        
        Args:
            other_molecules: List of other IGMs to calculate interactions with
            
        Returns:
            Total force vector acting on this molecule
        """
        total_force = np.zeros(3, dtype=np.float64)
        
        for molecule in other_molecules:
            if molecule.molecule_id == self.molecule_id:
                continue
                
            # Calculate semantic distance
            distance_vector = molecule.kinetic_state.semantic_position - self.kinetic_state.semantic_position
            distance = np.linalg.norm(distance_vector)
            
            if distance < 1e-10:  # Avoid division by zero
                continue
                
            # Semantic interaction force (modified Lennard-Jones for information)
            force_magnitude = self._calculate_semantic_interaction(molecule, distance)
            force_direction = distance_vector / distance
            
            force = force_magnitude * force_direction
            total_force += force
            
            # Record interaction for consciousness tracking
            self.interaction_history.append({
                'partner_id': molecule.molecule_id,
                'force_magnitude': force_magnitude,
                'distance': distance,
                'timestamp': time.time_ns()
            })
            
        return total_force
        
    def _calculate_semantic_interaction(self, other: 'InformationGasMolecule', distance: float) -> float:
        """
        Calculate semantic interaction force between two information gas molecules.
        
        Uses modified Lennard-Jones potential adapted for semantic interactions:
        F = 4ε[(σ/r)¹³ - (σ/r)⁷] where ε is semantic affinity, σ is meaning radius
        """
        # Semantic affinity based on entropy difference
        entropy_diff = abs(self.thermodynamic_state.info_entropy - 
                          other.thermodynamic_state.info_entropy)
        semantic_affinity = np.exp(-entropy_diff / self.thermodynamic_state.processing_temperature)
        
        # Meaning interaction radius
        meaning_radius = (self.kinetic_state.meaning_cross_section + 
                         other.kinetic_state.meaning_cross_section) / 2
        
        # Normalized distance
        r_norm = distance / meaning_radius
        
        # Modified Lennard-Jones force for semantic interactions
        if r_norm < 0.1:  # Very close - strong repulsion
            force = 1000 * semantic_affinity / (distance ** 2)
        else:
            # Attractive/repulsive force based on semantic compatibility
            force = 4 * semantic_affinity * (
                13 * (r_norm ** (-14)) - 7 * (r_norm ** (-8))
            ) / meaning_radius
            
        return force
        
    def update_dynamics(self, dt: float, external_forces: np.ndarray = None) -> None:
        """
        Update gas molecular dynamics according to Newton's equations.
        
        Args:
            dt: Time step for integration
            external_forces: Additional external forces acting on the molecule
        """
        if external_forces is None:
            external_forces = np.zeros(3)
            
        # Total force = accumulated semantic forces + external forces
        total_force = self.kinetic_state.force_accumulator + external_forces
        
        # Newton's second law: F = ma → a = F/m
        acceleration = total_force / self.effective_mass
        
        # Update velocity (Verlet integration for stability)
        self.kinetic_state.info_velocity += acceleration * dt
        
        # Update position  
        self.kinetic_state.semantic_position += self.kinetic_state.info_velocity * dt
        
        # Update thermodynamic properties based on kinetic energy
        kinetic_energy = 0.5 * self.effective_mass * np.linalg.norm(self.kinetic_state.info_velocity) ** 2
        self.thermodynamic_state.semantic_energy = kinetic_energy + self._potential_energy()
        
        # Update processing temperature based on kinetic energy
        # Temperature ∝ kinetic energy (equipartition theorem)
        self.thermodynamic_state.processing_temperature = (2/3) * kinetic_energy / self.boltzmann_constant
        
        # Update consciousness level based on thermodynamic stability
        self.consciousness_level = self._calculate_consciousness_level()
        
        # Reset force accumulator
        self.kinetic_state.force_accumulator = np.zeros(3)
        
    def _potential_energy(self) -> float:
        """Calculate potential energy based on semantic position."""
        # Potential energy in semantic space (harmonic oscillator model)
        return 0.5 * np.sum(self.kinetic_state.semantic_position ** 2)
        
    def _calculate_consciousness_level(self) -> float:
        """
        Calculate consciousness level based on thermodynamic and kinetic state.
        
        Consciousness emerges from the stability and coherence of the 
        gas molecular state in semantic space.
        """
        # Consciousness based on thermodynamic stability
        entropy_factor = np.exp(-self.thermodynamic_state.info_entropy / 10)
        energy_factor = 1 / (1 + self.thermodynamic_state.semantic_energy)
        stability_factor = 1 / (1 + np.linalg.norm(self.kinetic_state.info_velocity))
        
        # Interaction coherence (more interactions = higher consciousness)
        interaction_factor = min(len(self.interaction_history) / 100, 1.0)
        
        consciousness = entropy_factor * energy_factor * stability_factor * interaction_factor
        
        return np.clip(consciousness, 0.0, 1.0)
        
    def get_equilibrium_target(self) -> np.ndarray:
        """
        Get the target equilibrium position for this gas molecule.
        
        Returns:
            Target equilibrium position in semantic space
        """
        # Equilibrium position minimizes potential energy
        # For harmonic oscillator, equilibrium is at origin
        return np.zeros(3)
        
    def calculate_variance_from_equilibrium(self) -> float:
        """
        Calculate variance of current state from equilibrium.
        
        Returns:
            Variance measure (lower = closer to equilibrium)
        """
        equilibrium_target = self.get_equilibrium_target()
        position_variance = np.linalg.norm(self.kinetic_state.semantic_position - equilibrium_target) ** 2
        velocity_variance = np.linalg.norm(self.kinetic_state.info_velocity) ** 2
        
        total_variance = position_variance + velocity_variance
        return total_variance
        
    def apply_consciousness_enhancement(self, enhancement_factor: float = 1.2) -> None:
        """
        Apply consciousness enhancement to improve processing capabilities.
        
        Args:
            enhancement_factor: Factor by which to enhance consciousness
        """
        # Enhance processing temperature for better resource allocation
        self.thermodynamic_state.processing_temperature *= enhancement_factor
        
        # Increase semantic energy for better interactions
        self.thermodynamic_state.semantic_energy *= enhancement_factor
        
        # Improve meaning cross-section for better semantic interactions
        self.kinetic_state.meaning_cross_section *= enhancement_factor
        
        # Recalculate consciousness level
        self.consciousness_level = self._calculate_consciousness_level()
        
    def get_state_vector(self) -> np.ndarray:
        """
        Get complete state vector for equilibrium analysis.
        
        Returns:
            State vector containing position, velocity, and thermodynamic state
        """
        state = np.concatenate([
            self.kinetic_state.semantic_position,
            self.kinetic_state.info_velocity,
            [self.thermodynamic_state.semantic_energy,
             self.thermodynamic_state.info_entropy,
             self.thermodynamic_state.processing_temperature]
        ])
        return state
        
    def set_state_vector(self, state: np.ndarray) -> None:
        """
        Set complete state from state vector.
        
        Args:
            state: State vector containing all molecular properties
        """
        self.kinetic_state.semantic_position = state[0:3]
        self.kinetic_state.info_velocity = state[3:6] 
        self.thermodynamic_state.semantic_energy = state[6]
        self.thermodynamic_state.info_entropy = state[7]
        self.thermodynamic_state.processing_temperature = state[8]
        
        # Recalculate derived properties
        self.effective_mass = self._calculate_effective_mass()
        self.consciousness_level = self._calculate_consciousness_level()
        
    def __repr__(self) -> str:
        """String representation of the Information Gas Molecule."""
        return (f"InformationGasMolecule(id={self.molecule_id}, "
                f"consciousness={self.consciousness_level:.3f}, "
                f"energy={self.thermodynamic_state.semantic_energy:.3f}, "
                f"entropy={self.thermodynamic_state.info_entropy:.3f}, "
                f"temperature={self.thermodynamic_state.processing_temperature:.3f})")
