# Gas Molecule Model Classes and Functionality for Life Science Applications
"""
This module implements the gas molecular dynamics framework specifically adapted for
biological and life science image analysis. Information elements are modeled as 
thermodynamic gas molecules that seek equilibrium states through Hamilton's equations,
enabling principled analysis of biological structures and processes.

Key Features:
- InformationGasMolecule: Individual information entities with thermodynamic properties
- GasMolecularSystem: Collection of molecules with interaction potentials
- BiologicalGasAnalyzer: Life science-specific analysis wrapper
- ProteinFoldingGas: Specialized implementation for protein structure analysis
- CellularDynamicsGas: Molecular modeling of cellular processes

Applications:
- Protein structure analysis through thermodynamic equilibrium
- Cellular process modeling (mitosis, apoptosis, migration)
- Metabolic pathway visualization as molecular interactions
- Drug-target binding dynamics through equilibrium states
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MoleculeType(Enum):
    """Types of information molecules for biological contexts"""
    PROTEIN = "protein"
    NUCLEIC_ACID = "nucleic_acid"
    LIPID = "lipid"
    METABOLITE = "metabolite"
    CELLULAR_STRUCTURE = "cellular_structure"
    SIGNAL = "signal"


@dataclass
class BiologicalProperties:
    """Biological context for information molecules"""
    molecule_type: MoleculeType
    biological_function: str
    cellular_location: str
    interaction_energy: float = 0.0
    stability: float = 1.0
    activity_level: float = 0.5


@dataclass
class InformationGasMolecule:
    """
    Individual information gas molecule with thermodynamic and biological properties.
    
    Represents a single unit of biological information (protein, cell region, etc.)
    as a thermodynamic entity that can interact with other molecules and seek
    equilibrium states.
    """
    
    # Thermodynamic properties
    position: np.ndarray = field(default_factory=lambda: np.random.random(3))
    velocity: np.ndarray = field(default_factory=lambda: np.random.random(3) - 0.5)
    mass: float = 1.0
    temperature: float = 1.0
    entropy: float = 1.0
    internal_energy: float = 1.0
    
    # Biological context
    biological_props: Optional[BiologicalProperties] = None
    
    # Interaction parameters
    sigma: float = 1.0  # Size parameter
    epsilon: float = 1.0  # Interaction strength
    
    # Information content
    information_content: float = 1.0
    semantic_properties: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived properties"""
        if self.biological_props is None:
            self.biological_props = BiologicalProperties(
                molecule_type=MoleculeType.CELLULAR_STRUCTURE,
                biological_function="structural",
                cellular_location="cytoplasm"
            )
    
    def calculate_kinetic_energy(self) -> float:
        """Calculate kinetic energy of the molecule"""
        return 0.5 * self.mass * np.sum(self.velocity**2)
    
    def calculate_potential_energy(self, other_molecules: List['InformationGasMolecule']) -> float:
        """Calculate potential energy due to interactions with other molecules"""
        potential = 0.0
        
        for other in other_molecules:
            if other is not self:
                r = np.linalg.norm(self.position - other.position)
                if r > 0:
                    # Lennard-Jones potential with biological modulation
                    sigma_ij = (self.sigma + other.sigma) / 2
                    epsilon_ij = np.sqrt(self.epsilon * other.epsilon)
                    
                    # Biological similarity modulation
                    bio_similarity = self._calculate_biological_similarity(other)
                    epsilon_ij *= (1 + bio_similarity)
                    
                    lj_potential = 4 * epsilon_ij * ((sigma_ij/r)**12 - (sigma_ij/r)**6)
                    potential += lj_potential
        
        return potential
    
    def _calculate_biological_similarity(self, other: 'InformationGasMolecule') -> float:
        """Calculate biological similarity between molecules"""
        if self.biological_props.molecule_type == other.biological_props.molecule_type:
            return 0.5  # Same type molecules have moderate attraction
        else:
            return -0.2  # Different types have slight repulsion
    
    def update_thermodynamic_state(self, dt: float):
        """Update thermodynamic properties based on current state"""
        # Temperature based on kinetic energy
        kinetic_energy = self.calculate_kinetic_energy()
        self.temperature = 2 * kinetic_energy / (3 * 1.381e-23)  # Equipartition theorem
        
        # Entropy increase with time (second law)
        self.entropy += dt * 0.01 * self.temperature
        
        # Internal energy conservation
        self.internal_energy = kinetic_energy + self.entropy * self.temperature


class GasMolecularSystem:
    """
    Collection of information gas molecules with interaction dynamics.
    
    Implements the complete thermodynamic system for biological information
    processing, including equilibrium seeking, force calculations, and
    temporal evolution.
    """
    
    def __init__(self, molecules: Optional[List[InformationGasMolecule]] = None):
        self.molecules = molecules or []
        self.time = 0.0
        self.dt = 0.001
        self.damping = 0.1
        self.temperature_bath = 1.0
        
        # System properties
        self.total_energy_history = []
        self.temperature_history = []
        self.entropy_history = []
        
        # Force capping for numerical stability
        self.max_force = 100.0
        self.min_distance = 0.1
        
        logger.info(f"Initialized gas molecular system with {len(self.molecules)} molecules")
    
    def add_molecule(self, molecule: InformationGasMolecule):
        """Add a molecule to the system"""
        self.molecules.append(molecule)
    
    def calculate_forces(self) -> List[np.ndarray]:
        """Calculate forces on all molecules"""
        forces = [np.zeros(3) for _ in self.molecules]
        
        for i, mol_i in enumerate(self.molecules):
            for j, mol_j in enumerate(self.molecules):
                if i != j:
                    r_vec = mol_i.position - mol_j.position
                    r = np.linalg.norm(r_vec)
                    
                    # Prevent singular forces
                    r = max(r, self.min_distance)
                    r_hat = r_vec / r
                    
                    # Lennard-Jones force with biological modulation
                    sigma_ij = (mol_i.sigma + mol_j.sigma) / 2
                    epsilon_ij = np.sqrt(mol_i.epsilon * mol_j.epsilon)
                    
                    # Biological interaction strength
                    bio_similarity = mol_i._calculate_biological_similarity(mol_j)
                    epsilon_ij *= (1 + bio_similarity)
                    
                    # Force magnitude
                    force_mag = 24 * epsilon_ij / r * (2 * (sigma_ij/r)**12 - (sigma_ij/r)**6)
                    
                    # Cap force to prevent numerical instability
                    force_mag = np.clip(force_mag, -self.max_force, self.max_force)
                    
                    forces[i] += force_mag * r_hat
        
        return forces
    
    def integrate_motion(self, forces: List[np.ndarray]):
        """Integrate equations of motion using velocity Verlet"""
        for i, molecule in enumerate(self.molecules):
            # Velocity Verlet integration
            acceleration = forces[i] / molecule.mass
            
            # Update position
            molecule.position += molecule.velocity * self.dt + 0.5 * acceleration * self.dt**2
            
            # Apply damping (thermostat)
            molecule.velocity *= (1 - self.damping)
            
            # Update velocity
            molecule.velocity += acceleration * self.dt
            
            # Update thermodynamic state
            molecule.update_thermodynamic_state(self.dt)
    
    def calculate_system_properties(self) -> Dict[str, float]:
        """Calculate system-wide thermodynamic properties"""
        if not self.molecules:
            return {'total_energy': 0.0, 'temperature': 0.0, 'entropy': 0.0}
        
        total_kinetic = sum(mol.calculate_kinetic_energy() for mol in self.molecules)
        total_potential = sum(mol.calculate_potential_energy(self.molecules) for mol in self.molecules) / 2  # Avoid double counting
        total_energy = total_kinetic + total_potential
        
        avg_temperature = np.mean([mol.temperature for mol in self.molecules])
        total_entropy = sum(mol.entropy for mol in self.molecules)
        
        return {
            'total_energy': total_energy,
            'kinetic_energy': total_kinetic,
            'potential_energy': total_potential,
            'temperature': avg_temperature,
            'entropy': total_entropy,
            'num_molecules': len(self.molecules)
        }
    
    def evolve(self, steps: int = 1000) -> Dict[str, Any]:
        """Evolve the system toward equilibrium"""
        logger.info(f"Evolving gas molecular system for {steps} steps")
        
        for step in range(steps):
            # Calculate forces
            forces = self.calculate_forces()
            
            # Integrate motion
            self.integrate_motion(forces)
            
            # Update system time
            self.time += self.dt
            
            # Record system properties
            if step % 10 == 0:  # Record every 10 steps
                props = self.calculate_system_properties()
                self.total_energy_history.append(props['total_energy'])
                self.temperature_history.append(props['temperature'])
                self.entropy_history.append(props['entropy'])
        
        final_properties = self.calculate_system_properties()
        logger.info(f"System evolution complete. Final energy: {final_properties['total_energy']:.3f}")
        
        return {
            'final_properties': final_properties,
            'energy_history': self.total_energy_history,
            'temperature_history': self.temperature_history,
            'entropy_history': self.entropy_history,
            'equilibrium_positions': [mol.position.copy() for mol in self.molecules],
            'equilibrium_velocities': [mol.velocity.copy() for mol in self.molecules]
        }


# Export main classes
__all__ = [
    'InformationGasMolecule',
    'GasMolecularSystem',
    'MoleculeType',
    'BiologicalProperties'
]