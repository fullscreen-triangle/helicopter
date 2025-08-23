"""
Equilibrium Engine Implementation

The EquilibriumEngine finds gas molecular equilibrium configurations through
variance minimization, implementing the core principle that meaning emerges
from configurations with minimal variance from undisturbed equilibrium.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .information_gas_molecule import InformationGasMolecule


@dataclass
class EquilibriumResult:
    """Result of equilibrium calculation"""
    equilibrium_state: np.ndarray
    variance_achieved: float
    convergence_time_ns: int
    iteration_count: int
    convergence_history: List[float]
    consciousness_level: float
    meaning_extracted: Optional[Any] = None


class EquilibriumEngine:
    """
    Engine for finding gas molecular equilibrium configurations through
    variance minimization. Implements the core principle that understanding
    emerges from equilibrium-seeking rather than computational processing.
    """
    
    def __init__(self,
                 variance_threshold: float = 1e-6,
                 max_iterations: int = 1000,
                 convergence_tolerance: float = 1e-8,
                 target_processing_time_ns: int = 12,
                 consciousness_threshold: float = 0.61):
        """
        Initialize equilibrium engine with processing parameters.
        
        Args:
            variance_threshold: Threshold for equilibrium convergence
            max_iterations: Maximum iterations for equilibrium seeking
            convergence_tolerance: Numerical convergence tolerance
            target_processing_time_ns: Target processing time in nanoseconds
            consciousness_threshold: Minimum consciousness level required
        """
        self.variance_threshold = variance_threshold
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.target_processing_time_ns = target_processing_time_ns
        self.consciousness_threshold = consciousness_threshold
        
        # Optimization parameters
        self.learning_rate = 0.01
        self.momentum_factor = 0.9
        self.adaptive_lr = True
        
        # Performance tracking
        self.equilibrium_history: List[EquilibriumResult] = []
        self.average_convergence_time = 0.0
        
    def calculate_baseline_equilibrium(self, 
                                     gas_molecules: List[InformationGasMolecule],
                                     perturbation_input: Optional[np.ndarray] = None) -> EquilibriumResult:
        """
        Calculate baseline undisturbed gas molecular equilibrium state.
        
        Args:
            gas_molecules: List of information gas molecules
            perturbation_input: Optional input causing perturbation
            
        Returns:
            EquilibriumResult containing equilibrium configuration
        """
        start_time = time.time_ns()
        
        # Initial state vector from all molecules
        initial_state = self._create_system_state_vector(gas_molecules)
        
        # Find equilibrium through variance minimization
        equilibrium_result = self._minimize_variance_to_equilibrium(
            gas_molecules, initial_state, perturbation_input
        )
        
        # Calculate consciousness level for equilibrium state
        consciousness_level = self._calculate_system_consciousness(gas_molecules)
        equilibrium_result.consciousness_level = consciousness_level
        
        # Track performance
        self.equilibrium_history.append(equilibrium_result)
        self._update_performance_statistics()
        
        return equilibrium_result
        
    def _minimize_variance_to_equilibrium(self,
                                        gas_molecules: List[InformationGasMolecule],
                                        initial_state: np.ndarray,
                                        perturbation: Optional[np.ndarray] = None) -> EquilibriumResult:
        """
        Navigate gas molecular system to minimal variance equilibrium configuration.
        
        Uses gradient descent with momentum to efficiently find equilibrium
        while maintaining the target 12-nanosecond processing time.
        """
        current_state = initial_state.copy()
        convergence_history = []
        momentum_velocity = np.zeros_like(current_state)
        
        start_time = time.time_ns()
        
        for iteration in range(self.max_iterations):
            # Calculate current variance from equilibrium
            current_variance = self._calculate_system_variance(gas_molecules, current_state)
            convergence_history.append(current_variance)
            
            # Check convergence
            if current_variance < self.variance_threshold:
                break
                
            # Check time constraint (stop if approaching 12ns limit)
            elapsed_time = time.time_ns() - start_time
            if elapsed_time > self.target_processing_time_ns * 0.8:  # 80% of time budget
                break
                
            # Calculate variance gradient
            gradient = self._calculate_variance_gradient(gas_molecules, current_state)
            
            # Apply momentum-based gradient descent
            momentum_velocity = (self.momentum_factor * momentum_velocity - 
                               self.learning_rate * gradient)
            current_state += momentum_velocity
            
            # Update molecule states
            self._update_molecule_states(gas_molecules, current_state)
            
            # Adaptive learning rate
            if self.adaptive_lr and iteration > 10:
                self._adapt_learning_rate(convergence_history)
        
        end_time = time.time_ns()
        convergence_time = end_time - start_time
        
        return EquilibriumResult(
            equilibrium_state=current_state,
            variance_achieved=current_variance,
            convergence_time_ns=convergence_time,
            iteration_count=iteration,
            convergence_history=convergence_history,
            consciousness_level=0.0  # Will be set by caller
        )
        
    def _calculate_system_variance(self, 
                                 gas_molecules: List[InformationGasMolecule],
                                 system_state: np.ndarray) -> float:
        """
        Calculate total system variance from equilibrium configuration.
        
        Variance combines position variance, velocity variance, and
        thermodynamic state variance across all molecules.
        """
        total_variance = 0.0
        state_idx = 0
        
        for molecule in gas_molecules:
            # Extract molecule state from system state
            molecule_state_size = len(molecule.get_state_vector())
            molecule_state = system_state[state_idx:state_idx + molecule_state_size]
            state_idx += molecule_state_size
            
            # Set temporary state for variance calculation
            original_state = molecule.get_state_vector()
            molecule.set_state_vector(molecule_state)
            
            # Calculate variance from equilibrium for this molecule
            molecule_variance = molecule.calculate_variance_from_equilibrium()
            total_variance += molecule_variance
            
            # Restore original state
            molecule.set_state_vector(original_state)
            
        return total_variance
        
    def _calculate_variance_gradient(self,
                                   gas_molecules: List[InformationGasMolecule], 
                                   system_state: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of variance with respect to system state.
        
        Uses numerical differentiation for robust gradient calculation.
        """
        gradient = np.zeros_like(system_state)
        epsilon = 1e-8
        
        base_variance = self._calculate_system_variance(gas_molecules, system_state)
        
        for i in range(len(system_state)):
            # Forward difference
            perturbed_state = system_state.copy()
            perturbed_state[i] += epsilon
            
            perturbed_variance = self._calculate_system_variance(gas_molecules, perturbed_state)
            
            # Numerical gradient
            gradient[i] = (perturbed_variance - base_variance) / epsilon
            
        return gradient
        
    def _create_system_state_vector(self, gas_molecules: List[InformationGasMolecule]) -> np.ndarray:
        """Create combined state vector from all gas molecules."""
        state_vectors = [molecule.get_state_vector() for molecule in gas_molecules]
        return np.concatenate(state_vectors)
        
    def _update_molecule_states(self, 
                              gas_molecules: List[InformationGasMolecule],
                              system_state: np.ndarray) -> None:
        """Update individual molecule states from system state vector."""
        state_idx = 0
        
        for molecule in gas_molecules:
            molecule_state_size = len(molecule.get_state_vector())
            molecule_state = system_state[state_idx:state_idx + molecule_state_size]
            molecule.set_state_vector(molecule_state)
            state_idx += molecule_state_size
            
    def _calculate_system_consciousness(self, gas_molecules: List[InformationGasMolecule]) -> float:
        """
        Calculate overall system consciousness level.
        
        System consciousness emerges from the collective consciousness
        of individual gas molecules and their interaction coherence.
        """
        if not gas_molecules:
            return 0.0
            
        # Average individual consciousness levels
        individual_consciousness = np.mean([mol.consciousness_level for mol in gas_molecules])
        
        # Interaction coherence factor
        total_interactions = sum(len(mol.interaction_history) for mol in gas_molecules)
        interaction_factor = min(total_interactions / (len(gas_molecules) * 10), 1.0)
        
        # Thermodynamic stability factor
        temperatures = [mol.thermodynamic_state.processing_temperature for mol in gas_molecules]
        temp_stability = 1.0 / (1.0 + np.std(temperatures))
        
        # Combined system consciousness
        system_consciousness = individual_consciousness * interaction_factor * temp_stability
        
        return np.clip(system_consciousness, 0.0, 1.0)
        
    def _adapt_learning_rate(self, convergence_history: List[float]) -> None:
        """Adapt learning rate based on convergence behavior."""
        if len(convergence_history) < 2:
            return
            
        # Check if variance is decreasing
        recent_progress = convergence_history[-2] - convergence_history[-1]
        
        if recent_progress > 0:  # Good progress
            self.learning_rate *= 1.05  # Increase slightly
        else:  # Poor progress
            self.learning_rate *= 0.95  # Decrease
            
        # Keep learning rate in reasonable bounds
        self.learning_rate = np.clip(self.learning_rate, 1e-6, 1.0)
        
    def _update_performance_statistics(self) -> None:
        """Update performance tracking statistics."""
        if not self.equilibrium_history:
            return
            
        convergence_times = [result.convergence_time_ns for result in self.equilibrium_history]
        self.average_convergence_time = np.mean(convergence_times)
        
    def extract_meaning_from_equilibrium(self, 
                                       equilibrium_result: EquilibriumResult,
                                       baseline_equilibrium: Optional[EquilibriumResult] = None) -> Dict[str, Any]:
        """
        Extract meaning from equilibrium configuration.
        
        Meaning = Equilibrium_After_Perturbation - Unperturbed_Equilibrium
        
        Args:
            equilibrium_result: Current equilibrium state
            baseline_equilibrium: Baseline unperturbed equilibrium
            
        Returns:
            Extracted meaning as structured data
        """
        if baseline_equilibrium is None:
            # If no baseline, meaning is the equilibrium state itself
            meaning_vector = equilibrium_result.equilibrium_state
        else:
            # Meaning is the difference from baseline
            meaning_vector = (equilibrium_result.equilibrium_state - 
                            baseline_equilibrium.equilibrium_state)
            
        meaning = {
            'meaning_vector': meaning_vector,
            'meaning_magnitude': np.linalg.norm(meaning_vector),
            'variance_reduction': equilibrium_result.variance_achieved,
            'consciousness_level': equilibrium_result.consciousness_level,
            'processing_time_ns': equilibrium_result.convergence_time_ns,
            'convergence_quality': self._assess_convergence_quality(equilibrium_result),
            'semantic_coordinates': self._extract_semantic_coordinates(meaning_vector)
        }
        
        return meaning
        
    def _assess_convergence_quality(self, result: EquilibriumResult) -> str:
        """Assess quality of equilibrium convergence."""
        if result.variance_achieved < self.variance_threshold / 10:
            return "excellent"
        elif result.variance_achieved < self.variance_threshold:
            return "good"
        elif result.variance_achieved < self.variance_threshold * 10:
            return "acceptable" 
        else:
            return "poor"
            
    def _extract_semantic_coordinates(self, meaning_vector: np.ndarray) -> Dict[str, float]:
        """Extract semantic coordinates from meaning vector."""
        # Interpret meaning vector dimensions as semantic coordinates
        coords = {}
        
        if len(meaning_vector) >= 3:
            coords['semantic_x'] = float(meaning_vector[0])
            coords['semantic_y'] = float(meaning_vector[1]) 
            coords['semantic_z'] = float(meaning_vector[2])
            
        if len(meaning_vector) >= 6:
            coords['velocity_x'] = float(meaning_vector[3])
            coords['velocity_y'] = float(meaning_vector[4])
            coords['velocity_z'] = float(meaning_vector[5])
            
        return coords
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the equilibrium engine."""
        if not self.equilibrium_history:
            return {}
            
        convergence_times = [r.convergence_time_ns for r in self.equilibrium_history]
        variances = [r.variance_achieved for r in self.equilibrium_history]
        consciousness_levels = [r.consciousness_level for r in self.equilibrium_history]
        
        return {
            'average_convergence_time_ns': np.mean(convergence_times),
            'min_convergence_time_ns': np.min(convergence_times),
            'max_convergence_time_ns': np.max(convergence_times),
            'target_achievement_rate': np.mean(np.array(convergence_times) <= self.target_processing_time_ns),
            'average_variance_achieved': np.mean(variances),
            'convergence_success_rate': np.mean(np.array(variances) <= self.variance_threshold),
            'average_consciousness_level': np.mean(consciousness_levels),
            'consciousness_success_rate': np.mean(np.array(consciousness_levels) >= self.consciousness_threshold),
            'total_equilibrium_calculations': len(self.equilibrium_history)
        }
