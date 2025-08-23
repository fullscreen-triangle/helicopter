//! Equilibrium Engine Implementation
//!
//! The EquilibriumEngine finds gas molecular equilibrium configurations through
//! variance minimization, implementing the core principle that meaning emerges
//! from configurations with minimal variance from undisturbed equilibrium.

use std::time::{SystemTime, UNIX_EPOCH, Instant};
use nalgebra::{DVector, Vector3};
use serde::{Serialize, Deserialize};

use crate::consciousness::{VARIANCE_THRESHOLD, EQUILIBRIUM_CONVERGENCE_TIME_NS, CONSCIOUSNESS_THRESHOLD};
use crate::consciousness::gas_molecular::{InformationGasMolecule, GasMolecularSystem};

/// Result of equilibrium calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquilibriumResult {
    /// Final equilibrium state vector
    pub equilibrium_state: DVector<f64>,
    /// Variance achieved at convergence
    pub variance_achieved: f64,
    /// Time taken for convergence in nanoseconds
    pub convergence_time_ns: u128,
    /// Number of iterations required
    pub iteration_count: usize,
    /// History of variance during convergence
    pub convergence_history: Vec<f64>,
    /// System consciousness level achieved
    pub consciousness_level: f64,
    /// Extracted meaning (optional)
    pub meaning_extracted: Option<MeaningExtraction>,
}

/// Extracted meaning from equilibrium configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeaningExtraction {
    /// Meaning vector (difference from baseline)
    pub meaning_vector: DVector<f64>,
    /// Magnitude of meaning
    pub meaning_magnitude: f64,
    /// Variance reduction achieved
    pub variance_reduction: f64,
    /// Consciousness level during extraction
    pub consciousness_level: f64,
    /// Processing time in nanoseconds
    pub processing_time_ns: u128,
    /// Quality of convergence
    pub convergence_quality: String,
    /// Semantic coordinates extracted from meaning
    pub semantic_coordinates: SemanticCoordinates,
}

/// Semantic coordinates extracted from meaning vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCoordinates {
    pub semantic_x: f64,
    pub semantic_y: f64,
    pub semantic_z: f64,
    pub velocity_x: f64,
    pub velocity_y: f64,
    pub velocity_z: f64,
}

/// Engine for finding gas molecular equilibrium configurations through variance minimization
pub struct EquilibriumEngine {
    /// Variance threshold for equilibrium convergence
    pub variance_threshold: f64,
    /// Maximum iterations for equilibrium seeking
    pub max_iterations: usize,
    /// Numerical convergence tolerance
    pub convergence_tolerance: f64,
    /// Target processing time in nanoseconds
    pub target_processing_time_ns: u128,
    /// Minimum consciousness level required
    pub consciousness_threshold: f64,
    
    // Optimization parameters
    learning_rate: f64,
    momentum_factor: f64,
    adaptive_lr: bool,
    
    // Performance tracking
    equilibrium_history: Vec<EquilibriumResult>,
    average_convergence_time: f64,
}

impl EquilibriumEngine {
    /// Create a new equilibrium engine with specified parameters
    pub fn new(
        variance_threshold: Option<f64>,
        max_iterations: Option<usize>,
        convergence_tolerance: Option<f64>,
        target_processing_time_ns: Option<u128>,
        consciousness_threshold: Option<f64>,
    ) -> Self {
        Self {
            variance_threshold: variance_threshold.unwrap_or(VARIANCE_THRESHOLD),
            max_iterations: max_iterations.unwrap_or(1000),
            convergence_tolerance: convergence_tolerance.unwrap_or(1e-8),
            target_processing_time_ns: target_processing_time_ns.unwrap_or(EQUILIBRIUM_CONVERGENCE_TIME_NS as u128),
            consciousness_threshold: consciousness_threshold.unwrap_or(CONSCIOUSNESS_THRESHOLD),
            
            learning_rate: 0.01,
            momentum_factor: 0.9,
            adaptive_lr: true,
            
            equilibrium_history: Vec::new(),
            average_convergence_time: 0.0,
        }
    }

    /// Calculate baseline undisturbed gas molecular equilibrium state
    pub fn calculate_baseline_equilibrium(
        &mut self,
        gas_molecules: &mut Vec<InformationGasMolecule>,
        perturbation_input: Option<&DVector<f64>>,
    ) -> EquilibriumResult {
        let start_time = Instant::now();

        // Create initial system state vector from all molecules
        let initial_state = self.create_system_state_vector(gas_molecules);

        // Find equilibrium through variance minimization
        let mut equilibrium_result = self.minimize_variance_to_equilibrium(
            gas_molecules,
            &initial_state,
            perturbation_input,
        );

        // Calculate system consciousness level
        let mut system = GasMolecularSystem::new(gas_molecules.clone());
        system.update_system_properties();
        equilibrium_result.consciousness_level = system.system_consciousness_level;

        // Track performance
        self.equilibrium_history.push(equilibrium_result.clone());
        self.update_performance_statistics();

        equilibrium_result
    }

    /// Navigate gas molecular system to minimal variance equilibrium configuration
    fn minimize_variance_to_equilibrium(
        &self,
        gas_molecules: &mut Vec<InformationGasMolecule>,
        initial_state: &DVector<f64>,
        _perturbation: Option<&DVector<f64>>,
    ) -> EquilibriumResult {
        let mut current_state = initial_state.clone();
        let mut convergence_history = Vec::new();
        let mut momentum_velocity = DVector::zeros(current_state.len());

        let start_time = Instant::now();

        for iteration in 0..self.max_iterations {
            // Calculate current variance from equilibrium
            let current_variance = self.calculate_system_variance(gas_molecules, &current_state);
            convergence_history.push(current_variance);

            // Check convergence
            if current_variance < self.variance_threshold {
                break;
            }

            // Check time constraint (stop if approaching target time limit)
            let elapsed_time = start_time.elapsed().as_nanos();
            if elapsed_time > (self.target_processing_time_ns as f64 * 0.8) as u128 {
                break;
            }

            // Calculate variance gradient
            let gradient = self.calculate_variance_gradient(gas_molecules, &current_state);

            // Apply momentum-based gradient descent
            momentum_velocity = &momentum_velocity * self.momentum_factor - &gradient * self.learning_rate;
            current_state += &momentum_velocity;

            // Update molecule states
            self.update_molecule_states(gas_molecules, &current_state);

            // Adaptive learning rate adjustment
            if self.adaptive_lr && iteration > 10 && iteration % 50 == 0 {
                self.adapt_learning_rate(&convergence_history);
            }
        }

        let end_time = start_time.elapsed().as_nanos();

        EquilibriumResult {
            equilibrium_state: current_state,
            variance_achieved: convergence_history.last().copied().unwrap_or(f64::INFINITY),
            convergence_time_ns: end_time,
            iteration_count: convergence_history.len(),
            convergence_history,
            consciousness_level: 0.0, // Will be set by caller
            meaning_extracted: None,
        }
    }

    /// Calculate total system variance from equilibrium configuration
    fn calculate_system_variance(
        &self,
        gas_molecules: &[InformationGasMolecule],
        system_state: &DVector<f64>,
    ) -> f64 {
        let mut total_variance = 0.0;
        let mut state_idx = 0;

        for molecule in gas_molecules {
            let molecule_state_size = 9; // Position(3) + velocity(3) + thermo(3)
            
            if state_idx + molecule_state_size <= system_state.len() {
                let molecule_state = system_state.rows(state_idx, molecule_state_size);
                
                // Calculate variance from equilibrium for this molecule
                let equilibrium_target = Vector3::zeros(); // Assuming origin is equilibrium
                let position = Vector3::new(molecule_state[0], molecule_state[1], molecule_state[2]);
                let velocity = Vector3::new(molecule_state[3], molecule_state[4], molecule_state[5]);
                
                let position_variance = (position - equilibrium_target).norm_squared();
                let velocity_variance = velocity.norm_squared();
                
                total_variance += position_variance + velocity_variance;
            }

            state_idx += molecule_state_size;
        }

        total_variance
    }

    /// Calculate gradient of variance with respect to system state
    fn calculate_variance_gradient(
        &self,
        gas_molecules: &[InformationGasMolecule],
        system_state: &DVector<f64>,
    ) -> DVector<f64> {
        let mut gradient = DVector::zeros(system_state.len());
        let epsilon = 1e-8;

        let base_variance = self.calculate_system_variance(gas_molecules, system_state);

        for i in 0..system_state.len() {
            // Forward difference
            let mut perturbed_state = system_state.clone();
            perturbed_state[i] += epsilon;

            let perturbed_variance = self.calculate_system_variance(gas_molecules, &perturbed_state);

            // Numerical gradient
            gradient[i] = (perturbed_variance - base_variance) / epsilon;
        }

        gradient
    }

    /// Create combined state vector from all gas molecules
    fn create_system_state_vector(&self, gas_molecules: &[InformationGasMolecule]) -> DVector<f64> {
        let mut state_vec = Vec::new();

        for molecule in gas_molecules {
            let mol_state = molecule.get_state_vector();
            state_vec.extend(mol_state.iter());
        }

        DVector::from_vec(state_vec)
    }

    /// Update individual molecule states from system state vector
    fn update_molecule_states(&self, gas_molecules: &mut [InformationGasMolecule], system_state: &DVector<f64>) {
        let mut state_idx = 0;

        for molecule in gas_molecules {
            let molecule_state_size = 9; // Position(3) + velocity(3) + thermo(3)
            
            if state_idx + molecule_state_size <= system_state.len() {
                let molecule_state = DVector::from_iterator(
                    molecule_state_size,
                    system_state.iter().skip(state_idx).take(molecule_state_size).copied(),
                );
                molecule.set_state_vector(&molecule_state);
            }
            
            state_idx += molecule_state_size;
        }
    }

    /// Adapt learning rate based on convergence behavior
    fn adapt_learning_rate(&mut self, convergence_history: &[f64]) {
        if convergence_history.len() < 2 {
            return;
        }

        // Check if variance is decreasing
        let recent_progress = convergence_history[convergence_history.len() - 2] - convergence_history[convergence_history.len() - 1];

        if recent_progress > 0.0 {
            // Good progress - increase learning rate slightly
            self.learning_rate *= 1.05;
        } else {
            // Poor progress - decrease learning rate
            self.learning_rate *= 0.95;
        }

        // Keep learning rate in reasonable bounds
        self.learning_rate = self.learning_rate.max(1e-6).min(1.0);
    }

    /// Update performance tracking statistics
    fn update_performance_statistics(&mut self) {
        if self.equilibrium_history.is_empty() {
            return;
        }

        let convergence_times: Vec<f64> = self.equilibrium_history
            .iter()
            .map(|result| result.convergence_time_ns as f64)
            .collect();

        self.average_convergence_time = convergence_times.iter().sum::<f64>() / convergence_times.len() as f64;
    }

    /// Extract meaning from equilibrium configuration
    /// 
    /// Meaning = Equilibrium_After_Perturbation - Unperturbed_Equilibrium
    pub fn extract_meaning_from_equilibrium(
        &self,
        equilibrium_result: &EquilibriumResult,
        baseline_equilibrium: Option<&EquilibriumResult>,
    ) -> MeaningExtraction {
        let meaning_vector = if let Some(baseline) = baseline_equilibrium {
            // Meaning is the difference from baseline
            &equilibrium_result.equilibrium_state - &baseline.equilibrium_state
        } else {
            // If no baseline, meaning is the equilibrium state itself
            equilibrium_result.equilibrium_state.clone()
        };

        let meaning_magnitude = meaning_vector.norm();
        
        // Extract semantic coordinates
        let semantic_coordinates = if meaning_vector.len() >= 6 {
            SemanticCoordinates {
                semantic_x: meaning_vector[0],
                semantic_y: meaning_vector[1],
                semantic_z: meaning_vector[2],
                velocity_x: meaning_vector[3],
                velocity_y: meaning_vector[4],
                velocity_z: meaning_vector[5],
            }
        } else {
            SemanticCoordinates {
                semantic_x: meaning_vector.get(0).copied().unwrap_or(0.0),
                semantic_y: meaning_vector.get(1).copied().unwrap_or(0.0),
                semantic_z: meaning_vector.get(2).copied().unwrap_or(0.0),
                velocity_x: 0.0,
                velocity_y: 0.0,
                velocity_z: 0.0,
            }
        };

        // Assess convergence quality
        let convergence_quality = if equilibrium_result.variance_achieved < self.variance_threshold / 10.0 {
            "excellent".to_string()
        } else if equilibrium_result.variance_achieved < self.variance_threshold {
            "good".to_string()
        } else if equilibrium_result.variance_achieved < self.variance_threshold * 10.0 {
            "acceptable".to_string()
        } else {
            "poor".to_string()
        };

        MeaningExtraction {
            meaning_vector,
            meaning_magnitude,
            variance_reduction: equilibrium_result.variance_achieved,
            consciousness_level: equilibrium_result.consciousness_level,
            processing_time_ns: equilibrium_result.convergence_time_ns,
            convergence_quality,
            semantic_coordinates,
        }
    }

    /// Get performance metrics for the equilibrium engine
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        if self.equilibrium_history.is_empty() {
            return PerformanceMetrics::default();
        }

        let convergence_times: Vec<u128> = self.equilibrium_history
            .iter()
            .map(|r| r.convergence_time_ns)
            .collect();

        let variances: Vec<f64> = self.equilibrium_history
            .iter()
            .map(|r| r.variance_achieved)
            .collect();

        let consciousness_levels: Vec<f64> = self.equilibrium_history
            .iter()
            .map(|r| r.consciousness_level)
            .collect();

        let target_achievement_rate = convergence_times
            .iter()
            .filter(|&&time| time <= self.target_processing_time_ns)
            .count() as f64 / convergence_times.len() as f64;

        let convergence_success_rate = variances
            .iter()
            .filter(|&&variance| variance <= self.variance_threshold)
            .count() as f64 / variances.len() as f64;

        let consciousness_success_rate = consciousness_levels
            .iter()
            .filter(|&&level| level >= self.consciousness_threshold)
            .count() as f64 / consciousness_levels.len() as f64;

        PerformanceMetrics {
            average_convergence_time_ns: convergence_times.iter().sum::<u128>() as f64 / convergence_times.len() as f64,
            min_convergence_time_ns: *convergence_times.iter().min().unwrap_or(&0),
            max_convergence_time_ns: *convergence_times.iter().max().unwrap_or(&0),
            target_achievement_rate,
            average_variance_achieved: variances.iter().sum::<f64>() / variances.len() as f64,
            convergence_success_rate,
            average_consciousness_level: consciousness_levels.iter().sum::<f64>() / consciousness_levels.len() as f64,
            consciousness_success_rate,
            total_equilibrium_calculations: self.equilibrium_history.len(),
        }
    }
}

/// Performance metrics for equilibrium engine
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceMetrics {
    pub average_convergence_time_ns: f64,
    pub min_convergence_time_ns: u128,
    pub max_convergence_time_ns: u128,
    pub target_achievement_rate: f64,
    pub average_variance_achieved: f64,
    pub convergence_success_rate: f64,
    pub average_consciousness_level: f64,
    pub consciousness_success_rate: f64,
    pub total_equilibrium_calculations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::gas_molecular::InformationGasMolecule;
    use nalgebra::Vector3;

    #[test]
    fn test_equilibrium_engine_creation() {
        let engine = EquilibriumEngine::new(None, None, None, None, None);
        
        assert_eq!(engine.variance_threshold, VARIANCE_THRESHOLD);
        assert_eq!(engine.consciousness_threshold, CONSCIOUSNESS_THRESHOLD);
        assert!(engine.learning_rate > 0.0);
    }

    #[test]
    fn test_system_state_vector_creation() {
        let molecules = vec![
            InformationGasMolecule::new(
                3.0, 1.5, 300.0,
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.1, 0.0, 0.0),
                1.0, 1.0, 1.0,
                None,
            ),
            InformationGasMolecule::new(
                2.5, 1.2, 290.0,
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(0.0, 0.1, 0.0),
                1.0, 1.0, 1.0,
                None,
            ),
        ];

        let engine = EquilibriumEngine::new(None, None, None, None, None);
        let state_vector = engine.create_system_state_vector(&molecules);

        // Each molecule contributes 9 values (3 pos + 3 vel + 3 thermo)
        assert_eq!(state_vector.len(), 18);
    }

    #[test]
    fn test_variance_calculation() {
        let molecules = vec![
            InformationGasMolecule::new(
                2.0, 1.0, 280.0,
                Vector3::new(0.5, 0.5, 0.5),
                Vector3::new(0.1, 0.1, 0.1),
                1.0, 1.0, 1.0,
                None,
            ),
        ];

        let engine = EquilibriumEngine::new(None, None, None, None, None);
        let state_vector = engine.create_system_state_vector(&molecules);
        let variance = engine.calculate_system_variance(&molecules, &state_vector);

        assert!(variance >= 0.0);
    }

    #[test]
    fn test_equilibrium_calculation() {
        let mut molecules = vec![
            InformationGasMolecule::new(
                1.5, 0.8, 250.0,
                Vector3::new(0.2, 0.2, 0.2),
                Vector3::new(0.05, 0.05, 0.05),
                1.0, 1.0, 1.0,
                None,
            ),
        ];

        let mut engine = EquilibriumEngine::new(
            Some(1e-4), // Relaxed threshold for test
            Some(100),  // Limited iterations for test
            None, 
            Some(1_000_000), // 1ms target for test
            Some(0.3), // Lower consciousness threshold
        );

        let result = engine.calculate_baseline_equilibrium(&mut molecules, None);

        assert!(result.variance_achieved >= 0.0);
        assert!(result.consciousness_level >= 0.0 && result.consciousness_level <= 1.0);
        assert!(result.convergence_time_ns > 0);
        assert!(!result.convergence_history.is_empty());
    }

    #[test]
    fn test_meaning_extraction() {
        let equilibrium_result = EquilibriumResult {
            equilibrium_state: DVector::from_vec(vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 5.0, 2.0, 300.0]),
            variance_achieved: 1e-5,
            convergence_time_ns: 15_000,
            iteration_count: 50,
            convergence_history: vec![1.0, 0.5, 1e-5],
            consciousness_level: 0.75,
            meaning_extracted: None,
        };

        let engine = EquilibriumEngine::new(None, None, None, None, None);
        let meaning = engine.extract_meaning_from_equilibrium(&equilibrium_result, None);

        assert!(meaning.meaning_magnitude > 0.0);
        assert_eq!(meaning.convergence_quality, "excellent");
        assert_eq!(meaning.semantic_coordinates.semantic_x, 1.0);
        assert_eq!(meaning.semantic_coordinates.velocity_x, 0.1);
    }
}
