//! # Oscillatory Field
//!
//! Implementation of the fundamental oscillatory field processing engine.
//! This module implements the core oscillatory equation and provides the foundation
//! for all oscillatory reality processing.
//!
//! ## Mathematical Foundation
//!
//! The fundamental oscillatory equation:
//! ```
//! ∂²Φ/∂t² + ω²Φ = 𝒩[Φ] + 𝒞[Φ]
//! ```
//!
//! Where:
//! - `Φ` = Oscillatory field
//! - `ω` = Oscillatory frequency
//! - `𝒩[Φ]` = Nonlinear self-interaction terms
//! - `𝒞[Φ]` = Coherence enhancement terms

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::oscillatory::{
    OscillatoryResult, OscillatoryError, OscillatoryFieldType, 
    OscillatoryProcessingMode, CosmologicalStructureType
};

/// Oscillatory Field - Fundamental oscillatory processing engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryField {
    /// Field identifier
    pub id: String,
    
    /// Field type
    pub field_type: OscillatoryFieldType,
    
    /// Oscillatory frequency (ω)
    pub frequency: f64,
    
    /// Current field values (Φ)
    pub field_values: Vec<f64>,
    
    /// Field derivatives (∂Φ/∂t)
    pub field_derivatives: Vec<f64>,
    
    /// Second derivatives (∂²Φ/∂t²)
    pub field_second_derivatives: Vec<f64>,
    
    /// Nonlinear interaction terms (𝒩[Φ])
    pub nonlinear_terms: Vec<f64>,
    
    /// Coherence enhancement terms (𝒞[Φ])
    pub coherence_terms: Vec<f64>,
    
    /// Field dimensions
    pub dimensions: (usize, usize),
    
    /// Processing mode
    pub processing_mode: OscillatoryProcessingMode,
    
    /// Cosmological structure mapping
    pub cosmological_structure: HashMap<CosmologicalStructureType, Vec<usize>>,
    
    /// Field processing statistics
    pub processing_stats: OscillatoryFieldStats,
    
    /// Time evolution parameters
    pub time_evolution: TimeEvolutionParameters,
}

/// Time evolution parameters for oscillatory field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeEvolutionParameters {
    /// Current time
    pub current_time: f64,
    
    /// Time step
    pub time_step: f64,
    
    /// Evolution iterations
    pub evolution_iterations: u64,
    
    /// Stability threshold
    pub stability_threshold: f64,
    
    /// Convergence criteria
    pub convergence_criteria: f64,
}

/// Oscillatory field processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryFieldStats {
    /// Field evolution steps
    pub evolution_steps: u64,
    
    /// Nonlinear interactions processed
    pub nonlinear_interactions: u64,
    
    /// Coherence enhancements applied
    pub coherence_enhancements: u64,
    
    /// Computational reduction achieved
    pub computational_reduction: f64,
    
    /// Field coherence level
    pub field_coherence: f64,
    
    /// Processing efficiency
    pub processing_efficiency: f64,
    
    /// Energy conservation level
    pub energy_conservation: f64,
}

/// Oscillatory field configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryFieldConfig {
    /// Field dimensions
    pub dimensions: (usize, usize),
    
    /// Initial frequency
    pub initial_frequency: f64,
    
    /// Field type
    pub field_type: OscillatoryFieldType,
    
    /// Processing mode
    pub processing_mode: OscillatoryProcessingMode,
    
    /// Nonlinear interaction strength
    pub nonlinear_strength: f64,
    
    /// Coherence enhancement strength
    pub coherence_strength: f64,
    
    /// Time evolution parameters
    pub time_evolution: TimeEvolutionParameters,
}

impl OscillatoryField {
    /// Create new oscillatory field
    pub fn new(id: String, config: OscillatoryFieldConfig) -> OscillatoryResult<Self> {
        let total_points = config.dimensions.0 * config.dimensions.1;
        
        if total_points == 0 {
            return Err(OscillatoryError::OscillatoryFieldInitializationFailed(
                "Field dimensions cannot be zero".to_string()
            ));
        }

        let field = Self {
            id,
            field_type: config.field_type,
            frequency: config.initial_frequency,
            field_values: vec![0.0; total_points],
            field_derivatives: vec![0.0; total_points],
            field_second_derivatives: vec![0.0; total_points],
            nonlinear_terms: vec![0.0; total_points],
            coherence_terms: vec![0.0; total_points],
            dimensions: config.dimensions,
            processing_mode: config.processing_mode,
            cosmological_structure: HashMap::new(),
            processing_stats: OscillatoryFieldStats::default(),
            time_evolution: config.time_evolution,
        };

        Ok(field)
    }

    /// Initialize field with oscillatory pattern
    pub fn initialize_oscillatory_pattern(&mut self, pattern: &[f64]) -> OscillatoryResult<()> {
        if pattern.len() != self.field_values.len() {
            return Err(OscillatoryError::OscillatoryFieldInitializationFailed(
                format!("Pattern length {} does not match field size {}", 
                       pattern.len(), self.field_values.len())
            ));
        }

        // Initialize field values with pattern
        self.field_values.copy_from_slice(pattern);

        // Initialize cosmological structure
        self.initialize_cosmological_structure()?;

        // Calculate initial derivatives
        self.calculate_field_derivatives()?;

        Ok(())
    }

    /// Initialize cosmological structure mapping
    fn initialize_cosmological_structure(&mut self) -> OscillatoryResult<()> {
        let total_points = self.field_values.len();
        
        // 95% dark matter/energy (unoccupied oscillatory modes)
        let dark_matter_count = (total_points as f64 * 
                               crate::oscillatory::constants::DARK_MATTER_PERCENTAGE) as usize;
        
        // 5% ordinary matter (coherent oscillatory confluences)
        let ordinary_matter_count = (total_points as f64 * 
                                   crate::oscillatory::constants::ORDINARY_MATTER_PERCENTAGE) as usize;
        
        // 0.01% sequential states (processed by consciousness)
        let sequential_states_count = (total_points as f64 * 
                                     crate::oscillatory::constants::SEQUENTIAL_STATES_PERCENTAGE) as usize;

        // Assign indices to cosmological structures
        let mut dark_matter_indices = Vec::new();
        let mut ordinary_matter_indices = Vec::new();
        let mut sequential_state_indices = Vec::new();

        // Dark matter indices (first 95%)
        for i in 0..dark_matter_count {
            dark_matter_indices.push(i);
        }

        // Ordinary matter indices (next 5%)
        for i in dark_matter_count..(dark_matter_count + ordinary_matter_count) {
            ordinary_matter_indices.push(i);
        }

        // Sequential state indices (remaining 0.01%)
        for i in (dark_matter_count + ordinary_matter_count)..
                 (dark_matter_count + ordinary_matter_count + sequential_states_count) {
            sequential_state_indices.push(i);
        }

        // Store cosmological structure mappings
        self.cosmological_structure.insert(
            CosmologicalStructureType::DarkMatter, 
            dark_matter_indices
        );
        self.cosmological_structure.insert(
            CosmologicalStructureType::OrdinaryMatter, 
            ordinary_matter_indices
        );
        self.cosmological_structure.insert(
            CosmologicalStructureType::SequentialState, 
            sequential_state_indices
        );

        Ok(())
    }

    /// Evolve oscillatory field according to fundamental equation
    pub fn evolve_field(&mut self, time_steps: u64) -> OscillatoryResult<()> {
        for _ in 0..time_steps {
            // Calculate current derivatives
            self.calculate_field_derivatives()?;

            // Calculate nonlinear interaction terms
            self.calculate_nonlinear_terms()?;

            // Calculate coherence enhancement terms
            self.calculate_coherence_terms()?;

            // Apply fundamental oscillatory equation
            self.apply_oscillatory_equation()?;

            // Update time evolution
            self.update_time_evolution();

            // Update processing statistics
            self.update_processing_stats();
        }

        Ok(())
    }

    /// Calculate field derivatives (∂Φ/∂t and ∂²Φ/∂t²)
    fn calculate_field_derivatives(&mut self) -> OscillatoryResult<()> {
        let (width, height) = self.dimensions;
        
        // Calculate first derivatives (∂Φ/∂t)
        for i in 0..self.field_values.len() {
            let row = i / width;
            let col = i % width;
            
            // Spatial derivatives approximation
            let mut derivative = 0.0;
            
            // Left neighbor
            if col > 0 {
                derivative += self.field_values[i - 1];
            }
            
            // Right neighbor
            if col < width - 1 {
                derivative += self.field_values[i + 1];
            }
            
            // Top neighbor
            if row > 0 {
                derivative += self.field_values[i - width];
            }
            
            // Bottom neighbor
            if row < height - 1 {
                derivative += self.field_values[i + width];
            }
            
            // Central difference approximation
            derivative = (derivative - 4.0 * self.field_values[i]) / 4.0;
            
            self.field_derivatives[i] = derivative;
        }

        // Calculate second derivatives (∂²Φ/∂t²)
        for i in 0..self.field_derivatives.len() {
            let row = i / width;
            let col = i % width;
            
            let mut second_derivative = 0.0;
            
            // Similar spatial second derivative calculation
            if col > 0 {
                second_derivative += self.field_derivatives[i - 1];
            }
            
            if col < width - 1 {
                second_derivative += self.field_derivatives[i + 1];
            }
            
            if row > 0 {
                second_derivative += self.field_derivatives[i - width];
            }
            
            if row < height - 1 {
                second_derivative += self.field_derivatives[i + width];
            }
            
            second_derivative = (second_derivative - 4.0 * self.field_derivatives[i]) / 4.0;
            
            self.field_second_derivatives[i] = second_derivative;
        }

        Ok(())
    }

    /// Calculate nonlinear interaction terms (𝒩[Φ])
    fn calculate_nonlinear_terms(&mut self) -> OscillatoryResult<()> {
        for i in 0..self.field_values.len() {
            let phi = self.field_values[i];
            
            // Nonlinear self-interaction term
            // 𝒩[Φ] = -λΦ³ + μΦ⁵ (example nonlinear terms)
            let lambda = crate::oscillatory::constants::NONLINEAR_INTERACTION_STRENGTH;
            let mu = lambda * 0.1;
            
            self.nonlinear_terms[i] = -lambda * phi.powi(3) + mu * phi.powi(5);
        }

        self.processing_stats.nonlinear_interactions += 1;
        Ok(())
    }

    /// Calculate coherence enhancement terms (𝒞[Φ])
    fn calculate_coherence_terms(&mut self) -> OscillatoryResult<()> {
        let (width, height) = self.dimensions;
        
        for i in 0..self.field_values.len() {
            let row = i / width;
            let col = i % width;
            
            // Calculate local coherence
            let mut local_coherence = 0.0;
            let mut neighbor_count = 0;
            
            // Check all neighbors
            for dr in -1..=1 {
                for dc in -1..=1 {
                    if dr == 0 && dc == 0 { continue; }
                    
                    let nr = row as i32 + dr;
                    let nc = col as i32 + dc;
                    
                    if nr >= 0 && nr < height as i32 && nc >= 0 && nc < width as i32 {
                        let ni = nr as usize * width + nc as usize;
                        local_coherence += self.field_values[ni] * self.field_values[i];
                        neighbor_count += 1;
                    }
                }
            }
            
            if neighbor_count > 0 {
                local_coherence /= neighbor_count as f64;
            }
            
            // Coherence enhancement term
            let coherence_threshold = crate::oscillatory::constants::COHERENCE_ENHANCEMENT_THRESHOLD;
            self.coherence_terms[i] = if local_coherence > coherence_threshold {
                local_coherence * 0.1 // Enhance coherent regions
            } else {
                -local_coherence * 0.05 // Suppress incoherent regions
            };
        }

        self.processing_stats.coherence_enhancements += 1;
        Ok(())
    }

    /// Apply fundamental oscillatory equation
    fn apply_oscillatory_equation(&mut self) -> OscillatoryResult<()> {
        let dt = self.time_evolution.time_step;
        let omega_squared = self.frequency * self.frequency;
        
        for i in 0..self.field_values.len() {
            // Apply the fundamental oscillatory equation:
            // ∂²Φ/∂t² + ω²Φ = 𝒩[Φ] + 𝒞[Φ]
            let acceleration = -omega_squared * self.field_values[i] + 
                              self.nonlinear_terms[i] + 
                              self.coherence_terms[i];
            
            // Update field using Verlet integration
            let new_value = 2.0 * self.field_values[i] - 
                           (self.field_values[i] - self.field_derivatives[i] * dt) + 
                           acceleration * dt * dt;
            
            self.field_values[i] = new_value;
        }

        Ok(())
    }

    /// Update time evolution parameters
    fn update_time_evolution(&mut self) {
        self.time_evolution.current_time += self.time_evolution.time_step;
        self.time_evolution.evolution_iterations += 1;
    }

    /// Update processing statistics
    fn update_processing_stats(&mut self) {
        self.processing_stats.evolution_steps += 1;
        
        // Calculate field coherence
        self.processing_stats.field_coherence = self.calculate_field_coherence();
        
        // Calculate computational reduction achieved
        self.processing_stats.computational_reduction = 
            self.calculate_computational_reduction();
        
        // Calculate processing efficiency
        self.processing_stats.processing_efficiency = 
            self.calculate_processing_efficiency();
        
        // Calculate energy conservation
        self.processing_stats.energy_conservation = 
            self.calculate_energy_conservation();
    }

    /// Calculate overall field coherence
    fn calculate_field_coherence(&self) -> f64 {
        let mut total_coherence = 0.0;
        let mut coherent_points = 0;
        
        for &coherence_term in &self.coherence_terms {
            if coherence_term > 0.0 {
                total_coherence += coherence_term;
                coherent_points += 1;
            }
        }
        
        if coherent_points > 0 {
            total_coherence / coherent_points as f64
        } else {
            0.0
        }
    }

    /// Calculate computational reduction achieved
    fn calculate_computational_reduction(&self) -> f64 {
        match self.processing_mode {
            OscillatoryProcessingMode::Approximation => {
                crate::oscillatory::constants::COMPUTATIONAL_REDUCTION_FACTOR
            }
            OscillatoryProcessingMode::SequentialStatesOnly => {
                1.0 / crate::oscillatory::constants::SEQUENTIAL_STATES_PERCENTAGE
            }
            OscillatoryProcessingMode::OrdinaryMatterOnly => {
                1.0 / crate::oscillatory::constants::ORDINARY_MATTER_PERCENTAGE
            }
            OscillatoryProcessingMode::DarkMatterOnly => {
                1.0 / crate::oscillatory::constants::DARK_MATTER_PERCENTAGE
            }
            OscillatoryProcessingMode::Full => 1.0,
        }
    }

    /// Calculate processing efficiency
    fn calculate_processing_efficiency(&self) -> f64 {
        let coherence_factor = self.processing_stats.field_coherence;
        let reduction_factor = self.processing_stats.computational_reduction;
        
        (coherence_factor * reduction_factor.log10()) / 10.0
    }

    /// Calculate energy conservation level
    fn calculate_energy_conservation(&self) -> f64 {
        // Calculate total energy: kinetic + potential
        let kinetic_energy: f64 = self.field_derivatives.iter()
            .map(|&v| v * v)
            .sum::<f64>() * 0.5;
        
        let potential_energy: f64 = self.field_values.iter()
            .map(|&v| 0.5 * self.frequency * self.frequency * v * v)
            .sum::<f64>();
        
        let total_energy = kinetic_energy + potential_energy;
        
        // Energy conservation (normalized)
        if total_energy > 0.0 {
            (1.0 / (1.0 + (total_energy - 1.0).abs())).min(1.0)
        } else {
            0.0
        }
    }

    /// Process oscillatory field based on cosmological structure
    pub fn process_cosmological_structure(&mut self, 
                                        structure_type: CosmologicalStructureType) -> OscillatoryResult<Vec<f64>> {
        let indices = self.cosmological_structure.get(&structure_type)
            .ok_or_else(|| OscillatoryError::CosmologicalStructureInconsistency(
                format!("Structure type {:?} not found", structure_type)
            ))?;

        let mut processed_values = Vec::new();
        
        for &index in indices {
            if index < self.field_values.len() {
                processed_values.push(self.field_values[index]);
            }
        }

        Ok(processed_values)
    }

    /// Get field capabilities
    pub fn get_field_capabilities(&self) -> OscillatoryFieldCapabilities {
        OscillatoryFieldCapabilities {
            field_type: self.field_type.clone(),
            processing_mode: self.processing_mode.clone(),
            dimensions: self.dimensions,
            frequency: self.frequency,
            field_coherence: self.processing_stats.field_coherence,
            computational_reduction: self.processing_stats.computational_reduction,
            processing_efficiency: self.processing_stats.processing_efficiency,
            energy_conservation: self.processing_stats.energy_conservation,
        }
    }
}

/// Oscillatory field capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryFieldCapabilities {
    pub field_type: OscillatoryFieldType,
    pub processing_mode: OscillatoryProcessingMode,
    pub dimensions: (usize, usize),
    pub frequency: f64,
    pub field_coherence: f64,
    pub computational_reduction: f64,
    pub processing_efficiency: f64,
    pub energy_conservation: f64,
}

// Default implementations
impl Default for TimeEvolutionParameters {
    fn default() -> Self {
        Self {
            current_time: 0.0,
            time_step: 0.01,
            evolution_iterations: 0,
            stability_threshold: 0.01,
            convergence_criteria: 1e-6,
        }
    }
}

impl Default for OscillatoryFieldStats {
    fn default() -> Self {
        Self {
            evolution_steps: 0,
            nonlinear_interactions: 0,
            coherence_enhancements: 0,
            computational_reduction: 1.0,
            field_coherence: 0.0,
            processing_efficiency: 0.0,
            energy_conservation: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oscillatory_field_creation() {
        let config = OscillatoryFieldConfig {
            dimensions: (10, 10),
            initial_frequency: 1.0,
            field_type: OscillatoryFieldType::Continuous,
            processing_mode: OscillatoryProcessingMode::Full,
            nonlinear_strength: 0.1,
            coherence_strength: 0.1,
            time_evolution: TimeEvolutionParameters::default(),
        };

        let field = OscillatoryField::new("FIELD_001".to_string(), config);
        assert!(field.is_ok());

        let field = field.unwrap();
        assert_eq!(field.id, "FIELD_001");
        assert_eq!(field.dimensions, (10, 10));
        assert_eq!(field.field_values.len(), 100);
    }

    #[test]
    fn test_oscillatory_pattern_initialization() {
        let config = OscillatoryFieldConfig {
            dimensions: (5, 5),
            initial_frequency: 1.0,
            field_type: OscillatoryFieldType::Continuous,
            processing_mode: OscillatoryProcessingMode::Full,
            nonlinear_strength: 0.1,
            coherence_strength: 0.1,
            time_evolution: TimeEvolutionParameters::default(),
        };

        let mut field = OscillatoryField::new("FIELD_TEST".to_string(), config).unwrap();
        let pattern = vec![0.5; 25];

        let result = field.initialize_oscillatory_pattern(&pattern);
        assert!(result.is_ok());

        assert_eq!(field.field_values, pattern);
    }

    #[test]
    fn test_field_evolution() {
        let config = OscillatoryFieldConfig {
            dimensions: (5, 5),
            initial_frequency: 1.0,
            field_type: OscillatoryFieldType::Continuous,
            processing_mode: OscillatoryProcessingMode::Full,
            nonlinear_strength: 0.1,
            coherence_strength: 0.1,
            time_evolution: TimeEvolutionParameters::default(),
        };

        let mut field = OscillatoryField::new("FIELD_EVOLUTION".to_string(), config).unwrap();
        let pattern = vec![0.5; 25];
        field.initialize_oscillatory_pattern(&pattern).unwrap();

        let result = field.evolve_field(10);
        assert!(result.is_ok());

        assert_eq!(field.processing_stats.evolution_steps, 10);
        assert!(field.processing_stats.nonlinear_interactions > 0);
        assert!(field.processing_stats.coherence_enhancements > 0);
    }

    #[test]
    fn test_cosmological_structure_processing() {
        let config = OscillatoryFieldConfig {
            dimensions: (10, 10),
            initial_frequency: 1.0,
            field_type: OscillatoryFieldType::Continuous,
            processing_mode: OscillatoryProcessingMode::Approximation,
            nonlinear_strength: 0.1,
            coherence_strength: 0.1,
            time_evolution: TimeEvolutionParameters::default(),
        };

        let mut field = OscillatoryField::new("FIELD_COSMOLOGICAL".to_string(), config).unwrap();
        let pattern = vec![0.5; 100];
        field.initialize_oscillatory_pattern(&pattern).unwrap();

        let dark_matter_result = field.process_cosmological_structure(
            CosmologicalStructureType::DarkMatter
        );
        assert!(dark_matter_result.is_ok());

        let ordinary_matter_result = field.process_cosmological_structure(
            CosmologicalStructureType::OrdinaryMatter
        );
        assert!(ordinary_matter_result.is_ok());

        // Dark matter should have more values than ordinary matter (95% vs 5%)
        assert!(dark_matter_result.unwrap().len() > ordinary_matter_result.unwrap().len());
    }
} 