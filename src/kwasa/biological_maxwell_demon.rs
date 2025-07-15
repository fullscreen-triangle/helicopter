//! # Biological Maxwell's Demon (BMD)
//!
//! Core information catalyst that discretizes continuous oscillatory reality into named semantic units.
//! BMDs operate through fire-adapted consciousness enhancements to provide 322% processing improvements.
//!
//! ## Mathematical Foundation
//!
//! A BMD operates through the semantic catalysis function:
//! ```
//! iCat_semantic = ℑ_input ○ ℑ_output ○ ℑ_agency
//! ```
//!
//! ## Processing Levels
//!
//! BMDs operate at three distinct scales:
//! - **Molecular Level**: Token/phoneme processing through character recognition
//! - **Neural Level**: Sentence structure processing through syntax/semantic parsing
//! - **Cognitive Level**: Discourse processing through contextual integration

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::kwasa::{KwasaResult, KwasaError};

/// Biological Maxwell's Demon - Core information catalyst
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalMaxwellDemon {
    /// Unique identifier for this BMD
    pub id: String,
    
    /// Processing level (molecular, neural, cognitive)
    pub processing_level: BmdProcessingLevel,
    
    /// Current consciousness threshold
    pub consciousness_threshold: f64,
    
    /// Fire-adapted enhancement factor
    pub fire_adapted_enhancement: f64,
    
    /// Semantic catalysis state
    pub catalysis_state: SemanticCatalysisState,
    
    /// Naming function mappings
    pub naming_functions: HashMap<String, String>,
    
    /// Agency assertion capability
    pub agency_assertion_strength: f64,
    
    /// Information processing statistics
    pub processing_stats: BmdProcessingStats,
    
    /// Pattern recognition improvement factor
    pub pattern_recognition_factor: f64,
    
    /// Communication complexity enhancement
    pub communication_enhancement: f64,
}

/// Processing levels for BMDs
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BmdProcessingLevel {
    /// Molecular-level processing (tokens/phonemes)
    Molecular,
    /// Neural-level processing (sentence structures)
    Neural,
    /// Cognitive-level processing (discourse/context)
    Cognitive,
}

/// Semantic catalysis state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCatalysisState {
    /// Input pattern recognition state
    pub input_recognition: f64,
    /// Output channeling coordination
    pub output_coordination: f64,
    /// Agency assertion level
    pub agency_assertion: f64,
    /// Overall catalysis effectiveness
    pub catalysis_effectiveness: f64,
}

/// BMD processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmdProcessingStats {
    /// Total semantic units processed
    pub semantic_units_processed: u64,
    /// Reality discretization events
    pub reality_discretizations: u64,
    /// Successful naming function applications
    pub naming_function_applications: u64,
    /// Agency assertion events
    pub agency_assertions: u64,
    /// Processing efficiency (0.0-1.0)
    pub processing_efficiency: f64,
    /// Fire-adapted enhancements applied
    pub fire_adapted_enhancements: u64,
}

impl BiologicalMaxwellDemon {
    /// Create a new BMD with specified processing level
    pub fn new(
        id: String,
        processing_level: BmdProcessingLevel,
        fire_adapted_enhancement: f64,
    ) -> Self {
        Self {
            id,
            processing_level,
            consciousness_threshold: crate::kwasa::constants::FIRE_ADAPTED_CONSCIOUSNESS_THRESHOLD,
            fire_adapted_enhancement,
            catalysis_state: SemanticCatalysisState {
                input_recognition: 0.0,
                output_coordination: 0.0,
                agency_assertion: 0.0,
                catalysis_effectiveness: 0.0,
            },
            naming_functions: HashMap::new(),
            agency_assertion_strength: 0.0,
            processing_stats: BmdProcessingStats {
                semantic_units_processed: 0,
                reality_discretizations: 0,
                naming_function_applications: 0,
                agency_assertions: 0,
                processing_efficiency: 0.0,
                fire_adapted_enhancements: 0,
            },
            pattern_recognition_factor: crate::kwasa::constants::PATTERN_RECOGNITION_IMPROVEMENT,
            communication_enhancement: crate::kwasa::constants::COMMUNICATION_COMPLEXITY_ENHANCEMENT,
        }
    }

    /// Process continuous oscillatory input through semantic catalysis
    pub fn process_oscillatory_input(&mut self, oscillatory_data: &[f64]) -> KwasaResult<Vec<String>> {
        // Check consciousness threshold
        if self.consciousness_threshold < crate::kwasa::constants::FIRE_ADAPTED_CONSCIOUSNESS_THRESHOLD {
            return Err(KwasaError::ConsciousnessThresholdNotReached(self.consciousness_threshold));
        }

        // Apply fire-adapted enhancement
        let enhanced_data = self.apply_fire_adapted_enhancement(oscillatory_data)?;

        // Perform semantic catalysis
        let semantic_units = self.semantic_catalysis(&enhanced_data)?;

        // Apply naming functions
        let named_units = self.apply_naming_functions(&semantic_units)?;

        // Update processing statistics
        self.update_processing_stats(&named_units);

        Ok(named_units)
    }

    /// Apply fire-adapted consciousness enhancement
    fn apply_fire_adapted_enhancement(&mut self, data: &[f64]) -> KwasaResult<Vec<f64>> {
        let mut enhanced_data = Vec::with_capacity(data.len());

        for &value in data {
            // Apply fire-adapted processing improvement (322%)
            let enhanced_value = value * self.fire_adapted_enhancement * 
                crate::kwasa::constants::FIRE_ADAPTED_PROCESSING_IMPROVEMENT;
            enhanced_data.push(enhanced_value);
        }

        self.processing_stats.fire_adapted_enhancements += 1;
        Ok(enhanced_data)
    }

    /// Perform semantic catalysis on enhanced data
    fn semantic_catalysis(&mut self, data: &[f64]) -> KwasaResult<Vec<String>> {
        let mut semantic_units = Vec::new();

        // Process data based on BMD level
        match self.processing_level {
            BmdProcessingLevel::Molecular => {
                semantic_units = self.molecular_level_processing(data)?;
            }
            BmdProcessingLevel::Neural => {
                semantic_units = self.neural_level_processing(data)?;
            }
            BmdProcessingLevel::Cognitive => {
                semantic_units = self.cognitive_level_processing(data)?;
            }
        }

        // Update catalysis state
        self.update_catalysis_state(&semantic_units);

        Ok(semantic_units)
    }

    /// Molecular-level processing (tokens/phonemes)
    fn molecular_level_processing(&self, data: &[f64]) -> KwasaResult<Vec<String>> {
        let mut tokens = Vec::new();

        for (i, &value) in data.iter().enumerate() {
            // Convert oscillatory value to discrete token
            let token = if value > 0.7 {
                format!("HIGH_ENERGY_TOKEN_{}", i)
            } else if value > 0.3 {
                format!("MED_ENERGY_TOKEN_{}", i)
            } else {
                format!("LOW_ENERGY_TOKEN_{}", i)
            };
            tokens.push(token);
        }

        Ok(tokens)
    }

    /// Neural-level processing (sentence structures)
    fn neural_level_processing(&self, data: &[f64]) -> KwasaResult<Vec<String>> {
        let mut structures = Vec::new();

        // Group data into syntactic structures
        for chunk in data.chunks(5) {
            let avg_energy = chunk.iter().sum::<f64>() / chunk.len() as f64;
            let structure = if avg_energy > 0.6 {
                format!("COMPLEX_STRUCTURE_{:.3}", avg_energy)
            } else if avg_energy > 0.4 {
                format!("SIMPLE_STRUCTURE_{:.3}", avg_energy)
            } else {
                format!("BASIC_STRUCTURE_{:.3}", avg_energy)
            };
            structures.push(structure);
        }

        Ok(structures)
    }

    /// Cognitive-level processing (discourse/context)
    fn cognitive_level_processing(&self, data: &[f64]) -> KwasaResult<Vec<String>> {
        let mut contexts = Vec::new();

        // Process contextual information
        let total_energy: f64 = data.iter().sum();
        let avg_energy = total_energy / data.len() as f64;
        let coherence = self.calculate_coherence(data);

        let context = if coherence > 0.8 && avg_energy > 0.7 {
            format!("HIGH_COHERENCE_CONTEXT_{}_{:.3}", data.len(), coherence)
        } else if coherence > 0.5 && avg_energy > 0.4 {
            format!("MED_COHERENCE_CONTEXT_{}_{:.3}", data.len(), coherence)
        } else {
            format!("LOW_COHERENCE_CONTEXT_{}_{:.3}", data.len(), coherence)
        };

        contexts.push(context);
        Ok(contexts)
    }

    /// Calculate coherence of oscillatory data
    fn calculate_coherence(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();

        // Higher coherence means lower variance relative to mean
        if std_dev == 0.0 {
            1.0
        } else {
            (1.0 / (1.0 + std_dev / mean.abs())).min(1.0).max(0.0)
        }
    }

    /// Apply naming functions to semantic units
    fn apply_naming_functions(&mut self, semantic_units: &[String]) -> KwasaResult<Vec<String>> {
        let mut named_units = Vec::new();

        for unit in semantic_units {
            // Check if we have a naming function for this unit type
            let base_type = unit.split('_').next().unwrap_or(unit);
            
            let named_unit = if let Some(name) = self.naming_functions.get(base_type) {
                format!("{}:{}", name, unit)
            } else {
                // Create new naming function
                let name = format!("BMD_{}_{}", self.id, base_type);
                self.naming_functions.insert(base_type.to_string(), name.clone());
                format!("{}:{}", name, unit)
            };

            named_units.push(named_unit);
        }

        self.processing_stats.naming_function_applications += named_units.len() as u64;
        Ok(named_units)
    }

    /// Update catalysis state based on processing results
    fn update_catalysis_state(&mut self, semantic_units: &[String]) {
        let units_count = semantic_units.len() as f64;
        
        // Update input recognition based on successful unit generation
        self.catalysis_state.input_recognition = (units_count / 10.0).min(1.0);
        
        // Update output coordination based on naming function success
        self.catalysis_state.output_coordination = 
            (self.naming_functions.len() as f64 / 100.0).min(1.0);
        
        // Update agency assertion based on fire-adapted enhancement
        self.catalysis_state.agency_assertion = 
            (self.fire_adapted_enhancement / 10.0).min(1.0);
        
        // Calculate overall effectiveness
        self.catalysis_state.catalysis_effectiveness = 
            (self.catalysis_state.input_recognition + 
             self.catalysis_state.output_coordination + 
             self.catalysis_state.agency_assertion) / 3.0;
    }

    /// Update processing statistics
    fn update_processing_stats(&mut self, named_units: &[String]) {
        self.processing_stats.semantic_units_processed += named_units.len() as u64;
        self.processing_stats.reality_discretizations += 1;
        
        // Calculate processing efficiency
        let total_operations = self.processing_stats.semantic_units_processed + 
                               self.processing_stats.reality_discretizations;
        let successful_operations = self.processing_stats.naming_function_applications + 
                                    self.processing_stats.agency_assertions;
        
        self.processing_stats.processing_efficiency = 
            if total_operations > 0 {
                successful_operations as f64 / total_operations as f64
            } else {
                0.0
            };
    }

    /// Assert agency to modify reality
    pub fn assert_agency(&mut self, modification_request: &str) -> KwasaResult<String> {
        // Check if consciousness threshold allows agency assertion
        if self.consciousness_threshold < crate::kwasa::constants::FIRE_ADAPTED_CONSCIOUSNESS_THRESHOLD {
            return Err(KwasaError::AgencyAssertionFailed(
                "Consciousness threshold too low for agency assertion".to_string()
            ));
        }

        // Apply agency assertion strength
        let enhanced_modification = format!(
            "BMD_{}:AGENCY_ASSERTION[{}]:{}",
            self.id,
            self.agency_assertion_strength,
            modification_request
        );

        self.processing_stats.agency_assertions += 1;
        self.agency_assertion_strength += 0.1; // Increase with each assertion

        Ok(enhanced_modification)
    }

    /// Get current processing capabilities
    pub fn get_processing_capabilities(&self) -> BmdCapabilities {
        BmdCapabilities {
            consciousness_threshold: self.consciousness_threshold,
            fire_adapted_enhancement: self.fire_adapted_enhancement,
            pattern_recognition_factor: self.pattern_recognition_factor,
            communication_enhancement: self.communication_enhancement,
            processing_efficiency: self.processing_stats.processing_efficiency,
            catalysis_effectiveness: self.catalysis_state.catalysis_effectiveness,
        }
    }
}

/// BMD processing capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmdCapabilities {
    pub consciousness_threshold: f64,
    pub fire_adapted_enhancement: f64,
    pub pattern_recognition_factor: f64,
    pub communication_enhancement: f64,
    pub processing_efficiency: f64,
    pub catalysis_effectiveness: f64,
}

impl Default for SemanticCatalysisState {
    fn default() -> Self {
        Self {
            input_recognition: 0.0,
            output_coordination: 0.0,
            agency_assertion: 0.0,
            catalysis_effectiveness: 0.0,
        }
    }
}

impl Default for BmdProcessingStats {
    fn default() -> Self {
        Self {
            semantic_units_processed: 0,
            reality_discretizations: 0,
            naming_function_applications: 0,
            agency_assertions: 0,
            processing_efficiency: 0.0,
            fire_adapted_enhancements: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bmd_creation() {
        let bmd = BiologicalMaxwellDemon::new(
            "BMD_001".to_string(),
            BmdProcessingLevel::Molecular,
            1.0,
        );

        assert_eq!(bmd.id, "BMD_001");
        assert_eq!(bmd.processing_level, BmdProcessingLevel::Molecular);
        assert_eq!(bmd.consciousness_threshold, 0.61);
    }

    #[test]
    fn test_oscillatory_processing() {
        let mut bmd = BiologicalMaxwellDemon::new(
            "BMD_TEST".to_string(),
            BmdProcessingLevel::Molecular,
            2.0,
        );

        let test_data = vec![0.8, 0.5, 0.2, 0.9, 0.1];
        let result = bmd.process_oscillatory_input(&test_data);

        assert!(result.is_ok());
        let named_units = result.unwrap();
        assert_eq!(named_units.len(), 5);
    }

    #[test]
    fn test_coherence_calculation() {
        let bmd = BiologicalMaxwellDemon::new(
            "BMD_COHERENCE".to_string(),
            BmdProcessingLevel::Cognitive,
            1.0,
        );

        let coherent_data = vec![0.5, 0.5, 0.5, 0.5];
        let coherence = bmd.calculate_coherence(&coherent_data);
        assert!(coherence > 0.9);

        let incoherent_data = vec![0.1, 0.9, 0.2, 0.8];
        let coherence = bmd.calculate_coherence(&incoherent_data);
        assert!(coherence < 0.5);
    }

    #[test]
    fn test_agency_assertion() {
        let mut bmd = BiologicalMaxwellDemon::new(
            "BMD_AGENCY".to_string(),
            BmdProcessingLevel::Cognitive,
            1.0,
        );

        let result = bmd.assert_agency("MODIFY_REALITY");
        assert!(result.is_ok());
        assert!(result.unwrap().contains("AGENCY_ASSERTION"));
    }
} 