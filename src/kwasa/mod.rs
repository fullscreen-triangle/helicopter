//! # Kwasa-Kwasa Framework
//!
//! Implementation of consciousness-aware semantic computation through Biological Maxwell's Demons (BMDs).
//! This module implements the revolutionary framework for information catalysis and reality discretization.
//!
//! ## Core Components
//!
//! - **Biological Maxwell's Demons (BMDs)**: Information catalysts that discretize continuous reality
//! - **Fire-Adapted Consciousness**: 322% processing improvement through evolutionary enhancements
//! - **Semantic Catalysis**: Reality-direct processing through BMD networks
//! - **Naming Functions**: Reality discretization through naming system control
//! - **Agency Assertion**: Coordinated reality modification through agency
//!
//! ## Mathematical Foundation
//!
//! The Kwasa-Kwasa framework operates on the principle:
//! ```
//! iCat_semantic = ℑ_input ○ ℑ_output ○ ℑ_agency
//! ```
//!
//! Where:
//! - `ℑ_input` = Pattern recognition (naming function)
//! - `ℑ_output` = Output channeling (flow coordination)
//! - `ℑ_agency` = Agency assertion (naming modification)

pub mod biological_maxwell_demon;
pub mod bmd_network;
pub mod fire_adapted_consciousness;
pub mod semantic_catalysis;
pub mod naming_functions;
pub mod agency_assertion;
pub mod molecular_bmd;
pub mod neural_bmd;
pub mod cognitive_bmd;
pub mod consciousness_threshold;
pub mod fire_circle_communication;
pub mod kwasa_framework;

// Re-export main types
pub use biological_maxwell_demon::*;
pub use bmd_network::*;
pub use fire_adapted_consciousness::*;
pub use semantic_catalysis::*;
pub use naming_functions::*;
pub use agency_assertion::*;
pub use molecular_bmd::*;
pub use neural_bmd::*;
pub use cognitive_bmd::*;
pub use consciousness_threshold::*;
pub use fire_circle_communication::*;
pub use kwasa_framework::*;

/// Core result type for Kwasa-Kwasa operations
pub type KwasaResult<T> = Result<T, KwasaError>;

/// Error types for Kwasa-Kwasa framework
#[derive(Debug, Clone, PartialEq)]
pub enum KwasaError {
    /// BMD network initialization failed
    BmdNetworkInitializationFailed(String),
    /// Consciousness threshold not reached
    ConsciousnessThresholdNotReached(f64),
    /// Fire-adapted enhancement failed
    FireAdaptedEnhancementFailed(String),
    /// Semantic catalysis error
    SemanticCatalysisError(String),
    /// Naming function error
    NamingFunctionError(String),
    /// Agency assertion failed
    AgencyAssertionFailed(String),
    /// Reality discretization error
    RealityDiscretizationError(String),
    /// Information catalysis error
    InformationCatalysisError(String),
}

impl std::fmt::Display for KwasaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KwasaError::BmdNetworkInitializationFailed(msg) => {
                write!(f, "BMD network initialization failed: {}", msg)
            }
            KwasaError::ConsciousnessThresholdNotReached(threshold) => {
                write!(f, "Consciousness threshold not reached: {}", threshold)
            }
            KwasaError::FireAdaptedEnhancementFailed(msg) => {
                write!(f, "Fire-adapted enhancement failed: {}", msg)
            }
            KwasaError::SemanticCatalysisError(msg) => {
                write!(f, "Semantic catalysis error: {}", msg)
            }
            KwasaError::NamingFunctionError(msg) => {
                write!(f, "Naming function error: {}", msg)
            }
            KwasaError::AgencyAssertionFailed(msg) => {
                write!(f, "Agency assertion failed: {}", msg)
            }
            KwasaError::RealityDiscretizationError(msg) => {
                write!(f, "Reality discretization error: {}", msg)
            }
            KwasaError::InformationCatalysisError(msg) => {
                write!(f, "Information catalysis error: {}", msg)
            }
        }
    }
}

impl std::error::Error for KwasaError {}

/// Constants for the Kwasa-Kwasa framework
pub mod constants {
    /// Fire-adapted consciousness threshold (Θ_c = 0.61)
    pub const FIRE_ADAPTED_CONSCIOUSNESS_THRESHOLD: f64 = 0.61;
    
    /// Baseline consciousness threshold (Θ_c = 0.4)
    pub const BASELINE_CONSCIOUSNESS_THRESHOLD: f64 = 0.4;
    
    /// Fire-adapted processing improvement factor (322%)
    pub const FIRE_ADAPTED_PROCESSING_IMPROVEMENT: f64 = 3.22;
    
    /// Survival advantage factor (460%)
    pub const SURVIVAL_ADVANTAGE_FACTOR: f64 = 4.60;
    
    /// Communication complexity enhancement (79.3×)
    pub const COMMUNICATION_COMPLEXITY_ENHANCEMENT: f64 = 79.3;
    
    /// Pattern recognition improvement (346%)
    pub const PATTERN_RECOGNITION_IMPROVEMENT: f64 = 3.46;
    
    /// Coherence time enhancement (247ms vs 89ms baseline)
    pub const FIRE_ADAPTED_COHERENCE_TIME: f64 = 247.0; // milliseconds
    pub const BASELINE_COHERENCE_TIME: f64 = 89.0; // milliseconds
    
    /// Cognitive capacity enhancement (4.22×)
    pub const COGNITIVE_CAPACITY_ENHANCEMENT: f64 = 4.22;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kwasa_error_display() {
        let error = KwasaError::ConsciousnessThresholdNotReached(0.5);
        assert_eq!(format!("{}", error), "Consciousness threshold not reached: 0.5");
    }

    #[test]
    fn test_fire_adapted_constants() {
        assert_eq!(constants::FIRE_ADAPTED_CONSCIOUSNESS_THRESHOLD, 0.61);
        assert_eq!(constants::FIRE_ADAPTED_PROCESSING_IMPROVEMENT, 3.22);
        assert_eq!(constants::SURVIVAL_ADVANTAGE_FACTOR, 4.60);
    }
} 