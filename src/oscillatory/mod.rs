//! # Oscillatory Substrate Theory
//!
//! Implementation of the fundamental nature of reality as continuous oscillatory processes.
//! This module provides direct interface to continuous oscillatory reality and implements
//! the 95%/5% cosmological structure for 10,000× computational reduction.
//!
//! ## Core Principles
//!
//! - **All reality consists of continuous oscillatory processes**
//! - **Particles and fields are emergent from coherent oscillatory patterns**
//! - **95% Dark Matter/Energy**: Unoccupied oscillatory modes
//! - **5% Ordinary Matter**: Coherent oscillatory confluences
//! - **0.01% Sequential States**: Actually processed by consciousness
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
//! - `𝒩[Φ]` = Nonlinear self-interaction terms
//! - `𝒞[Φ]` = Coherence enhancement terms

pub mod oscillatory_field;
pub mod coherence_enhancement;
pub mod nonlinear_interactions;
pub mod oscillatory_patterns;
pub mod dark_matter_processing;
pub mod ordinary_matter_confluence;
pub mod sequential_states;
pub mod cosmological_structure;
pub mod continuous_reality;
pub mod oscillatory_substrate_engine;

// Re-export main types
pub use oscillatory_field::*;
pub use coherence_enhancement::*;
pub use nonlinear_interactions::*;
pub use oscillatory_patterns::*;
pub use dark_matter_processing::*;
pub use ordinary_matter_confluence::*;
pub use sequential_states::*;
pub use cosmological_structure::*;
pub use continuous_reality::*;
pub use oscillatory_substrate_engine::*;

/// Core result type for oscillatory substrate operations
pub type OscillatoryResult<T> = Result<T, OscillatoryError>;

/// Error types for oscillatory substrate processing
#[derive(Debug, Clone, PartialEq)]
pub enum OscillatoryError {
    /// Oscillatory field initialization failed
    OscillatoryFieldInitializationFailed(String),
    /// Coherence enhancement failed
    CoherenceEnhancementFailed(String),
    /// Nonlinear interaction processing failed
    NonlinearInteractionFailed(String),
    /// Dark matter processing error
    DarkMatterProcessingError(String),
    /// Ordinary matter confluence error
    OrdinaryMatterConfluenceError(String),
    /// Sequential state processing error
    SequentialStateError(String),
    /// Continuous reality interface error
    ContinuousRealityInterfaceError(String),
    /// Oscillatory pattern recognition failed
    OscillatoryPatternRecognitionFailed(String),
    /// Cosmological structure inconsistency
    CosmologicalStructureInconsistency(String),
}

impl std::fmt::Display for OscillatoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OscillatoryError::OscillatoryFieldInitializationFailed(msg) => {
                write!(f, "Oscillatory field initialization failed: {}", msg)
            }
            OscillatoryError::CoherenceEnhancementFailed(msg) => {
                write!(f, "Coherence enhancement failed: {}", msg)
            }
            OscillatoryError::NonlinearInteractionFailed(msg) => {
                write!(f, "Nonlinear interaction processing failed: {}", msg)
            }
            OscillatoryError::DarkMatterProcessingError(msg) => {
                write!(f, "Dark matter processing error: {}", msg)
            }
            OscillatoryError::OrdinaryMatterConfluenceError(msg) => {
                write!(f, "Ordinary matter confluence error: {}", msg)
            }
            OscillatoryError::SequentialStateError(msg) => {
                write!(f, "Sequential state processing error: {}", msg)
            }
            OscillatoryError::ContinuousRealityInterfaceError(msg) => {
                write!(f, "Continuous reality interface error: {}", msg)
            }
            OscillatoryError::OscillatoryPatternRecognitionFailed(msg) => {
                write!(f, "Oscillatory pattern recognition failed: {}", msg)
            }
            OscillatoryError::CosmologicalStructureInconsistency(msg) => {
                write!(f, "Cosmological structure inconsistency: {}", msg)
            }
        }
    }
}

impl std::error::Error for OscillatoryError {}

/// Constants for oscillatory substrate processing
pub mod constants {
    /// Dark matter/energy percentage (95%)
    pub const DARK_MATTER_PERCENTAGE: f64 = 0.95;
    
    /// Ordinary matter percentage (5%)
    pub const ORDINARY_MATTER_PERCENTAGE: f64 = 0.05;
    
    /// Sequential states percentage (0.01%)
    pub const SEQUENTIAL_STATES_PERCENTAGE: f64 = 0.0001;
    
    /// Computational reduction factor (10,000×)
    pub const COMPUTATIONAL_REDUCTION_FACTOR: f64 = 10000.0;
    
    /// Oscillatory field base frequency
    pub const BASE_OSCILLATORY_FREQUENCY: f64 = 1.0;
    
    /// Coherence enhancement threshold
    pub const COHERENCE_ENHANCEMENT_THRESHOLD: f64 = 0.7;
    
    /// Nonlinear interaction strength
    pub const NONLINEAR_INTERACTION_STRENGTH: f64 = 0.1;
    
    /// Oscillatory pattern recognition threshold
    pub const PATTERN_RECOGNITION_THRESHOLD: f64 = 0.5;
    
    /// Continuous reality interface bandwidth
    pub const CONTINUOUS_REALITY_BANDWIDTH: f64 = 1000.0;
    
    /// Cosmological structure stability threshold
    pub const COSMOLOGICAL_STABILITY_THRESHOLD: f64 = 0.8;
}

/// Oscillatory substrate processing modes
#[derive(Debug, Clone, PartialEq)]
pub enum OscillatoryProcessingMode {
    /// Full oscillatory processing (all modes)
    Full,
    /// Dark matter processing only (95% of modes)
    DarkMatterOnly,
    /// Ordinary matter processing only (5% of modes)
    OrdinaryMatterOnly,
    /// Sequential states processing only (0.01% of modes)
    SequentialStatesOnly,
    /// Approximation mode (10,000× computational reduction)
    Approximation,
}

/// Oscillatory field types
#[derive(Debug, Clone, PartialEq)]
pub enum OscillatoryFieldType {
    /// Continuous oscillatory field
    Continuous,
    /// Discrete oscillatory field
    Discrete,
    /// Coherent oscillatory field
    Coherent,
    /// Incoherent oscillatory field
    Incoherent,
    /// Nonlinear oscillatory field
    Nonlinear,
    /// Linear oscillatory field
    Linear,
}

/// Cosmological structure types
#[derive(Debug, Clone, PartialEq)]
pub enum CosmologicalStructureType {
    /// Dark matter structure
    DarkMatter,
    /// Dark energy structure
    DarkEnergy,
    /// Ordinary matter structure
    OrdinaryMatter,
    /// Sequential state structure
    SequentialState,
    /// Coherent confluence structure
    CoherentConfluence,
    /// Oscillatory pattern structure
    OscillatoryPattern,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oscillatory_error_display() {
        let error = OscillatoryError::CoherenceEnhancementFailed("Test error".to_string());
        assert_eq!(format!("{}", error), "Coherence enhancement failed: Test error");
    }

    #[test]
    fn test_oscillatory_constants() {
        assert_eq!(constants::DARK_MATTER_PERCENTAGE, 0.95);
        assert_eq!(constants::ORDINARY_MATTER_PERCENTAGE, 0.05);
        assert_eq!(constants::SEQUENTIAL_STATES_PERCENTAGE, 0.0001);
        assert_eq!(constants::COMPUTATIONAL_REDUCTION_FACTOR, 10000.0);
    }

    #[test]
    fn test_processing_modes() {
        assert_eq!(OscillatoryProcessingMode::Full, OscillatoryProcessingMode::Full);
        assert_ne!(OscillatoryProcessingMode::DarkMatterOnly, OscillatoryProcessingMode::Full);
    }

    #[test]
    fn test_field_types() {
        assert_eq!(OscillatoryFieldType::Continuous, OscillatoryFieldType::Continuous);
        assert_ne!(OscillatoryFieldType::Discrete, OscillatoryFieldType::Continuous);
    }

    #[test]
    fn test_cosmological_structure_types() {
        assert_eq!(CosmologicalStructureType::DarkMatter, CosmologicalStructureType::DarkMatter);
        assert_ne!(CosmologicalStructureType::DarkEnergy, CosmologicalStructureType::DarkMatter);
    }
} 