//! Consciousness-Aware Computer Vision Processing
//! 
//! This module implements consciousness-aware visual processing through gas molecular
//! information dynamics, cross-modal BMD validation, and dual-mode processing architecture.
//! 
//! The framework operates on the principle that meaning emerges from gas molecular
//! configurations with minimal variance from undisturbed equilibrium, eliminating
//! the need for semantic dictionaries or computational lookup.

pub mod gas_molecular;
pub mod equilibrium;
pub mod bmd_validation;
pub mod moon_landing;
pub mod variance_analysis;
pub mod consciousness_validation;

pub use gas_molecular::{InformationGasMolecule, ThermodynamicState, KineticState};
pub use equilibrium::{EquilibriumEngine, EquilibriumResult};
pub use variance_analysis::{VarianceAnalyzer, VarianceSnapshot};

/// Consciousness processing constants
pub const CONSCIOUSNESS_THRESHOLD: f64 = 0.61;
pub const EQUILIBRIUM_CONVERGENCE_TIME_NS: u64 = 12;
pub const VARIANCE_THRESHOLD: f64 = 1e-6;
pub const BMD_CONVERGENCE_RATE: f64 = 0.95;

/// Semantic Boltzmann constant for information thermodynamics
pub const SEMANTIC_BOLTZMANN_CONSTANT: f64 = 1.380649e-23;

/// Universal gas constant for information processing
pub const INFORMATION_GAS_CONSTANT: f64 = 8.314462618;

/// Information speed constant for semantic energy calculations
pub const INFO_SPEED_CONSTANT: f64 = 2.998e8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert!(CONSCIOUSNESS_THRESHOLD > 0.0 && CONSCIOUSNESS_THRESHOLD < 1.0);
        assert!(VARIANCE_THRESHOLD > 0.0);
        assert!(BMD_CONVERGENCE_RATE > 0.0 && BMD_CONVERGENCE_RATE < 1.0);
    }
}
