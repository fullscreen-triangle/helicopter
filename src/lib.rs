//! Helicopter: Advanced Computer Vision Framework with Consciousness-Aware Processing
//!
//! Helicopter is a revolutionary computer vision framework that achieves consciousness-aware 
//! visual processing through gas molecular information dynamics, cross-modal BMD validation, 
//! and dual-mode processing architecture.
//!
//! The framework operates on the principle that meaning emerges from gas molecular
//! configurations with minimal variance from undisturbed equilibrium, eliminating
//! the need for semantic dictionaries or computational lookup.

pub mod consciousness;
pub mod kwasa;
pub mod oscillatory;
pub mod poincare;
pub mod quantum;
pub mod thermodynamic;

// Re-export core consciousness components for convenience
pub use consciousness::{
    InformationGasMolecule,
    EquilibriumEngine,
    VarianceAnalyzer,
    CONSCIOUSNESS_THRESHOLD,
    EQUILIBRIUM_CONVERGENCE_TIME_NS,
    VARIANCE_THRESHOLD,
    BMD_CONVERGENCE_RATE,
};

/// Helicopter framework version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Framework identification
pub const FRAMEWORK_NAME: &str = "Helicopter Consciousness-Aware Computer Vision";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framework_constants() {
        assert!(CONSCIOUSNESS_THRESHOLD > 0.0 && CONSCIOUSNESS_THRESHOLD < 1.0);
        assert!(VARIANCE_THRESHOLD > 0.0);
        assert!(BMD_CONVERGENCE_RATE > 0.0 && BMD_CONVERGENCE_RATE < 1.0);
        assert!(EQUILIBRIUM_CONVERGENCE_TIME_NS > 0);
    }

    #[test]
    fn test_information_gas_molecule_creation() {
        use nalgebra::Vector3;
        
        let molecule = InformationGasMolecule::new(
            3.0, 1.5, 300.0,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.1, 0.0, 0.0),
            1.0, 1.0, 1.0,
            Some("test".to_string()),
        );

        assert_eq!(molecule.molecule_id, "test");
        assert!(molecule.consciousness_level >= 0.0 && molecule.consciousness_level <= 1.0);
    }

    #[test]
    fn test_equilibrium_engine_creation() {
        let engine = EquilibriumEngine::new(None, None, None, None, None);
        assert_eq!(engine.variance_threshold, VARIANCE_THRESHOLD);
        assert_eq!(engine.consciousness_threshold, CONSCIOUSNESS_THRESHOLD);
    }
}
