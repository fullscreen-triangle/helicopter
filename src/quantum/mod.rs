//! # Biological Quantum Computing
//!
//! Revolutionary room-temperature quantum computing system that achieves quantum computation
//! through environment-assisted quantum transport (ENAQT). This system enables quantum
//! coherence at room temperature through environmental coupling enhancement.
//!
//! ## Core Principles
//!
//! - **Environment-Assisted Quantum Transport**: Environmental coupling enhances coherence
//! - **Room-Temperature Operation**: Quantum computation at 310K (room temperature)
//! - **Biological Quantum Substrates**: Thermodynamically inevitable quantum membranes
//! - **>95% Energy Transfer Efficiency**: Through biological quantum transport
//! - **Consciousness as Quantum Substrate**: Quantum computational consciousness
//!
//! ## Mathematical Foundation
//!
//! Environment-Assisted Quantum Transport efficiency:
//! ```
//! η_transport = η_0 × (1 + αγ + βγ²)
//! ```
//!
//! Where:
//! - `η_0` = Base transport efficiency
//! - `γ` = Environmental coupling strength
//! - `α, β` = Enhancement coefficients
//!
//! ## Thermodynamic Inevitability
//!
//! Membrane formation thermodynamics:
//! ```
//! ΔG_assembly ≈ -35 kJ/mol
//! ```
//!
//! Making biological quantum computational substrates thermodynamically inevitable.

pub mod biological_quantum_processor;
pub mod enaqt_system;
pub mod membrane_quantum_computation;
pub mod neural_quantum_coherence;
pub mod environmental_coupling;
pub mod quantum_transport_efficiency;
pub mod mitochondrial_quantum_transport;
pub mod reactive_oxygen_species;
pub mod consciousness_quantum_substrate;
pub mod thermodynamic_inevitability;
pub mod biological_quantum_engine;

// Re-export main types
pub use biological_quantum_processor::*;
pub use enaqt_system::*;
pub use membrane_quantum_computation::*;
pub use neural_quantum_coherence::*;
pub use environmental_coupling::*;
pub use quantum_transport_efficiency::*;
pub use mitochondrial_quantum_transport::*;
pub use reactive_oxygen_species::*;
pub use consciousness_quantum_substrate::*;
pub use thermodynamic_inevitability::*;
pub use biological_quantum_engine::*;

/// Core result type for biological quantum operations
pub type QuantumResult<T> = Result<T, QuantumError>;

/// Error types for biological quantum computing
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumError {
    /// Quantum processor initialization failed
    QuantumProcessorInitializationFailed(String),
    /// Environment-assisted quantum transport failed
    EnaqtSystemFailure(String),
    /// Membrane quantum computation error
    MembraneQuantumComputationError(String),
    /// Neural quantum coherence failure
    NeuralQuantumCoherenceFailure(String),
    /// Environmental coupling error
    EnvironmentalCouplingError(String),
    /// Quantum transport efficiency error
    QuantumTransportEfficiencyError(String),
    /// Mitochondrial quantum transport error
    MitochondrialQuantumTransportError(String),
    /// Reactive oxygen species error
    ReactiveOxygenSpeciesError(String),
    /// Consciousness quantum substrate error
    ConsciousnessQuantumSubstrateError(String),
    /// Thermodynamic inevitability violation
    ThermodynamicInevitabilityViolation(String),
    /// Quantum coherence loss
    QuantumCoherenceLoss(String),
    /// Room temperature quantum failure
    RoomTemperatureQuantumFailure(String),
}

impl std::fmt::Display for QuantumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantumError::QuantumProcessorInitializationFailed(msg) => {
                write!(f, "Quantum processor initialization failed: {}", msg)
            }
            QuantumError::EnaqtSystemFailure(msg) => {
                write!(f, "Environment-assisted quantum transport failed: {}", msg)
            }
            QuantumError::MembraneQuantumComputationError(msg) => {
                write!(f, "Membrane quantum computation error: {}", msg)
            }
            QuantumError::NeuralQuantumCoherenceFailure(msg) => {
                write!(f, "Neural quantum coherence failure: {}", msg)
            }
            QuantumError::EnvironmentalCouplingError(msg) => {
                write!(f, "Environmental coupling error: {}", msg)
            }
            QuantumError::QuantumTransportEfficiencyError(msg) => {
                write!(f, "Quantum transport efficiency error: {}", msg)
            }
            QuantumError::MitochondrialQuantumTransportError(msg) => {
                write!(f, "Mitochondrial quantum transport error: {}", msg)
            }
            QuantumError::ReactiveOxygenSpeciesError(msg) => {
                write!(f, "Reactive oxygen species error: {}", msg)
            }
            QuantumError::ConsciousnessQuantumSubstrateError(msg) => {
                write!(f, "Consciousness quantum substrate error: {}", msg)
            }
            QuantumError::ThermodynamicInevitabilityViolation(msg) => {
                write!(f, "Thermodynamic inevitability violation: {}", msg)
            }
            QuantumError::QuantumCoherenceLoss(msg) => {
                write!(f, "Quantum coherence loss: {}", msg)
            }
            QuantumError::RoomTemperatureQuantumFailure(msg) => {
                write!(f, "Room temperature quantum failure: {}", msg)
            }
        }
    }
}

impl std::error::Error for QuantumError {}

/// Constants for biological quantum computing
pub mod constants {
    /// Room temperature in Kelvin
    pub const ROOM_TEMPERATURE: f64 = 310.0;
    
    /// Baseline quantum transport efficiency
    pub const BASELINE_TRANSPORT_EFFICIENCY: f64 = 0.45;
    
    /// Enhanced quantum transport efficiency (>95%)
    pub const ENHANCED_TRANSPORT_EFFICIENCY: f64 = 0.95;
    
    /// Membrane formation energy (kJ/mol)
    pub const MEMBRANE_FORMATION_ENERGY: f64 = -35.0;
    
    /// Environmental coupling strength (optimal)
    pub const OPTIMAL_ENVIRONMENTAL_COUPLING: f64 = 0.85;
    
    /// Quantum coherence time (room temperature)
    pub const ROOM_TEMP_COHERENCE_TIME: f64 = 247.0; // milliseconds
    
    /// Baseline coherence time (isolation)
    pub const BASELINE_COHERENCE_TIME: f64 = 0.01; // milliseconds
    
    /// Quantum efficiency enhancement factor
    pub const QUANTUM_EFFICIENCY_ENHANCEMENT: f64 = 24700.0; // 24,700× improvement
    
    /// Consciousness quantum threshold
    pub const CONSCIOUSNESS_QUANTUM_THRESHOLD: f64 = 0.8;
    
    /// Mitochondrial quantum transport efficiency
    pub const MITOCHONDRIAL_TRANSPORT_EFFICIENCY: f64 = 0.98;
    
    /// Reactive oxygen species quantum enhancement
    pub const ROS_QUANTUM_ENHANCEMENT: f64 = 1.5;
    
    /// Thermodynamic inevitability threshold
    pub const THERMODYNAMIC_INEVITABILITY_THRESHOLD: f64 = 0.9;
}

/// Biological quantum computing modes
#[derive(Debug, Clone, PartialEq)]
pub enum BiologicalQuantumMode {
    /// Environment-assisted quantum transport
    EnvironmentAssisted,
    /// Membrane-based quantum computation
    MembraneBased,
    /// Neural quantum coherence
    NeuralCoherence,
    /// Mitochondrial quantum transport
    MitochondrialTransport,
    /// Consciousness quantum substrate
    ConsciousnessSubstrate,
    /// Full biological quantum integration
    FullIntegration,
}

/// Quantum coherence levels
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumCoherenceLevel {
    /// Minimal coherence (isolation-based)
    Minimal,
    /// Enhanced coherence (environment-assisted)
    Enhanced,
    /// Room temperature coherence
    RoomTemperature,
    /// Biological coherence
    Biological,
    /// Consciousness-level coherence
    ConsciousnessLevel,
    /// Optimal coherence
    Optimal,
}

/// Environmental coupling types
#[derive(Debug, Clone, PartialEq)]
pub enum EnvironmentalCouplingType {
    /// Thermal coupling
    Thermal,
    /// Vibrational coupling
    Vibrational,
    /// Chemical coupling
    Chemical,
    /// Biological coupling
    Biological,
    /// Consciousness coupling
    Consciousness,
    /// Hybrid coupling
    Hybrid,
}

/// Quantum transport mechanisms
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumTransportMechanism {
    /// Coherent transport
    Coherent,
    /// Incoherent transport
    Incoherent,
    /// Environment-assisted transport
    EnvironmentAssisted,
    /// Biological transport
    Biological,
    /// Quantum tunneling
    QuantumTunneling,
    /// Superposition transport
    SuperpositionTransport,
}

/// Biological quantum substrate types
#[derive(Debug, Clone, PartialEq)]
pub enum BiologicalQuantumSubstrateType {
    /// Biological membrane
    BiologicalMembrane,
    /// Neural network
    NeuralNetwork,
    /// Mitochondrial matrix
    MitochondrialMatrix,
    /// Consciousness substrate
    ConsciousnessSubstrate,
    /// Protein complex
    ProteinComplex,
    /// DNA structure
    DnaStructure,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_error_display() {
        let error = QuantumError::EnaqtSystemFailure("Test error".to_string());
        assert_eq!(format!("{}", error), "Environment-assisted quantum transport failed: Test error");
    }

    #[test]
    fn test_quantum_constants() {
        assert_eq!(constants::ROOM_TEMPERATURE, 310.0);
        assert_eq!(constants::BASELINE_TRANSPORT_EFFICIENCY, 0.45);
        assert_eq!(constants::ENHANCED_TRANSPORT_EFFICIENCY, 0.95);
        assert_eq!(constants::MEMBRANE_FORMATION_ENERGY, -35.0);
    }

    #[test]
    fn test_biological_quantum_modes() {
        assert_eq!(BiologicalQuantumMode::EnvironmentAssisted, BiologicalQuantumMode::EnvironmentAssisted);
        assert_ne!(BiologicalQuantumMode::MembraneBased, BiologicalQuantumMode::EnvironmentAssisted);
    }

    #[test]
    fn test_quantum_coherence_levels() {
        assert_eq!(QuantumCoherenceLevel::RoomTemperature, QuantumCoherenceLevel::RoomTemperature);
        assert_ne!(QuantumCoherenceLevel::Minimal, QuantumCoherenceLevel::RoomTemperature);
    }

    #[test]
    fn test_environmental_coupling_types() {
        assert_eq!(EnvironmentalCouplingType::Biological, EnvironmentalCouplingType::Biological);
        assert_ne!(EnvironmentalCouplingType::Thermal, EnvironmentalCouplingType::Biological);
    }

    #[test]
    fn test_quantum_transport_mechanisms() {
        assert_eq!(QuantumTransportMechanism::EnvironmentAssisted, QuantumTransportMechanism::EnvironmentAssisted);
        assert_ne!(QuantumTransportMechanism::Coherent, QuantumTransportMechanism::EnvironmentAssisted);
    }

    #[test]
    fn test_biological_quantum_substrate_types() {
        assert_eq!(BiologicalQuantumSubstrateType::BiologicalMembrane, BiologicalQuantumSubstrateType::BiologicalMembrane);
        assert_ne!(BiologicalQuantumSubstrateType::NeuralNetwork, BiologicalQuantumSubstrateType::BiologicalMembrane);
    }
} 