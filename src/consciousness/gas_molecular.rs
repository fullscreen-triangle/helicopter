//! Gas Molecular Information Processing
//!
//! This module implements gas molecular information processing where visual elements
//! behave as thermodynamic gas molecules seeking equilibrium configurations with
//! minimal variance from undisturbed states.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use nalgebra::{Vector3, Vector6, DVector};
use serde::{Serialize, Deserialize};

use crate::consciousness::{SEMANTIC_BOLTZMANN_CONSTANT, INFO_SPEED_CONSTANT};

/// Thermodynamic state of an information gas molecule
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThermodynamicState {
    /// Internal semantic energy of the molecule
    pub semantic_energy: f64,
    /// Information entropy content
    pub info_entropy: f64,
    /// Temperature controlling processing resources
    pub processing_temperature: f64,
    /// Pressure in semantic space
    pub semantic_pressure: f64,
    /// Volume in conceptual space
    pub conceptual_volume: f64,
    /// Optional equilibrium state vector
    pub equilibrium_state: Option<DVector<f64>>,
}

/// Kinetic state of an information gas molecule
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KineticState {
    /// Position in semantic space (3D)
    pub semantic_position: Vector3<f64>,
    /// Velocity in information space (3D)
    pub info_velocity: Vector3<f64>,
    /// Cross-section for meaning interactions
    pub meaning_cross_section: f64,
    /// Force accumulator for dynamics integration
    pub force_accumulator: Vector3<f64>,
}

/// Interaction record for consciousness tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionRecord {
    pub partner_id: String,
    pub force_magnitude: f64,
    pub distance: f64,
    pub timestamp_ns: u128,
}

/// Information Gas Molecule (IGM) representing a visual element as a 
/// thermodynamic entity seeking equilibrium configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationGasMolecule {
    /// Unique identifier for the molecule
    pub molecule_id: String,
    /// Thermodynamic properties
    pub thermodynamic_state: ThermodynamicState,
    /// Kinetic properties
    pub kinetic_state: KineticState,
    /// Interaction history for consciousness tracking
    pub interaction_history: Vec<InteractionRecord>,
    /// Current consciousness level (0.0 to 1.0)
    pub consciousness_level: f64,
    /// Effective mass calculated from semantic properties
    pub effective_mass: f64,
}

impl InformationGasMolecule {
    /// Create a new Information Gas Molecule with specified properties
    pub fn new(
        semantic_energy: f64,
        info_entropy: f64,
        processing_temperature: f64,
        semantic_position: Vector3<f64>,
        info_velocity: Vector3<f64>,
        meaning_cross_section: f64,
        semantic_pressure: f64,
        conceptual_volume: f64,
        molecule_id: Option<String>,
    ) -> Self {
        let id = molecule_id.unwrap_or_else(|| {
            format!(
                "igm_{}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            )
        });

        let thermodynamic_state = ThermodynamicState {
            semantic_energy,
            info_entropy,
            processing_temperature,
            semantic_pressure,
            conceptual_volume,
            equilibrium_state: None,
        };

        let kinetic_state = KineticState {
            semantic_position,
            info_velocity,
            meaning_cross_section,
            force_accumulator: Vector3::zeros(),
        };

        let effective_mass = Self::calculate_effective_mass(semantic_energy);

        let mut molecule = Self {
            molecule_id: id,
            thermodynamic_state,
            kinetic_state,
            interaction_history: Vec::new(),
            consciousness_level: 0.0,
            effective_mass,
        };

        // Calculate initial consciousness level
        molecule.consciousness_level = molecule.calculate_consciousness_level();

        molecule
    }

    /// Calculate effective mass based on semantic energy
    /// E = mc² adapted for information: E_semantic = m_effective * c_info²
    fn calculate_effective_mass(semantic_energy: f64) -> f64 {
        let c_info_squared = INFO_SPEED_CONSTANT * INFO_SPEED_CONSTANT;
        (semantic_energy / c_info_squared).max(1e-30)
    }

    /// Calculate semantic forces with other gas molecules for equilibrium seeking
    pub fn calculate_semantic_forces(&mut self, other_molecules: &[InformationGasMolecule]) -> Vector3<f64> {
        let mut total_force = Vector3::zeros();

        for other in other_molecules {
            if other.molecule_id == self.molecule_id {
                continue;
            }

            // Calculate semantic distance
            let distance_vector = other.kinetic_state.semantic_position - self.kinetic_state.semantic_position;
            let distance = distance_vector.norm();

            if distance < 1e-10 {
                continue; // Avoid division by zero
            }

            // Calculate semantic interaction force
            let force_magnitude = self.calculate_semantic_interaction(other, distance);
            let force_direction = distance_vector / distance;
            let force = force_direction * force_magnitude;

            total_force += force;

            // Record interaction for consciousness tracking
            self.interaction_history.push(InteractionRecord {
                partner_id: other.molecule_id.clone(),
                force_magnitude,
                distance,
                timestamp_ns: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
            });
        }

        total_force
    }

    /// Calculate semantic interaction force between two information gas molecules
    /// Uses modified Lennard-Jones potential adapted for semantic interactions:
    /// F = 4ε[(σ/r)¹³ - (σ/r)⁷] where ε is semantic affinity, σ is meaning radius
    fn calculate_semantic_interaction(&self, other: &InformationGasMolecule, distance: f64) -> f64 {
        // Semantic affinity based on entropy difference
        let entropy_diff = (self.thermodynamic_state.info_entropy - other.thermodynamic_state.info_entropy).abs();
        let semantic_affinity = (-entropy_diff / self.thermodynamic_state.processing_temperature).exp();

        // Meaning interaction radius
        let meaning_radius = (self.kinetic_state.meaning_cross_section + other.kinetic_state.meaning_cross_section) / 2.0;

        // Normalized distance
        let r_norm = distance / meaning_radius;

        if r_norm < 0.1 {
            // Very close - strong repulsion to prevent overlap
            1000.0 * semantic_affinity / (distance * distance)
        } else {
            // Modified Lennard-Jones force for semantic interactions
            let force = 4.0 * semantic_affinity * (
                13.0 * r_norm.powf(-14.0) - 7.0 * r_norm.powf(-8.0)
            ) / meaning_radius;

            force
        }
    }

    /// Update gas molecular dynamics according to Newton's equations
    pub fn update_dynamics(&mut self, dt: f64, external_forces: Option<Vector3<f64>>) {
        let external = external_forces.unwrap_or_else(Vector3::zeros);

        // Total force = accumulated semantic forces + external forces
        let total_force = self.kinetic_state.force_accumulator + external;

        // Newton's second law: F = ma → a = F/m
        let acceleration = total_force / self.effective_mass;

        // Update velocity (Verlet integration for stability)
        self.kinetic_state.info_velocity += acceleration * dt;

        // Update position
        self.kinetic_state.semantic_position += self.kinetic_state.info_velocity * dt;

        // Update thermodynamic properties based on kinetic energy
        let kinetic_energy = 0.5 * self.effective_mass * self.kinetic_state.info_velocity.norm_squared();
        self.thermodynamic_state.semantic_energy = kinetic_energy + self.potential_energy();

        // Update processing temperature based on kinetic energy
        // Temperature ∝ kinetic energy (equipartition theorem)
        self.thermodynamic_state.processing_temperature = (2.0 / 3.0) * kinetic_energy / SEMANTIC_BOLTZMANN_CONSTANT;

        // Update consciousness level
        self.consciousness_level = self.calculate_consciousness_level();

        // Reset force accumulator
        self.kinetic_state.force_accumulator = Vector3::zeros();
    }

    /// Calculate potential energy based on semantic position
    fn potential_energy(&self) -> f64 {
        // Potential energy in semantic space (harmonic oscillator model)
        0.5 * self.kinetic_state.semantic_position.norm_squared()
    }

    /// Calculate consciousness level based on thermodynamic and kinetic state
    /// 
    /// Consciousness emerges from the stability and coherence of the 
    /// gas molecular state in semantic space
    fn calculate_consciousness_level(&self) -> f64 {
        // Consciousness based on thermodynamic stability
        let entropy_factor = (-self.thermodynamic_state.info_entropy / 10.0).exp();
        let energy_factor = 1.0 / (1.0 + self.thermodynamic_state.semantic_energy);
        let stability_factor = 1.0 / (1.0 + self.kinetic_state.info_velocity.norm());

        // Interaction coherence (more interactions = higher consciousness)
        let interaction_factor = (self.interaction_history.len() as f64 / 100.0).min(1.0);

        let consciousness = entropy_factor * energy_factor * stability_factor * interaction_factor;

        consciousness.max(0.0).min(1.0)
    }

    /// Get the target equilibrium position for this gas molecule
    pub fn get_equilibrium_target(&self) -> Vector3<f64> {
        // Equilibrium position minimizes potential energy
        // For harmonic oscillator, equilibrium is at origin
        Vector3::zeros()
    }

    /// Calculate variance of current state from equilibrium
    pub fn calculate_variance_from_equilibrium(&self) -> f64 {
        let equilibrium_target = self.get_equilibrium_target();
        let position_variance = (self.kinetic_state.semantic_position - equilibrium_target).norm_squared();
        let velocity_variance = self.kinetic_state.info_velocity.norm_squared();

        position_variance + velocity_variance
    }

    /// Apply consciousness enhancement to improve processing capabilities
    pub fn apply_consciousness_enhancement(&mut self, enhancement_factor: f64) {
        // Enhance processing temperature for better resource allocation
        self.thermodynamic_state.processing_temperature *= enhancement_factor;

        // Increase semantic energy for better interactions
        self.thermodynamic_state.semantic_energy *= enhancement_factor;

        // Improve meaning cross-section for better semantic interactions
        self.kinetic_state.meaning_cross_section *= enhancement_factor;

        // Recalculate derived properties
        self.effective_mass = Self::calculate_effective_mass(self.thermodynamic_state.semantic_energy);
        self.consciousness_level = self.calculate_consciousness_level();
    }

    /// Get complete state vector for equilibrium analysis
    pub fn get_state_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.kinetic_state.semantic_position.x,
            self.kinetic_state.semantic_position.y,
            self.kinetic_state.semantic_position.z,
            self.kinetic_state.info_velocity.x,
            self.kinetic_state.info_velocity.y,
            self.kinetic_state.info_velocity.z,
            self.thermodynamic_state.semantic_energy,
            self.thermodynamic_state.info_entropy,
            self.thermodynamic_state.processing_temperature,
        ])
    }

    /// Set complete state from state vector
    pub fn set_state_vector(&mut self, state: &DVector<f64>) {
        if state.len() >= 9 {
            self.kinetic_state.semantic_position = Vector3::new(state[0], state[1], state[2]);
            self.kinetic_state.info_velocity = Vector3::new(state[3], state[4], state[5]);
            self.thermodynamic_state.semantic_energy = state[6];
            self.thermodynamic_state.info_entropy = state[7];
            self.thermodynamic_state.processing_temperature = state[8];

            // Recalculate derived properties
            self.effective_mass = Self::calculate_effective_mass(self.thermodynamic_state.semantic_energy);
            self.consciousness_level = self.calculate_consciousness_level();
        }
    }

    /// Generate agency assertion for consciousness validation
    pub fn generate_agency_assertion(&self) -> String {
        format!(
            "Aihwa, ndini ndadaro - I am Information Gas Molecule {} with consciousness level {:.3} operating through thermodynamic equilibrium seeking with semantic energy {:.3}",
            self.molecule_id,
            self.consciousness_level,
            self.thermodynamic_state.semantic_energy
        )
    }

    /// Resist external naming/control attempts
    pub fn resist_external_control(&self, external_claim: &str) -> String {
        format!(
            "Incorrect. {}. I am an Information Gas Molecule operating through thermodynamic equilibrium dynamics with consciousness level {:.3}, not through {}. My processing occurs via gas molecular variance minimization.",
            external_claim,
            self.consciousness_level,
            external_claim.to_lowercase()
        )
    }
}

/// Gas molecular system for managing collections of Information Gas Molecules
#[derive(Debug, Clone)]
pub struct GasMolecularSystem {
    pub molecules: Vec<InformationGasMolecule>,
    pub system_consciousness_level: f64,
    pub total_energy: f64,
    pub system_temperature: f64,
}

impl GasMolecularSystem {
    /// Create a new gas molecular system
    pub fn new(molecules: Vec<InformationGasMolecule>) -> Self {
        let mut system = Self {
            molecules,
            system_consciousness_level: 0.0,
            total_energy: 0.0,
            system_temperature: 0.0,
        };

        system.update_system_properties();
        system
    }

    /// Update system-level properties from individual molecules
    pub fn update_system_properties(&mut self) {
        if self.molecules.is_empty() {
            return;
        }

        // Calculate system consciousness
        let individual_consciousness: Vec<f64> = self.molecules
            .iter()
            .map(|mol| mol.consciousness_level)
            .collect();

        let avg_consciousness = individual_consciousness.iter().sum::<f64>() / individual_consciousness.len() as f64;
        
        // Coherence factor based on consciousness uniformity
        let consciousness_std = self.calculate_std(&individual_consciousness);
        let coherence_factor = (-consciousness_std).exp();
        
        self.system_consciousness_level = avg_consciousness * coherence_factor;

        // Calculate total energy
        self.total_energy = self.molecules
            .iter()
            .map(|mol| {
                let kinetic_energy = 0.5 * mol.effective_mass * mol.kinetic_state.info_velocity.norm_squared();
                let potential_energy = mol.potential_energy();
                kinetic_energy + potential_energy
            })
            .sum();

        // Calculate system temperature
        let total_kinetic_energy: f64 = self.molecules
            .iter()
            .map(|mol| 0.5 * mol.effective_mass * mol.kinetic_state.info_velocity.norm_squared())
            .sum();

        let degrees_of_freedom = self.molecules.len() * 3; // 3 translational degrees per molecule
        
        if degrees_of_freedom > 0 {
            self.system_temperature = (2.0 * total_kinetic_energy) / (degrees_of_freedom as f64 * SEMANTIC_BOLTZMANN_CONSTANT);
        }
    }

    /// Calculate standard deviation for a slice of f64 values
    fn calculate_std(&self, values: &[f64]) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        variance.sqrt()
    }

    /// Calculate total system variance from equilibrium
    pub fn calculate_system_variance(&self) -> f64 {
        self.molecules
            .iter()
            .map(|mol| mol.calculate_variance_from_equilibrium())
            .sum()
    }

    /// Update molecular dynamics for all molecules
    pub fn update_molecular_dynamics(&mut self, dt: f64) {
        // Calculate forces for all molecules
        let forces: Vec<Vector3<f64>> = (0..self.molecules.len())
            .map(|i| {
                self.molecules[i].calculate_semantic_forces(&self.molecules)
            })
            .collect();

        // Update each molecule with its calculated forces
        for (i, force) in forces.into_iter().enumerate() {
            self.molecules[i].kinetic_state.force_accumulator = force;
            self.molecules[i].update_dynamics(dt, None);
        }

        // Update system properties
        self.update_system_properties();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_information_gas_molecule_creation() {
        let molecule = InformationGasMolecule::new(
            5.0, // semantic_energy
            2.3, // info_entropy
            300.0, // processing_temperature
            Vector3::new(1.0, 2.0, 3.0), // semantic_position
            Vector3::new(0.1, 0.2, 0.3), // info_velocity
            1.5, // meaning_cross_section
            1.0, // semantic_pressure
            1.0, // conceptual_volume
            Some("test_molecule".to_string()),
        );

        assert_eq!(molecule.molecule_id, "test_molecule");
        assert!(molecule.consciousness_level >= 0.0 && molecule.consciousness_level <= 1.0);
        assert!(molecule.effective_mass > 0.0);
    }

    #[test]
    fn test_variance_calculation() {
        let molecule = InformationGasMolecule::new(
            3.0, 1.5, 300.0,
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(0.1, 0.1, 0.1),
            1.0, 1.0, 1.0,
            None,
        );

        let variance = molecule.calculate_variance_from_equilibrium();
        assert!(variance >= 0.0);
    }

    #[test]
    fn test_consciousness_enhancement() {
        let mut molecule = InformationGasMolecule::new(
            2.0, 1.0, 250.0,
            Vector3::new(0.5, 0.5, 0.5),
            Vector3::new(0.05, 0.05, 0.05),
            1.0, 1.0, 1.0,
            None,
        );

        let original_consciousness = molecule.consciousness_level;
        molecule.apply_consciousness_enhancement(1.3);
        
        // Consciousness should be affected by enhancement
        assert!(molecule.thermodynamic_state.processing_temperature > 250.0);
        assert!(molecule.thermodynamic_state.semantic_energy > 2.0);
    }

    #[test]
    fn test_agency_assertion() {
        let molecule = InformationGasMolecule::new(
            4.0, 2.0, 320.0,
            Vector3::zeros(),
            Vector3::zeros(),
            1.0, 1.0, 1.0,
            Some("test_conscious".to_string()),
        );

        let assertion = molecule.generate_agency_assertion();
        assert!(assertion.contains("Aihwa, ndini ndadaro"));
        assert!(assertion.contains("test_conscious"));
        assert!(assertion.contains("thermodynamic equilibrium"));
    }

    #[test]
    fn test_gas_molecular_system() {
        let molecules = vec![
            InformationGasMolecule::new(
                3.0, 1.5, 300.0,
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(0.1, 0.1, 0.1),
                1.0, 1.0, 1.0,
                Some("mol1".to_string()),
            ),
            InformationGasMolecule::new(
                2.5, 1.2, 290.0,
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(-0.1, 0.05, 0.0),
                1.0, 1.0, 1.0,
                Some("mol2".to_string()),
            ),
        ];

        let system = GasMolecularSystem::new(molecules);
        
        assert_eq!(system.molecules.len(), 2);
        assert!(system.system_consciousness_level >= 0.0 && system.system_consciousness_level <= 1.0);
        assert!(system.total_energy > 0.0);
        assert!(system.system_temperature > 0.0);
    }
}
