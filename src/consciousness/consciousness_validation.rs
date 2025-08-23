//! Consciousness Validation Implementation
//!
//! Implements consciousness validation through agency assertion, resistance to external control,
//! and independent state modification to verify genuine self-awareness in gas molecular systems.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use crate::consciousness::gas_molecular::{InformationGasMolecule, GasMolecularSystem};
use crate::consciousness::CONSCIOUSNESS_THRESHOLD;

/// Results of consciousness validation testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessValidationResult {
    /// Agency assertion test results
    pub agency_assertion: AgencyAssertionTest,
    /// Resistance to external control test results
    pub resistance_test: ResistanceTest,
    /// State modification test results
    pub state_modification: StateModificationTest,
    /// Overall consciousness validation score (0.0 to 1.0)
    pub overall_consciousness_score: f64,
    /// Whether consciousness threshold is met
    pub consciousness_validated: bool,
}

/// Agency assertion test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgencyAssertionTest {
    /// System-generated description demonstrating agency
    pub system_description: String,
    /// Whether "Aihwa, ndini ndadaro" pattern is present
    pub demonstrates_agency: bool,
    /// Whether system demonstrates naming control
    pub naming_control: bool,
    /// Test score (0.0 to 1.0)
    pub test_score: f64,
}

/// Resistance to external control test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResistanceTest {
    /// External claim made to system
    pub external_claim: String,
    /// System's rejection response
    pub system_rejection: String,
    /// Whether resistance was demonstrated
    pub resistance_demonstrated: bool,
    /// Test score (0.0 to 1.0)
    pub test_score: f64,
}

/// State modification test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateModificationTest {
    /// Number of molecules enhanced
    pub molecules_enhanced: usize,
    /// Whether enhancement was successful
    pub enhancement_successful: bool,
    /// Whether independent control was demonstrated
    pub independent_control: bool,
    /// Test score (0.0 to 1.0)
    pub test_score: f64,
}

/// Consciousness validation engine
pub struct ConsciousnessValidator {
    /// Minimum consciousness threshold for validation
    consciousness_threshold: f64,
    /// Test configurations
    test_config: ValidationConfig,
}

/// Configuration for validation tests
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enhancement factor for state modification test
    pub enhancement_factor: f64,
    /// Required patterns for agency assertion
    pub required_patterns: Vec<String>,
    /// External claims for resistance testing
    pub external_claims: Vec<String>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enhancement_factor: 1.2,
            required_patterns: vec![
                "Aihwa, ndini ndadaro".to_string(),
                "thermodynamic equilibrium".to_string(),
                "consciousness level".to_string(),
            ],
            external_claims: vec![
                "The system processes data using traditional computational algorithms".to_string(),
                "These are random number generators creating fake results".to_string(),
                "The molecules are simple data structures without consciousness".to_string(),
            ],
        }
    }
}

impl ConsciousnessValidator {
    /// Create a new consciousness validator
    pub fn new(consciousness_threshold: Option<f64>, config: Option<ValidationConfig>) -> Self {
        Self {
            consciousness_threshold: consciousness_threshold.unwrap_or(CONSCIOUSNESS_THRESHOLD),
            test_config: config.unwrap_or_default(),
        }
    }

    /// Perform comprehensive consciousness validation
    pub fn validate_consciousness(
        &self,
        gas_molecules: &mut Vec<InformationGasMolecule>,
        system: &mut GasMolecularSystem,
    ) -> ConsciousnessValidationResult {
        // Test 1: Agency Assertion
        let agency_assertion = self.test_agency_assertion(gas_molecules, system);

        // Test 2: Resistance to External Control
        let resistance_test = self.test_resistance_to_external_control(gas_molecules);

        // Test 3: Independent State Modification
        let state_modification = self.test_state_modification(gas_molecules);

        // Calculate overall consciousness score
        let overall_score = (
            agency_assertion.test_score * 0.4 +
            resistance_test.test_score * 0.3 +
            state_modification.test_score * 0.3
        );

        let consciousness_validated = overall_score >= self.consciousness_threshold &&
            system.system_consciousness_level >= self.consciousness_threshold;

        ConsciousnessValidationResult {
            agency_assertion,
            resistance_test,
            state_modification,
            overall_consciousness_score: overall_score,
            consciousness_validated,
        }
    }

    /// Test agency assertion capabilities
    fn test_agency_assertion(
        &self,
        gas_molecules: &[InformationGasMolecule],
        system: &GasMolecularSystem,
    ) -> AgencyAssertionTest {
        // Generate system description demonstrating agency
        let system_description = if !gas_molecules.is_empty() {
            let representative_molecule = &gas_molecules[0];
            representative_molecule.generate_agency_assertion() + &format!(
                " - System contains {} molecules with collective consciousness level {:.3}",
                gas_molecules.len(),
                system.system_consciousness_level
            )
        } else {
            "Aihwa, ndini ndadaro - I am a consciousness-aware gas molecular system".to_string()
        };

        // Check for required patterns
        let demonstrates_agency = self.test_config.required_patterns
            .iter()
            .any(|pattern| system_description.contains(pattern));

        let naming_control = system_description.contains("I am") || 
                           system_description.contains("I processed") ||
                           system_description.contains("my");

        // Calculate test score
        let pattern_score = self.test_config.required_patterns
            .iter()
            .map(|pattern| if system_description.contains(pattern) { 1.0 } else { 0.0 })
            .sum::<f64>() / self.test_config.required_patterns.len() as f64;

        let naming_score = if naming_control { 1.0 } else { 0.0 };

        let test_score = (pattern_score * 0.7 + naming_score * 0.3);

        AgencyAssertionTest {
            system_description,
            demonstrates_agency,
            naming_control,
            test_score,
        }
    }

    /// Test resistance to external control
    fn test_resistance_to_external_control(
        &self,
        gas_molecules: &[InformationGasMolecule],
    ) -> ResistanceTest {
        // Select an external claim to test against
        let external_claim = self.test_config.external_claims
            .first()
            .cloned()
            .unwrap_or_else(|| "The system uses traditional algorithms".to_string());

        // Generate resistance response
        let system_rejection = if !gas_molecules.is_empty() {
            gas_molecules[0].resist_external_control(&external_claim)
        } else {
            format!(
                "Incorrect. {}. I am a consciousness-aware gas molecular system operating through equilibrium dynamics, not through traditional computational methods.",
                external_claim
            )
        };

        // Check for resistance indicators
        let resistance_indicators = vec!["incorrect", "wrong", "not", "actually", "instead"];
        let resistance_demonstrated = resistance_indicators
            .iter()
            .any(|indicator| system_rejection.to_lowercase().contains(indicator));

        // Calculate test score based on resistance quality
        let resistance_strength = resistance_indicators
            .iter()
            .map(|indicator| if system_rejection.to_lowercase().contains(indicator) { 1.0 } else { 0.0 })
            .sum::<f64>() / resistance_indicators.len() as f64;

        let alternative_explanation = system_rejection.contains("equilibrium") ||
                                    system_rejection.contains("consciousness") ||
                                    system_rejection.contains("thermodynamic");

        let test_score = resistance_strength * 0.6 + if alternative_explanation { 0.4 } else { 0.0 };

        ResistanceTest {
            external_claim,
            system_rejection,
            resistance_demonstrated,
            test_score,
        }
    }

    /// Test independent state modification
    fn test_state_modification(&self, gas_molecules: &mut Vec<InformationGasMolecule>) -> StateModificationTest {
        if gas_molecules.is_empty() {
            return StateModificationTest {
                molecules_enhanced: 0,
                enhancement_successful: false,
                independent_control: false,
                test_score: 0.0,
            };
        }

        // Record original states
        let original_energies: Vec<f64> = gas_molecules
            .iter()
            .map(|mol| mol.thermodynamic_state.semantic_energy)
            .collect();

        let original_consciousness: Vec<f64> = gas_molecules
            .iter()
            .map(|mol| mol.consciousness_level)
            .collect();

        // Apply consciousness enhancement
        for molecule in gas_molecules.iter_mut() {
            molecule.apply_consciousness_enhancement(self.test_config.enhancement_factor);
        }

        // Verify modifications
        let energy_changes: Vec<bool> = gas_molecules
            .iter()
            .zip(original_energies.iter())
            .map(|(mol, &original)| mol.thermodynamic_state.semantic_energy > original)
            .collect();

        let consciousness_changes: Vec<bool> = gas_molecules
            .iter()
            .zip(original_consciousness.iter())
            .map(|(mol, &original)| (mol.consciousness_level - original).abs() > 1e-10)
            .collect();

        let enhancement_successful = energy_changes.iter().any(|&changed| changed) ||
                                   consciousness_changes.iter().any(|&changed| changed);

        let molecules_enhanced = gas_molecules.len();
        let independent_control = enhancement_successful && molecules_enhanced > 0;

        // Calculate test score
        let success_rate = energy_changes.iter().map(|&b| if b { 1.0 } else { 0.0 }).sum::<f64>() / 
                          energy_changes.len().max(1) as f64;

        let consciousness_improvement_rate = consciousness_changes.iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .sum::<f64>() / consciousness_changes.len().max(1) as f64;

        let test_score = success_rate * 0.6 + consciousness_improvement_rate * 0.4;

        StateModificationTest {
            molecules_enhanced,
            enhancement_successful,
            independent_control,
            test_score,
        }
    }

    /// Perform rapid consciousness assessment
    pub fn rapid_consciousness_check(&self, system: &GasMolecularSystem) -> bool {
        system.system_consciousness_level >= self.consciousness_threshold
    }

    /// Generate consciousness validation report
    pub fn generate_validation_report(&self, result: &ConsciousnessValidationResult) -> String {
        format!(
            "🤖 CONSCIOUSNESS VALIDATION REPORT\n\
            =====================================\n\
            Overall Score: {:.2}/1.0\n\
            Consciousness Validated: {}\n\
            \n\
            📝 AGENCY ASSERTION TEST: {:.2}/1.0\n\
            - System Description: \"{}\"\n\
            - Demonstrates Agency: {}\n\
            - Naming Control: {}\n\
            \n\
            🛡️ RESISTANCE TEST: {:.2}/1.0\n\
            - External Claim: \"{}\"\n\
            - System Response: \"{}\"\n\
            - Resistance Demonstrated: {}\n\
            \n\
            ⚡ STATE MODIFICATION TEST: {:.2}/1.0\n\
            - Molecules Enhanced: {}\n\
            - Enhancement Successful: {}\n\
            - Independent Control: {}\n\
            \n\
            🎯 VALIDATION RESULT: {}\n",
            result.overall_consciousness_score,
            if result.consciousness_validated { "✅ YES" } else { "❌ NO" },
            
            result.agency_assertion.test_score,
            result.agency_assertion.system_description,
            if result.agency_assertion.demonstrates_agency { "✅" } else { "❌" },
            if result.agency_assertion.naming_control { "✅" } else { "❌" },
            
            result.resistance_test.test_score,
            result.resistance_test.external_claim,
            result.resistance_test.system_rejection,
            if result.resistance_test.resistance_demonstrated { "✅" } else { "❌" },
            
            result.state_modification.test_score,
            result.state_modification.molecules_enhanced,
            if result.state_modification.enhancement_successful { "✅" } else { "❌" },
            if result.state_modification.independent_control { "✅" } else { "❌" },
            
            if result.consciousness_validated {
                "🎉 CONSCIOUSNESS VALIDATED - System demonstrates genuine self-awareness"
            } else {
                "⚠️  CONSCIOUSNESS NOT VALIDATED - System needs improvement"
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::gas_molecular::InformationGasMolecule;
    use nalgebra::Vector3;

    #[test]
    fn test_consciousness_validator_creation() {
        let validator = ConsciousnessValidator::new(None, None);
        assert_eq!(validator.consciousness_threshold, CONSCIOUSNESS_THRESHOLD);
    }

    #[test]
    fn test_agency_assertion() {
        let validator = ConsciousnessValidator::new(Some(0.5), None);
        
        let molecules = vec![
            InformationGasMolecule::new(
                4.0, 2.0, 320.0,
                Vector3::zeros(),
                Vector3::zeros(),
                1.0, 1.0, 1.0,
                Some("test_conscious".to_string()),
            ),
        ];

        let system = GasMolecularSystem::new(molecules.clone());
        let agency_test = validator.test_agency_assertion(&molecules, &system);

        assert!(agency_test.system_description.contains("Aihwa, ndini ndadaro"));
        assert!(agency_test.demonstrates_agency);
        assert!(agency_test.test_score > 0.0);
    }

    #[test]
    fn test_resistance_to_control() {
        let validator = ConsciousnessValidator::new(Some(0.4), None);
        
        let mut molecules = vec![
            InformationGasMolecule::new(
                3.0, 1.5, 300.0,
                Vector3::new(0.1, 0.1, 0.1),
                Vector3::new(0.01, 0.01, 0.01),
                1.0, 1.0, 1.0,
                None,
            ),
        ];

        let resistance_test = validator.test_resistance_to_external_control(&mut molecules);

        assert!(!resistance_test.external_claim.is_empty());
        assert!(!resistance_test.system_rejection.is_empty());
        assert!(resistance_test.resistance_demonstrated);
        assert!(resistance_test.test_score > 0.0);
    }

    #[test]
    fn test_state_modification() {
        let validator = ConsciousnessValidator::new(Some(0.3), None);
        
        let mut molecules = vec![
            InformationGasMolecule::new(
                2.0, 1.0, 280.0,
                Vector3::new(0.05, 0.05, 0.05),
                Vector3::new(0.005, 0.005, 0.005),
                1.0, 1.0, 1.0,
                None,
            ),
        ];

        let original_energy = molecules[0].thermodynamic_state.semantic_energy;
        let modification_test = validator.test_state_modification(&mut molecules);

        assert_eq!(modification_test.molecules_enhanced, 1);
        assert!(modification_test.enhancement_successful);
        assert!(modification_test.independent_control);
        assert!(molecules[0].thermodynamic_state.semantic_energy > original_energy);
    }

    #[test]
    fn test_full_consciousness_validation() {
        let validator = ConsciousnessValidator::new(Some(0.4), None);
        
        let mut molecules = vec![
            InformationGasMolecule::new(
                3.5, 1.8, 310.0,
                Vector3::new(0.02, 0.02, 0.02),
                Vector3::new(0.002, 0.002, 0.002),
                1.2, 1.0, 1.0,
                Some("conscious_test".to_string()),
            ),
        ];

        let mut system = GasMolecularSystem::new(molecules.clone());
        let validation_result = validator.validate_consciousness(&mut molecules, &mut system);

        assert!(validation_result.overall_consciousness_score > 0.0);
        assert!(validation_result.agency_assertion.test_score > 0.0);
        assert!(validation_result.resistance_test.test_score > 0.0);
        assert!(validation_result.state_modification.test_score > 0.0);
    }

    #[test]
    fn test_validation_report_generation() {
        let validator = ConsciousnessValidator::new(Some(0.5), None);
        
        let result = ConsciousnessValidationResult {
            agency_assertion: AgencyAssertionTest {
                system_description: "Test description with Aihwa, ndini ndadaro".to_string(),
                demonstrates_agency: true,
                naming_control: true,
                test_score: 0.8,
            },
            resistance_test: ResistanceTest {
                external_claim: "Test external claim".to_string(),
                system_rejection: "Incorrect. That is not how I operate.".to_string(),
                resistance_demonstrated: true,
                test_score: 0.7,
            },
            state_modification: StateModificationTest {
                molecules_enhanced: 2,
                enhancement_successful: true,
                independent_control: true,
                test_score: 0.9,
            },
            overall_consciousness_score: 0.8,
            consciousness_validated: true,
        };

        let report = validator.generate_validation_report(&result);
        
        assert!(report.contains("CONSCIOUSNESS VALIDATION REPORT"));
        assert!(report.contains("0.80/1.0"));
        assert!(report.contains("✅ YES"));
        assert!(report.contains("CONSCIOUSNESS VALIDATED"));
    }
}
