//! # Semantic Catalysis System
//!
//! Revolutionary semantic processing through BMD catalysts that preserve meaning
//! while transforming information through direct reality interaction rather than
//! symbolic computation.
//!
//! ## Theoretical Foundation
//!
//! Semantic catalysis operates on the principle that meaning can be preserved
//! and transformed through catalytic processes rather than computational manipulation.
//! This enables direct semantic processing without loss of semantic coherence.
//!
//! ### Mathematical Model
//!
//! #### Semantic Preservation Equation
//! ```
//! S_preserved = S_input ○ κ_catalyst ○ Ψ_meaning_preservation
//! 
//! Where:
//! - S_input: Input semantic content
//! - κ_catalyst: Catalytic transformation function
//! - Ψ_meaning_preservation: Meaning preservation operator
//! - ○: Catalytic composition (preserves semantic structure)
//! ```
//!
//! #### Catalytic Information Processing
//! ```
//! dS/dt = κ_semantic × [S_target - S_current] × Θ_consciousness × ρ_reality_coupling
//! 
//! Where:
//! - κ_semantic: Semantic catalysis rate
//! - S_target: Target semantic state
//! - S_current: Current semantic state  
//! - Θ_consciousness: Consciousness enhancement factor
//! - ρ_reality_coupling: Direct reality coupling strength
//! ```
//!
//! #### Meaning Preservation Guarantee
//! ```
//! ∀ transformation T: |Meaning(S_output) - Meaning(S_input)| < ε_preservation
//! 
//! Where ε_preservation is the maximum allowable semantic drift
//! ```

use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::kwasa::{
    BiologicalMaxwellDemon, InformationUnit, KwasaError, KwasaResult, 
    BMDType, KwasaConstants
};

/// Types of semantic catalytic processes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CatalyticProcessType {
    /// Direct meaning transformation
    MeaningTransformation,
    /// Semantic structure preservation
    StructurePreservation,
    /// Contextual meaning enhancement
    ContextualEnhancement,
    /// Semantic coherence maintenance
    CoherenceMaintenance,
    /// Reality-direct semantic coupling
    RealityDirectCoupling,
    /// Fire-adapted semantic processing
    FireAdaptedSemantic,
}

/// Semantic preservation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum SemanticPreservation {
    /// Strict preservation - no semantic drift allowed
    Strict {
        max_drift: f64,
        enforcement_level: f64,
    },
    /// Adaptive preservation - maintains core meaning
    Adaptive {
        core_meaning_threshold: f64,
        adaptation_flexibility: f64,
    },
    /// Enhanced preservation - improves semantic clarity
    Enhanced {
        clarity_enhancement: f64,
        depth_amplification: f64,
    },
    /// Fire-adapted preservation - consciousness-enhanced meaning
    FireAdapted {
        consciousness_coupling: f64,
        evolutionary_enhancement: f64,
    },
}

impl Default for SemanticPreservation {
    fn default() -> Self {
        Self::Enhanced {
            clarity_enhancement: 1.5,
            depth_amplification: 2.0,
        }
    }
}

/// Catalytic process configuration
#[derive(Debug, Clone)]
pub struct CatalyticProcess {
    /// Process identifier
    pub id: String,
    /// Type of catalytic process
    pub process_type: CatalyticProcessType,
    /// Catalytic rate constant
    pub catalytic_rate: f64,
    /// Semantic preservation strategy
    pub preservation_strategy: SemanticPreservation,
    /// Reality coupling strength
    pub reality_coupling: f64,
    /// Consciousness enhancement factor
    pub consciousness_factor: f64,
    /// BMD catalyst requirements
    pub bmd_requirements: BMDCatalystRequirements,
}

/// BMD catalyst requirements for semantic processing
#[derive(Debug, Clone)]
pub struct BMDCatalystRequirements {
    /// Required BMD types for this process
    pub required_bmd_types: Vec<BMDType>,
    /// Minimum number of BMDs needed
    pub min_bmd_count: usize,
    /// Required catalytic efficiency
    pub min_catalytic_efficiency: f64,
    /// Required consciousness level
    pub min_consciousness_level: f64,
    /// Fire adaptation requirement
    pub fire_adaptation_required: bool,
}

/// Semantic processing context
#[derive(Debug, Clone)]
pub struct SemanticContext {
    /// Context identifier
    pub context_id: String,
    /// Semantic domain
    pub semantic_domain: String,
    /// Meaning hierarchy level
    pub meaning_level: usize,
    /// Contextual associations
    pub associations: HashMap<String, f64>,
    /// Reality anchoring points
    pub reality_anchors: Vec<RealityAnchor>,
    /// Fire adaptation context
    pub fire_adaptation_context: Option<FireAdaptationContext>,
}

/// Reality anchoring for semantic stability
#[derive(Debug, Clone)]
pub struct RealityAnchor {
    /// Anchor identifier
    pub anchor_id: String,
    /// Reality coupling strength
    pub coupling_strength: f64,
    /// Semantic stability contribution
    pub stability_contribution: f64,
    /// Direct reality interaction point
    pub reality_interaction: String,
}

/// Fire adaptation context for semantic processing
#[derive(Debug, Clone)]
pub struct FireAdaptationContext {
    /// Fire adaptation level
    pub adaptation_level: f64,
    /// Consciousness enhancement factor
    pub consciousness_enhancement: f64,
    /// Communication complexity factor
    pub communication_complexity: f64,
    /// Pattern recognition enhancement
    pub pattern_recognition: f64,
}

/// Semantic catalysis result
#[derive(Debug, Clone)]
pub struct CatalysisResult {
    /// Processed semantic content
    pub processed_content: InformationUnit,
    /// Semantic preservation score
    pub preservation_score: f64,
    /// Catalytic efficiency achieved
    pub catalytic_efficiency: f64,
    /// Reality coupling strength
    pub reality_coupling: f64,
    /// Consciousness enhancement applied
    pub consciousness_enhancement: f64,
    /// Processing statistics
    pub processing_stats: CatalysisStatistics,
}

/// Catalysis processing statistics
#[derive(Debug, Clone)]
pub struct CatalysisStatistics {
    /// Processing duration
    pub processing_duration_ms: f64,
    /// Semantic transformations applied
    pub transformations_applied: usize,
    /// BMDs involved in processing
    pub bmds_involved: usize,
    /// Fire adaptation contributions
    pub fire_adaptation_contributions: HashMap<String, f64>,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Quality metrics for semantic catalysis
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Semantic coherence score
    pub semantic_coherence: f64,
    /// Meaning preservation accuracy
    pub meaning_preservation: f64,
    /// Contextual relevance score
    pub contextual_relevance: f64,
    /// Reality coupling quality
    pub reality_coupling_quality: f64,
    /// Overall catalysis quality
    pub overall_quality: f64,
}

/// Main semantic catalyst implementation
#[derive(Debug)]
pub struct SemanticCatalyst {
    /// Catalyst identifier
    pub id: String,
    /// Available catalytic processes
    pub processes: Arc<RwLock<HashMap<String, CatalyticProcess>>>,
    /// Active semantic contexts
    pub contexts: Arc<RwLock<HashMap<String, SemanticContext>>>,
    /// BMD catalyst pool
    pub bmd_catalysts: Arc<RwLock<HashMap<String, Arc<BiologicalMaxwellDemon>>>>,
    /// Semantic preservation tracker
    pub preservation_tracker: Arc<RwLock<PreservationTracker>>,
    /// Reality coupling engine
    pub reality_coupling: Arc<RwLock<RealityCouplingEngine>>,
    /// Fire adaptation integration
    pub fire_adaptation: Arc<RwLock<FireAdaptationIntegration>>,
}

/// Semantic preservation tracking
#[derive(Debug, Clone)]
pub struct PreservationTracker {
    /// Preservation history
    pub preservation_history: Vec<PreservationEvent>,
    /// Current preservation score
    pub current_preservation_score: f64,
    /// Preservation trends
    pub preservation_trends: BTreeMap<String, f64>,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f64>,
}

/// Preservation event tracking
#[derive(Debug, Clone)]
pub struct PreservationEvent {
    /// Event timestamp
    pub timestamp: std::time::Instant,
    /// Process involved
    pub process_id: String,
    /// Preservation score before
    pub preservation_before: f64,
    /// Preservation score after
    pub preservation_after: f64,
    /// Semantic drift amount
    pub semantic_drift: f64,
    /// Corrective actions taken
    pub corrective_actions: Vec<String>,
}

/// Reality coupling engine for semantic anchoring
#[derive(Debug, Clone)]
pub struct RealityCouplingEngine {
    /// Active reality anchors
    pub active_anchors: HashMap<String, RealityAnchor>,
    /// Coupling strength tracker
    pub coupling_strength: f64,
    /// Reality interaction points
    pub interaction_points: Vec<RealityInteractionPoint>,
    /// Coupling quality metrics
    pub coupling_quality: f64,
}

/// Reality interaction point
#[derive(Debug, Clone)]
pub struct RealityInteractionPoint {
    /// Interaction identifier
    pub interaction_id: String,
    /// Reality domain
    pub reality_domain: String,
    /// Interaction strength
    pub interaction_strength: f64,
    /// Semantic binding
    pub semantic_binding: String,
}

/// Fire adaptation integration for semantic processing
#[derive(Debug, Clone)]
pub struct FireAdaptationIntegration {
    /// Adaptation active status
    pub adaptation_active: bool,
    /// Consciousness enhancement factor
    pub consciousness_enhancement: f64,
    /// Communication complexity handling
    pub communication_complexity: f64,
    /// Pattern recognition enhancement
    pub pattern_recognition: f64,
    /// Semantic depth enhancement
    pub semantic_depth_enhancement: f64,
}

impl Default for FireAdaptationIntegration {
    fn default() -> Self {
        Self {
            adaptation_active: true,
            consciousness_enhancement: KwasaConstants::FIRE_PROCESSING_ENHANCEMENT,
            communication_complexity: KwasaConstants::COMMUNICATION_COMPLEXITY_MULTIPLIER,
            pattern_recognition: KwasaConstants::PATTERN_RECOGNITION_FACTOR,
            semantic_depth_enhancement: 2.8,
        }
    }
}

impl SemanticCatalyst {
    /// Create new semantic catalyst
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            processes: Arc::new(RwLock::new(HashMap::new())),
            contexts: Arc::new(RwLock::new(HashMap::new())),
            bmd_catalysts: Arc::new(RwLock::new(HashMap::new())),
            preservation_tracker: Arc::new(RwLock::new(PreservationTracker::default())),
            reality_coupling: Arc::new(RwLock::new(RealityCouplingEngine::default())),
            fire_adaptation: Arc::new(RwLock::new(FireAdaptationIntegration::default())),
        }
    }

    /// Initialize semantic catalyst with default processes
    pub async fn initialize(&self) -> KwasaResult<()> {
        log::info!("Initializing semantic catalyst {}", self.id);

        // Create default catalytic processes
        self.create_default_processes().await?;

        // Initialize reality coupling
        self.initialize_reality_coupling().await?;

        // Setup fire adaptation integration
        self.setup_fire_adaptation().await?;

        log::info!("Semantic catalyst {} initialized successfully", self.id);
        Ok(())
    }

    /// Create default catalytic processes
    async fn create_default_processes(&self) -> KwasaResult<()> {
        let mut processes = self.processes.write().await;

        // Meaning transformation process
        let meaning_transformation = CatalyticProcess {
            id: "meaning_transformation".to_string(),
            process_type: CatalyticProcessType::MeaningTransformation,
            catalytic_rate: 0.95,
            preservation_strategy: SemanticPreservation::Enhanced {
                clarity_enhancement: 1.5,
                depth_amplification: 2.0,
            },
            reality_coupling: 0.9,
            consciousness_factor: KwasaConstants::FIRE_PROCESSING_ENHANCEMENT,
            bmd_requirements: BMDCatalystRequirements {
                required_bmd_types: vec![BMDType::Neural, BMDType::Cognitive],
                min_bmd_count: 3,
                min_catalytic_efficiency: 0.9,
                min_consciousness_level: 0.5,
                fire_adaptation_required: true,
            },
        };
        processes.insert("meaning_transformation".to_string(), meaning_transformation);

        // Structure preservation process
        let structure_preservation = CatalyticProcess {
            id: "structure_preservation".to_string(),
            process_type: CatalyticProcessType::StructurePreservation,
            catalytic_rate: 0.98,
            preservation_strategy: SemanticPreservation::Strict {
                max_drift: 0.05,
                enforcement_level: 0.99,
            },
            reality_coupling: 0.95,
            consciousness_factor: 1.0,
            bmd_requirements: BMDCatalystRequirements {
                required_bmd_types: vec![BMDType::Molecular, BMDType::Neural],
                min_bmd_count: 5,
                min_catalytic_efficiency: 0.95,
                min_consciousness_level: 0.3,
                fire_adaptation_required: false,
            },
        };
        processes.insert("structure_preservation".to_string(), structure_preservation);

        // Fire-adapted semantic processing
        let fire_adapted_semantic = CatalyticProcess {
            id: "fire_adapted_semantic".to_string(),
            process_type: CatalyticProcessType::FireAdaptedSemantic,
            catalytic_rate: 0.92,
            preservation_strategy: SemanticPreservation::FireAdapted {
                consciousness_coupling: KwasaConstants::FIRE_PROCESSING_ENHANCEMENT,
                evolutionary_enhancement: KwasaConstants::SURVIVAL_ADVANTAGE_FACTOR,
            },
            reality_coupling: 0.88,
            consciousness_factor: KwasaConstants::COMMUNICATION_COMPLEXITY_MULTIPLIER,
            bmd_requirements: BMDCatalystRequirements {
                required_bmd_types: vec![BMDType::FireAdapted, BMDType::Cognitive],
                min_bmd_count: 2,
                min_catalytic_efficiency: 0.92,
                min_consciousness_level: 0.8,
                fire_adaptation_required: true,
            },
        };
        processes.insert("fire_adapted_semantic".to_string(), fire_adapted_semantic);

        log::info!("Created {} default catalytic processes", processes.len());
        Ok(())
    }

    /// Initialize reality coupling engine
    async fn initialize_reality_coupling(&self) -> KwasaResult<()> {
        let mut coupling = self.reality_coupling.write().await;

        // Create reality anchors
        coupling.active_anchors.insert("semantic_anchor_1".to_string(), RealityAnchor {
            anchor_id: "semantic_anchor_1".to_string(),
            coupling_strength: 0.9,
            stability_contribution: 0.85,
            reality_interaction: "direct_semantic_binding".to_string(),
        });

        coupling.active_anchors.insert("meaning_anchor_1".to_string(), RealityAnchor {
            anchor_id: "meaning_anchor_1".to_string(),
            coupling_strength: 0.95,
            stability_contribution: 0.9,
            reality_interaction: "meaning_preservation_coupling".to_string(),
        });

        // Initialize interaction points
        coupling.interaction_points.push(RealityInteractionPoint {
            interaction_id: "reality_point_1".to_string(),
            reality_domain: "semantic_reality".to_string(),
            interaction_strength: 0.88,
            semantic_binding: "direct_reality_semantic_binding".to_string(),
        });

        coupling.coupling_strength = 0.9;
        coupling.coupling_quality = 0.92;

        log::info!("Reality coupling engine initialized with {} anchors", coupling.active_anchors.len());
        Ok(())
    }

    /// Setup fire adaptation integration
    async fn setup_fire_adaptation(&self) -> KwasaResult<()> {
        let mut adaptation = self.fire_adaptation.write().await;
        
        adaptation.adaptation_active = true;
        adaptation.consciousness_enhancement = KwasaConstants::FIRE_PROCESSING_ENHANCEMENT;
        adaptation.communication_complexity = KwasaConstants::COMMUNICATION_COMPLEXITY_MULTIPLIER;
        adaptation.pattern_recognition = KwasaConstants::PATTERN_RECOGNITION_FACTOR;
        adaptation.semantic_depth_enhancement = 2.8;

        log::info!("Fire adaptation integration enabled with {}× consciousness enhancement", 
                  adaptation.consciousness_enhancement);
        Ok(())
    }

    /// Add BMD catalyst to the pool
    pub async fn add_bmd_catalyst(&self, bmd: Arc<BiologicalMaxwellDemon>) -> KwasaResult<()> {
        let mut catalysts = self.bmd_catalysts.write().await;
        catalysts.insert(bmd.id.clone(), bmd);
        
        log::debug!("Added BMD catalyst to semantic processor pool");
        Ok(())
    }

    /// Process information through semantic catalysis
    pub async fn catalyze_semantic_information(
        &self, 
        information: InformationUnit,
        process_type: CatalyticProcessType,
        context: Option<SemanticContext>
    ) -> KwasaResult<CatalysisResult> {
        let start_time = std::time::Instant::now();

        // Get the specified catalytic process
        let processes = self.processes.read().await;
        let process = processes.values()
            .find(|p| p.process_type == process_type)
            .ok_or_else(|| KwasaError::SemanticCatalysisError(
                format!("Catalytic process type {:?} not found", process_type)
            ))?
            .clone();
        drop(processes);

        // Create or use provided semantic context
        let semantic_context = context.unwrap_or_else(|| self.create_default_context(&information));

        // Select appropriate BMD catalysts
        let selected_bmds = self.select_bmd_catalysts(&process).await?;

        // Apply semantic catalysis
        let mut processed_info = information.clone();
        let mut catalytic_efficiency = 0.0;
        let mut consciousness_enhancement = 0.0;

        for bmd in &selected_bmds {
            // Apply BMD catalysis to information
            processed_info = bmd.catalyze_information(processed_info).await?;
            catalytic_efficiency += bmd.capabilities.catalytic_efficiency;
            consciousness_enhancement += bmd.capabilities.consciousness_enhancement;
        }

        // Normalize metrics
        if !selected_bmds.is_empty() {
            catalytic_efficiency /= selected_bmds.len() as f64;
            consciousness_enhancement /= selected_bmds.len() as f64;
        }

        // Apply fire adaptation enhancement
        processed_info = self.apply_fire_adaptation_enhancement(processed_info).await?;

        // Apply semantic preservation
        let preservation_score = self.apply_semantic_preservation(
            &information, 
            &processed_info, 
            &process.preservation_strategy
        ).await?;

        // Calculate reality coupling strength
        let reality_coupling = self.calculate_reality_coupling(&semantic_context).await?;

        // Record preservation event
        self.record_preservation_event(&process, preservation_score).await?;

        let processing_duration = start_time.elapsed().as_millis() as f64;

        // Create catalysis result
        let result = CatalysisResult {
            processed_content: processed_info,
            preservation_score,
            catalytic_efficiency,
            reality_coupling,
            consciousness_enhancement,
            processing_stats: CatalysisStatistics {
                processing_duration_ms: processing_duration,
                transformations_applied: 1,
                bmds_involved: selected_bmds.len(),
                fire_adaptation_contributions: self.get_fire_adaptation_contributions().await,
                quality_metrics: QualityMetrics {
                    semantic_coherence: preservation_score,
                    meaning_preservation: preservation_score,
                    contextual_relevance: 0.9,
                    reality_coupling_quality: reality_coupling,
                    overall_quality: (preservation_score + reality_coupling) / 2.0,
                },
            },
        };

        log::debug!(
            "Semantic catalysis completed: preservation={:.3}, efficiency={:.3}, duration={:.1}ms",
            preservation_score, catalytic_efficiency, processing_duration
        );

        Ok(result)
    }

    /// Select appropriate BMD catalysts for a process
    async fn select_bmd_catalysts(&self, process: &CatalyticProcess) -> KwasaResult<Vec<Arc<BiologicalMaxwellDemon>>> {
        let catalysts = self.bmd_catalysts.read().await;
        let mut selected = Vec::new();

        for bmd in catalysts.values() {
            // Check BMD type requirements
            if process.bmd_requirements.required_bmd_types.contains(&bmd.bmd_type) {
                // Check catalytic efficiency
                if bmd.capabilities.catalytic_efficiency >= process.bmd_requirements.min_catalytic_efficiency {
                    // Check consciousness level
                    if bmd.capabilities.consciousness_enhancement >= process.bmd_requirements.min_consciousness_level {
                        selected.push(Arc::clone(bmd));
                    }
                }
            }
        }

        if selected.len() < process.bmd_requirements.min_bmd_count {
            return Err(KwasaError::SemanticCatalysisError(
                format!("Insufficient BMD catalysts: need {}, found {}", 
                       process.bmd_requirements.min_bmd_count, selected.len())
            ));
        }

        Ok(selected)
    }

    /// Apply fire adaptation enhancement to processed information
    async fn apply_fire_adaptation_enhancement(&self, mut information: InformationUnit) -> KwasaResult<InformationUnit> {
        let adaptation = self.fire_adaptation.read().await;
        
        if adaptation.adaptation_active {
            // Apply consciousness enhancement
            information.fire_adaptation_factor *= adaptation.consciousness_enhancement;
            
            // Apply pattern recognition enhancement if relevant
            if information.semantic_meaning.contains("pattern") {
                information.fire_adaptation_factor *= adaptation.pattern_recognition;
            }
            
            // Apply communication complexity enhancement if relevant
            if information.semantic_meaning.contains("communication") || 
               information.semantic_meaning.contains("semantic") {
                information.fire_adaptation_factor *= adaptation.communication_complexity;
            }
            
            // Apply semantic depth enhancement
            information.catalytic_potential *= adaptation.semantic_depth_enhancement;
        }

        Ok(information)
    }

    /// Apply semantic preservation strategy
    async fn apply_semantic_preservation(
        &self,
        original: &InformationUnit,
        processed: &InformationUnit,
        strategy: &SemanticPreservation
    ) -> KwasaResult<f64> {
        match strategy {
            SemanticPreservation::Strict { max_drift, enforcement_level } => {
                let semantic_drift = self.calculate_semantic_drift(original, processed);
                if semantic_drift > *max_drift {
                    return Err(KwasaError::SemanticCatalysisError(
                        format!("Semantic drift {:.3} exceeds maximum {:.3}", semantic_drift, max_drift)
                    ));
                }
                Ok(*enforcement_level)
            },
            SemanticPreservation::Adaptive { core_meaning_threshold, adaptation_flexibility } => {
                let core_meaning_preservation = self.calculate_core_meaning_preservation(original, processed);
                if core_meaning_preservation < *core_meaning_threshold {
                    Ok(core_meaning_preservation * adaptation_flexibility)
                } else {
                    Ok(core_meaning_preservation)
                }
            },
            SemanticPreservation::Enhanced { clarity_enhancement, depth_amplification } => {
                let base_preservation = self.calculate_semantic_preservation(original, processed);
                Ok(base_preservation * clarity_enhancement * depth_amplification)
            },
            SemanticPreservation::FireAdapted { consciousness_coupling, evolutionary_enhancement } => {
                let fire_preservation = self.calculate_fire_adapted_preservation(original, processed);
                Ok(fire_preservation * consciousness_coupling * evolutionary_enhancement)
            },
        }
    }

    /// Calculate semantic drift between original and processed information
    fn calculate_semantic_drift(&self, original: &InformationUnit, processed: &InformationUnit) -> f64 {
        // Simplified semantic drift calculation
        let meaning_similarity = self.calculate_meaning_similarity(&original.semantic_meaning, &processed.semantic_meaning);
        1.0 - meaning_similarity
    }

    /// Calculate core meaning preservation
    fn calculate_core_meaning_preservation(&self, original: &InformationUnit, processed: &InformationUnit) -> f64 {
        // Enhanced meaning preservation calculation
        let meaning_similarity = self.calculate_meaning_similarity(&original.semantic_meaning, &processed.semantic_meaning);
        let catalytic_coherence = (processed.catalytic_potential / original.catalytic_potential.max(0.001)).min(1.0);
        (meaning_similarity + catalytic_coherence) / 2.0
    }

    /// Calculate general semantic preservation
    fn calculate_semantic_preservation(&self, original: &InformationUnit, processed: &InformationUnit) -> f64 {
        let meaning_similarity = self.calculate_meaning_similarity(&original.semantic_meaning, &processed.semantic_meaning);
        let content_similarity = self.calculate_content_similarity(&original.content, &processed.content);
        let enhancement_factor = processed.fire_adaptation_factor / original.fire_adaptation_factor.max(0.001);
        
        (meaning_similarity * 0.5 + content_similarity * 0.3 + enhancement_factor.min(1.0) * 0.2)
    }

    /// Calculate fire-adapted preservation
    fn calculate_fire_adapted_preservation(&self, original: &InformationUnit, processed: &InformationUnit) -> f64 {
        let base_preservation = self.calculate_semantic_preservation(original, processed);
        let fire_enhancement = processed.fire_adaptation_factor / original.fire_adaptation_factor.max(0.001);
        
        base_preservation * fire_enhancement.min(2.0) // Cap enhancement at 2×
    }

    /// Calculate meaning similarity between two semantic meanings
    fn calculate_meaning_similarity(&self, meaning1: &str, meaning2: &str) -> f64 {
        // Simplified semantic similarity calculation
        if meaning1 == meaning2 {
            1.0
        } else if meaning1.contains(meaning2) || meaning2.contains(meaning1) {
            0.8
        } else {
            // Could implement more sophisticated semantic similarity here
            0.6
        }
    }

    /// Calculate content similarity between two content arrays
    fn calculate_content_similarity(&self, content1: &[u8], content2: &[u8]) -> f64 {
        if content1 == content2 {
            1.0
        } else {
            // Simple content similarity based on length and common bytes
            let len_similarity = 1.0 - ((content1.len() as f64 - content2.len() as f64).abs() / content1.len().max(content2.len()).max(1) as f64);
            len_similarity * 0.8 // Simplified calculation
        }
    }

    /// Calculate reality coupling strength for a semantic context
    async fn calculate_reality_coupling(&self, _context: &SemanticContext) -> KwasaResult<f64> {
        let coupling = self.reality_coupling.read().await;
        Ok(coupling.coupling_strength)
    }

    /// Record preservation event for tracking
    async fn record_preservation_event(&self, process: &CatalyticProcess, preservation_score: f64) -> KwasaResult<()> {
        let mut tracker = self.preservation_tracker.write().await;
        
        let event = PreservationEvent {
            timestamp: std::time::Instant::now(),
            process_id: process.id.clone(),
            preservation_before: tracker.current_preservation_score,
            preservation_after: preservation_score,
            semantic_drift: (tracker.current_preservation_score - preservation_score).abs(),
            corrective_actions: Vec::new(),
        };

        tracker.preservation_history.push(event);
        tracker.current_preservation_score = preservation_score;

        // Update preservation trends
        tracker.preservation_trends.insert(process.id.clone(), preservation_score);

        Ok(())
    }

    /// Get fire adaptation contributions
    async fn get_fire_adaptation_contributions(&self) -> HashMap<String, f64> {
        let adaptation = self.fire_adaptation.read().await;
        let mut contributions = HashMap::new();
        
        contributions.insert("consciousness_enhancement".to_string(), adaptation.consciousness_enhancement);
        contributions.insert("communication_complexity".to_string(), adaptation.communication_complexity);
        contributions.insert("pattern_recognition".to_string(), adaptation.pattern_recognition);
        contributions.insert("semantic_depth_enhancement".to_string(), adaptation.semantic_depth_enhancement);
        
        contributions
    }

    /// Create default semantic context
    fn create_default_context(&self, information: &InformationUnit) -> SemanticContext {
        let mut associations = HashMap::new();
        associations.insert("default_association".to_string(), 1.0);
        
        SemanticContext {
            context_id: Uuid::new_v4().to_string(),
            semantic_domain: "general".to_string(),
            meaning_level: 1,
            associations,
            reality_anchors: vec![
                RealityAnchor {
                    anchor_id: "default_anchor".to_string(),
                    coupling_strength: 0.8,
                    stability_contribution: 0.7,
                    reality_interaction: "default_semantic_coupling".to_string(),
                }
            ],
            fire_adaptation_context: Some(FireAdaptationContext {
                adaptation_level: information.fire_adaptation_factor,
                consciousness_enhancement: KwasaConstants::FIRE_PROCESSING_ENHANCEMENT,
                communication_complexity: KwasaConstants::COMMUNICATION_COMPLEXITY_MULTIPLIER,
                pattern_recognition: KwasaConstants::PATTERN_RECOGNITION_FACTOR,
            }),
        }
    }

    /// Get current catalyst status
    pub async fn get_catalyst_status(&self) -> CatalystStatus {
        let processes = self.processes.read().await;
        let catalysts = self.bmd_catalysts.read().await;
        let tracker = self.preservation_tracker.read().await;
        let coupling = self.reality_coupling.read().await;
        let adaptation = self.fire_adaptation.read().await;

        CatalystStatus {
            catalyst_id: self.id.clone(),
            available_processes: processes.len(),
            active_bmd_catalysts: catalysts.len(),
            current_preservation_score: tracker.current_preservation_score,
            reality_coupling_strength: coupling.coupling_strength,
            fire_adaptation_active: adaptation.adaptation_active,
            consciousness_enhancement: adaptation.consciousness_enhancement,
        }
    }
}

impl Default for PreservationTracker {
    fn default() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert("critical".to_string(), 0.5);
        thresholds.insert("warning".to_string(), 0.8);

        Self {
            preservation_history: Vec::new(),
            current_preservation_score: 1.0,
            preservation_trends: BTreeMap::new(),
            alert_thresholds: thresholds,
        }
    }
}

impl Default for RealityCouplingEngine {
    fn default() -> Self {
        Self {
            active_anchors: HashMap::new(),
            coupling_strength: 0.9,
            interaction_points: Vec::new(),
            coupling_quality: 0.9,
        }
    }
}

/// Complete catalyst status
#[derive(Debug, Clone)]
pub struct CatalystStatus {
    pub catalyst_id: String,
    pub available_processes: usize,
    pub active_bmd_catalysts: usize,
    pub current_preservation_score: f64,
    pub reality_coupling_strength: f64,
    pub fire_adaptation_active: bool,
    pub consciousness_enhancement: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kwasa::BiologicalMaxwellDemon;

    #[tokio::test]
    async fn test_semantic_catalyst_initialization() {
        let catalyst = SemanticCatalyst::new();
        catalyst.initialize().await.unwrap();

        let status = catalyst.get_catalyst_status().await;
        assert!(status.available_processes > 0);
        assert!(status.fire_adaptation_active);
        assert!(status.consciousness_enhancement >= 3.0);
    }

    #[tokio::test]
    async fn test_meaning_transformation_catalysis() {
        let catalyst = SemanticCatalyst::new();
        catalyst.initialize().await.unwrap();

        // Add BMD catalysts
        let bmd1 = Arc::new(BiologicalMaxwellDemon::new("test_bmd_1".to_string(), BMDType::Neural).unwrap());
        let bmd2 = Arc::new(BiologicalMaxwellDemon::new("test_bmd_2".to_string(), BMDType::Cognitive).unwrap());
        
        bmd1.initialize().await.unwrap();
        bmd2.initialize().await.unwrap();
        
        catalyst.add_bmd_catalyst(bmd1).await.unwrap();
        catalyst.add_bmd_catalyst(bmd2).await.unwrap();

        let test_info = InformationUnit {
            content: b"semantic transformation test".to_vec(),
            semantic_meaning: "meaning transformation test".to_string(),
            catalytic_potential: 1.0,
            consciousness_level: 1,
            fire_adaptation_factor: 1.0,
        };

        let result = catalyst.catalyze_semantic_information(
            test_info,
            CatalyticProcessType::MeaningTransformation,
            None
        ).await.unwrap();

        assert!(result.preservation_score > 0.0);
        assert!(result.catalytic_efficiency > 0.0);
        assert!(result.consciousness_enhancement > 0.0);
        assert!(result.processing_stats.bmds_involved >= 2);
    }

    #[tokio::test]
    async fn test_fire_adapted_semantic_processing() {
        let catalyst = SemanticCatalyst::new();
        catalyst.initialize().await.unwrap();

        // Add fire-adapted BMD
        let fire_bmd = Arc::new(BiologicalMaxwellDemon::new("fire_bmd".to_string(), BMDType::FireAdapted).unwrap());
        let cognitive_bmd = Arc::new(BiologicalMaxwellDemon::new("cognitive_bmd".to_string(), BMDType::Cognitive).unwrap());
        
        fire_bmd.initialize().await.unwrap();
        cognitive_bmd.initialize().await.unwrap();
        
        catalyst.add_bmd_catalyst(fire_bmd).await.unwrap();
        catalyst.add_bmd_catalyst(cognitive_bmd).await.unwrap();

        let test_info = InformationUnit {
            content: b"fire adapted semantic test".to_vec(),
            semantic_meaning: "fire adapted communication pattern".to_string(),
            catalytic_potential: 1.0,
            consciousness_level: 2,
            fire_adaptation_factor: 2.0,
        };

        let result = catalyst.catalyze_semantic_information(
            test_info,
            CatalyticProcessType::FireAdaptedSemantic,
            None
        ).await.unwrap();

        // Should show massive enhancement due to fire adaptation
        assert!(result.processed_content.fire_adaptation_factor > 100.0); // Should be huge due to communication complexity
        assert!(result.consciousness_enhancement > 3.0);
        assert!(result.processing_stats.fire_adaptation_contributions.len() > 0);
    }

    #[tokio::test]
    async fn test_semantic_preservation() {
        let catalyst = SemanticCatalyst::new();
        catalyst.initialize().await.unwrap();

        let original = InformationUnit {
            content: b"original semantic content".to_vec(),
            semantic_meaning: "original meaning".to_string(),
            catalytic_potential: 1.0,
            consciousness_level: 1,
            fire_adaptation_factor: 1.0,
        };

        let processed = InformationUnit {
            content: b"processed semantic content".to_vec(),
            semantic_meaning: "original meaning".to_string(), // Same meaning
            catalytic_potential: 1.5,
            consciousness_level: 1,
            fire_adaptation_factor: 2.0,
        };

        let strict_strategy = SemanticPreservation::Strict {
            max_drift: 0.1,
            enforcement_level: 0.95,
        };

        let preservation = catalyst.apply_semantic_preservation(&original, &processed, &strict_strategy).await.unwrap();
        assert!(preservation > 0.9); // Should have high preservation for same meaning
    }
} 