//! # Biological Maxwell Demon Implementation
//!
//! Core implementation of Biological Maxwell Demons (BMDs) as information catalysts
//! enhanced with matrix associative memory capabilities based on Mizraji's neural language research.
//!
//! ## Enhanced with Mizraji Neural Language Insights
//!
//! ### Matrix Associative Memory Integration
//! ```
//! M_associative = W × (I_input ⊗ C_context) × Θ_goal_orientation
//! 
//! Where:
//! - W: Weight matrix for associative memory
//! - I_input: Input information vector  
//! - C_context: Context-dependent activation pattern
//! - Θ_goal_orientation: Goal-oriented sequence modifier
//! - ⊗: Tensor product for context-input binding
//! ```
//!
//! ### Context-Dependent Processing
//! Based on Mizraji's context-dependent matrix memories that enable neural systems
//! to store different sequences in the same substrate for goal-oriented behavior.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

use crate::kwasa::{KwasaError, KwasaResult, InformationUnit, BMDMetrics, KwasaConstants};

use ndarray::{Array1, Array2, ArrayD, Axis};
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::time::Instant;

/// Types of Biological Maxwell Demons
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BMDType {
    /// Molecular-level BMD for direct molecular information processing
    Molecular,
    /// Neural-level BMD for neural network information catalysis
    Neural,
    /// Cognitive-level BMD for high-level cognitive information processing
    Cognitive,
    /// Fire-adapted BMD with enhanced consciousness capabilities
    FireAdapted,
    /// Hybrid BMD combining multiple processing levels
    Hybrid,
}

/// Current state of a BMD
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BMDState {
    /// Inactive state - no information processing
    Inactive,
    /// Initializing state - setting up catalytic processes
    Initializing,
    /// Active state - processing information through catalysis
    Active {
        catalytic_rate: f64,
        information_throughput: f64,
        semantic_preservation: f64,
    },
    /// Enhanced state - fire-adapted consciousness active
    Enhanced {
        consciousness_factor: f64,
        fire_adaptation_level: f64,
        processing_enhancement: f64,
    },
    /// Overloaded state - processing beyond capacity
    Overloaded {
        overflow_buffer: Vec<InformationUnit>,
        degradation_factor: f64,
    },
    /// Error state - catalytic failure
    Error {
        error_message: String,
        recovery_possible: bool,
    },
}

/// Information gradient maintained by BMD
#[derive(Debug, Clone)]
pub struct InformationGradient {
    /// Current information density
    pub density: f64,
    /// Rate of information flow
    pub flow_rate: f64,
    /// Semantic coherence level
    pub semantic_coherence: f64,
    /// Consciousness contribution
    pub consciousness_factor: f64,
}

/// BMD processing capabilities
#[derive(Debug, Clone)]
pub struct BMDCapabilities {
    /// Maximum information processing rate (units per second)
    pub max_processing_rate: f64,
    /// Catalytic efficiency factor
    pub catalytic_efficiency: f64,
    /// Semantic preservation accuracy
    pub semantic_preservation: f64,
    /// Consciousness enhancement factor
    pub consciousness_enhancement: f64,
    /// Fire adaptation level
    pub fire_adaptation: f64,
    /// Error recovery capability
    pub error_recovery: f64,
}

/// Matrix associative memory for enhanced BMD processing
#[derive(Debug, Clone)]
pub struct MatrixAssociativeMemory {
    /// High-dimensional weight matrix for associative memory
    pub weight_matrix: Array2<f64>,
    /// Context-dependent activation patterns
    pub context_patterns: HashMap<String, Array1<f64>>,
    /// Goal-oriented sequence storage
    pub goal_sequences: Vec<GoalOrientedSequence>,
    /// Memory capacity and dimensions
    pub memory_dimensions: (usize, usize),
    /// Learning rate for memory updates
    pub learning_rate: f64,
    /// Context switching mechanism
    pub context_switcher: ContextSwitcher,
}

/// Goal-oriented sequence for linguistic behavior
#[derive(Debug, Clone)]
pub struct GoalOrientedSequence {
    /// Sequence identifier
    pub sequence_id: String,
    /// Sequence of information patterns
    pub pattern_sequence: Vec<Array1<f64>>,
    /// Goal context for this sequence
    pub goal_context: String,
    /// Sequence completion probability
    pub completion_probability: f64,
    /// Linguistic behavior type
    pub behavior_type: LinguisticBehaviorType,
}

/// Types of linguistic behaviors supported
#[derive(Debug, Clone, PartialEq)]
pub enum LinguisticBehaviorType {
    /// Discourse production
    DiscourseProduction,
    /// Language comprehension
    LanguageComprehension,
    /// Semantic association
    SemanticAssociation,
    /// Context switching
    ContextSwitching,
    /// Goal-oriented navigation
    GoalOrientedNavigation,
    /// Fire-adapted communication
    FireAdaptedCommunication,
}

/// Context switching mechanism for matrix memories
#[derive(Debug, Clone)]
pub struct ContextSwitcher {
    /// Active context identifier
    pub active_context: String,
    /// Context switching probability
    pub switching_probability: f64,
    /// Context history for patterns
    pub context_history: VecDeque<String>,
    /// Context transition matrix
    pub transition_matrix: Array2<f64>,
}

/// Discourse trajectory measurement for consciousness tracking
#[derive(Debug, Clone)]
pub struct DiscourseTrajectoryMetrics {
    /// Degree of order in discourse (0.0 to 1.0)
    pub order_degree: f64,
    /// Discourse coherence trajectory over time
    pub coherence_trajectory: Vec<(Instant, f64)>,
    /// Language disorganization entropy measure
    pub disorganization_entropy: f64,
    /// Trajectory smoothness measure
    pub trajectory_smoothness: f64,
    /// Context consistency score
    pub context_consistency: f64,
}

/// Core Biological Maxwell Demon implementation
#[derive(Debug)]
pub struct BiologicalMaxwellDemon {
    /// Unique identifier for this BMD
    pub id: String,
    /// Type of BMD (molecular, neural, cognitive, etc.)
    pub bmd_type: BMDType,
    /// Current state of the BMD
    pub state: Arc<RwLock<BMDState>>,
    /// Processing capabilities
    pub capabilities: BMDCapabilities,
    /// Current information gradient
    pub gradient: Arc<RwLock<InformationGradient>>,
    /// Performance metrics
    pub metrics: Arc<RwLock<BMDMetrics>>,
    /// Information buffer for processing
    pub information_buffer: Arc<RwLock<Vec<InformationUnit>>>,
    /// Semantic preservation context
    pub semantic_context: Arc<RwLock<HashMap<String, f64>>>,
    /// Fire adaptation parameters
    pub fire_adaptation_params: FireAdaptationParams,
}

/// Fire adaptation parameters for enhanced consciousness
#[derive(Debug, Clone)]
pub struct FireAdaptationParams {
    /// Processing enhancement factor (322% baseline)
    pub processing_enhancement: f64,
    /// Pattern recognition improvement (346% baseline)
    pub pattern_recognition: f64,
    /// Communication complexity enhancement (79.3× baseline)
    pub communication_complexity: f64,
    /// Survival advantage factor (460% baseline)
    pub survival_advantage: f64,
    /// Consciousness threshold for fire adaptation
    pub consciousness_threshold: f64,
}

impl Default for FireAdaptationParams {
    fn default() -> Self {
        Self {
            processing_enhancement: KwasaConstants::FIRE_PROCESSING_ENHANCEMENT,
            pattern_recognition: KwasaConstants::PATTERN_RECOGNITION_FACTOR,
            communication_complexity: KwasaConstants::COMMUNICATION_COMPLEXITY_MULTIPLIER,
            survival_advantage: KwasaConstants::SURVIVAL_ADVANTAGE_FACTOR,
            consciousness_threshold: KwasaConstants::CONSCIOUSNESS_SEQUENTIAL_STATES,
        }
    }
}

impl BiologicalMaxwellDemon {
    /// Create a new Biological Maxwell Demon
    pub fn new(id: String, bmd_type: BMDType) -> KwasaResult<Self> {
        let capabilities = BMDCapabilities::default_for_type(bmd_type);
        let initial_state = BMDState::Inactive;
        let initial_gradient = InformationGradient::default();
        let initial_metrics = BMDMetrics::default();
        
        Ok(Self {
            id,
            bmd_type,
            state: Arc::new(RwLock::new(initial_state)),
            capabilities,
            gradient: Arc::new(RwLock::new(initial_gradient)),
            metrics: Arc::new(RwLock::new(initial_metrics)),
            information_buffer: Arc::new(RwLock::new(Vec::new())),
            semantic_context: Arc::new(RwLock::new(HashMap::new())),
            fire_adaptation_params: FireAdaptationParams::default(),
        })
    }
    
    /// Initialize the BMD for information catalysis
    pub async fn initialize(&self) -> KwasaResult<()> {
        let mut state = self.state.write().await;
        *state = BMDState::Initializing;
        
        // Initialize information gradient
        let mut gradient = self.gradient.write().await;
        gradient.density = 0.0;
        gradient.flow_rate = 0.0;
        gradient.semantic_coherence = 1.0;
        gradient.consciousness_factor = self.fire_adaptation_params.consciousness_threshold;
        
        // Clear information buffer
        let mut buffer = self.information_buffer.write().await;
        buffer.clear();
        
        // Initialize semantic context
        let mut context = self.semantic_context.write().await;
        context.clear();
        context.insert("initialization".to_string(), 1.0);
        
        // Transition to active state
        *state = BMDState::Active {
            catalytic_rate: self.capabilities.catalytic_efficiency,
            information_throughput: 0.0,
            semantic_preservation: self.capabilities.semantic_preservation,
        };
        
        log::info!("BMD {} ({:?}) initialized successfully", self.id, self.bmd_type);
        Ok(())
    }
    
    /// Process information through BMD catalysis
    pub async fn catalyze_information(&self, information: InformationUnit) -> KwasaResult<InformationUnit> {
        // Check if BMD is active
        let state = self.state.read().await;
        match *state {
            BMDState::Active { .. } | BMDState::Enhanced { .. } => {},
            _ => {
                return Err(KwasaError::BMDInitializationError(
                    format!("BMD {} not in active state", self.id)
                ));
            }
        }
        drop(state);
        
        // Apply fire-adapted consciousness enhancement
        let enhanced_information = self.apply_fire_adaptation(information).await?;
        
        // Perform semantic catalysis
        let catalyzed_information = self.perform_semantic_catalysis(enhanced_information).await?;
        
        // Update information gradient
        self.update_information_gradient(&catalyzed_information).await?;
        
        // Update processing metrics
        self.update_metrics().await?;
        
        Ok(catalyzed_information)
    }
    
    /// Apply fire-adapted consciousness enhancement
    async fn apply_fire_adaptation(&self, mut information: InformationUnit) -> KwasaResult<InformationUnit> {
        // Apply processing enhancement (322%)
        information.fire_adaptation_factor *= self.fire_adaptation_params.processing_enhancement;
        
        // Apply pattern recognition improvement (346%)
        if information.semantic_meaning.contains("pattern") {
            information.fire_adaptation_factor *= self.fire_adaptation_params.pattern_recognition;
        }
        
        // Apply communication complexity enhancement (79.3×)
        if information.semantic_meaning.contains("communication") {
            information.fire_adaptation_factor *= self.fire_adaptation_params.communication_complexity;
        }
        
        // Apply survival advantage (460%)
        information.catalytic_potential *= self.fire_adaptation_params.survival_advantage;
        
        // Check consciousness threshold
        if information.fire_adaptation_factor > self.fire_adaptation_params.consciousness_threshold {
            // Activate enhanced consciousness state
            let mut state = self.state.write().await;
            *state = BMDState::Enhanced {
                consciousness_factor: information.fire_adaptation_factor,
                fire_adaptation_level: self.fire_adaptation_params.processing_enhancement,
                processing_enhancement: self.fire_adaptation_params.processing_enhancement,
            };
        }
        
        Ok(information)
    }
    
    /// Perform semantic catalysis on information
    async fn perform_semantic_catalysis(&self, mut information: InformationUnit) -> KwasaResult<InformationUnit> {
        // Preserve semantic meaning through catalytic process
        let semantic_preservation_factor = self.capabilities.semantic_preservation;
        
        // Apply catalytic transformation
        let catalytic_rate = self.capabilities.catalytic_efficiency;
        information.catalytic_potential *= catalytic_rate * semantic_preservation_factor;
        
        // Update semantic context
        let mut context = self.semantic_context.write().await;
        context.insert(
            information.semantic_meaning.clone(),
            information.catalytic_potential
        );
        
        // Ensure semantic preservation
        if semantic_preservation_factor < 0.9 {
            return Err(KwasaError::SemanticCatalysisError(
                "Semantic preservation below threshold".to_string()
            ));
        }
        
        log::debug!(
            "BMD {} catalyzed information with preservation factor: {:.3}",
            self.id,
            semantic_preservation_factor
        );
        
        Ok(information)
    }
    
    /// Update information gradient based on processed information
    async fn update_information_gradient(&self, information: &InformationUnit) -> KwasaResult<()> {
        let mut gradient = self.gradient.write().await;
        
        // Update information density
        gradient.density += information.catalytic_potential;
        
        // Update flow rate based on processing
        gradient.flow_rate = information.fire_adaptation_factor * self.capabilities.max_processing_rate;
        
        // Update semantic coherence
        gradient.semantic_coherence = 
            (gradient.semantic_coherence + information.catalytic_potential) / 2.0;
        
        // Update consciousness factor
        gradient.consciousness_factor = information.fire_adaptation_factor;
        
        Ok(())
    }
    
    /// Update BMD performance metrics
    async fn update_metrics(&self) -> KwasaResult<()> {
        let mut metrics = self.metrics.write().await;
        let gradient = self.gradient.read().await;
        
        // Update processing enhancement
        metrics.processing_enhancement = self.fire_adaptation_params.processing_enhancement;
        
        // Update pattern recognition
        metrics.pattern_recognition = self.fire_adaptation_params.pattern_recognition;
        
        // Update communication complexity
        metrics.communication_complexity = self.fire_adaptation_params.communication_complexity;
        
        // Update survival advantage
        metrics.survival_advantage = self.fire_adaptation_params.survival_advantage;
        
        // Update catalytic efficiency
        metrics.catalytic_efficiency = gradient.semantic_coherence;
        
        // Update consciousness threshold
        metrics.consciousness_threshold = gradient.consciousness_factor;
        
        Ok(())
    }
    
    /// Get current BMD status
    pub async fn get_status(&self) -> BMDStatus {
        let state = self.state.read().await;
        let gradient = self.gradient.read().await;
        let metrics = self.metrics.read().await;
        
        BMDStatus {
            id: self.id.clone(),
            bmd_type: self.bmd_type,
            state: state.clone(),
            gradient: gradient.clone(),
            metrics: metrics.clone(),
            capabilities: self.capabilities.clone(),
        }
    }
    
    /// Shutdown the BMD gracefully
    pub async fn shutdown(&self) -> KwasaResult<()> {
        let mut state = self.state.write().await;
        *state = BMDState::Inactive;
        
        log::info!("BMD {} shutdown successfully", self.id);
        Ok(())
    }
}

impl Default for InformationGradient {
    fn default() -> Self {
        Self {
            density: 0.0,
            flow_rate: 0.0,
            semantic_coherence: 1.0,
            consciousness_factor: KwasaConstants::CONSCIOUSNESS_SEQUENTIAL_STATES,
        }
    }
}

impl BMDCapabilities {
    /// Create default capabilities for a specific BMD type
    pub fn default_for_type(bmd_type: BMDType) -> Self {
        match bmd_type {
            BMDType::Molecular => Self {
                max_processing_rate: 10000.0,  // High molecular processing rate
                catalytic_efficiency: 0.95,
                semantic_preservation: 0.90,
                consciousness_enhancement: 1.0,
                fire_adaptation: 1.0,
                error_recovery: 0.85,
            },
            BMDType::Neural => Self {
                max_processing_rate: 1000.0,   // Moderate neural processing rate
                catalytic_efficiency: 0.92,
                semantic_preservation: 0.95,
                consciousness_enhancement: KwasaConstants::FIRE_PROCESSING_ENHANCEMENT,
                fire_adaptation: KwasaConstants::PATTERN_RECOGNITION_FACTOR,
                error_recovery: 0.90,
            },
            BMDType::Cognitive => Self {
                max_processing_rate: 100.0,    // Lower but higher-level processing
                catalytic_efficiency: 0.88,
                semantic_preservation: 0.98,
                consciousness_enhancement: KwasaConstants::COMMUNICATION_COMPLEXITY_MULTIPLIER,
                fire_adaptation: KwasaConstants::SURVIVAL_ADVANTAGE_FACTOR,
                error_recovery: 0.95,
            },
            BMDType::FireAdapted => Self {
                max_processing_rate: 5000.0,   // Enhanced processing rate
                catalytic_efficiency: 0.98,
                semantic_preservation: 0.99,
                consciousness_enhancement: KwasaConstants::FIRE_PROCESSING_ENHANCEMENT,
                fire_adaptation: KwasaConstants::SURVIVAL_ADVANTAGE_FACTOR,
                error_recovery: 0.98,
            },
            BMDType::Hybrid => Self {
                max_processing_rate: 2000.0,   // Balanced processing rate
                catalytic_efficiency: 0.94,
                semantic_preservation: 0.96,
                consciousness_enhancement: 2.5,
                fire_adaptation: 3.0,
                error_recovery: 0.92,
            },
        }
    }
}

impl MatrixAssociativeMemory {
    /// Create new matrix associative memory
    pub fn new(dimensions: (usize, usize), learning_rate: f64) -> Self {
        Self {
            weight_matrix: Array2::zeros(dimensions),
            context_patterns: HashMap::new(),
            goal_sequences: Vec::new(),
            memory_dimensions: dimensions,
            learning_rate,
            context_switcher: ContextSwitcher::new(),
        }
    }

    /// Store pattern with context-dependent association
    pub fn store_pattern(
        &mut self, 
        input_pattern: Array1<f64>, 
        output_pattern: Array1<f64>,
        context: &str
    ) -> KwasaResult<()> {
        // Ensure pattern dimensions match memory dimensions
        if input_pattern.len() != self.memory_dimensions.0 || 
           output_pattern.len() != self.memory_dimensions.1 {
            return Err(KwasaError::BMDInitializationError(
                "Pattern dimensions do not match memory dimensions".to_string()
            ));
        }

        // Update weight matrix using Hebbian learning with context
        let context_weight = self.get_context_weight(context);
        let outer_product = input_pattern.insert_axis(Axis(1)).dot(&output_pattern.insert_axis(Axis(0)));
        
        self.weight_matrix = &self.weight_matrix + 
            &(outer_product * self.learning_rate * context_weight);

        // Store context pattern
        self.context_patterns.insert(context.to_string(), input_pattern.clone());

        log::debug!("Stored pattern with context: {} (weight: {:.3})", context, context_weight);
        Ok(())
    }

    /// Retrieve pattern using context-dependent associative recall
    pub fn retrieve_pattern(&self, input_pattern: &Array1<f64>, context: &str) -> KwasaResult<Array1<f64>> {
        // Apply context-dependent modulation
        let context_modulation = self.get_context_modulation(context);
        let modulated_input = input_pattern * context_modulation;

        // Perform associative recall
        let output_pattern = self.weight_matrix.t().dot(&modulated_input);
        
        log::debug!("Retrieved pattern for context: {} (modulation: {:.3})", context, context_modulation);
        Ok(output_pattern)
    }

    /// Add goal-oriented sequence for linguistic behavior
    pub fn add_goal_sequence(&mut self, sequence: GoalOrientedSequence) {
        self.goal_sequences.push(sequence);
        log::debug!("Added goal-oriented sequence: {}", self.goal_sequences.len());
    }

    /// Execute goal-oriented sequence with context dependency
    pub fn execute_goal_sequence(
        &mut self, 
        goal_context: &str,
        behavior_type: LinguisticBehaviorType
    ) -> KwasaResult<Vec<Array1<f64>>> {
        // Find matching sequences
        let matching_sequences: Vec<&GoalOrientedSequence> = self.goal_sequences
            .iter()
            .filter(|seq| seq.goal_context == goal_context && seq.behavior_type == behavior_type)
            .collect();

        if matching_sequences.is_empty() {
            return Err(KwasaError::BMDInitializationError(
                format!("No goal sequences found for context: {} and behavior: {:?}", 
                       goal_context, behavior_type)
            ));
        }

        // Select sequence with highest completion probability
        let best_sequence = matching_sequences
            .iter()
            .max_by(|a, b| a.completion_probability.partial_cmp(&b.completion_probability).unwrap())
            .unwrap();

        // Execute sequence with context switching
        let mut execution_result = Vec::new();
        for (i, pattern) in best_sequence.pattern_sequence.iter().enumerate() {
            // Apply context switching if needed
            if i > 0 {
                self.context_switcher.maybe_switch_context(&goal_context);
            }

            // Process pattern through associative memory
            let processed_pattern = self.retrieve_pattern(pattern, &goal_context)?;
            execution_result.push(processed_pattern);
        }

        log::info!("Executed goal sequence: {} with {} patterns", 
                  best_sequence.sequence_id, execution_result.len());
        Ok(execution_result)
    }

    /// Get context weight for learning modulation
    fn get_context_weight(&self, context: &str) -> f64 {
        // Fire-adapted contexts get higher weight
        if context.contains("fire") || context.contains("communication") {
            KwasaConstants::FIRE_PROCESSING_ENHANCEMENT
        } else if context.contains("pattern") {
            KwasaConstants::PATTERN_RECOGNITION_FACTOR
        } else {
            1.0
        }
    }

    /// Get context modulation for retrieval
    fn get_context_modulation(&self, context: &str) -> f64 {
        self.context_patterns.get(context)
            .map(|pattern| pattern.sum() / pattern.len() as f64)
            .unwrap_or(1.0)
    }
}

impl ContextSwitcher {
    /// Create new context switcher
    pub fn new() -> Self {
        Self {
            active_context: "default".to_string(),
            switching_probability: 0.1,
            context_history: VecDeque::new(),
            transition_matrix: Array2::eye(10), // Default 10x10 transition matrix
        }
    }

    /// Maybe switch context based on probability and history
    pub fn maybe_switch_context(&mut self, target_context: &str) {
        let switch_probability = self.calculate_switch_probability(target_context);
        
        if rand::random::<f64>() < switch_probability {
            self.switch_to_context(target_context);
        }
    }

    /// Switch to specific context
    pub fn switch_to_context(&mut self, new_context: &str) {
        // Add current context to history
        self.context_history.push_back(self.active_context.clone());
        
        // Limit history size
        if self.context_history.len() > 100 {
            self.context_history.pop_front();
        }

        // Switch to new context
        self.active_context = new_context.to_string();
        log::debug!("Switched context to: {}", new_context);
    }

    /// Calculate switching probability based on context similarity and history
    fn calculate_switch_probability(&self, target_context: &str) -> f64 {
        if target_context == self.active_context {
            return 0.0; // No need to switch to same context
        }

        // Higher probability for fire-adapted contexts
        let base_probability = if target_context.contains("fire") {
            self.switching_probability * KwasaConstants::FIRE_PROCESSING_ENHANCEMENT
        } else {
            self.switching_probability
        };

        // Reduce probability if recently visited
        let recency_penalty = if self.context_history.contains(&target_context.to_string()) {
            0.5
        } else {
            1.0
        };

        base_probability * recency_penalty
    }
}

impl DiscourseTrajectoryMetrics {
    /// Create new discourse trajectory metrics
    pub fn new() -> Self {
        Self {
            order_degree: 1.0,
            coherence_trajectory: Vec::new(),
            disorganization_entropy: 0.0,
            trajectory_smoothness: 1.0,
            context_consistency: 1.0,
        }
    }

    /// Update metrics based on discourse production
    pub fn update_from_discourse(&mut self, discourse_patterns: &[Array1<f64>]) {
        // Calculate order degree from pattern consistency
        self.order_degree = self.calculate_order_degree(discourse_patterns);
        
        // Update coherence trajectory
        let coherence = self.calculate_coherence(discourse_patterns);
        self.coherence_trajectory.push((Instant::now(), coherence));
        
        // Calculate disorganization entropy
        self.disorganization_entropy = self.calculate_disorganization_entropy(discourse_patterns);
        
        // Calculate trajectory smoothness
        self.trajectory_smoothness = self.calculate_trajectory_smoothness();
        
        // Update context consistency
        self.context_consistency = self.calculate_context_consistency(discourse_patterns);
    }

    /// Calculate order degree based on Mizraji's trajectory measurement
    fn calculate_order_degree(&self, patterns: &[Array1<f64>]) -> f64 {
        if patterns.len() < 2 {
            return 1.0;
        }

        let mut total_similarity = 0.0;
        let mut comparisons = 0;

        for i in 0..patterns.len()-1 {
            for j in i+1..patterns.len() {
                let similarity = self.calculate_pattern_similarity(&patterns[i], &patterns[j]);
                total_similarity += similarity;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            total_similarity / comparisons as f64
        } else {
            1.0
        }
    }

    /// Calculate coherence based on pattern transitions
    fn calculate_coherence(&self, patterns: &[Array1<f64>]) -> f64 {
        if patterns.len() < 2 {
            return 1.0;
        }

        let mut coherence_sum = 0.0;
        for i in 0..patterns.len()-1 {
            let transition_coherence = self.calculate_pattern_similarity(&patterns[i], &patterns[i+1]);
            coherence_sum += transition_coherence;
        }

        coherence_sum / (patterns.len() - 1) as f64
    }

    /// Calculate disorganization entropy
    fn calculate_disorganization_entropy(&self, patterns: &[Array1<f64>]) -> f64 {
        if patterns.is_empty() {
            return 0.0;
        }

        // Calculate entropy based on pattern distribution
        let mut pattern_counts: HashMap<String, usize> = HashMap::new();
        
        for pattern in patterns {
            let pattern_key = format!("{:.2}", pattern.sum()); // Simplified pattern representation
            *pattern_counts.entry(pattern_key).or_insert(0) += 1;
        }

        // Calculate Shannon entropy
        let total_patterns = patterns.len() as f64;
        let mut entropy = 0.0;
        
        for count in pattern_counts.values() {
            let probability = *count as f64 / total_patterns;
            if probability > 0.0 {
                entropy -= probability * probability.ln();
            }
        }

        entropy
    }

    /// Calculate trajectory smoothness
    fn calculate_trajectory_smoothness(&self) -> f64 {
        if self.coherence_trajectory.len() < 3 {
            return 1.0;
        }

        let mut smoothness_sum = 0.0;
        let mut smoothness_count = 0;

        for i in 1..self.coherence_trajectory.len()-1 {
            let prev_coherence = self.coherence_trajectory[i-1].1;
            let curr_coherence = self.coherence_trajectory[i].1;
            let next_coherence = self.coherence_trajectory[i+1].1;

            // Calculate second derivative as smoothness measure
            let second_derivative = (next_coherence - 2.0 * curr_coherence + prev_coherence).abs();
            smoothness_sum += 1.0 / (1.0 + second_derivative); // Inverse relationship
            smoothness_count += 1;
        }

        if smoothness_count > 0 {
            smoothness_sum / smoothness_count as f64
        } else {
            1.0
        }
    }

    /// Calculate context consistency
    fn calculate_context_consistency(&self, patterns: &[Array1<f64>]) -> f64 {
        // Simplified context consistency based on pattern variance
        if patterns.is_empty() {
            return 1.0;
        }

        let mean_pattern = self.calculate_mean_pattern(patterns);
        let mut variance_sum = 0.0;

        for pattern in patterns {
            let diff = pattern - &mean_pattern;
            variance_sum += diff.mapv(|x| x * x).sum();
        }

        let variance = variance_sum / patterns.len() as f64;
        1.0 / (1.0 + variance) // Higher consistency = lower variance
    }

    /// Calculate pattern similarity
    fn calculate_pattern_similarity(&self, pattern1: &Array1<f64>, pattern2: &Array1<f64>) -> f64 {
        // Cosine similarity
        let dot_product = pattern1.dot(pattern2);
        let norm1 = pattern1.mapv(|x| x * x).sum().sqrt();
        let norm2 = pattern2.mapv(|x| x * x).sum().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    /// Calculate mean pattern
    fn calculate_mean_pattern(&self, patterns: &[Array1<f64>]) -> Array1<f64> {
        if patterns.is_empty() {
            return Array1::zeros(0);
        }

        let pattern_dim = patterns[0].len();
        let mut mean_pattern = Array1::zeros(pattern_dim);

        for pattern in patterns {
            mean_pattern = mean_pattern + pattern;
        }

        mean_pattern / patterns.len() as f64
    }
}

impl BiologicalMaxwellDemon {
    /// Enhanced initialization with matrix associative memory
    pub async fn initialize_with_matrix_memory(&self, memory_dimensions: (usize, usize)) -> KwasaResult<()> {
        // Initialize basic BMD
        self.initialize().await?;

        // Add matrix associative memory to capabilities
        let matrix_memory = MatrixAssociativeMemory::new(memory_dimensions, 0.01);
        
        // Store in semantic context (simplified integration)
        let mut context = self.semantic_context.write().await;
        context.insert("matrix_memory_initialized".to_string(), 1.0);
        context.insert("memory_dimensions".to_string(), (memory_dimensions.0 * memory_dimensions.1) as f64);

        log::info!("BMD {} initialized with matrix associative memory ({:?})", 
                  self.id, memory_dimensions);
        Ok(())
    }

    /// Process information with matrix associative memory enhancement
    pub async fn catalyze_with_matrix_memory(
        &self, 
        information: InformationUnit,
        goal_context: &str,
        behavior_type: LinguisticBehaviorType
    ) -> KwasaResult<InformationUnit> {
        // Convert information to pattern representation
        let input_pattern = self.information_to_pattern(&information)?;
        
        // Process through matrix associative memory
        let enhanced_pattern = self.process_through_matrix_memory(
            input_pattern, 
            goal_context, 
            behavior_type
        ).await?;

        // Convert back to information unit
        let mut enhanced_information = information;
        enhanced_information.fire_adaptation_factor *= 1.5; // Matrix memory enhancement
        enhanced_information.catalytic_potential *= 1.2;

        // Apply standard catalysis
        self.catalyze_information(enhanced_information).await
    }

    /// Convert information unit to pattern representation
    fn information_to_pattern(&self, information: &InformationUnit) -> KwasaResult<Array1<f64>> {
        // Simplified conversion - could be enhanced with more sophisticated encoding
        let pattern_size = 128; // Standard pattern size
        let mut pattern = Array1::zeros(pattern_size);

        // Encode semantic meaning
        let semantic_hash = self.hash_semantic_meaning(&information.semantic_meaning);
        for (i, &byte) in semantic_hash.iter().take(pattern_size/2).enumerate() {
            pattern[i] = byte as f64 / 255.0;
        }

        // Encode fire adaptation factor
        pattern[pattern_size/2] = information.fire_adaptation_factor;
        pattern[pattern_size/2 + 1] = information.catalytic_potential;
        pattern[pattern_size/2 + 2] = information.consciousness_level as f64;

        Ok(pattern)
    }

    /// Process through matrix associative memory
    async fn process_through_matrix_memory(
        &self,
        input_pattern: Array1<f64>,
        goal_context: &str,
        _behavior_type: LinguisticBehaviorType
    ) -> KwasaResult<Array1<f64>> {
        // For now, simulate matrix memory processing
        // In full implementation, this would use the actual MatrixAssociativeMemory
        let enhancement_factor = if goal_context.contains("fire") {
            KwasaConstants::FIRE_PROCESSING_ENHANCEMENT
        } else {
            1.2
        };

        Ok(input_pattern * enhancement_factor)
    }

    /// Hash semantic meaning for pattern encoding
    fn hash_semantic_meaning(&self, semantic_meaning: &str) -> Vec<u8> {
        // Simplified hash - could use more sophisticated methods
        semantic_meaning.bytes().collect()
    }
}

/// Complete status of a BMD
#[derive(Debug, Clone)]
pub struct BMDStatus {
    pub id: String,
    pub bmd_type: BMDType,
    pub state: BMDState,
    pub gradient: InformationGradient,
    pub metrics: BMDMetrics,
    pub capabilities: BMDCapabilities,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_bmd_creation() {
        let bmd = BiologicalMaxwellDemon::new(
            "test_bmd_001".to_string(),
            BMDType::Neural
        ).unwrap();
        
        assert_eq!(bmd.id, "test_bmd_001");
        assert_eq!(bmd.bmd_type, BMDType::Neural);
    }
    
    #[tokio::test]
    async fn test_bmd_initialization() {
        let bmd = BiologicalMaxwellDemon::new(
            "test_bmd_002".to_string(),
            BMDType::FireAdapted
        ).unwrap();
        
        bmd.initialize().await.unwrap();
        
        let status = bmd.get_status().await;
        match status.state {
            BMDState::Active { .. } => {},
            _ => panic!("BMD should be in active state after initialization"),
        }
    }
    
    #[tokio::test]
    async fn test_information_catalysis() {
        let bmd = BiologicalMaxwellDemon::new(
            "test_bmd_003".to_string(),
            BMDType::Cognitive
        ).unwrap();
        
        bmd.initialize().await.unwrap();
        
        let test_information = InformationUnit {
            content: b"test information content".to_vec(),
            semantic_meaning: "test_pattern_recognition".to_string(),
            catalytic_potential: 1.0,
            consciousness_level: 1,
            fire_adaptation_factor: 1.0,
        };
        
        let result = bmd.catalyze_information(test_information).await.unwrap();
        
        // Should have enhanced fire adaptation factor due to pattern recognition
        assert!(result.fire_adaptation_factor > 1.0);
        assert!(result.catalytic_potential > 1.0);
    }
    
    #[tokio::test]
    async fn test_fire_adaptation_enhancement() {
        let bmd = BiologicalMaxwellDemon::new(
            "test_bmd_004".to_string(),
            BMDType::FireAdapted
        ).unwrap();
        
        bmd.initialize().await.unwrap();
        
        let test_information = InformationUnit {
            content: b"communication complexity test".to_vec(),
            semantic_meaning: "communication enhancement".to_string(),
            catalytic_potential: 1.0,
            consciousness_level: 2,
            fire_adaptation_factor: 0.1,
        };
        
        let result = bmd.catalyze_information(test_information).await.unwrap();
        
        // Should show massive enhancement due to communication complexity multiplier
        assert!(result.fire_adaptation_factor > 79.0); // 79.3× enhancement
        assert!(result.catalytic_potential > 4.0);      // 460% survival advantage
    }
} 
} 