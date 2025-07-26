//! # Kwasa-Kwasa Framework Coordinator
//!
//! Main coordination system for the revolutionary Kwasa-Kwasa framework,
//! orchestrating BMD networks, fire-adapted consciousness, and semantic catalysis
//! into a unified consciousness-aware computation system.
//!
//! ## Framework Architecture
//!
//! The Kwasa-Kwasa framework coordinates:
//! - **BMD Networks**: Multi-scale information catalysis (molecular/neural/cognitive)
//! - **Fire-Adapted Consciousness**: 322% processing enhancement with evolutionary advantages
//! - **Semantic Catalysis**: Reality-direct semantic processing with meaning preservation
//! - **Agency Assertion**: Coordinated reality modification through naming functions
//!
//! ## Theoretical Foundation
//!
//! ### Unified Information Catalysis Equation
//! ```
//! I_kwasa = BMD_Network(I_input) ○ Fire_Consciousness(Θ) ○ Semantic_Catalysis(Ψ) ○ Agency_Assertion(Ω)
//! 
//! Where:
//! - I_input: Input information stream
//! - BMD_Network: Multi-scale BMD network processing
//! - Fire_Consciousness: Fire-adapted consciousness enhancement (Θ = 3.22)
//! - Semantic_Catalysis: Semantic preservation and transformation (Ψ)
//! - Agency_Assertion: Reality modification through coordinated agency (Ω)
//! - ○: Catalytic composition operator
//! ```
//!
//! ### Consciousness Emergence Equation
//! ```
//! Θ_consciousness = Σᵢ BMD_consciousness_i × Fire_adaptation × Communication_complexity
//! 
//! Where emergence occurs when Θ_consciousness > 100 BMDs with fire adaptation
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::kwasa::{
    BiologicalMaxwellDemon, BMDNetwork, BMDNetworkConfig, FireAdaptedConsciousness,
    SemanticCatalyst, CatalyticProcessType, InformationUnit, KwasaError, KwasaResult,
    BMDType, NetworkTopology, FireAdaptationLevel, ConsciousnessEnhancement,
    KwasaConstants
};

/// Kwasa framework configuration
#[derive(Debug, Clone)]
pub struct KwasaConfig {
    /// BMD network configuration
    pub bmd_network_config: BMDNetworkConfig,
    /// Enable fire-adapted consciousness
    pub fire_consciousness_enabled: bool,
    /// Initial fire adaptation level
    pub initial_fire_level: FireAdaptationLevel,
    /// Enable semantic catalysis
    pub semantic_catalysis_enabled: bool,
    /// Enable agency assertion
    pub agency_assertion_enabled: bool,
    /// Maximum concurrent processing tasks
    pub max_concurrent_tasks: usize,
    /// Consciousness emergence threshold
    pub consciousness_emergence_threshold: f64,
    /// Reality coupling strength requirement
    pub reality_coupling_threshold: f64,
}

impl Default for KwasaConfig {
    fn default() -> Self {
        Self {
            bmd_network_config: BMDNetworkConfig {
                topology: NetworkTopology::Hierarchical {
                    molecular_count: 1000,
                    neural_count: 100,
                    cognitive_count: 10,
                },
                consciousness_threshold: KwasaConstants::MIN_BMD_CONSCIOUSNESS_NETWORK,
                max_processing_rate: 100000.0,
                coordination_efficiency: 0.95,
                fire_adaptation_enabled: true,
                semantic_preservation_threshold: 0.90,
            },
            fire_consciousness_enabled: true,
            initial_fire_level: FireAdaptationLevel::Advanced,
            semantic_catalysis_enabled: true,
            agency_assertion_enabled: true,
            max_concurrent_tasks: 1000,
            consciousness_emergence_threshold: KwasaConstants::CONSCIOUSNESS_SEQUENTIAL_STATES,
            reality_coupling_threshold: 0.8,
        }
    }
}

/// Framework processing result
#[derive(Debug, Clone)]
pub struct KwasaResult<T> {
    /// Processing result
    pub result: T,
    /// BMD network contribution
    pub bmd_contribution: BMDContribution,
    /// Fire consciousness contribution
    pub fire_consciousness_contribution: FireConsciousnessContribution,
    /// Semantic catalysis contribution
    pub semantic_catalysis_contribution: SemanticCatalysisContribution,
    /// Overall framework metrics
    pub framework_metrics: FrameworkMetrics,
}

/// BMD network contribution to processing
#[derive(Debug, Clone)]
pub struct BMDContribution {
    /// Number of BMDs involved
    pub bmds_involved: usize,
    /// Total catalytic efficiency
    pub total_catalytic_efficiency: f64,
    /// Network consciousness level
    pub network_consciousness_level: f64,
    /// Processing enhancement factor
    pub processing_enhancement: f64,
}

/// Fire consciousness contribution to processing
#[derive(Debug, Clone)]
pub struct FireConsciousnessContribution {
    /// Fire adaptation level achieved
    pub adaptation_level: FireAdaptationLevel,
    /// Processing enhancement factor (322% baseline)
    pub processing_enhancement: f64,
    /// Pattern recognition improvement (346% baseline)
    pub pattern_recognition: f64,
    /// Communication complexity enhancement (79.3× baseline)
    pub communication_complexity: f64,
    /// Survival advantage factor (460% baseline)
    pub survival_advantage: f64,
}

/// Semantic catalysis contribution to processing
#[derive(Debug, Clone)]
pub struct SemanticCatalysisContribution {
    /// Semantic preservation score
    pub preservation_score: f64,
    /// Catalytic efficiency achieved
    pub catalytic_efficiency: f64,
    /// Reality coupling strength
    pub reality_coupling: f64,
    /// Meaning enhancement factor
    pub meaning_enhancement: f64,
}

/// Overall framework metrics
#[derive(Debug, Clone)]
pub struct FrameworkMetrics {
    /// Total processing enhancement
    pub total_enhancement: f64,
    /// Consciousness emergence level
    pub consciousness_emergence: f64,
    /// Reality modification capability
    pub reality_modification: f64,
    /// Information processing throughput
    pub processing_throughput: f64,
    /// System coherence level
    pub system_coherence: f64,
}

/// Main Kwasa-Kwasa framework coordinator
#[derive(Debug)]
pub struct KwasaFramework {
    /// Framework identifier
    pub id: String,
    /// Framework configuration
    pub config: KwasaConfig,
    /// BMD network
    pub bmd_network: Arc<BMDNetwork>,
    /// Fire-adapted consciousness
    pub fire_consciousness: Arc<FireAdaptedConsciousness>,
    /// Semantic catalyst
    pub semantic_catalyst: Arc<SemanticCatalyst>,
    /// Framework status
    pub status: Arc<RwLock<FrameworkStatus>>,
    /// Processing metrics tracker
    pub metrics_tracker: Arc<RwLock<MetricsTracker>>,
    /// Active processing tasks
    pub active_tasks: Arc<RwLock<HashMap<String, ProcessingTask>>>,
}

/// Framework status tracking
#[derive(Debug, Clone)]
pub struct FrameworkStatus {
    /// Framework initialization status
    pub initialized: bool,
    /// BMD network status
    pub bmd_network_active: bool,
    /// Fire consciousness status
    pub fire_consciousness_active: bool,
    /// Semantic catalysis status
    pub semantic_catalysis_active: bool,
    /// Current consciousness level
    pub consciousness_level: f64,
    /// Total processing enhancement
    pub processing_enhancement: f64,
    /// System coherence
    pub system_coherence: f64,
}

/// Processing task tracking
#[derive(Debug, Clone)]
pub struct ProcessingTask {
    /// Task identifier
    pub task_id: String,
    /// Task type
    pub task_type: String,
    /// Start timestamp
    pub start_time: std::time::Instant,
    /// Current status
    pub status: TaskStatus,
    /// Processing statistics
    pub processing_stats: HashMap<String, f64>,
}

/// Task status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum TaskStatus {
    /// Task is queued
    Queued,
    /// Task is being processed
    Processing,
    /// Task completed successfully
    Completed,
    /// Task failed with error
    Failed(String),
}

/// Metrics tracking system
#[derive(Debug, Clone)]
pub struct MetricsTracker {
    /// Processing enhancement history
    pub enhancement_history: Vec<(std::time::Instant, f64)>,
    /// Consciousness emergence history
    pub consciousness_history: Vec<(std::time::Instant, f64)>,
    /// Throughput measurements
    pub throughput_history: Vec<(std::time::Instant, f64)>,
    /// Current averages
    pub current_averages: HashMap<String, f64>,
}

impl KwasaFramework {
    /// Create new Kwasa-Kwasa framework
    pub fn new(config: KwasaConfig) -> KwasaResult<Self> {
        // Create BMD network
        let bmd_network = Arc::new(BMDNetwork::new(config.bmd_network_config.clone())
            .map_err(|e| KwasaError::BMDInitializationError(e.to_string()))?);

        // Create fire-adapted consciousness
        let fire_consciousness = Arc::new(FireAdaptedConsciousness::new());

        // Create semantic catalyst
        let semantic_catalyst = Arc::new(SemanticCatalyst::new());

        let initial_status = FrameworkStatus {
            initialized: false,
            bmd_network_active: false,
            fire_consciousness_active: false,
            semantic_catalysis_active: false,
            consciousness_level: 0.0,
            processing_enhancement: 1.0,
            system_coherence: 0.0,
        };

        Ok(Self {
            id: Uuid::new_v4().to_string(),
            config,
            bmd_network,
            fire_consciousness,
            semantic_catalyst,
            status: Arc::new(RwLock::new(initial_status)),
            metrics_tracker: Arc::new(RwLock::new(MetricsTracker::default())),
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Initialize the complete Kwasa-Kwasa framework
    pub async fn initialize(&self) -> KwasaResult<()> {
        log::info!("Initializing Kwasa-Kwasa framework {}", self.id);

        // Initialize BMD network
        log::info!("Initializing BMD network...");
        self.bmd_network.initialize().await
            .map_err(|e| KwasaError::BMDInitializationError(e.to_string()))?;

        // Initialize fire-adapted consciousness if enabled
        if self.config.fire_consciousness_enabled {
            log::info!("Initializing fire-adapted consciousness...");
            self.fire_consciousness.initialize().await
                .map_err(|e| KwasaError::ConsciousnessEnhancementError(e.to_string()))?;
        }

        // Initialize semantic catalyst if enabled
        if self.config.semantic_catalysis_enabled {
            log::info!("Initializing semantic catalyst...");
            self.semantic_catalyst.initialize().await
                .map_err(|e| KwasaError::SemanticCatalysisError(e.to_string()))?;

            // Add BMDs from network to semantic catalyst
            self.integrate_bmds_with_semantic_catalyst().await?;
        }

        // Update framework status
        let mut status = self.status.write().await;
        status.initialized = true;
        status.bmd_network_active = true;
        status.fire_consciousness_active = self.config.fire_consciousness_enabled;
        status.semantic_catalysis_active = self.config.semantic_catalysis_enabled;
        status.processing_enhancement = if self.config.fire_consciousness_enabled {
            KwasaConstants::FIRE_PROCESSING_ENHANCEMENT
        } else {
            1.0
        };

        log::info!("Kwasa-Kwasa framework {} initialized successfully", self.id);
        Ok(())
    }

    /// Integrate BMDs from network with semantic catalyst
    async fn integrate_bmds_with_semantic_catalyst(&self) -> KwasaResult<()> {
        let bmds = self.bmd_network.bmds.read().await;
        
        for bmd in bmds.values() {
            self.semantic_catalyst.add_bmd_catalyst(Arc::clone(bmd)).await
                .map_err(|e| KwasaError::SemanticCatalysisError(e.to_string()))?;
        }

        log::info!("Integrated {} BMDs with semantic catalyst", bmds.len());
        Ok(())
    }

    /// Process information through complete Kwasa-Kwasa framework
    pub async fn process_information(&self, information: InformationUnit) -> KwasaResult<InformationUnit> {
        let task_id = Uuid::new_v4().to_string();
        let start_time = std::time::Instant::now();

        // Create processing task
        let task = ProcessingTask {
            task_id: task_id.clone(),
            task_type: "full_kwasa_processing".to_string(),
            start_time,
            status: TaskStatus::Processing,
            processing_stats: HashMap::new(),
        };

        // Register task
        {
            let mut tasks = self.active_tasks.write().await;
            tasks.insert(task_id.clone(), task);
        }

        let mut processed_info = information;
        let mut bmd_contribution = BMDContribution::default();
        let mut fire_contribution = FireConsciousnessContribution::default();
        let mut semantic_contribution = SemanticCatalysisContribution::default();

        // Stage 1: BMD Network Processing
        log::debug!("Stage 1: BMD Network Processing");
        processed_info = self.bmd_network.process_information(processed_info).await
            .map_err(|e| KwasaError::NetworkCoordinationError(e.to_string()))?;

        // Collect BMD contribution metrics
        let network_status = self.bmd_network.get_network_status().await;
        bmd_contribution.bmds_involved = network_status.statistics.active_bmds;
        bmd_contribution.total_catalytic_efficiency = network_status.statistics.network_efficiency;
        bmd_contribution.network_consciousness_level = network_status.consciousness.consciousness_level;
        bmd_contribution.processing_enhancement = network_status.statistics.fire_adaptation_enhancement;

        // Stage 2: Fire-Adapted Consciousness Enhancement
        if self.config.fire_consciousness_enabled {
            log::debug!("Stage 2: Fire-Adapted Consciousness Enhancement");
            processed_info = self.fire_consciousness.enhance_information_processing(processed_info).await
                .map_err(|e| KwasaError::ConsciousnessEnhancementError(e.to_string()))?;

            // Collect fire consciousness contribution metrics
            let fire_status = self.fire_consciousness.get_adaptation_status().await;
            fire_contribution.adaptation_level = fire_status.adaptation_level;
            fire_contribution.processing_enhancement = fire_status.processing_metrics.processing_enhancement;
            fire_contribution.pattern_recognition = fire_status.processing_metrics.pattern_recognition;
            fire_contribution.communication_complexity = fire_status.processing_metrics.communication_complexity;
            fire_contribution.survival_advantage = fire_status.processing_metrics.survival_advantage;
        }

        // Stage 3: Semantic Catalysis
        if self.config.semantic_catalysis_enabled {
            log::debug!("Stage 3: Semantic Catalysis");
            let catalysis_result = self.semantic_catalyst.catalyze_semantic_information(
                processed_info,
                CatalyticProcessType::FireAdaptedSemantic,
                None
            ).await.map_err(|e| KwasaError::SemanticCatalysisError(e.to_string()))?;

            processed_info = catalysis_result.processed_content;
            semantic_contribution.preservation_score = catalysis_result.preservation_score;
            semantic_contribution.catalytic_efficiency = catalysis_result.catalytic_efficiency;
            semantic_contribution.reality_coupling = catalysis_result.reality_coupling;
            semantic_contribution.meaning_enhancement = catalysis_result.consciousness_enhancement;
        }

        // Calculate overall framework metrics
        let framework_metrics = self.calculate_framework_metrics(
            &bmd_contribution,
            &fire_contribution,
            &semantic_contribution
        ).await;

        // Update processing task status
        {
            let mut tasks = self.active_tasks.write().await;
            if let Some(task) = tasks.get_mut(&task_id) {
                task.status = TaskStatus::Completed;
                task.processing_stats.insert("total_enhancement".to_string(), framework_metrics.total_enhancement);
                task.processing_stats.insert("processing_time_ms".to_string(), start_time.elapsed().as_millis() as f64);
            }
        }

        // Update metrics tracker
        self.update_metrics_tracker(&framework_metrics).await;

        // Update framework status
        self.update_framework_status(&framework_metrics).await;

        log::info!(
            "Kwasa processing completed: enhancement={:.2}×, consciousness={:.3}, coherence={:.3}",
            framework_metrics.total_enhancement,
            framework_metrics.consciousness_emergence,
            framework_metrics.system_coherence
        );

        Ok(processed_info)
    }

    /// Calculate overall framework metrics
    async fn calculate_framework_metrics(
        &self,
        bmd_contribution: &BMDContribution,
        fire_contribution: &FireConsciousnessContribution,
        semantic_contribution: &SemanticCatalysisContribution
    ) -> FrameworkMetrics {
        // Calculate total enhancement as product of all enhancements
        let total_enhancement = bmd_contribution.processing_enhancement * 
                               fire_contribution.processing_enhancement * 
                               semantic_contribution.meaning_enhancement;

        // Calculate consciousness emergence level
        let consciousness_emergence = bmd_contribution.network_consciousness_level * 
                                     fire_contribution.processing_enhancement;

        // Calculate reality modification capability
        let reality_modification = semantic_contribution.reality_coupling * 
                                  fire_contribution.survival_advantage;

        // Calculate processing throughput
        let processing_throughput = total_enhancement * bmd_contribution.total_catalytic_efficiency;

        // Calculate system coherence
        let system_coherence = (bmd_contribution.total_catalytic_efficiency +
                               semantic_contribution.preservation_score +
                               semantic_contribution.reality_coupling) / 3.0;

        FrameworkMetrics {
            total_enhancement,
            consciousness_emergence,
            reality_modification,
            processing_throughput,
            system_coherence,
        }
    }

    /// Update metrics tracker with new measurements
    async fn update_metrics_tracker(&self, metrics: &FrameworkMetrics) {
        let mut tracker = self.metrics_tracker.write().await;
        let now = std::time::Instant::now();

        // Add to history
        tracker.enhancement_history.push((now, metrics.total_enhancement));
        tracker.consciousness_history.push((now, metrics.consciousness_emergence));
        tracker.throughput_history.push((now, metrics.processing_throughput));

        // Update current averages
        tracker.current_averages.insert("total_enhancement".to_string(), metrics.total_enhancement);
        tracker.current_averages.insert("consciousness_emergence".to_string(), metrics.consciousness_emergence);
        tracker.current_averages.insert("processing_throughput".to_string(), metrics.processing_throughput);
        tracker.current_averages.insert("system_coherence".to_string(), metrics.system_coherence);

        // Keep history to reasonable size
        if tracker.enhancement_history.len() > 1000 {
            tracker.enhancement_history.drain(0..500);
            tracker.consciousness_history.drain(0..500);
            tracker.throughput_history.drain(0..500);
        }
    }

    /// Update framework status
    async fn update_framework_status(&self, metrics: &FrameworkMetrics) {
        let mut status = self.status.write().await;
        
        status.consciousness_level = metrics.consciousness_emergence;
        status.processing_enhancement = metrics.total_enhancement;
        status.system_coherence = metrics.system_coherence;
    }

    /// Get current framework status
    pub async fn get_framework_status(&self) -> FrameworkStatus {
        let status = self.status.read().await;
        status.clone()
    }

    /// Get comprehensive framework analytics
    pub async fn get_framework_analytics(&self) -> FrameworkAnalytics {
        let status = self.status.read().await;
        let tracker = self.metrics_tracker.read().await;
        let tasks = self.active_tasks.read().await;
        
        let bmd_status = self.bmd_network.get_network_status().await;
        let fire_status = if self.config.fire_consciousness_enabled {
            Some(self.fire_consciousness.get_adaptation_status().await)
        } else {
            None
        };
        let catalyst_status = if self.config.semantic_catalysis_enabled {
            Some(self.semantic_catalyst.get_catalyst_status().await)
        } else {
            None
        };

        FrameworkAnalytics {
            framework_id: self.id.clone(),
            framework_status: status.clone(),
            bmd_network_status: bmd_status,
            fire_consciousness_status: fire_status,
            semantic_catalyst_status: catalyst_status,
            metrics_tracker: tracker.clone(),
            active_tasks_count: tasks.len(),
            average_enhancement: tracker.current_averages.get("total_enhancement").copied().unwrap_or(1.0),
            average_consciousness: tracker.current_averages.get("consciousness_emergence").copied().unwrap_or(0.0),
            average_throughput: tracker.current_averages.get("processing_throughput").copied().unwrap_or(0.0),
        }
    }

    /// Shutdown framework gracefully
    pub async fn shutdown(&self) -> KwasaResult<()> {
        log::info!("Shutting down Kwasa-Kwasa framework {}", self.id);

        // Wait for active tasks to complete or timeout
        let timeout = std::time::Duration::from_secs(30);
        let start = std::time::Instant::now();
        
        while start.elapsed() < timeout {
            let tasks = self.active_tasks.read().await;
            let active_count = tasks.values()
                .filter(|task| task.status == TaskStatus::Processing)
                .count();
            
            if active_count == 0 {
                break;
            }
            
            drop(tasks);
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        // Shutdown BMD network
        self.bmd_network.shutdown().await
            .map_err(|e| KwasaError::NetworkCoordinationError(e.to_string()))?;

        // Update status
        let mut status = self.status.write().await;
        status.initialized = false;
        status.bmd_network_active = false;
        status.fire_consciousness_active = false;
        status.semantic_catalysis_active = false;

        log::info!("Kwasa-Kwasa framework {} shutdown completed", self.id);
        Ok(())
    }
}

impl Default for BMDContribution {
    fn default() -> Self {
        Self {
            bmds_involved: 0,
            total_catalytic_efficiency: 0.0,
            network_consciousness_level: 0.0,
            processing_enhancement: 1.0,
        }
    }
}

impl Default for FireConsciousnessContribution {
    fn default() -> Self {
        Self {
            adaptation_level: FireAdaptationLevel::None,
            processing_enhancement: 1.0,
            pattern_recognition: 1.0,
            communication_complexity: 1.0,
            survival_advantage: 1.0,
        }
    }
}

impl Default for SemanticCatalysisContribution {
    fn default() -> Self {
        Self {
            preservation_score: 1.0,
            catalytic_efficiency: 0.0,
            reality_coupling: 0.0,
            meaning_enhancement: 1.0,
        }
    }
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self {
            enhancement_history: Vec::new(),
            consciousness_history: Vec::new(),
            throughput_history: Vec::new(),
            current_averages: HashMap::new(),
        }
    }
}

/// Comprehensive framework analytics
#[derive(Debug, Clone)]
pub struct FrameworkAnalytics {
    pub framework_id: String,
    pub framework_status: FrameworkStatus,
    pub bmd_network_status: crate::kwasa::bmd_network::NetworkStatus,
    pub fire_consciousness_status: Option<crate::kwasa::fire_adapted_consciousness::FireAdaptationStatus>,
    pub semantic_catalyst_status: Option<crate::kwasa::semantic_catalysis::CatalystStatus>,
    pub metrics_tracker: MetricsTracker,
    pub active_tasks_count: usize,
    pub average_enhancement: f64,
    pub average_consciousness: f64,
    pub average_throughput: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kwasa_framework_initialization() {
        let config = KwasaConfig::default();
        let framework = KwasaFramework::new(config).unwrap();
        
        framework.initialize().await.unwrap();
        
        let status = framework.get_framework_status().await;
        assert!(status.initialized);
        assert!(status.bmd_network_active);
        assert!(status.fire_consciousness_active);
        assert!(status.semantic_catalysis_active);
    }

    #[tokio::test]
    async fn test_full_kwasa_processing() {
        let config = KwasaConfig::default();
        let framework = KwasaFramework::new(config).unwrap();
        framework.initialize().await.unwrap();

        let test_info = InformationUnit {
            content: b"full kwasa processing test".to_vec(),
            semantic_meaning: "comprehensive kwasa framework test with fire adaptation".to_string(),
            catalytic_potential: 1.0,
            consciousness_level: 2,
            fire_adaptation_factor: 1.0,
        };

        let result = framework.process_information(test_info).await.unwrap();
        
        // Should show massive enhancement from all systems combined
        assert!(result.fire_adaptation_factor > 100.0); // Combined enhancements
        assert!(result.catalytic_potential > 4.0); // Survival advantage
        
        let analytics = framework.get_framework_analytics().await;
        assert!(analytics.average_enhancement > 50.0); // Significant total enhancement
        assert!(analytics.average_consciousness > 0.0);
    }

    #[tokio::test]
    async fn test_framework_metrics_tracking() {
        let config = KwasaConfig::default();
        let framework = KwasaFramework::new(config).unwrap();
        framework.initialize().await.unwrap();

        // Process multiple information units
        for i in 0..5 {
            let test_info = InformationUnit {
                content: format!("test info {}", i).into_bytes(),
                semantic_meaning: format!("test pattern communication {}", i),
                catalytic_potential: 1.0,
                consciousness_level: 1,
                fire_adaptation_factor: 1.0,
            };

            framework.process_information(test_info).await.unwrap();
        }

        let analytics = framework.get_framework_analytics().await;
        
        // Should have metrics history
        assert!(analytics.metrics_tracker.enhancement_history.len() == 5);
        assert!(analytics.metrics_tracker.consciousness_history.len() == 5);
        assert!(analytics.average_enhancement > 1.0);
    }

    #[tokio::test]
    async fn test_framework_shutdown() {
        let config = KwasaConfig::default();
        let framework = KwasaFramework::new(config).unwrap();
        framework.initialize().await.unwrap();

        framework.shutdown().await.unwrap();
        
        let status = framework.get_framework_status().await;
        assert!(!status.initialized);
        assert!(!status.bmd_network_active);
    }
} 