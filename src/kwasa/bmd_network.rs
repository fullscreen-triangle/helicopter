//! # BMD Network Coordination System
//!
//! Multi-level BMD network coordination across molecular, neural, and cognitive scales.
//! This module implements the hierarchical BMD network architecture that enables
//! consciousness emergence through coordinated information catalysis.
//!
//! ## Network Architecture
//!
//! The BMD network operates across three primary scales:
//!
//! ### Molecular BMD Network (Scale 1: 10^-9m)
//! - 1000+ molecular BMDs for direct molecular information processing
//! - 10^12 Hz processing frequency (molecular timescale)
//! - Direct chemical structure → information conversion
//!
//! ### Neural BMD Network (Scale 2: 10^-3m)  
//! - 100+ neural BMDs for neural network information catalysis
//! - 10^3 Hz processing frequency (neural timescale)
//! - Pattern recognition and semantic processing
//!
//! ### Cognitive BMD Network (Scale 3: 10^0m)
//! - 10+ cognitive BMDs for high-level cognitive processing
//! - 10^1 Hz processing frequency (cognitive timescale)
//! - Abstract reasoning and consciousness emergence
//!
//! ## Mathematical Model
//!
//! ### Network Information Flow
//! ```
//! I_network(t) = Σᵢ BMDᵢ(I_input) ○ Ψ_coordination ○ Θ_consciousness
//! 
//! Where:
//! - I_network(t): Total network information flow
//! - BMDᵢ: Individual BMD contribution
//! - Ψ_coordination: Network coordination function
//! - Θ_consciousness: Consciousness emergence factor
//! ```
//!
//! ### Consciousness Emergence Threshold
//! ```
//! Θ_consciousness = {
//!   0,  if |BMD_active| < 100
//!   log(|BMD_active| / 100), if |BMD_active| ≥ 100
//! }
//! ```

use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::kwasa::{
    BiologicalMaxwellDemon, BMDType, BMDState, BMDStatus, InformationUnit, 
    KwasaError, KwasaResult, BMDMetrics, KwasaConstants
};

/// Network topology configurations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NetworkTopology {
    /// Hierarchical topology with clear scale separation
    Hierarchical {
        molecular_count: usize,
        neural_count: usize,
        cognitive_count: usize,
    },
    /// Mesh topology with all-to-all connections
    Mesh {
        total_bmds: usize,
        connection_density: f64,
    },
    /// Ring topology for coordinated processing
    Ring {
        bmds_per_ring: usize,
        ring_count: usize,
    },
    /// Adaptive topology that changes based on processing needs
    Adaptive {
        min_bmds: usize,
        max_bmds: usize,
        adaptation_threshold: f64,
    },
}

/// BMD network configuration
#[derive(Debug, Clone)]
pub struct BMDNetworkConfig {
    /// Network topology
    pub topology: NetworkTopology,
    /// Consciousness emergence threshold
    pub consciousness_threshold: usize,
    /// Maximum information processing rate
    pub max_processing_rate: f64,
    /// Network coordination efficiency
    pub coordination_efficiency: f64,
    /// Fire adaptation enabled
    pub fire_adaptation_enabled: bool,
    /// Semantic preservation requirement
    pub semantic_preservation_threshold: f64,
}

impl Default for BMDNetworkConfig {
    fn default() -> Self {
        Self {
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
        }
    }
}

/// Network connection between BMDs
#[derive(Debug, Clone)]
pub struct BMDConnection {
    /// Source BMD ID
    pub source_id: String,
    /// Target BMD ID
    pub target_id: String,
    /// Connection weight (information flow capacity)
    pub weight: f64,
    /// Connection type
    pub connection_type: ConnectionType,
    /// Current information flow
    pub current_flow: f64,
}

/// Types of connections between BMDs
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionType {
    /// Direct information flow
    Direct,
    /// Catalytic enhancement
    Catalytic,
    /// Consciousness coordination
    Consciousness,
    /// Fire adaptation feedback
    FireAdaptation,
    /// Semantic preservation
    Semantic,
}

/// Network processing statistics
#[derive(Debug, Clone)]
pub struct NetworkStatistics {
    /// Total BMDs in network
    pub total_bmds: usize,
    /// Active BMDs
    pub active_bmds: usize,
    /// Total information throughput
    pub total_throughput: f64,
    /// Average semantic preservation
    pub avg_semantic_preservation: f64,
    /// Consciousness emergence level
    pub consciousness_level: f64,
    /// Fire adaptation enhancement
    pub fire_adaptation_enhancement: f64,
    /// Network efficiency
    pub network_efficiency: f64,
}

/// Main BMD Network coordination system
#[derive(Debug)]
pub struct BMDNetwork {
    /// Network identifier
    pub id: String,
    /// Network configuration
    pub config: BMDNetworkConfig,
    /// All BMDs in the network
    pub bmds: Arc<RwLock<HashMap<String, Arc<BiologicalMaxwellDemon>>>>,
    /// Network connections
    pub connections: Arc<RwLock<Vec<BMDConnection>>>,
    /// Processing semaphore for rate limiting
    pub processing_semaphore: Arc<Semaphore>,
    /// Network statistics
    pub statistics: Arc<RwLock<NetworkStatistics>>,
    /// Consciousness emergence tracking
    pub consciousness_tracker: Arc<RwLock<ConsciousnessTracker>>,
    /// Information routing table
    pub routing_table: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

/// Consciousness emergence tracking
#[derive(Debug, Clone)]
pub struct ConsciousnessTracker {
    /// Current consciousness level
    pub consciousness_level: f64,
    /// Consciousness emergence threshold
    pub emergence_threshold: f64,
    /// Active consciousness centers
    pub consciousness_centers: Vec<String>,
    /// Fire adaptation consciousness contribution
    pub fire_adaptation_contribution: f64,
    /// Temporal coherence of consciousness
    pub temporal_coherence: f64,
}

impl Default for ConsciousnessTracker {
    fn default() -> Self {
        Self {
            consciousness_level: 0.0,
            emergence_threshold: KwasaConstants::CONSCIOUSNESS_SEQUENTIAL_STATES,
            consciousness_centers: Vec::new(),
            fire_adaptation_contribution: 0.0,
            temporal_coherence: 0.0,
        }
    }
}

impl BMDNetwork {
    /// Create a new BMD network
    pub fn new(config: BMDNetworkConfig) -> KwasaResult<Self> {
        let max_concurrent = match &config.topology {
            NetworkTopology::Hierarchical { molecular_count, neural_count, cognitive_count } => {
                molecular_count + neural_count + cognitive_count
            },
            NetworkTopology::Mesh { total_bmds, .. } => *total_bmds,
            NetworkTopology::Ring { bmds_per_ring, ring_count } => bmds_per_ring * ring_count,
            NetworkTopology::Adaptive { max_bmds, .. } => *max_bmds,
        };

        let initial_stats = NetworkStatistics {
            total_bmds: 0,
            active_bmds: 0,
            total_throughput: 0.0,
            avg_semantic_preservation: 0.0,
            consciousness_level: 0.0,
            fire_adaptation_enhancement: 0.0,
            network_efficiency: 0.0,
        };

        Ok(Self {
            id: Uuid::new_v4().to_string(),
            config,
            bmds: Arc::new(RwLock::new(HashMap::new())),
            connections: Arc::new(RwLock::new(Vec::new())),
            processing_semaphore: Arc::new(Semaphore::new(max_concurrent)),
            statistics: Arc::new(RwLock::new(initial_stats)),
            consciousness_tracker: Arc::new(RwLock::new(ConsciousnessTracker::default())),
            routing_table: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Initialize the BMD network according to configuration
    pub async fn initialize(&self) -> KwasaResult<()> {
        log::info!("Initializing BMD network {} with topology: {:?}", self.id, self.config.topology);

        match &self.config.topology {
            NetworkTopology::Hierarchical { molecular_count, neural_count, cognitive_count } => {
                self.initialize_hierarchical_topology(*molecular_count, *neural_count, *cognitive_count).await?;
            },
            NetworkTopology::Mesh { total_bmds, connection_density } => {
                self.initialize_mesh_topology(*total_bmds, *connection_density).await?;
            },
            NetworkTopology::Ring { bmds_per_ring, ring_count } => {
                self.initialize_ring_topology(*bmds_per_ring, *ring_count).await?;
            },
            NetworkTopology::Adaptive { min_bmds, max_bmds, adaptation_threshold } => {
                self.initialize_adaptive_topology(*min_bmds, *max_bmds, *adaptation_threshold).await?;
            },
        }

        // Initialize all BMDs
        self.initialize_all_bmds().await?;
        
        // Update network statistics
        self.update_network_statistics().await?;

        log::info!("BMD network {} initialized successfully", self.id);
        Ok(())
    }

    /// Initialize hierarchical topology
    async fn initialize_hierarchical_topology(
        &self, 
        molecular_count: usize, 
        neural_count: usize, 
        cognitive_count: usize
    ) -> KwasaResult<()> {
        let mut bmds = self.bmds.write().await;
        let mut connections = self.connections.write().await;

        // Create molecular BMDs
        for i in 0..molecular_count {
            let bmd_id = format!("molecular_bmd_{:04}", i);
            let bmd = Arc::new(BiologicalMaxwellDemon::new(bmd_id.clone(), BMDType::Molecular)?);
            bmds.insert(bmd_id.clone(), bmd);
        }

        // Create neural BMDs
        for i in 0..neural_count {
            let bmd_id = format!("neural_bmd_{:04}", i);
            let bmd = Arc::new(BiologicalMaxwellDemon::new(bmd_id.clone(), BMDType::Neural)?);
            bmds.insert(bmd_id.clone(), bmd);
        }

        // Create cognitive BMDs
        for i in 0..cognitive_count {
            let bmd_id = format!("cognitive_bmd_{:04}", i);
            let bmd = Arc::new(BiologicalMaxwellDemon::new(bmd_id.clone(), BMDType::Cognitive)?);
            bmds.insert(bmd_id.clone(), bmd);
        }

        // Create hierarchical connections
        // Molecular → Neural connections
        for mol_idx in 0..molecular_count {
            let neural_idx = mol_idx / (molecular_count / neural_count.max(1));
            if neural_idx < neural_count {
                let connection = BMDConnection {
                    source_id: format!("molecular_bmd_{:04}", mol_idx),
                    target_id: format!("neural_bmd_{:04}", neural_idx),
                    weight: 1.0,
                    connection_type: ConnectionType::Direct,
                    current_flow: 0.0,
                };
                connections.push(connection);
            }
        }

        // Neural → Cognitive connections
        for neural_idx in 0..neural_count {
            let cognitive_idx = neural_idx / (neural_count / cognitive_count.max(1));
            if cognitive_idx < cognitive_count {
                let connection = BMDConnection {
                    source_id: format!("neural_bmd_{:04}", neural_idx),
                    target_id: format!("cognitive_bmd_{:04}", cognitive_idx),
                    weight: 1.0,
                    connection_type: ConnectionType::Consciousness,
                    current_flow: 0.0,
                };
                connections.push(connection);
            }
        }

        log::info!("Created hierarchical topology: {} molecular, {} neural, {} cognitive BMDs", 
                  molecular_count, neural_count, cognitive_count);
        Ok(())
    }

    /// Initialize mesh topology
    async fn initialize_mesh_topology(&self, total_bmds: usize, connection_density: f64) -> KwasaResult<()> {
        let mut bmds = self.bmds.write().await;
        let mut connections = self.connections.write().await;

        // Create BMDs with mixed types
        for i in 0..total_bmds {
            let bmd_type = match i % 3 {
                0 => BMDType::Molecular,
                1 => BMDType::Neural,
                _ => BMDType::Cognitive,
            };
            
            let bmd_id = format!("mesh_bmd_{:04}", i);
            let bmd = Arc::new(BiologicalMaxwellDemon::new(bmd_id.clone(), bmd_type)?);
            bmds.insert(bmd_id.clone(), bmd);
        }

        // Create mesh connections based on density
        let total_possible_connections = total_bmds * (total_bmds - 1) / 2;
        let connections_to_create = (total_possible_connections as f64 * connection_density) as usize;

        for _ in 0..connections_to_create {
            let source_idx = rand::random::<usize>() % total_bmds;
            let mut target_idx = rand::random::<usize>() % total_bmds;
            while target_idx == source_idx {
                target_idx = rand::random::<usize>() % total_bmds;
            }

            let connection = BMDConnection {
                source_id: format!("mesh_bmd_{:04}", source_idx),
                target_id: format!("mesh_bmd_{:04}", target_idx),
                weight: rand::random::<f64>(),
                connection_type: ConnectionType::Direct,
                current_flow: 0.0,
            };
            connections.push(connection);
        }

        log::info!("Created mesh topology: {} BMDs with {:.2}% connection density", 
                  total_bmds, connection_density * 100.0);
        Ok(())
    }

    /// Initialize ring topology
    async fn initialize_ring_topology(&self, bmds_per_ring: usize, ring_count: usize) -> KwasaResult<()> {
        let mut bmds = self.bmds.write().await;
        let mut connections = self.connections.write().await;

        // Create BMDs in rings
        for ring in 0..ring_count {
            for pos in 0..bmds_per_ring {
                let bmd_id = format!("ring_{}_bmd_{:04}", ring, pos);
                let bmd_type = if ring == 0 { BMDType::Molecular } 
                              else if ring == 1 { BMDType::Neural } 
                              else { BMDType::Cognitive };
                
                let bmd = Arc::new(BiologicalMaxwellDemon::new(bmd_id.clone(), bmd_type)?);
                bmds.insert(bmd_id.clone(), bmd);

                // Create ring connections
                let next_pos = (pos + 1) % bmds_per_ring;
                let connection = BMDConnection {
                    source_id: bmd_id.clone(),
                    target_id: format!("ring_{}_bmd_{:04}", ring, next_pos),
                    weight: 1.0,
                    connection_type: ConnectionType::Direct,
                    current_flow: 0.0,
                };
                connections.push(connection);
            }
        }

        // Create inter-ring connections
        for ring in 0..ring_count.saturating_sub(1) {
            for pos in 0..bmds_per_ring {
                let connection = BMDConnection {
                    source_id: format!("ring_{}_bmd_{:04}", ring, pos),
                    target_id: format!("ring_{}_bmd_{:04}", ring + 1, pos),
                    weight: 0.5,
                    connection_type: ConnectionType::Consciousness,
                    current_flow: 0.0,
                };
                connections.push(connection);
            }
        }

        log::info!("Created ring topology: {} rings with {} BMDs each", ring_count, bmds_per_ring);
        Ok(())
    }

    /// Initialize adaptive topology
    async fn initialize_adaptive_topology(
        &self, 
        min_bmds: usize, 
        _max_bmds: usize, 
        _adaptation_threshold: f64
    ) -> KwasaResult<()> {
        // Start with minimum BMDs, will adapt based on load
        let mut bmds = self.bmds.write().await;

        for i in 0..min_bmds {
            let bmd_type = if i < min_bmds / 3 { BMDType::Molecular }
                          else if i < 2 * min_bmds / 3 { BMDType::Neural }
                          else { BMDType::Cognitive };
            
            let bmd_id = format!("adaptive_bmd_{:04}", i);
            let bmd = Arc::new(BiologicalMaxwellDemon::new(bmd_id.clone(), bmd_type)?);
            bmds.insert(bmd_id.clone(), bmd);
        }

        log::info!("Created adaptive topology starting with {} BMDs", min_bmds);
        Ok(())
    }

    /// Initialize all BMDs in the network
    async fn initialize_all_bmds(&self) -> KwasaResult<()> {
        let bmds = self.bmds.read().await;
        let mut init_tasks = Vec::new();

        for (bmd_id, bmd) in bmds.iter() {
            let bmd_clone = Arc::clone(bmd);
            let id_clone = bmd_id.clone();
            
            let task = tokio::spawn(async move {
                if let Err(e) = bmd_clone.initialize().await {
                    log::error!("Failed to initialize BMD {}: {}", id_clone, e);
                    return Err(e);
                }
                Ok(())
            });
            init_tasks.push(task);
        }

        // Wait for all BMDs to initialize
        for task in init_tasks {
            task.await.map_err(|e| KwasaError::BMDInitializationError(e.to_string()))??;
        }

        log::info!("All BMDs in network {} initialized successfully", self.id);
        Ok(())
    }

    /// Process information through the BMD network
    pub async fn process_information(&self, information: InformationUnit) -> KwasaResult<InformationUnit> {
        // Acquire processing permit
        let _permit = self.processing_semaphore.acquire().await
            .map_err(|e| KwasaError::NetworkCoordinationError(e.to_string()))?;

        // Route information through network based on topology
        let processed_information = match &self.config.topology {
            NetworkTopology::Hierarchical { .. } => {
                self.process_through_hierarchy(information).await?
            },
            _ => {
                self.process_through_mesh(information).await?
            },
        };

        // Update consciousness tracking
        self.update_consciousness_tracking(&processed_information).await?;

        // Update network statistics
        self.update_network_statistics().await?;

        Ok(processed_information)
    }

    /// Process information through hierarchical topology
    async fn process_through_hierarchy(&self, mut information: InformationUnit) -> KwasaResult<InformationUnit> {
        let bmds = self.bmds.read().await;

        // Stage 1: Molecular processing
        if let Some(molecular_bmd) = bmds.values().find(|bmd| matches!(bmd.bmd_type, BMDType::Molecular)) {
            information = molecular_bmd.catalyze_information(information).await?;
        }

        // Stage 2: Neural processing  
        if let Some(neural_bmd) = bmds.values().find(|bmd| matches!(bmd.bmd_type, BMDType::Neural)) {
            information = neural_bmd.catalyze_information(information).await?;
        }

        // Stage 3: Cognitive processing
        if let Some(cognitive_bmd) = bmds.values().find(|bmd| matches!(bmd.bmd_type, BMDType::Cognitive)) {
            information = cognitive_bmd.catalyze_information(information).await?;
        }

        Ok(information)
    }

    /// Process information through mesh/other topologies
    async fn process_through_mesh(&self, mut information: InformationUnit) -> KwasaResult<InformationUnit> {
        let bmds = self.bmds.read().await;
        
        // Process through available BMDs in parallel
        let mut processing_tasks = Vec::new();
        
        for (bmd_id, bmd) in bmds.iter().take(3) { // Limit to prevent overwhelm
            let bmd_clone = Arc::clone(bmd);
            let info_clone = information.clone();
            let id_clone = bmd_id.clone();
            
            let task = tokio::spawn(async move {
                bmd_clone.catalyze_information(info_clone).await
                    .map_err(|e| (id_clone, e))
            });
            processing_tasks.push(task);
        }

        // Collect results and aggregate
        let mut processed_results = Vec::new();
        for task in processing_tasks {
            match task.await {
                Ok(Ok(result)) => processed_results.push(result),
                Ok(Err((bmd_id, e))) => {
                    log::warn!("BMD {} processing failed: {}", bmd_id, e);
                },
                Err(e) => {
                    log::error!("Task execution failed: {}", e);
                }
            }
        }

        // Aggregate results (simple average for now)
        if !processed_results.is_empty() {
            let avg_catalytic_potential = processed_results.iter()
                .map(|r| r.catalytic_potential)
                .sum::<f64>() / processed_results.len() as f64;
            
            let avg_fire_adaptation = processed_results.iter()
                .map(|r| r.fire_adaptation_factor)
                .sum::<f64>() / processed_results.len() as f64;

            information.catalytic_potential = avg_catalytic_potential;
            information.fire_adaptation_factor = avg_fire_adaptation;
        }

        Ok(information)
    }

    /// Update consciousness tracking based on processed information
    async fn update_consciousness_tracking(&self, information: &InformationUnit) -> KwasaResult<()> {
        let mut tracker = self.consciousness_tracker.write().await;
        let bmds = self.bmds.read().await;

        // Count active BMDs
        let active_bmd_count = bmds.len(); // Simplified - should check actual BMD states

        // Update consciousness level based on network size and fire adaptation
        if active_bmd_count >= self.config.consciousness_threshold {
            tracker.consciousness_level = (active_bmd_count as f64 / self.config.consciousness_threshold as f64).ln();
            tracker.fire_adaptation_contribution = information.fire_adaptation_factor;
            tracker.temporal_coherence = information.catalytic_potential;
            
            // Add consciousness centers for highly adapted information
            if information.fire_adaptation_factor > KwasaConstants::FIRE_PROCESSING_ENHANCEMENT {
                tracker.consciousness_centers.push(format!("center_{}", tracker.consciousness_centers.len()));
            }
        }

        Ok(())
    }

    /// Update network statistics
    async fn update_network_statistics(&self) -> KwasaResult<()> {
        let mut stats = self.statistics.write().await;
        let bmds = self.bmds.read().await;
        let tracker = self.consciousness_tracker.read().await;

        stats.total_bmds = bmds.len();
        stats.active_bmds = bmds.len(); // Simplified
        stats.consciousness_level = tracker.consciousness_level;
        stats.fire_adaptation_enhancement = tracker.fire_adaptation_contribution;
        
        // Calculate network efficiency based on consciousness emergence
        stats.network_efficiency = if stats.active_bmds >= self.config.consciousness_threshold {
            0.95 * (1.0 + tracker.consciousness_level)
        } else {
            0.5
        };

        Ok(())
    }

    /// Get current network status
    pub async fn get_network_status(&self) -> NetworkStatus {
        let stats = self.statistics.read().await;
        let tracker = self.consciousness_tracker.read().await;

        NetworkStatus {
            network_id: self.id.clone(),
            config: self.config.clone(),
            statistics: stats.clone(),
            consciousness: tracker.clone(),
            total_connections: self.connections.read().await.len(),
        }
    }

    /// Shutdown the network gracefully
    pub async fn shutdown(&self) -> KwasaResult<()> {
        let bmds = self.bmds.read().await;
        let mut shutdown_tasks = Vec::new();

        for (bmd_id, bmd) in bmds.iter() {
            let bmd_clone = Arc::clone(bmd);
            let id_clone = bmd_id.clone();
            
            let task = tokio::spawn(async move {
                if let Err(e) = bmd_clone.shutdown().await {
                    log::error!("Failed to shutdown BMD {}: {}", id_clone, e);
                }
            });
            shutdown_tasks.push(task);
        }

        // Wait for all BMDs to shutdown
        for task in shutdown_tasks {
            let _ = task.await;
        }

        log::info!("BMD network {} shutdown successfully", self.id);
        Ok(())
    }
}

/// Complete network status
#[derive(Debug, Clone)]
pub struct NetworkStatus {
    pub network_id: String,
    pub config: BMDNetworkConfig,
    pub statistics: NetworkStatistics,
    pub consciousness: ConsciousnessTracker,
    pub total_connections: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hierarchical_network_creation() {
        let config = BMDNetworkConfig {
            topology: NetworkTopology::Hierarchical {
                molecular_count: 10,
                neural_count: 5,
                cognitive_count: 2,
            },
            ..Default::default()
        };

        let network = BMDNetwork::new(config).unwrap();
        network.initialize().await.unwrap();

        let status = network.get_network_status().await;
        assert_eq!(status.statistics.total_bmds, 17); // 10 + 5 + 2
    }

    #[tokio::test]
    async fn test_consciousness_emergence() {
        let config = BMDNetworkConfig {
            topology: NetworkTopology::Hierarchical {
                molecular_count: 80,
                neural_count: 20,
                cognitive_count: 10,
            },
            consciousness_threshold: 100,
            ..Default::default()
        };

        let network = BMDNetwork::new(config).unwrap();
        network.initialize().await.unwrap();

        let test_information = InformationUnit {
            content: b"consciousness test".to_vec(),
            semantic_meaning: "consciousness emergence test".to_string(),
            catalytic_potential: 1.0,
            consciousness_level: 3,
            fire_adaptation_factor: 4.0,
        };

        let _result = network.process_information(test_information).await.unwrap();
        
        let status = network.get_network_status().await;
        assert!(status.statistics.total_bmds >= status.config.consciousness_threshold);
        assert!(status.consciousness.consciousness_level > 0.0);
    }
} 