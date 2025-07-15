//! # BMD Network
//!
//! Multi-level BMD coordination system that orchestrates information processing across
//! molecular, neural, and cognitive scales. The network enables hierarchical semantic
//! catalysis with fire-adapted consciousness enhancements.
//!
//! ## Network Architecture
//!
//! ```
//! Cognitive BMDs (Discourse/Context)
//!        ↑
//! Neural BMDs (Sentence Structure)
//!        ↑
//! Molecular BMDs (Tokens/Phonemes)
//!        ↑
//! Continuous Oscillatory Reality
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::kwasa::{
    BiologicalMaxwellDemon, BmdProcessingLevel, BmdCapabilities,
    KwasaResult, KwasaError
};

/// BMD Network - Multi-level BMD coordination system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmdNetwork {
    /// Network identifier
    pub id: String,
    
    /// Molecular-level BMDs (token/phoneme processing)
    pub molecular_bmds: Vec<BiologicalMaxwellDemon>,
    
    /// Neural-level BMDs (sentence structure processing)
    pub neural_bmds: Vec<BiologicalMaxwellDemon>,
    
    /// Cognitive-level BMDs (discourse/context processing)
    pub cognitive_bmds: Vec<BiologicalMaxwellDemon>,
    
    /// Network-wide consciousness threshold
    pub network_consciousness_threshold: f64,
    
    /// Fire-adapted enhancement coordination
    pub fire_adapted_coordination: FireAdaptedCoordination,
    
    /// Inter-BMD communication channels
    pub communication_channels: HashMap<String, CommunicationChannel>,
    
    /// Network processing statistics
    pub network_stats: NetworkProcessingStats,
    
    /// Hierarchical processing state
    pub hierarchical_state: HierarchicalProcessingState,
}

/// Fire-adapted coordination across BMD levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireAdaptedCoordination {
    /// Molecular-level fire enhancement
    pub molecular_fire_enhancement: f64,
    
    /// Neural-level fire enhancement
    pub neural_fire_enhancement: f64,
    
    /// Cognitive-level fire enhancement
    pub cognitive_fire_enhancement: f64,
    
    /// Cross-level enhancement synchronization
    pub cross_level_synchronization: f64,
    
    /// Fire circle communication effectiveness
    pub fire_circle_effectiveness: f64,
}

/// Communication channel between BMDs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationChannel {
    /// Source BMD identifier
    pub source_bmd_id: String,
    
    /// Target BMD identifier
    pub target_bmd_id: String,
    
    /// Communication bandwidth
    pub bandwidth: f64,
    
    /// Message queue
    pub message_queue: Vec<InterBmdMessage>,
    
    /// Channel effectiveness
    pub effectiveness: f64,
}

/// Inter-BMD message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterBmdMessage {
    /// Message identifier
    pub id: String,
    
    /// Source BMD
    pub source: String,
    
    /// Target BMD
    pub target: String,
    
    /// Message content (semantic units)
    pub content: Vec<String>,
    
    /// Processing level
    pub processing_level: BmdProcessingLevel,
    
    /// Consciousness enhancement data
    pub consciousness_data: f64,
    
    /// Timestamp
    pub timestamp: u64,
}

/// Network processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkProcessingStats {
    /// Total semantic units processed across network
    pub total_semantic_units: u64,
    
    /// Hierarchical processing cycles
    pub hierarchical_cycles: u64,
    
    /// Cross-level communications
    pub cross_level_communications: u64,
    
    /// Network efficiency
    pub network_efficiency: f64,
    
    /// Fire-adapted enhancements coordinated
    pub fire_adapted_enhancements: u64,
    
    /// Consciousness threshold breaches
    pub consciousness_threshold_breaches: u64,
}

/// Hierarchical processing state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalProcessingState {
    /// Molecular level processing state
    pub molecular_state: LevelProcessingState,
    
    /// Neural level processing state
    pub neural_state: LevelProcessingState,
    
    /// Cognitive level processing state
    pub cognitive_state: LevelProcessingState,
    
    /// Inter-level synchronization
    pub inter_level_sync: f64,
}

/// Processing state at each level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelProcessingState {
    /// Number of active BMDs at this level
    pub active_bmds: usize,
    
    /// Current processing load
    pub processing_load: f64,
    
    /// Level efficiency
    pub level_efficiency: f64,
    
    /// Fire-adapted enhancement level
    pub fire_enhancement_level: f64,
}

impl BmdNetwork {
    /// Create a new BMD network
    pub fn new(id: String) -> Self {
        Self {
            id,
            molecular_bmds: Vec::new(),
            neural_bmds: Vec::new(),
            cognitive_bmds: Vec::new(),
            network_consciousness_threshold: crate::kwasa::constants::FIRE_ADAPTED_CONSCIOUSNESS_THRESHOLD,
            fire_adapted_coordination: FireAdaptedCoordination::default(),
            communication_channels: HashMap::new(),
            network_stats: NetworkProcessingStats::default(),
            hierarchical_state: HierarchicalProcessingState::default(),
        }
    }

    /// Add a BMD to the network
    pub fn add_bmd(&mut self, bmd: BiologicalMaxwellDemon) -> KwasaResult<()> {
        let bmd_id = bmd.id.clone();
        
        match bmd.processing_level {
            BmdProcessingLevel::Molecular => {
                self.molecular_bmds.push(bmd);
                self.hierarchical_state.molecular_state.active_bmds += 1;
            }
            BmdProcessingLevel::Neural => {
                self.neural_bmds.push(bmd);
                self.hierarchical_state.neural_state.active_bmds += 1;
            }
            BmdProcessingLevel::Cognitive => {
                self.cognitive_bmds.push(bmd);
                self.hierarchical_state.cognitive_state.active_bmds += 1;
            }
        }

        // Create communication channels for new BMD
        self.create_communication_channels(&bmd_id)?;

        Ok(())
    }

    /// Process oscillatory input through hierarchical BMD network
    pub fn process_hierarchical_input(&mut self, oscillatory_data: &[f64]) -> KwasaResult<Vec<String>> {
        // Check network consciousness threshold
        if self.network_consciousness_threshold < crate::kwasa::constants::FIRE_ADAPTED_CONSCIOUSNESS_THRESHOLD {
            return Err(KwasaError::ConsciousnessThresholdNotReached(self.network_consciousness_threshold));
        }

        // Process through hierarchical levels
        let molecular_results = self.process_molecular_level(oscillatory_data)?;
        let neural_results = self.process_neural_level(&molecular_results)?;
        let cognitive_results = self.process_cognitive_level(&neural_results)?;

        // Update network statistics
        self.update_network_stats(&cognitive_results);

        // Coordinate fire-adapted enhancements
        self.coordinate_fire_adapted_enhancements();

        Ok(cognitive_results)
    }

    /// Process data at molecular level
    fn process_molecular_level(&mut self, data: &[f64]) -> KwasaResult<Vec<String>> {
        let mut all_results = Vec::new();

        for bmd in &mut self.molecular_bmds {
            let results = bmd.process_oscillatory_input(data)?;
            all_results.extend(results);
        }

        // Update molecular state
        self.hierarchical_state.molecular_state.processing_load = data.len() as f64;
        self.hierarchical_state.molecular_state.level_efficiency = 
            self.calculate_level_efficiency(&self.molecular_bmds);

        Ok(all_results)
    }

    /// Process data at neural level
    fn process_neural_level(&mut self, molecular_results: &[String]) -> KwasaResult<Vec<String>> {
        let mut all_results = Vec::new();

        // Convert molecular results to neural input
        let neural_input = self.convert_molecular_to_neural(molecular_results)?;

        for bmd in &mut self.neural_bmds {
            let results = bmd.process_oscillatory_input(&neural_input)?;
            all_results.extend(results);
        }

        // Update neural state
        self.hierarchical_state.neural_state.processing_load = neural_input.len() as f64;
        self.hierarchical_state.neural_state.level_efficiency = 
            self.calculate_level_efficiency(&self.neural_bmds);

        Ok(all_results)
    }

    /// Process data at cognitive level
    fn process_cognitive_level(&mut self, neural_results: &[String]) -> KwasaResult<Vec<String>> {
        let mut all_results = Vec::new();

        // Convert neural results to cognitive input
        let cognitive_input = self.convert_neural_to_cognitive(neural_results)?;

        for bmd in &mut self.cognitive_bmds {
            let results = bmd.process_oscillatory_input(&cognitive_input)?;
            all_results.extend(results);
        }

        // Update cognitive state
        self.hierarchical_state.cognitive_state.processing_load = cognitive_input.len() as f64;
        self.hierarchical_state.cognitive_state.level_efficiency = 
            self.calculate_level_efficiency(&self.cognitive_bmds);

        Ok(all_results)
    }

    /// Convert molecular semantic units to neural input
    fn convert_molecular_to_neural(&self, molecular_results: &[String]) -> KwasaResult<Vec<f64>> {
        let mut neural_input = Vec::new();

        for result in molecular_results {
            // Extract energy level from molecular semantic unit
            let energy = if result.contains("HIGH_ENERGY") {
                0.8
            } else if result.contains("MED_ENERGY") {
                0.5
            } else {
                0.2
            };
            neural_input.push(energy);
        }

        Ok(neural_input)
    }

    /// Convert neural semantic units to cognitive input
    fn convert_neural_to_cognitive(&self, neural_results: &[String]) -> KwasaResult<Vec<f64>> {
        let mut cognitive_input = Vec::new();

        for result in neural_results {
            // Extract complexity from neural semantic unit
            let complexity = if result.contains("COMPLEX_STRUCTURE") {
                0.9
            } else if result.contains("SIMPLE_STRUCTURE") {
                0.6
            } else {
                0.3
            };
            cognitive_input.push(complexity);
        }

        Ok(cognitive_input)
    }

    /// Calculate efficiency for a level
    fn calculate_level_efficiency(&self, bmds: &[BiologicalMaxwellDemon]) -> f64 {
        if bmds.is_empty() {
            return 0.0;
        }

        let total_efficiency: f64 = bmds.iter()
            .map(|bmd| bmd.processing_stats.processing_efficiency)
            .sum();

        total_efficiency / bmds.len() as f64
    }

    /// Create communication channels for new BMD
    fn create_communication_channels(&mut self, bmd_id: &str) -> KwasaResult<()> {
        // Create channels to all existing BMDs
        let all_bmd_ids: Vec<String> = self.molecular_bmds.iter()
            .chain(self.neural_bmds.iter())
            .chain(self.cognitive_bmds.iter())
            .map(|bmd| bmd.id.clone())
            .collect();

        for existing_id in all_bmd_ids {
            if existing_id != bmd_id {
                // Create bidirectional channels
                let channel_id_1 = format!("{}_{}", bmd_id, existing_id);
                let channel_id_2 = format!("{}_{}", existing_id, bmd_id);

                let channel_1 = CommunicationChannel {
                    source_bmd_id: bmd_id.to_string(),
                    target_bmd_id: existing_id.clone(),
                    bandwidth: 1.0,
                    message_queue: Vec::new(),
                    effectiveness: 1.0,
                };

                let channel_2 = CommunicationChannel {
                    source_bmd_id: existing_id,
                    target_bmd_id: bmd_id.to_string(),
                    bandwidth: 1.0,
                    message_queue: Vec::new(),
                    effectiveness: 1.0,
                };

                self.communication_channels.insert(channel_id_1, channel_1);
                self.communication_channels.insert(channel_id_2, channel_2);
            }
        }

        Ok(())
    }

    /// Coordinate fire-adapted enhancements across levels
    fn coordinate_fire_adapted_enhancements(&mut self) {
        // Calculate average fire enhancement across levels
        let molecular_avg = self.calculate_average_fire_enhancement(&self.molecular_bmds);
        let neural_avg = self.calculate_average_fire_enhancement(&self.neural_bmds);
        let cognitive_avg = self.calculate_average_fire_enhancement(&self.cognitive_bmds);

        // Update coordination
        self.fire_adapted_coordination.molecular_fire_enhancement = molecular_avg;
        self.fire_adapted_coordination.neural_fire_enhancement = neural_avg;
        self.fire_adapted_coordination.cognitive_fire_enhancement = cognitive_avg;

        // Calculate cross-level synchronization
        let enhancements = vec![molecular_avg, neural_avg, cognitive_avg];
        let avg_enhancement = enhancements.iter().sum::<f64>() / enhancements.len() as f64;
        let variance = enhancements.iter()
            .map(|x| (x - avg_enhancement).powi(2))
            .sum::<f64>() / enhancements.len() as f64;

        // Higher synchronization means lower variance
        self.fire_adapted_coordination.cross_level_synchronization = 
            (1.0 / (1.0 + variance)).min(1.0);

        // Update fire circle effectiveness
        self.fire_adapted_coordination.fire_circle_effectiveness = 
            self.fire_adapted_coordination.cross_level_synchronization * 
            crate::kwasa::constants::COMMUNICATION_COMPLEXITY_ENHANCEMENT / 100.0;
    }

    /// Calculate average fire enhancement for a level
    fn calculate_average_fire_enhancement(&self, bmds: &[BiologicalMaxwellDemon]) -> f64 {
        if bmds.is_empty() {
            return 0.0;
        }

        let total_enhancement: f64 = bmds.iter()
            .map(|bmd| bmd.fire_adapted_enhancement)
            .sum();

        total_enhancement / bmds.len() as f64
    }

    /// Update network statistics
    fn update_network_stats(&mut self, results: &[String]) {
        self.network_stats.total_semantic_units += results.len() as u64;
        self.network_stats.hierarchical_cycles += 1;

        // Calculate network efficiency
        let level_efficiencies = vec![
            self.hierarchical_state.molecular_state.level_efficiency,
            self.hierarchical_state.neural_state.level_efficiency,
            self.hierarchical_state.cognitive_state.level_efficiency,
        ];

        self.network_stats.network_efficiency = 
            level_efficiencies.iter().sum::<f64>() / level_efficiencies.len() as f64;

        // Update inter-level synchronization
        self.hierarchical_state.inter_level_sync = 
            self.fire_adapted_coordination.cross_level_synchronization;
    }

    /// Send message between BMDs
    pub fn send_inter_bmd_message(&mut self, message: InterBmdMessage) -> KwasaResult<()> {
        let channel_id = format!("{}_{}", message.source, message.target);
        
        if let Some(channel) = self.communication_channels.get_mut(&channel_id) {
            channel.message_queue.push(message);
            self.network_stats.cross_level_communications += 1;
            Ok(())
        } else {
            Err(KwasaError::BmdNetworkInitializationFailed(
                format!("Communication channel {} not found", channel_id)
            ))
        }
    }

    /// Get network capabilities
    pub fn get_network_capabilities(&self) -> NetworkCapabilities {
        let total_bmds = self.molecular_bmds.len() + self.neural_bmds.len() + self.cognitive_bmds.len();
        
        NetworkCapabilities {
            total_bmds,
            network_consciousness_threshold: self.network_consciousness_threshold,
            fire_adapted_coordination: self.fire_adapted_coordination.clone(),
            network_efficiency: self.network_stats.network_efficiency,
            hierarchical_processing_capability: self.hierarchical_state.inter_level_sync,
        }
    }
}

/// Network capabilities summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCapabilities {
    pub total_bmds: usize,
    pub network_consciousness_threshold: f64,
    pub fire_adapted_coordination: FireAdaptedCoordination,
    pub network_efficiency: f64,
    pub hierarchical_processing_capability: f64,
}

// Default implementations
impl Default for FireAdaptedCoordination {
    fn default() -> Self {
        Self {
            molecular_fire_enhancement: 1.0,
            neural_fire_enhancement: 1.0,
            cognitive_fire_enhancement: 1.0,
            cross_level_synchronization: 0.0,
            fire_circle_effectiveness: 0.0,
        }
    }
}

impl Default for NetworkProcessingStats {
    fn default() -> Self {
        Self {
            total_semantic_units: 0,
            hierarchical_cycles: 0,
            cross_level_communications: 0,
            network_efficiency: 0.0,
            fire_adapted_enhancements: 0,
            consciousness_threshold_breaches: 0,
        }
    }
}

impl Default for HierarchicalProcessingState {
    fn default() -> Self {
        Self {
            molecular_state: LevelProcessingState::default(),
            neural_state: LevelProcessingState::default(),
            cognitive_state: LevelProcessingState::default(),
            inter_level_sync: 0.0,
        }
    }
}

impl Default for LevelProcessingState {
    fn default() -> Self {
        Self {
            active_bmds: 0,
            processing_load: 0.0,
            level_efficiency: 0.0,
            fire_enhancement_level: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kwasa::BiologicalMaxwellDemon;

    #[test]
    fn test_network_creation() {
        let network = BmdNetwork::new("NETWORK_001".to_string());
        assert_eq!(network.id, "NETWORK_001");
        assert_eq!(network.molecular_bmds.len(), 0);
        assert_eq!(network.neural_bmds.len(), 0);
        assert_eq!(network.cognitive_bmds.len(), 0);
    }

    #[test]
    fn test_add_bmd() {
        let mut network = BmdNetwork::new("NETWORK_TEST".to_string());
        let bmd = BiologicalMaxwellDemon::new(
            "BMD_001".to_string(),
            BmdProcessingLevel::Molecular,
            1.0,
        );

        let result = network.add_bmd(bmd);
        assert!(result.is_ok());
        assert_eq!(network.molecular_bmds.len(), 1);
        assert_eq!(network.hierarchical_state.molecular_state.active_bmds, 1);
    }

    #[test]
    fn test_hierarchical_processing() {
        let mut network = BmdNetwork::new("NETWORK_HIERARCHICAL".to_string());
        
        // Add BMDs at each level
        let molecular_bmd = BiologicalMaxwellDemon::new(
            "BMD_MOL_001".to_string(),
            BmdProcessingLevel::Molecular,
            1.0,
        );
        let neural_bmd = BiologicalMaxwellDemon::new(
            "BMD_NEU_001".to_string(),
            BmdProcessingLevel::Neural,
            1.0,
        );
        let cognitive_bmd = BiologicalMaxwellDemon::new(
            "BMD_COG_001".to_string(),
            BmdProcessingLevel::Cognitive,
            1.0,
        );

        network.add_bmd(molecular_bmd).unwrap();
        network.add_bmd(neural_bmd).unwrap();
        network.add_bmd(cognitive_bmd).unwrap();

        let test_data = vec![0.8, 0.5, 0.2, 0.9];
        let result = network.process_hierarchical_input(&test_data);
        
        assert!(result.is_ok());
        let results = result.unwrap();
        assert!(!results.is_empty());
    }
} 