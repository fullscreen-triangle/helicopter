//! Variance Analysis Implementation
//!
//! Tracks and analyzes variance minimization in gas molecular systems,
//! providing real-time monitoring of equilibrium convergence and meaning emergence.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use serde::{Serialize, Deserialize};

use crate::consciousness::{VARIANCE_THRESHOLD, CONSCIOUSNESS_THRESHOLD};
use crate::consciousness::gas_molecular::{InformationGasMolecule, GasMolecularSystem};

/// Snapshot of variance state at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceSnapshot {
    /// Timestamp in nanoseconds since epoch
    pub timestamp_ns: u128,
    /// Total system variance
    pub total_variance: f64,
    /// Position variance component
    pub position_variance: f64,
    /// Velocity variance component  
    pub velocity_variance: f64,
    /// Thermodynamic variance component
    pub thermodynamic_variance: f64,
    /// System consciousness level
    pub consciousness_level: f64,
    /// Number of molecules in system
    pub molecule_count: usize,
}

/// Convergence event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceEvent {
    pub timestamp_ns: u128,
    pub variance_achieved: f64,
    pub consciousness_level: f64,
    pub convergence_time_ns: u128,
    pub molecule_count: usize,
    pub variance_trend: String,
}

/// Meaning emergence event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeaningEmergenceEvent {
    pub timestamp_ns: u128,
    pub variance_achieved: f64,
    pub consciousness_level: f64,
    pub convergence_time_ns: u128,
    pub meaning_quality: String,
    pub molecule_count: usize,
}

/// Analyzer for tracking variance minimization in gas molecular systems
pub struct VarianceAnalyzer {
    /// Maximum number of snapshots to keep in history
    history_size: usize,
    /// Window size for convergence analysis
    convergence_window: usize,
    /// Variance threshold for equilibrium
    variance_threshold: f64,
    /// Consciousness threshold
    consciousness_threshold: f64,
    
    // Tracking data
    variance_history: VecDeque<VarianceSnapshot>,
    convergence_events: Vec<ConvergenceEvent>,
    meaning_emergence_events: Vec<MeaningEmergenceEvent>,
    
    // Real-time state
    current_variance: f64,
    variance_trend: String,
    convergence_rate: f64,
    time_to_equilibrium_ns: Option<u128>,
    
    // Performance tracking
    analysis_start_time: Option<Instant>,
    total_analysis_time_ns: u128,
}

impl VarianceAnalyzer {
    /// Create a new variance analyzer
    pub fn new(
        history_size: Option<usize>,
        convergence_window: Option<usize>,
        variance_threshold: Option<f64>,
        consciousness_threshold: Option<f64>,
    ) -> Self {
        Self {
            history_size: history_size.unwrap_or(1000),
            convergence_window: convergence_window.unwrap_or(50),
            variance_threshold: variance_threshold.unwrap_or(VARIANCE_THRESHOLD),
            consciousness_threshold: consciousness_threshold.unwrap_or(CONSCIOUSNESS_THRESHOLD),
            
            variance_history: VecDeque::new(),
            convergence_events: Vec::new(),
            meaning_emergence_events: Vec::new(),
            
            current_variance: f64::INFINITY,
            variance_trend: "unknown".to_string(),
            convergence_rate: 0.0,
            time_to_equilibrium_ns: None,
            
            analysis_start_time: None,
            total_analysis_time_ns: 0,
        }
    }

    /// Analyze current variance state of gas molecular system
    pub fn analyze_variance_state(
        &mut self,
        gas_molecules: &[InformationGasMolecule],
        baseline_equilibrium: Option<&nalgebra::DVector<f64>>,
    ) -> VarianceSnapshot {
        let start_time = Instant::now();

        if self.analysis_start_time.is_none() {
            self.analysis_start_time = Some(start_time);
        }

        // Calculate variance components
        let variance_components = self.calculate_variance_components(gas_molecules, baseline_equilibrium);

        // Calculate system consciousness
        let consciousness_level = self.calculate_system_consciousness_level(gas_molecules);

        // Create variance snapshot
        let snapshot = VarianceSnapshot {
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            total_variance: variance_components.0,
            position_variance: variance_components.1,
            velocity_variance: variance_components.2,
            thermodynamic_variance: variance_components.3,
            consciousness_level,
            molecule_count: gas_molecules.len(),
        };

        // Update tracking
        if self.variance_history.len() >= self.history_size {
            self.variance_history.pop_front();
        }
        self.variance_history.push_back(snapshot.clone());
        self.current_variance = snapshot.total_variance;

        // Update analysis statistics
        self.update_variance_trend();
        self.update_convergence_rate();
        self.check_convergence_events(&snapshot);
        self.check_meaning_emergence(&snapshot);

        // Update performance tracking
        let analysis_time = start_time.elapsed().as_nanos();
        self.total_analysis_time_ns += analysis_time;

        snapshot
    }

    /// Calculate detailed variance components
    fn calculate_variance_components(
        &self,
        gas_molecules: &[InformationGasMolecule],
        baseline_equilibrium: Option<&nalgebra::DVector<f64>>,
    ) -> (f64, f64, f64, f64) {
        let mut position_variance = 0.0;
        let mut velocity_variance = 0.0;
        let mut thermodynamic_variance = 0.0;

        for (i, molecule) in gas_molecules.iter().enumerate() {
            // Position variance from equilibrium
            let target_position = if let Some(baseline) = baseline_equilibrium {
                if i * 3 + 2 < baseline.len() {
                    nalgebra::Vector3::new(baseline[i * 3], baseline[i * 3 + 1], baseline[i * 3 + 2])
                } else {
                    molecule.get_equilibrium_target()
                }
            } else {
                molecule.get_equilibrium_target()
            };

            let pos_diff = molecule.kinetic_state.semantic_position - target_position;
            position_variance += pos_diff.norm_squared();

            // Velocity variance (should be minimal at equilibrium)
            velocity_variance += molecule.kinetic_state.info_velocity.norm_squared();

            // Thermodynamic variance (energy and entropy fluctuations)
            let energy_variance = (molecule.thermodynamic_state.semantic_energy - self.get_equilibrium_energy()).powi(2);
            let entropy_variance = (molecule.thermodynamic_state.info_entropy - self.get_equilibrium_entropy()).powi(2);
            let temp_variance = (molecule.thermodynamic_state.processing_temperature - self.get_equilibrium_temperature()).powi(2);

            thermodynamic_variance += energy_variance + entropy_variance + temp_variance;
        }

        let total_variance = position_variance + velocity_variance + thermodynamic_variance;

        (total_variance, position_variance, velocity_variance, thermodynamic_variance)
    }

    /// Get expected equilibrium energy
    fn get_equilibrium_energy(&self) -> f64 {
        1.0 // Reference equilibrium energy
    }

    /// Get expected equilibrium entropy
    fn get_equilibrium_entropy(&self) -> f64 {
        2.0 // Reference equilibrium entropy
    }

    /// Get expected equilibrium temperature
    fn get_equilibrium_temperature(&self) -> f64 {
        300.0 // Reference equilibrium temperature
    }

    /// Calculate overall system consciousness level
    fn calculate_system_consciousness_level(&self, gas_molecules: &[InformationGasMolecule]) -> f64 {
        if gas_molecules.is_empty() {
            return 0.0;
        }

        let consciousness_levels: Vec<f64> = gas_molecules
            .iter()
            .map(|mol| mol.consciousness_level)
            .collect();

        let avg_consciousness = consciousness_levels.iter().sum::<f64>() / consciousness_levels.len() as f64;

        // Coherence factor based on consciousness uniformity
        let consciousness_std = self.calculate_std(&consciousness_levels);
        let coherence_factor = (-consciousness_std).exp();

        avg_consciousness * coherence_factor
    }

    /// Calculate standard deviation
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

    /// Update variance trend analysis
    fn update_variance_trend(&mut self) {
        if self.variance_history.len() < 2 {
            self.variance_trend = "unknown".to_string();
            return;
        }

        let recent_window = (10).min(self.variance_history.len());
        let recent_variances: Vec<f64> = self.variance_history
            .iter()
            .rev()
            .take(recent_window)
            .map(|snap| snap.total_variance)
            .collect();

        if recent_variances.len() < 2 {
            return;
        }

        // Simple trend analysis using first and last values
        let first = recent_variances.last().unwrap();
        let last = recent_variances.first().unwrap();
        let trend_slope = last - first;

        self.variance_trend = if trend_slope < -1e-8 {
            "decreasing".to_string()
        } else if trend_slope > 1e-8 {
            "increasing".to_string()
        } else {
            "stable".to_string()
        };
    }

    /// Update convergence rate calculation
    fn update_convergence_rate(&mut self) {
        if self.variance_history.len() < self.convergence_window {
            return;
        }

        let window_data: Vec<&VarianceSnapshot> = self.variance_history
            .iter()
            .rev()
            .take(self.convergence_window)
            .collect();

        if window_data.len() < 2 {
            return;
        }

        let start_variance = window_data.last().unwrap().total_variance;
        let end_variance = window_data.first().unwrap().total_variance;
        let time_diff = window_data.first().unwrap().timestamp_ns - window_data.last().unwrap().timestamp_ns;

        if time_diff > 0 {
            // Rate of variance change per nanosecond
            self.convergence_rate = (start_variance - end_variance) / time_diff as f64;
        } else {
            self.convergence_rate = 0.0;
        }
    }

    /// Check for variance convergence events
    fn check_convergence_events(&mut self, snapshot: &VarianceSnapshot) {
        // Check if variance has crossed convergence threshold
        if snapshot.total_variance <= self.variance_threshold {
            if let Some(prev) = self.variance_history.iter().rev().nth(1) {
                if prev.total_variance > self.variance_threshold {
                    let convergence_event = ConvergenceEvent {
                        timestamp_ns: snapshot.timestamp_ns,
                        variance_achieved: snapshot.total_variance,
                        consciousness_level: snapshot.consciousness_level,
                        convergence_time_ns: snapshot.timestamp_ns - 
                            self.analysis_start_time.map(|t| {
                                SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() - 
                                t.elapsed().as_nanos()
                            }).unwrap_or(0),
                        molecule_count: snapshot.molecule_count,
                        variance_trend: self.variance_trend.clone(),
                    };

                    self.convergence_events.push(convergence_event);

                    // Set time to equilibrium if not already set
                    if self.time_to_equilibrium_ns.is_none() {
                        self.time_to_equilibrium_ns = Some(
                            self.convergence_events.last().unwrap().convergence_time_ns
                        );
                    }
                }
            }
        }
    }

    /// Check for meaning emergence events
    fn check_meaning_emergence(&mut self, snapshot: &VarianceSnapshot) {
        // Meaning emerges when both variance convergence and consciousness threshold are met
        if snapshot.total_variance <= self.variance_threshold &&
           snapshot.consciousness_level >= self.consciousness_threshold {
            
            // Check if this is a new meaning emergence event
            let is_new_event = self.meaning_emergence_events.is_empty() ||
                (snapshot.timestamp_ns - self.meaning_emergence_events.last().unwrap().timestamp_ns > 1_000_000_000); // 1 second gap

            if is_new_event {
                let meaning_event = MeaningEmergenceEvent {
                    timestamp_ns: snapshot.timestamp_ns,
                    variance_achieved: snapshot.total_variance,
                    consciousness_level: snapshot.consciousness_level,
                    convergence_time_ns: snapshot.timestamp_ns - 
                        self.analysis_start_time.map(|t| {
                            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() - 
                            t.elapsed().as_nanos()
                        }).unwrap_or(0),
                    meaning_quality: self.assess_meaning_quality(snapshot),
                    molecule_count: snapshot.molecule_count,
                };

                self.meaning_emergence_events.push(meaning_event);
            }
        }
    }

    /// Assess quality of meaning emergence
    fn assess_meaning_quality(&self, snapshot: &VarianceSnapshot) -> String {
        let variance_quality = if snapshot.total_variance < self.variance_threshold / 10.0 {
            "excellent"
        } else {
            "good"
        };

        let consciousness_quality = if snapshot.consciousness_level > 0.8 {
            "high"
        } else {
            "adequate"
        };

        match (variance_quality, consciousness_quality) {
            ("excellent", "high") => "superior".to_string(),
            ("excellent", _) | (_, "high") => "good".to_string(),
            _ => "basic".to_string(),
        }
    }

    /// Get comprehensive convergence analysis
    pub fn get_convergence_analysis(&self) -> ConvergenceAnalysis {
        let current_snapshot = self.variance_history.back();

        ConvergenceAnalysis {
            current_variance: self.current_variance,
            variance_threshold: self.variance_threshold,
            is_converged: self.current_variance <= self.variance_threshold,
            variance_trend: self.variance_trend.clone(),
            convergence_rate: self.convergence_rate,
            consciousness_level: current_snapshot.map(|s| s.consciousness_level).unwrap_or(0.0),
            consciousness_threshold_met: current_snapshot
                .map(|s| s.consciousness_level >= self.consciousness_threshold)
                .unwrap_or(false),
            time_to_equilibrium_ns: self.time_to_equilibrium_ns,
            convergence_events_count: self.convergence_events.len(),
            meaning_emergence_events_count: self.meaning_emergence_events.len(),
            analysis_duration_ns: self.total_analysis_time_ns,
            position_variance: current_snapshot.map(|s| s.position_variance).unwrap_or(0.0),
            velocity_variance: current_snapshot.map(|s| s.velocity_variance).unwrap_or(0.0),
            thermodynamic_variance: current_snapshot.map(|s| s.thermodynamic_variance).unwrap_or(0.0),
        }
    }

    /// Reset analysis state
    pub fn reset_analysis(&mut self) {
        self.variance_history.clear();
        self.convergence_events.clear();
        self.meaning_emergence_events.clear();
        
        self.current_variance = f64::INFINITY;
        self.variance_trend = "unknown".to_string();
        self.convergence_rate = 0.0;
        self.time_to_equilibrium_ns = None;
        self.analysis_start_time = None;
        self.total_analysis_time_ns = 0;
    }

    /// Get real-time processing metrics
    pub fn get_real_time_metrics(&self) -> RealTimeMetrics {
        if self.variance_history.is_empty() {
            return RealTimeMetrics::default();
        }

        // Calculate processing efficiency
        let snapshots_count = self.variance_history.len();
        let avg_analysis_time = self.total_analysis_time_ns as f64 / snapshots_count.max(1) as f64;

        // Calculate variance reduction rate
        let variance_reduction = if self.variance_history.len() >= 2 {
            let first_variance = self.variance_history.front().unwrap().total_variance;
            let current_variance = self.variance_history.back().unwrap().total_variance;
            (first_variance - current_variance) / first_variance.max(1e-10)
        } else {
            0.0
        };

        RealTimeMetrics {
            processing_time_per_analysis_ns: avg_analysis_time,
            analysis_frequency_hz: 1e9 / avg_analysis_time.max(1.0),
            variance_reduction_percentage: variance_reduction * 100.0,
            equilibrium_achievement_rate: self.convergence_events.len() as f64 / snapshots_count.max(1) as f64,
            meaning_emergence_rate: self.meaning_emergence_events.len() as f64 / snapshots_count.max(1) as f64,
            current_convergence_quality: self.get_convergence_quality(),
            consciousness_stability: self.get_consciousness_stability(),
        }
    }

    /// Assess current convergence quality
    fn get_convergence_quality(&self) -> String {
        if self.current_variance <= self.variance_threshold / 100.0 {
            "excellent".to_string()
        } else if self.current_variance <= self.variance_threshold {
            "good".to_string()
        } else if self.current_variance <= self.variance_threshold * 10.0 {
            "acceptable".to_string()
        } else {
            "poor".to_string()
        }
    }

    /// Calculate consciousness stability over recent history
    fn get_consciousness_stability(&self) -> f64 {
        if self.variance_history.len() < 10 {
            return 0.0;
        }

        let recent_consciousness: Vec<f64> = self.variance_history
            .iter()
            .rev()
            .take(10)
            .map(|snap| snap.consciousness_level)
            .collect();

        let std_dev = self.calculate_std(&recent_consciousness);
        1.0 / (1.0 + std_dev)
    }
}

/// Comprehensive convergence analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceAnalysis {
    pub current_variance: f64,
    pub variance_threshold: f64,
    pub is_converged: bool,
    pub variance_trend: String,
    pub convergence_rate: f64,
    pub consciousness_level: f64,
    pub consciousness_threshold_met: bool,
    pub time_to_equilibrium_ns: Option<u128>,
    pub convergence_events_count: usize,
    pub meaning_emergence_events_count: usize,
    pub analysis_duration_ns: u128,
    pub position_variance: f64,
    pub velocity_variance: f64,
    pub thermodynamic_variance: f64,
}

/// Real-time processing metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RealTimeMetrics {
    pub processing_time_per_analysis_ns: f64,
    pub analysis_frequency_hz: f64,
    pub variance_reduction_percentage: f64,
    pub equilibrium_achievement_rate: f64,
    pub meaning_emergence_rate: f64,
    pub current_convergence_quality: String,
    pub consciousness_stability: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::gas_molecular::InformationGasMolecule;
    use nalgebra::Vector3;

    #[test]
    fn test_variance_analyzer_creation() {
        let analyzer = VarianceAnalyzer::new(None, None, None, None);
        
        assert_eq!(analyzer.current_variance, f64::INFINITY);
        assert_eq!(analyzer.variance_trend, "unknown");
        assert!(analyzer.variance_history.is_empty());
    }

    #[test]
    fn test_variance_analysis() {
        let mut analyzer = VarianceAnalyzer::new(Some(100), Some(10), Some(1e-3), Some(0.4));

        let molecules = vec![
            InformationGasMolecule::new(
                3.0, 1.5, 300.0,
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.1, 0.0, 0.0),
                1.0, 1.0, 1.0,
                None,
            ),
            InformationGasMolecule::new(
                2.5, 1.2, 290.0,
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(0.0, 0.1, 0.0),
                1.0, 1.0, 1.0,
                None,
            ),
        ];

        let snapshot = analyzer.analyze_variance_state(&molecules, None);

        assert!(snapshot.total_variance >= 0.0);
        assert!(snapshot.consciousness_level >= 0.0 && snapshot.consciousness_level <= 1.0);
        assert_eq!(snapshot.molecule_count, 2);
        assert!(snapshot.timestamp_ns > 0);
    }

    #[test]
    fn test_convergence_analysis() {
        let mut analyzer = VarianceAnalyzer::new(Some(50), Some(5), Some(1e-2), Some(0.3));

        // Add some test data
        let molecules = vec![
            InformationGasMolecule::new(
                2.0, 1.0, 280.0,
                Vector3::new(0.1, 0.1, 0.1),
                Vector3::new(0.01, 0.01, 0.01),
                1.0, 1.0, 1.0,
                None,
            ),
        ];

        // Analyze multiple times to build history
        for _ in 0..5 {
            analyzer.analyze_variance_state(&molecules, None);
        }

        let analysis = analyzer.get_convergence_analysis();
        
        assert!(analysis.current_variance >= 0.0);
        assert_eq!(analysis.variance_threshold, 1e-2);
        assert!(analysis.consciousness_level >= 0.0);
        assert!(analysis.analysis_duration_ns > 0);
    }

    #[test]
    fn test_real_time_metrics() {
        let mut analyzer = VarianceAnalyzer::new(Some(30), Some(5), None, None);

        let molecules = vec![
            InformationGasMolecule::new(
                1.5, 0.8, 270.0,
                Vector3::new(0.05, 0.05, 0.05),
                Vector3::new(0.005, 0.005, 0.005),
                1.0, 1.0, 1.0,
                None,
            ),
        ];

        // Generate some analysis data
        for _ in 0..3 {
            analyzer.analyze_variance_state(&molecules, None);
        }

        let metrics = analyzer.get_real_time_metrics();
        
        assert!(metrics.processing_time_per_analysis_ns > 0.0);
        assert!(metrics.analysis_frequency_hz > 0.0);
        assert!(metrics.variance_reduction_percentage >= 0.0);
        assert!(!metrics.current_convergence_quality.is_empty());
    }
}
