"""
Variance Analyzer Implementation

Tracks and analyzes variance minimization in gas molecular systems,
providing real-time monitoring of equilibrium convergence and meaning emergence.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque

from .information_gas_molecule import InformationGasMolecule


@dataclass
class VarianceSnapshot:
    """Snapshot of variance state at a point in time"""
    timestamp_ns: int
    total_variance: float
    position_variance: float
    velocity_variance: float
    thermodynamic_variance: float
    consciousness_level: float
    molecule_count: int


class VarianceAnalyzer:
    """
    Analyzer for tracking variance minimization in gas molecular systems.
    
    Provides real-time monitoring of equilibrium convergence and identifies
    when meaning emergence occurs through variance minimization patterns.
    """
    
    def __init__(self,
                 history_size: int = 1000,
                 convergence_window: int = 50,
                 variance_threshold: float = 1e-6,
                 consciousness_threshold: float = 0.61):
        """
        Initialize variance analyzer.
        
        Args:
            history_size: Number of variance snapshots to maintain
            convergence_window: Window size for convergence analysis
            variance_threshold: Threshold for equilibrium convergence
            consciousness_threshold: Minimum consciousness level
        """
        self.history_size = history_size
        self.convergence_window = convergence_window
        self.variance_threshold = variance_threshold
        self.consciousness_threshold = consciousness_threshold
        
        # Variance tracking
        self.variance_history: deque = deque(maxlen=history_size)
        self.convergence_events: List[Dict[str, Any]] = []
        self.meaning_emergence_events: List[Dict[str, Any]] = []
        
        # Real-time statistics
        self.current_variance: float = float('inf')
        self.variance_trend: str = "unknown"  # "decreasing", "increasing", "stable"
        self.convergence_rate: float = 0.0
        self.time_to_equilibrium_ns: Optional[int] = None
        
        # Performance tracking
        self.analysis_start_time: Optional[int] = None
        self.total_analysis_time_ns: int = 0
        
    def analyze_variance_state(self, 
                             gas_molecules: List[InformationGasMolecule],
                             baseline_equilibrium: Optional[np.ndarray] = None) -> VarianceSnapshot:
        """
        Analyze current variance state of gas molecular system.
        
        Args:
            gas_molecules: List of information gas molecules
            baseline_equilibrium: Optional baseline equilibrium for comparison
            
        Returns:
            VarianceSnapshot containing current variance analysis
        """
        start_time = time.time_ns()
        
        if self.analysis_start_time is None:
            self.analysis_start_time = start_time
            
        # Calculate variance components
        variance_components = self._calculate_variance_components(gas_molecules, baseline_equilibrium)
        
        # Calculate system consciousness
        consciousness_level = self._calculate_system_consciousness_level(gas_molecules)
        
        # Create variance snapshot
        snapshot = VarianceSnapshot(
            timestamp_ns=start_time,
            total_variance=variance_components['total'],
            position_variance=variance_components['position'],
            velocity_variance=variance_components['velocity'],
            thermodynamic_variance=variance_components['thermodynamic'],
            consciousness_level=consciousness_level,
            molecule_count=len(gas_molecules)
        )
        
        # Update tracking
        self.variance_history.append(snapshot)
        self.current_variance = snapshot.total_variance
        
        # Update analysis statistics
        self._update_variance_trend()
        self._update_convergence_rate()
        self._check_convergence_events(snapshot)
        self._check_meaning_emergence(snapshot)
        
        # Update performance tracking
        analysis_time = time.time_ns() - start_time
        self.total_analysis_time_ns += analysis_time
        
        return snapshot
        
    def _calculate_variance_components(self,
                                     gas_molecules: List[InformationGasMolecule],
                                     baseline_equilibrium: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate detailed variance components."""
        position_variance = 0.0
        velocity_variance = 0.0
        thermodynamic_variance = 0.0
        
        for i, molecule in enumerate(gas_molecules):
            # Position variance from equilibrium
            if baseline_equilibrium is not None and i * 3 + 2 < len(baseline_equilibrium):
                target_position = baseline_equilibrium[i*3:(i+1)*3]
            else:
                target_position = molecule.get_equilibrium_target()
                
            pos_diff = molecule.kinetic_state.semantic_position - target_position
            position_variance += np.sum(pos_diff ** 2)
            
            # Velocity variance (should be minimal at equilibrium)
            velocity_variance += np.sum(molecule.kinetic_state.info_velocity ** 2)
            
            # Thermodynamic variance (energy and entropy fluctuations)
            energy_variance = (molecule.thermodynamic_state.semantic_energy - 
                             self._get_equilibrium_energy()) ** 2
            entropy_variance = (molecule.thermodynamic_state.info_entropy - 
                              self._get_equilibrium_entropy()) ** 2
            temp_variance = (molecule.thermodynamic_state.processing_temperature - 
                           self._get_equilibrium_temperature()) ** 2
            
            thermodynamic_variance += energy_variance + entropy_variance + temp_variance
            
        total_variance = position_variance + velocity_variance + thermodynamic_variance
        
        return {
            'total': total_variance,
            'position': position_variance,
            'velocity': velocity_variance,
            'thermodynamic': thermodynamic_variance
        }
        
    def _get_equilibrium_energy(self) -> float:
        """Get expected equilibrium energy."""
        return 1.0  # Reference equilibrium energy
        
    def _get_equilibrium_entropy(self) -> float:
        """Get expected equilibrium entropy."""
        return 2.0  # Reference equilibrium entropy
        
    def _get_equilibrium_temperature(self) -> float:
        """Get expected equilibrium temperature."""
        return 300.0  # Reference equilibrium temperature
        
    def _calculate_system_consciousness_level(self, gas_molecules: List[InformationGasMolecule]) -> float:
        """Calculate overall system consciousness level."""
        if not gas_molecules:
            return 0.0
            
        consciousness_levels = [mol.consciousness_level for mol in gas_molecules]
        avg_consciousness = np.mean(consciousness_levels)
        
        # Coherence factor based on consciousness uniformity
        consciousness_std = np.std(consciousness_levels)
        coherence_factor = np.exp(-consciousness_std)
        
        return avg_consciousness * coherence_factor
        
    def _update_variance_trend(self) -> None:
        """Update variance trend analysis."""
        if len(self.variance_history) < 2:
            self.variance_trend = "unknown"
            return
            
        recent_window = min(10, len(self.variance_history))
        recent_variances = [snap.total_variance for snap in list(self.variance_history)[-recent_window:]]
        
        if len(recent_variances) < 2:
            return
            
        # Calculate trend using linear regression
        x = np.arange(len(recent_variances))
        y = np.array(recent_variances)
        
        if np.std(x) > 0:
            slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
            
            if slope < -1e-8:
                self.variance_trend = "decreasing"
            elif slope > 1e-8:
                self.variance_trend = "increasing" 
            else:
                self.variance_trend = "stable"
        else:
            self.variance_trend = "stable"
            
    def _update_convergence_rate(self) -> None:
        """Update convergence rate calculation."""
        if len(self.variance_history) < self.convergence_window:
            return
            
        # Calculate rate of variance change over convergence window
        window_data = list(self.variance_history)[-self.convergence_window:]
        
        if len(window_data) < 2:
            return
            
        start_variance = window_data[0].total_variance
        end_variance = window_data[-1].total_variance
        time_diff = window_data[-1].timestamp_ns - window_data[0].timestamp_ns
        
        if time_diff > 0:
            # Rate of variance change per nanosecond
            self.convergence_rate = (start_variance - end_variance) / time_diff
        else:
            self.convergence_rate = 0.0
            
    def _check_convergence_events(self, snapshot: VarianceSnapshot) -> None:
        """Check for variance convergence events."""
        # Check if variance has crossed convergence threshold
        if (snapshot.total_variance <= self.variance_threshold and
            len(self.variance_history) > 1 and
            self.variance_history[-2].total_variance > self.variance_threshold):
            
            convergence_event = {
                'timestamp_ns': snapshot.timestamp_ns,
                'variance_achieved': snapshot.total_variance,
                'consciousness_level': snapshot.consciousness_level,
                'convergence_time_ns': snapshot.timestamp_ns - (self.analysis_start_time or 0),
                'molecule_count': snapshot.molecule_count,
                'variance_trend': self.variance_trend
            }
            
            self.convergence_events.append(convergence_event)
            
            # Estimate time to equilibrium
            if self.time_to_equilibrium_ns is None:
                self.time_to_equilibrium_ns = convergence_event['convergence_time_ns']
                
    def _check_meaning_emergence(self, snapshot: VarianceSnapshot) -> None:
        """Check for meaning emergence events."""
        # Meaning emerges when both variance convergence and consciousness threshold are met
        if (snapshot.total_variance <= self.variance_threshold and
            snapshot.consciousness_level >= self.consciousness_threshold):
            
            # Check if this is a new meaning emergence event
            if not self.meaning_emergence_events or (
                snapshot.timestamp_ns - self.meaning_emergence_events[-1]['timestamp_ns'] > 1e9):  # 1 second gap
                
                meaning_event = {
                    'timestamp_ns': snapshot.timestamp_ns,
                    'variance_achieved': snapshot.total_variance,
                    'consciousness_level': snapshot.consciousness_level,
                    'convergence_time_ns': snapshot.timestamp_ns - (self.analysis_start_time or 0),
                    'meaning_quality': self._assess_meaning_quality(snapshot),
                    'molecule_count': snapshot.molecule_count
                }
                
                self.meaning_emergence_events.append(meaning_event)
                
    def _assess_meaning_quality(self, snapshot: VarianceSnapshot) -> str:
        """Assess quality of meaning emergence."""
        variance_quality = "excellent" if snapshot.total_variance < self.variance_threshold / 10 else "good"
        consciousness_quality = "high" if snapshot.consciousness_level > 0.8 else "adequate"
        
        if variance_quality == "excellent" and consciousness_quality == "high":
            return "superior"
        elif variance_quality == "excellent" or consciousness_quality == "high":
            return "good"
        else:
            return "basic"
            
    def get_convergence_analysis(self) -> Dict[str, Any]:
        """Get comprehensive convergence analysis."""
        if not self.variance_history:
            return {}
            
        current_snapshot = self.variance_history[-1]
        
        analysis = {
            'current_variance': self.current_variance,
            'variance_threshold': self.variance_threshold,
            'is_converged': self.current_variance <= self.variance_threshold,
            'variance_trend': self.variance_trend,
            'convergence_rate': self.convergence_rate,
            'consciousness_level': current_snapshot.consciousness_level,
            'consciousness_threshold_met': current_snapshot.consciousness_level >= self.consciousness_threshold,
            'time_to_equilibrium_ns': self.time_to_equilibrium_ns,
            'convergence_events_count': len(self.convergence_events),
            'meaning_emergence_events_count': len(self.meaning_emergence_events),
            'analysis_duration_ns': self.total_analysis_time_ns
        }
        
        # Add variance component breakdown
        if len(self.variance_history) > 0:
            latest = self.variance_history[-1]
            analysis.update({
                'position_variance': latest.position_variance,
                'velocity_variance': latest.velocity_variance,
                'thermodynamic_variance': latest.thermodynamic_variance,
                'position_variance_ratio': latest.position_variance / max(latest.total_variance, 1e-10),
                'velocity_variance_ratio': latest.velocity_variance / max(latest.total_variance, 1e-10),
                'thermodynamic_variance_ratio': latest.thermodynamic_variance / max(latest.total_variance, 1e-10)
            })
            
        return analysis
        
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time processing metrics."""
        if not self.variance_history:
            return {}
            
        # Calculate processing efficiency
        snapshots_count = len(self.variance_history)
        avg_analysis_time = self.total_analysis_time_ns / max(snapshots_count, 1)
        
        # Calculate variance reduction rate
        if len(self.variance_history) >= 2:
            first_variance = self.variance_history[0].total_variance
            current_variance = self.variance_history[-1].total_variance
            variance_reduction = (first_variance - current_variance) / max(first_variance, 1e-10)
        else:
            variance_reduction = 0.0
            
        return {
            'processing_time_per_analysis_ns': avg_analysis_time,
            'analysis_frequency_hz': 1e9 / max(avg_analysis_time, 1),
            'variance_reduction_percentage': variance_reduction * 100,
            'equilibrium_achievement_rate': len(self.convergence_events) / max(snapshots_count, 1),
            'meaning_emergence_rate': len(self.meaning_emergence_events) / max(snapshots_count, 1),
            'current_convergence_quality': self._get_convergence_quality(),
            'consciousness_stability': self._get_consciousness_stability()
        }
        
    def _get_convergence_quality(self) -> str:
        """Assess current convergence quality."""
        if self.current_variance <= self.variance_threshold / 100:
            return "excellent"
        elif self.current_variance <= self.variance_threshold:
            return "good"
        elif self.current_variance <= self.variance_threshold * 10:
            return "acceptable"
        else:
            return "poor"
            
    def _get_consciousness_stability(self) -> float:
        """Calculate consciousness stability over recent history."""
        if len(self.variance_history) < 10:
            return 0.0
            
        recent_consciousness = [snap.consciousness_level for snap in list(self.variance_history)[-10:]]
        return 1.0 / (1.0 + np.std(recent_consciousness))
        
    def reset_analysis(self) -> None:
        """Reset variance analysis state."""
        self.variance_history.clear()
        self.convergence_events.clear()
        self.meaning_emergence_events.clear()
        
        self.current_variance = float('inf')
        self.variance_trend = "unknown"
        self.convergence_rate = 0.0
        self.time_to_equilibrium_ns = None
        self.analysis_start_time = None
        self.total_analysis_time_ns = 0
        
    def export_analysis_data(self) -> Dict[str, Any]:
        """Export complete analysis data for external processing."""
        return {
            'variance_history': [
                {
                    'timestamp_ns': snap.timestamp_ns,
                    'total_variance': snap.total_variance,
                    'position_variance': snap.position_variance,
                    'velocity_variance': snap.velocity_variance,
                    'thermodynamic_variance': snap.thermodynamic_variance,
                    'consciousness_level': snap.consciousness_level,
                    'molecule_count': snap.molecule_count
                }
                for snap in self.variance_history
            ],
            'convergence_events': self.convergence_events,
            'meaning_emergence_events': self.meaning_emergence_events,
            'analysis_summary': self.get_convergence_analysis(),
            'real_time_metrics': self.get_real_time_metrics()
        }
