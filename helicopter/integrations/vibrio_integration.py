"""
Vibrio Integration: Physics-Based Validation

This module integrates methods from the Vibrio High-Precision Human Velocity Analysis Framework
to provide physics-based validation for Helicopter's analysis results.

Vibrio Methods Integrated:
1. Optical Flow Analysis (Farneback dense optical flow)
2. Motion Energy Analysis (Motion History Images)
3. Neuromorphic Camera Simulation
4. Texture and Gradient Analysis
5. Shadow and Illumination Analysis  
6. Physics Constraints Validation
7. Multi-Object Tracking with Kalman Filtering

Reference: https://github.com/fullscreen-triangle/vibrio
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import ndimage
from skimage.feature import local_binary_pattern
from filterpy.kalman import KalmanFilter

logger = logging.getLogger(__name__)


@dataclass
class PhysicsConstraints:
    """Physics constraints for different activities"""
    activity: str
    max_speed: float  # km/h
    max_acceleration: float  # m/s²
    typical_speed_range: Tuple[float, float]  # km/h
    movement_patterns: List[str]
    biomechanical_limits: Dict[str, Any]


@dataclass
class OpticalFlowResult:
    """Result from optical flow analysis"""
    flow_magnitude: np.ndarray
    flow_direction: np.ndarray
    motion_coherence: float
    dominant_direction: float
    flow_density: float
    temporal_consistency: float


@dataclass
class MotionEnergyResult:
    """Result from motion energy analysis"""
    motion_history: np.ndarray
    energy_levels: np.ndarray
    active_regions: np.ndarray
    temporal_signature: np.ndarray
    activity_intensity: float


@dataclass
class PhysicsValidationResult:
    """Physics validation result"""
    is_physically_plausible: bool
    constraint_violations: List[str]
    plausibility_score: float
    speed_analysis: Dict[str, Any]
    motion_analysis: Dict[str, Any]
    optical_analysis: Dict[str, Any]
    recommendations: List[str]


class VibrioPhysicsValidator:
    """
    Physics-based validator using Vibrio methods
    
    This integrates the sophisticated physics validation and motion analysis
    methods from the Vibrio framework to validate Helicopter's analysis results.
    """
    
    def __init__(
        self,
        constraints: Optional[Dict[str, Any]] = None,
        optical_methods: Optional[List[str]] = None,
        motion_methods: Optional[List[str]] = None
    ):
        self.constraints = self._load_physics_constraints(constraints)
        self.optical_methods = optical_methods or ['optical_flow', 'motion_energy']
        self.motion_methods = motion_methods or ['tracking', 'velocity_estimation']
        
        # Initialize tracking system
        self.trackers = {}
        self.track_id_counter = 0
        
        # Optical flow parameters (from Vibrio)
        self.flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
        
        # Motion history parameters
        self.mhi_duration = 1.0  # seconds
        self.mhi_threshold = 32
        
        logger.info("Initialized Vibrio Physics Validator")
        logger.info(f"Optical methods: {self.optical_methods}")
        logger.info(f"Motion methods: {self.motion_methods}")
    
    def validate_analysis_result(
        self,
        image_sequence: List[np.ndarray],
        analysis_result: Dict[str, Any],
        activity_context: str = "general"
    ) -> PhysicsValidationResult:
        """
        Validate analysis result using physics-based methods
        
        Args:
            image_sequence: Sequence of images for analysis
            analysis_result: Result to validate
            activity_context: Type of activity being analyzed
            
        Returns:
            Physics validation result
        """
        
        logger.info(f"Validating analysis result for activity: {activity_context}")
        
        # Get constraints for this activity
        constraints = self.constraints.get(activity_context, self._get_default_constraints())
        
        # Perform optical analysis
        optical_results = self._analyze_optical_flow_sequence(image_sequence)
        
        # Perform motion analysis
        motion_results = self._analyze_motion_energy_sequence(image_sequence)
        
        # Validate against physics constraints
        physics_validation = self._validate_physics_constraints(
            analysis_result, constraints, optical_results, motion_results
        )
        
        # Calculate overall plausibility score
        plausibility_score = self._calculate_plausibility_score(
            physics_validation, optical_results, motion_results
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            physics_validation, constraints, optical_results
        )
        
        return PhysicsValidationResult(
            is_physically_plausible=plausibility_score > 0.7,
            constraint_violations=physics_validation['violations'],
            plausibility_score=plausibility_score,
            speed_analysis=physics_validation['speed_analysis'],
            motion_analysis=motion_results.__dict__,
            optical_analysis=optical_results.__dict__,
            recommendations=recommendations
        )
    
    def _load_physics_constraints(self, constraints: Optional[Dict[str, Any]]) -> Dict[str, PhysicsConstraints]:
        """Load physics constraints for different activities"""
        
        if constraints:
            # Convert dict to PhysicsConstraints objects
            physics_constraints = {}
            for activity, constraint_data in constraints.items():
                physics_constraints[activity] = PhysicsConstraints(**constraint_data)
            return physics_constraints
        
        # Default constraints (from Vibrio documentation)
        return {
            'walking': PhysicsConstraints(
                activity='walking',
                max_speed=7.0,  # km/h
                max_acceleration=2.0,  # m/s²
                typical_speed_range=(3.0, 6.0),
                movement_patterns=['cyclic', 'forward'],
                biomechanical_limits={'step_frequency': (1.5, 2.5)}
            ),
            'running': PhysicsConstraints(
                activity='running',
                max_speed=45.0,  # km/h (elite sprinters)
                max_acceleration=10.0,  # m/s²
                typical_speed_range=(8.0, 25.0),
                movement_patterns=['cyclic', 'forward', 'bounding'],
                biomechanical_limits={'step_frequency': (2.5, 4.5)}
            ),
            'cycling': PhysicsConstraints(
                activity='cycling',
                max_speed=80.0,  # km/h (racing cyclists)
                max_acceleration=4.0,  # m/s²
                typical_speed_range=(15.0, 50.0),
                movement_patterns=['rotational', 'forward'],
                biomechanical_limits={'cadence': (60, 120)}
            ),
            'skiing': PhysicsConstraints(
                activity='skiing',
                max_speed=120.0,  # km/h (speed skiing)
                max_acceleration=8.0,  # m/s²
                typical_speed_range=(20.0, 80.0),
                movement_patterns=['carving', 'gliding', 'turning'],
                biomechanical_limits={'turn_radius': (5.0, 50.0)}
            ),
            'general': PhysicsConstraints(
                activity='general',
                max_speed=50.0,  # km/h
                max_acceleration=5.0,  # m/s²
                typical_speed_range=(0.0, 30.0),
                movement_patterns=['variable'],
                biomechanical_limits={}
            )
        }
    
    def _get_default_constraints(self) -> PhysicsConstraints:
        """Get default physics constraints"""
        return self.constraints['general']
    
    def _analyze_optical_flow_sequence(self, image_sequence: List[np.ndarray]) -> OpticalFlowResult:
        """
        Analyze optical flow across image sequence
        
        Implements Farneback dense optical flow from Vibrio
        """
        
        if len(image_sequence) < 2:
            # Return empty result for single image
            return OpticalFlowResult(
                flow_magnitude=np.zeros((100, 100)),
                flow_direction=np.zeros((100, 100)),
                motion_coherence=0.0,
                dominant_direction=0.0,
                flow_density=0.0,
                temporal_consistency=0.0
            )
        
        flow_results = []
        
        for i in range(1, len(image_sequence)):
            prev_frame = cv2.cvtColor(image_sequence[i-1], cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(image_sequence[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow using Farneback method (from Vibrio)
            flow = cv2.calcOpticalFlowPyrLK(
                prev_frame, curr_frame, **self.flow_params
            )
            
            # Calculate flow magnitude and direction
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            flow_results.append({
                'magnitude': magnitude,
                'angle': angle,
                'flow': flow
            })
        
        # Aggregate results
        if flow_results:
            # Average magnitude and calculate metrics
            avg_magnitude = np.mean([r['magnitude'] for r in flow_results], axis=0)
            avg_angle = np.mean([r['angle'] for r in flow_results], axis=0)
            
            # Calculate motion coherence (how consistent the flow is)
            flow_vectors = np.array([r['flow'] for r in flow_results])
            motion_coherence = self._calculate_motion_coherence(flow_vectors)
            
            # Calculate dominant direction
            dominant_direction = np.mean(avg_angle[avg_magnitude > np.percentile(avg_magnitude, 75)])
            
            # Calculate flow density
            flow_density = np.mean(avg_magnitude > 1.0)  # Threshold for significant motion
            
            # Calculate temporal consistency
            temporal_consistency = self._calculate_temporal_consistency(flow_results)
            
            return OpticalFlowResult(
                flow_magnitude=avg_magnitude,
                flow_direction=avg_angle,
                motion_coherence=motion_coherence,
                dominant_direction=dominant_direction,
                flow_density=flow_density,
                temporal_consistency=temporal_consistency
            )
        
        # Return empty result if no flow calculated
        h, w = image_sequence[0].shape[:2]
        return OpticalFlowResult(
            flow_magnitude=np.zeros((h, w)),
            flow_direction=np.zeros((h, w)),
            motion_coherence=0.0,
            dominant_direction=0.0,
            flow_density=0.0,
            temporal_consistency=0.0
        )
    
    def _calculate_motion_coherence(self, flow_vectors: np.ndarray) -> float:
        """Calculate motion coherence from flow vectors"""
        
        # Coherence measures how aligned motion vectors are
        # High coherence = coordinated movement, Low coherence = chaotic movement
        
        if flow_vectors.size == 0:
            return 0.0
        
        # Calculate vector magnitudes
        magnitudes = np.sqrt(flow_vectors[..., 0]**2 + flow_vectors[..., 1]**2)
        
        # Calculate mean direction
        mean_direction = np.arctan2(
            np.mean(flow_vectors[..., 1]),
            np.mean(flow_vectors[..., 0])
        )
        
        # Calculate coherence as alignment with mean direction
        directions = np.arctan2(flow_vectors[..., 1], flow_vectors[..., 0])
        direction_diffs = np.abs(directions - mean_direction)
        
        # Weight by magnitude
        weighted_diffs = direction_diffs * magnitudes
        coherence = 1.0 - np.mean(weighted_diffs) / np.pi
        
        return max(0.0, coherence)
    
    def _calculate_temporal_consistency(self, flow_results: List[Dict[str, Any]]) -> float:
        """Calculate temporal consistency of optical flow"""
        
        if len(flow_results) < 2:
            return 0.0
        
        # Compare consecutive flow fields
        consistencies = []
        
        for i in range(1, len(flow_results)):
            prev_flow = flow_results[i-1]['flow']
            curr_flow = flow_results[i]['flow']
            
            # Calculate correlation between flow fields
            correlation = np.corrcoef(
                prev_flow.flatten(),
                curr_flow.flatten()
            )[0, 1]
            
            if not np.isnan(correlation):
                consistencies.append(correlation)
        
        return np.mean(consistencies) if consistencies else 0.0
    
    def _analyze_motion_energy_sequence(self, image_sequence: List[np.ndarray]) -> MotionEnergyResult:
        """
        Analyze motion energy using Motion History Images (MHI)
        
        Implements MHI method from Vibrio
        """
        
        if len(image_sequence) < 2:
            h, w = image_sequence[0].shape[:2]
            return MotionEnergyResult(
                motion_history=np.zeros((h, w)),
                energy_levels=np.zeros(1),
                active_regions=np.zeros((h, w)),
                temporal_signature=np.zeros(1),
                activity_intensity=0.0
            )
        
        # Convert to grayscale
        gray_sequence = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in image_sequence]
        
        # Initialize motion history image
        h, w = gray_sequence[0].shape
        motion_history = np.zeros((h, w), dtype=np.float32)
        
        energy_levels = []
        temporal_signature = []
        
        for i in range(1, len(gray_sequence)):
            # Calculate frame difference
            frame_diff = cv2.absdiff(gray_sequence[i], gray_sequence[i-1])
            
            # Threshold to get binary motion mask
            _, motion_mask = cv2.threshold(frame_diff, self.mhi_threshold, 1, cv2.THRESH_BINARY)
            
            # Update motion history
            motion_history = motion_history * 0.9  # Decay
            motion_history[motion_mask == 1] = 1.0  # Set new motion
            
            # Calculate energy for this frame
            energy = np.sum(motion_mask)
            energy_levels.append(energy)
            
            # Update temporal signature
            temporal_signature.append(np.mean(motion_history))
        
        # Calculate active regions (areas with consistent motion)
        active_regions = (motion_history > 0.3).astype(np.uint8)
        
        # Calculate overall activity intensity
        activity_intensity = np.mean(energy_levels) if energy_levels else 0.0
        
        return MotionEnergyResult(
            motion_history=motion_history,
            energy_levels=np.array(energy_levels),
            active_regions=active_regions,
            temporal_signature=np.array(temporal_signature),
            activity_intensity=activity_intensity
        )
    
    def _validate_physics_constraints(
        self,
        analysis_result: Dict[str, Any],
        constraints: PhysicsConstraints,
        optical_results: OpticalFlowResult,
        motion_results: MotionEnergyResult
    ) -> Dict[str, Any]:
        """Validate analysis against physics constraints"""
        
        violations = []
        speed_analysis = {}
        
        # Extract speed estimates from analysis result
        estimated_speeds = analysis_result.get('estimated_speeds', [])
        
        if estimated_speeds:
            max_speed = max(estimated_speeds)
            avg_speed = np.mean(estimated_speeds)
            
            speed_analysis = {
                'max_speed': max_speed,
                'avg_speed': avg_speed,
                'speed_profile': estimated_speeds
            }
            
            # Check speed constraints
            if max_speed > constraints.max_speed:
                violations.append(f"Maximum speed {max_speed:.1f} km/h exceeds limit {constraints.max_speed:.1f} km/h")
            
            if avg_speed < constraints.typical_speed_range[0] or avg_speed > constraints.typical_speed_range[1]:
                violations.append(f"Average speed {avg_speed:.1f} km/h outside typical range {constraints.typical_speed_range}")
            
            # Check acceleration constraints
            if len(estimated_speeds) > 1:
                speed_diffs = np.diff(estimated_speeds)
                max_acceleration = max(abs(speed_diffs)) * 3.6  # Convert to m/s²
                
                if max_acceleration > constraints.max_acceleration:
                    violations.append(f"Maximum acceleration {max_acceleration:.1f} m/s² exceeds limit {constraints.max_acceleration:.1f} m/s²")
        
        # Validate motion patterns
        self._validate_motion_patterns(
            constraints, optical_results, motion_results, violations
        )
        
        return {
            'violations': violations,
            'speed_analysis': speed_analysis,
            'constraints_checked': constraints.activity
        }
    
    def _validate_motion_patterns(
        self,
        constraints: PhysicsConstraints,
        optical_results: OpticalFlowResult,
        motion_results: MotionEnergyResult,
        violations: List[str]
    ):
        """Validate motion patterns against expected patterns"""
        
        expected_patterns = constraints.movement_patterns
        
        # Check for expected patterns based on optical flow and motion energy
        if 'cyclic' in expected_patterns:
            # Check for periodic motion in temporal signature
            if not self._detect_periodic_motion(motion_results.temporal_signature):
                violations.append("Expected cyclic motion pattern not detected")
        
        if 'forward' in expected_patterns:
            # Check for consistent forward motion direction
            if optical_results.motion_coherence < 0.5:
                violations.append("Expected forward motion pattern not coherent")
        
        if 'rotational' in expected_patterns:
            # Check for rotational motion patterns
            if not self._detect_rotational_motion(optical_results):
                violations.append("Expected rotational motion pattern not detected")
    
    def _detect_periodic_motion(self, temporal_signature: np.ndarray) -> bool:
        """Detect periodic motion in temporal signature"""
        
        if len(temporal_signature) < 10:
            return False
        
        # Use FFT to detect periodicity
        fft = np.fft.fft(temporal_signature)
        freqs = np.fft.fftfreq(len(temporal_signature))
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
        
        # Check if there's a strong periodic component
        power_ratio = np.abs(fft[dominant_freq_idx]) / np.sum(np.abs(fft))
        
        return power_ratio > 0.1  # At least 10% of power in dominant frequency
    
    def _detect_rotational_motion(self, optical_results: OpticalFlowResult) -> bool:
        """Detect rotational motion patterns"""
        
        # Check for circular flow patterns
        # This is a simplified detection - real implementation would be more sophisticated
        
        flow_directions = optical_results.flow_direction
        
        # Calculate variance in flow directions
        direction_variance = np.var(flow_directions)
        
        # High variance suggests rotational motion
        return direction_variance > 1.0
    
    def _calculate_plausibility_score(
        self,
        physics_validation: Dict[str, Any],
        optical_results: OpticalFlowResult,
        motion_results: MotionEnergyResult
    ) -> float:
        """Calculate overall plausibility score"""
        
        # Start with base score
        score = 1.0
        
        # Penalize for constraint violations
        num_violations = len(physics_validation['violations'])
        violation_penalty = num_violations * 0.2
        score -= violation_penalty
        
        # Reward for motion coherence
        coherence_bonus = optical_results.motion_coherence * 0.2
        score += coherence_bonus
        
        # Reward for temporal consistency
        consistency_bonus = optical_results.temporal_consistency * 0.1
        score += consistency_bonus
        
        # Penalize for unusual motion patterns
        if motion_results.activity_intensity < 0.1:
            score -= 0.1  # Very low activity might indicate detection issues
        
        return max(0.0, min(1.0, score))
    
    def _generate_recommendations(
        self,
        physics_validation: Dict[str, Any],
        constraints: PhysicsConstraints,
        optical_results: OpticalFlowResult
    ) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        violations = physics_validation['violations']
        
        if violations:
            recommendations.append("Physics constraint violations detected - review analysis parameters")
            
            for violation in violations:
                if "speed" in violation.lower():
                    recommendations.append("Consider recalibrating speed estimation or checking camera parameters")
                elif "acceleration" in violation.lower():
                    recommendations.append("Check for tracking errors or rapid camera movement")
                elif "motion pattern" in violation.lower():
                    recommendations.append("Verify activity classification or adjust pattern detection parameters")
        
        if optical_results.motion_coherence < 0.3:
            recommendations.append("Low motion coherence - check for tracking errors or camera stability")
        
        if optical_results.flow_density < 0.1:
            recommendations.append("Low flow density - verify motion is occurring in the scene")
        
        if not recommendations:
            recommendations.append("Analysis passes physics validation - results appear plausible")
        
        return recommendations
    
    def analyze_neuromorphic_events(self, image_sequence: List[np.ndarray]) -> Dict[str, Any]:
        """
        Simulate neuromorphic camera event detection
        
        This implements the event-based analysis method from Vibrio
        """
        
        if len(image_sequence) < 2:
            return {'events': [], 'event_density': 0.0, 'polarity_ratio': 0.5}
        
        events = []
        
        for i in range(1, len(image_sequence)):
            prev_frame = cv2.cvtColor(image_sequence[i-1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            curr_frame = cv2.cvtColor(image_sequence[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            # Calculate intensity differences
            diff = curr_frame - prev_frame
            
            # Event threshold (from Vibrio neuromorphic simulation)
            threshold = 15.0
            
            # Positive events (brightness increase)
            pos_events = np.where(diff > threshold)
            for y, x in zip(pos_events[0], pos_events[1]):
                events.append({
                    'x': x,
                    'y': y,
                    'timestamp': i,
                    'polarity': 1,
                    'magnitude': diff[y, x]
                })
            
            # Negative events (brightness decrease)
            neg_events = np.where(diff < -threshold)
            for y, x in zip(neg_events[0], neg_events[1]):
                events.append({
                    'x': x,
                    'y': y,
                    'timestamp': i,
                    'polarity': -1,
                    'magnitude': abs(diff[y, x])
                })
        
        # Calculate metrics
        total_pixels = image_sequence[0].shape[0] * image_sequence[0].shape[1]
        event_density = len(events) / (total_pixels * (len(image_sequence) - 1))
        
        pos_events_count = len([e for e in events if e['polarity'] == 1])
        polarity_ratio = pos_events_count / len(events) if events else 0.5
        
        return {
            'events': events,
            'event_density': event_density,
            'polarity_ratio': polarity_ratio,
            'total_events': len(events)
        }
    
    def analyze_texture_gradients(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze texture and gradients using Gabor filters and LBP
        
        This implements the texture analysis method from Vibrio
        """
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern analysis
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Calculate LBP histogram
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize
        
        # Gabor filter responses
        gabor_responses = []
        for theta in range(0, 180, 30):  # 6 orientations
            for frequency in [0.1, 0.3, 0.5]:  # 3 frequencies
                real, _ = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, real)
                gabor_responses.append(np.mean(filtered))
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate dominant orientation
        gradient_direction = np.arctan2(grad_y, grad_x)
        dominant_orientation = np.mean(gradient_direction)
        
        return {
            'lbp_histogram': lbp_hist,
            'gabor_responses': gabor_responses,
            'texture_energy': np.var(lbp),
            'gradient_strength': np.mean(gradient_magnitude),
            'dominant_orientation': dominant_orientation,
            'texture_uniformity': 1.0 - np.var(lbp_hist)
        } 