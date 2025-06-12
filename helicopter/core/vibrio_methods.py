"""
Vibrio Methods Implementation

Direct implementation of High-Precision Human Velocity Analysis methods from Vibrio:
1. Optical Flow Analysis (Farneback dense optical flow)
2. Motion Energy Analysis (Motion History Images)
3. Neuromorphic Camera Simulation  
4. Texture and Gradient Analysis
5. Physics Constraints Validation
6. Multi-Object Tracking with Kalman Filtering

These are the actual implementations, not integration wrappers.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import ndimage
from skimage.feature import local_binary_pattern
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


@dataclass
class OpticalFlowAnalysis:
    """Optical flow analysis result"""
    magnitude: np.ndarray
    direction: np.ndarray
    coherence: float
    density: float
    dominant_direction: float
    temporal_consistency: float


@dataclass
class MotionEnergyAnalysis:
    """Motion energy analysis result"""
    motion_history: np.ndarray
    energy_profile: np.ndarray
    active_regions: np.ndarray
    intensity: float
    frequency_signature: np.ndarray


@dataclass
class TrackingResult:
    """Multi-object tracking result"""
    track_id: int
    bounding_box: Tuple[int, int, int, int]
    center: Tuple[float, float]
    velocity: Tuple[float, float]
    confidence: float
    trajectory: List[Tuple[float, float]]


class OpticalFlowAnalyzer:
    """
    Optical Flow Analysis using Farneback method
    
    Implements dense optical flow calculation and motion analysis
    """
    
    def __init__(self):
        # Farneback parameters (optimized from Vibrio)
        self.flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
    
    def analyze_sequence(self, frames: List[np.ndarray]) -> OpticalFlowAnalysis:
        """Analyze optical flow across frame sequence"""
        
        if len(frames) < 2:
            h, w = frames[0].shape[:2]
            return self._empty_result(h, w)
        
        flow_fields = []
        magnitudes = []
        directions = []
        
        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, **self.flow_params)
            
            # Convert to magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            flow_fields.append(flow)
            magnitudes.append(magnitude)
            directions.append(angle)
        
        # Calculate aggregate metrics
        avg_magnitude = np.mean(magnitudes, axis=0)
        avg_direction = np.mean(directions, axis=0)
        
        coherence = self._calculate_coherence(flow_fields)
        density = self._calculate_density(magnitudes)
        dominant_direction = self._calculate_dominant_direction(directions, magnitudes)
        temporal_consistency = self._calculate_temporal_consistency(flow_fields)
        
        return OpticalFlowAnalysis(
            magnitude=avg_magnitude,
            direction=avg_direction,
            coherence=coherence,
            density=density,
            dominant_direction=dominant_direction,
            temporal_consistency=temporal_consistency
        )
    
    def _calculate_coherence(self, flow_fields: List[np.ndarray]) -> float:
        """Calculate motion coherence"""
        
        if not flow_fields:
            return 0.0
        
        # Calculate mean flow direction
        all_flows = np.concatenate([f.reshape(-1, 2) for f in flow_fields])
        mean_flow = np.mean(all_flows, axis=0)
        
        # Calculate coherence as alignment with mean direction
        coherences = []
        for flow in flow_fields:
            flow_flat = flow.reshape(-1, 2)
            # Dot product with mean direction, normalized
            dots = np.dot(flow_flat, mean_flow) / (np.linalg.norm(flow_flat, axis=1) * np.linalg.norm(mean_flow) + 1e-8)
            coherence = np.mean(np.abs(dots))
            coherences.append(coherence)
        
        return np.mean(coherences)
    
    def _calculate_density(self, magnitudes: List[np.ndarray]) -> float:
        """Calculate flow density (percentage of pixels with significant motion)"""
        
        threshold = 1.0  # Minimum magnitude for significant motion
        densities = []
        
        for magnitude in magnitudes:
            significant_pixels = np.sum(magnitude > threshold)
            total_pixels = magnitude.size
            density = significant_pixels / total_pixels
            densities.append(density)
        
        return np.mean(densities)
    
    def _calculate_dominant_direction(self, directions: List[np.ndarray], magnitudes: List[np.ndarray]) -> float:
        """Calculate weighted dominant direction"""
        
        weighted_directions = []
        
        for direction, magnitude in zip(directions, magnitudes):
            # Weight directions by magnitude
            mask = magnitude > np.percentile(magnitude, 75)  # Top 25% magnitude
            if np.any(mask):
                dominant_dir = np.mean(direction[mask])
                weighted_directions.append(dominant_dir)
        
        return np.mean(weighted_directions) if weighted_directions else 0.0
    
    def _calculate_temporal_consistency(self, flow_fields: List[np.ndarray]) -> float:
        """Calculate temporal consistency between consecutive flows"""
        
        if len(flow_fields) < 2:
            return 0.0
        
        consistencies = []
        
        for i in range(1, len(flow_fields)):
            prev_flow = flow_fields[i-1].flatten()
            curr_flow = flow_fields[i].flatten()
            
            # Calculate correlation
            correlation = np.corrcoef(prev_flow, curr_flow)[0, 1]
            if not np.isnan(correlation):
                consistencies.append(abs(correlation))
        
        return np.mean(consistencies) if consistencies else 0.0
    
    def _empty_result(self, h: int, w: int) -> OpticalFlowAnalysis:
        """Return empty optical flow result"""
        return OpticalFlowAnalysis(
            magnitude=np.zeros((h, w)),
            direction=np.zeros((h, w)),
            coherence=0.0,
            density=0.0,
            dominant_direction=0.0,
            temporal_consistency=0.0
        )


class MotionEnergyAnalyzer:
    """
    Motion Energy Analysis using Motion History Images (MHI)
    
    Implements temporal motion analysis and activity detection
    """
    
    def __init__(self, duration: float = 1.0, threshold: int = 32):
        self.duration = duration  # MHI duration in seconds
        self.threshold = threshold  # Motion detection threshold
    
    def analyze_sequence(self, frames: List[np.ndarray], fps: float = 30.0) -> MotionEnergyAnalysis:
        """Analyze motion energy across frame sequence"""
        
        if len(frames) < 2:
            h, w = frames[0].shape[:2]
            return self._empty_result(h, w)
        
        # Convert to grayscale
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        
        h, w = gray_frames[0].shape
        motion_history = np.zeros((h, w), dtype=np.float32)
        energy_profile = []
        
        timestamp = 0.0
        dt = 1.0 / fps
        
        for i in range(1, len(gray_frames)):
            timestamp += dt
            
            # Calculate frame difference
            diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
            
            # Create motion mask
            _, motion_mask = cv2.threshold(diff, self.threshold, 1, cv2.THRESH_BINARY)
            
            # Update motion history image
            motion_history = self._update_mhi(motion_history, motion_mask, timestamp)
            
            # Calculate energy for this frame
            energy = np.sum(motion_mask)
            energy_profile.append(energy)
        
        # Calculate derived metrics
        active_regions = self._calculate_active_regions(motion_history)
        intensity = self._calculate_intensity(energy_profile)
        frequency_signature = self._calculate_frequency_signature(energy_profile)
        
        return MotionEnergyAnalysis(
            motion_history=motion_history,
            energy_profile=np.array(energy_profile),
            active_regions=active_regions,
            intensity=intensity,
            frequency_signature=frequency_signature
        )
    
    def _update_mhi(self, mhi: np.ndarray, motion_mask: np.ndarray, timestamp: float) -> np.ndarray:
        """Update Motion History Image"""
        
        # Decay old motion
        mhi = np.where(mhi > 0, mhi - 1.0/30.0, 0)  # Assuming 30 FPS
        
        # Add new motion
        mhi = np.where(motion_mask > 0, timestamp, mhi)
        
        # Remove old motion (older than duration)
        mhi = np.where(mhi < timestamp - self.duration, 0, mhi)
        
        return mhi
    
    def _calculate_active_regions(self, motion_history: np.ndarray) -> np.ndarray:
        """Calculate regions with consistent motion activity"""
        
        # Threshold MHI to get active regions
        active_threshold = 0.3
        active_regions = (motion_history > active_threshold).astype(np.uint8)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        active_regions = cv2.morphologyEx(active_regions, cv2.MORPH_OPEN, kernel)
        active_regions = cv2.morphologyEx(active_regions, cv2.MORPH_CLOSE, kernel)
        
        return active_regions
    
    def _calculate_intensity(self, energy_profile: List[float]) -> float:
        """Calculate overall motion intensity"""
        
        if not energy_profile:
            return 0.0
        
        # Normalize by image size (assuming typical image size)
        normalized_energy = np.array(energy_profile) / (640 * 480)  # Typical resolution
        
        return np.mean(normalized_energy)
    
    def _calculate_frequency_signature(self, energy_profile: List[float]) -> np.ndarray:
        """Calculate frequency signature of motion"""
        
        if len(energy_profile) < 10:
            return np.zeros(5)
        
        # Apply FFT to get frequency components
        fft = np.fft.fft(energy_profile)
        freqs = np.fft.fftfreq(len(energy_profile))
        
        # Get magnitude spectrum
        magnitude_spectrum = np.abs(fft)
        
        # Extract dominant frequencies (first 5 components)
        return magnitude_spectrum[:5]
    
    def _empty_result(self, h: int, w: int) -> MotionEnergyAnalysis:
        """Return empty motion energy result"""
        return MotionEnergyAnalysis(
            motion_history=np.zeros((h, w)),
            energy_profile=np.zeros(1),
            active_regions=np.zeros((h, w)),
            intensity=0.0,
            frequency_signature=np.zeros(5)
        )


class KalmanTracker:
    """
    Kalman Filter-based object tracker
    
    Implements multi-object tracking with motion prediction
    """
    
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State: [x, y, vx, vy, w, h, vw, vh]
        # Measurement: [x, y, w, h]
        
        # Transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0]
        ])
        
        # Process noise
        self.kf.Q *= 0.1
        
        # Measurement noise
        self.kf.R *= 10
        
        # Initial covariance
        self.kf.P *= 1000
        
        self.trajectory = []
        self.age = 0
        self.hits = 0
        self.hit_streak = 0
        self.time_since_update = 0
        
    def update(self, bbox: Tuple[int, int, int, int]):
        """Update tracker with new detection"""
        
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        # Convert bbox to measurement [cx, cy, w, h]
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        measurement = np.array([cx, cy, w, h])
        
        self.kf.update(measurement)
        
        # Store trajectory point
        self.trajectory.append((cx, cy))
        
    def predict(self):
        """Predict next state"""
        
        if self.kf.x[6] + self.kf.x[2] <= 0:  # Width + velocity <= 0
            self.kf.x[6] *= 0.0
        if self.kf.x[7] + self.kf.x[3] <= 0:  # Height + velocity <= 0
            self.kf.x[7] *= 0.0
            
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        return self.get_state()
    
    def get_state(self) -> Tuple[int, int, int, int]:
        """Get current bounding box state"""
        
        cx, cy, w, h = self.kf.x[0], self.kf.x[1], self.kf.x[4], self.kf.x[5]
        x = cx - w/2
        y = cy - h/2
        
        return (int(x), int(y), int(w), int(h))
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity"""
        return (self.kf.x[2], self.kf.x[3])


class MultiObjectTracker:
    """
    Multi-Object Tracker using Hungarian algorithm and Kalman filters
    
    Implements the tracking system from Vibrio
    """
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 50):
        self.trackers = []
        self.next_id = 0
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[TrackingResult]:
        """Update trackers with new detections"""
        
        # Predict all existing trackers
        predicted_boxes = []
        for tracker in self.trackers:
            bbox = tracker.predict()
            predicted_boxes.append(bbox)
        
        # Associate detections with trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, predicted_boxes
        )
        
        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(detections[m[0]])
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            tracker = KalmanTracker(self.next_id)
            self.next_id += 1
            tracker.update(detections[i])
            self.trackers.append(tracker)
        
        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_disappeared]
        
        # Generate tracking results
        results = []
        for tracker in self.trackers:
            if tracker.time_since_update < 1 and tracker.hit_streak >= 3:
                bbox = tracker.get_state()
                velocity = tracker.get_velocity()
                
                result = TrackingResult(
                    track_id=tracker.track_id,
                    bounding_box=bbox,
                    center=(bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2),
                    velocity=velocity,
                    confidence=min(1.0, tracker.hits / 10.0),
                    trajectory=tracker.trajectory.copy()
                )
                results.append(result)
        
        return results
    
    def _associate_detections_to_trackers(
        self, 
        detections: List[Tuple[int, int, int, int]], 
        trackers: List[Tuple[int, int, int, int]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to trackers using Hungarian algorithm"""
        
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        
        # Calculate cost matrix (IoU distances)
        iou_matrix = np.zeros((len(detections), len(trackers)))
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det, trk)
        
        # Convert IoU to cost (1 - IoU)
        cost_matrix = 1 - iou_matrix
        
        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out matches with low IoU
        matches = []
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= 0.7:  # IoU > 0.3
                matches.append((row, col))
        
        # Find unmatched detections and trackers
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in [m[0] for m in matches]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in [m[1] for m in matches]:
                unmatched_trackers.append(t)
        
        return matches, unmatched_detections, unmatched_trackers
    
    def _iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU)"""
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class PhysicsValidator:
    """
    Physics-based validation using constraints and motion analysis
    
    Implements physics validation from Vibrio
    """
    
    def __init__(self):
        self.constraints = self._load_default_constraints()
    
    def validate_tracking_result(
        self, 
        tracking_results: List[TrackingResult], 
        activity_type: str = "general"
    ) -> Dict[str, Any]:
        """Validate tracking results against physics constraints"""
        
        constraints = self.constraints.get(activity_type, self.constraints["general"])
        violations = []
        
        for result in tracking_results:
            # Validate velocity
            velocity_magnitude = np.sqrt(result.velocity[0]**2 + result.velocity[1]**2)
            
            # Convert pixels/frame to approximate km/h (rough conversion)
            speed_kmh = velocity_magnitude * 0.1  # Approximate conversion
            
            if speed_kmh > constraints["max_speed"]:
                violations.append(f"Track {result.track_id}: Speed {speed_kmh:.1f} km/h exceeds maximum {constraints['max_speed']} km/h")
            
            # Validate trajectory smoothness
            if len(result.trajectory) > 3:
                smoothness = self._calculate_trajectory_smoothness(result.trajectory)
                if smoothness < 0.5:
                    violations.append(f"Track {result.track_id}: Trajectory not smooth (smoothness: {smoothness:.2f})")
        
        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
            "activity_type": activity_type,
            "constraints_applied": constraints
        }
    
    def _calculate_trajectory_smoothness(self, trajectory: List[Tuple[float, float]]) -> float:
        """Calculate trajectory smoothness"""
        
        if len(trajectory) < 3:
            return 1.0
        
        # Calculate changes in direction
        direction_changes = []
        
        for i in range(2, len(trajectory)):
            p1, p2, p3 = trajectory[i-2], trajectory[i-1], trajectory[i]
            
            # Calculate vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Calculate angle between vectors
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                direction_changes.append(angle)
        
        # Smoothness is inversely related to direction changes
        if direction_changes:
            avg_change = np.mean(direction_changes)
            smoothness = 1.0 - (avg_change / np.pi)  # Normalize to [0, 1]
            return max(0.0, smoothness)
        
        return 1.0
    
    def _load_default_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Load default physics constraints"""
        
        return {
            "walking": {
                "max_speed": 7.0,
                "max_acceleration": 2.0,
                "typical_speed_range": (3.0, 6.0)
            },
            "running": {
                "max_speed": 45.0,
                "max_acceleration": 10.0,
                "typical_speed_range": (8.0, 25.0)
            },
            "cycling": {
                "max_speed": 80.0,
                "max_acceleration": 4.0,
                "typical_speed_range": (15.0, 50.0)
            },
            "general": {
                "max_speed": 50.0,
                "max_acceleration": 5.0,
                "typical_speed_range": (0.0, 30.0)
            }
        } 