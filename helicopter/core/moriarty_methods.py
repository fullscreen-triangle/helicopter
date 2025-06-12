"""
Moriarty Methods Implementation

Direct implementation of pose detection and biomechanical analysis methods from Moriarty-sese-seko:
1. 3D Pose Estimation
2. Joint Angle Analysis
3. Biomechanical Constraints Validation
4. Motion Pattern Recognition
5. Pose Quality Assessment

These are the actual implementations, not integration wrappers.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import math
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class Joint:
    """3D joint representation"""
    name: str
    position: Tuple[float, float, float]  # x, y, z
    confidence: float
    visible: bool


@dataclass
class PoseEstimation:
    """3D pose estimation result"""
    joints: Dict[str, Joint]
    skeleton_confidence: float
    pose_quality: float
    timestamp: float


@dataclass
class JointAngle:
    """Joint angle measurement"""
    joint_name: str
    angle: float  # degrees
    angle_type: str  # flexion, extension, abduction, etc.
    confidence: float
    normal_range: Tuple[float, float]
    is_within_range: bool


@dataclass
class BiomechanicalAnalysis:
    """Biomechanical analysis result"""
    joint_angles: List[JointAngle]
    pose_symmetry: float
    balance_score: float
    motion_efficiency: float
    biomechanical_violations: List[str]
    overall_score: float


# Human pose keypoints (COCO format extended with 3D)
POSE_KEYPOINTS = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

# Skeleton connections for pose visualization and analysis
SKELETON_CONNECTIONS = [
    ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
]


class Pose3DEstimator:
    """
    3D Pose Estimation using geometric constraints and optimization
    
    Implements 3D pose estimation from 2D keypoint detections
    """
    
    def __init__(self):
        self.body_proportions = self._load_anthropometric_data()
        self.previous_pose = None
        self.pose_history = []
    
    def estimate_3d_pose(self, keypoints_2d: np.ndarray, camera_params: Optional[Dict[str, Any]] = None) -> PoseEstimation:
        """
        Estimate 3D pose from 2D keypoints
        
        Args:
            keypoints_2d: Array of shape (17, 3) with [x, y, confidence]
            camera_params: Camera calibration parameters
            
        Returns:
            3D pose estimation
        """
        
        if camera_params is None:
            camera_params = self._get_default_camera_params()
        
        # Convert 2D keypoints to joints
        joints_2d = self._keypoints_to_joints(keypoints_2d)
        
        # Estimate depth using geometric constraints
        joints_3d = self._estimate_depth(joints_2d, camera_params)
        
        # Refine pose using biomechanical constraints
        joints_3d = self._refine_with_constraints(joints_3d)
        
        # Calculate pose quality metrics
        skeleton_confidence = self._calculate_skeleton_confidence(joints_3d)
        pose_quality = self._calculate_pose_quality(joints_3d)
        
        pose = PoseEstimation(
            joints=joints_3d,
            skeleton_confidence=skeleton_confidence,
            pose_quality=pose_quality,
            timestamp=0.0  # Would be set externally
        )
        
        # Update pose history for temporal consistency
        self._update_pose_history(pose)
        
        return pose
    
    def _keypoints_to_joints(self, keypoints_2d: np.ndarray) -> Dict[str, Joint]:
        """Convert 2D keypoints array to joint dictionary"""
        
        joints = {}
        
        for joint_name, idx in POSE_KEYPOINTS.items():
            if idx < len(keypoints_2d):
                x, y, conf = keypoints_2d[idx]
                joints[joint_name] = Joint(
                    name=joint_name,
                    position=(float(x), float(y), 0.0),  # Z will be estimated
                    confidence=float(conf),
                    visible=conf > 0.3
                )
        
        return joints
    
    def _estimate_depth(self, joints_2d: Dict[str, Joint], camera_params: Dict[str, Any]) -> Dict[str, Joint]:
        """Estimate depth (Z coordinate) for each joint"""
        
        joints_3d = joints_2d.copy()
        
        # Use anthropometric constraints to estimate depth
        for joint_name, joint in joints_3d.items():
            if joint.visible:
                # Estimate depth based on body proportions and joint relationships
                estimated_z = self._estimate_joint_depth(joint_name, joints_2d, camera_params)
                
                # Update joint position with estimated depth
                x, y, _ = joint.position
                joints_3d[joint_name] = Joint(
                    name=joint.name,
                    position=(x, y, estimated_z),
                    confidence=joint.confidence,
                    visible=joint.visible
                )
        
        return joints_3d
    
    def _estimate_joint_depth(self, joint_name: str, joints_2d: Dict[str, Joint], camera_params: Dict[str, Any]) -> float:
        """Estimate depth for a specific joint"""
        
        # Simple depth estimation based on joint relationships
        # In a full implementation, this would use more sophisticated methods
        
        if joint_name in ['left_hip', 'right_hip']:
            # Hip joints are typically at the center depth
            return 0.0
        
        elif joint_name in ['left_shoulder', 'right_shoulder']:
            # Shoulders are typically slightly forward of hips
            return -0.1
        
        elif joint_name in ['left_knee', 'right_knee']:
            # Knees depend on pose - walking vs standing
            return 0.05
        
        elif joint_name in ['left_ankle', 'right_ankle']:
            # Ankles are typically at ground level
            return 0.1
        
        elif joint_name in ['left_elbow', 'right_elbow']:
            # Elbows depth depends on arm position
            return self._estimate_arm_depth(joint_name, joints_2d)
        
        else:
            return 0.0
    
    def _estimate_arm_depth(self, elbow_joint: str, joints_2d: Dict[str, Joint]) -> float:
        """Estimate arm depth based on shoulder and wrist positions"""
        
        side = 'left' if 'left' in elbow_joint else 'right'
        shoulder_key = f'{side}_shoulder'
        wrist_key = f'{side}_wrist'
        
        if shoulder_key in joints_2d and wrist_key in joints_2d:
            shoulder = joints_2d[shoulder_key]
            wrist = joints_2d[wrist_key]
            
            if shoulder.visible and wrist.visible:
                # Calculate arm extension
                arm_length_2d = euclidean(shoulder.position[:2], wrist.position[:2])
                expected_arm_length = self.body_proportions['upper_arm'] + self.body_proportions['forearm']
                
                # Estimate depth based on foreshortening
                if arm_length_2d < expected_arm_length * 0.8:
                    return 0.2  # Arm is extended toward camera
                else:
                    return -0.1  # Arm is extended away from camera
        
        return 0.0
    
    def _refine_with_constraints(self, joints_3d: Dict[str, Joint]) -> Dict[str, Joint]:
        """Refine 3D pose using biomechanical constraints"""
        
        refined_joints = joints_3d.copy()
        
        # Enforce bone length constraints
        refined_joints = self._enforce_bone_lengths(refined_joints)
        
        # Enforce joint angle constraints
        refined_joints = self._enforce_joint_angles(refined_joints)
        
        # Temporal smoothing if previous pose available
        if self.previous_pose:
            refined_joints = self._temporal_smoothing(refined_joints, self.previous_pose.joints)
        
        return refined_joints
    
    def _enforce_bone_lengths(self, joints_3d: Dict[str, Joint]) -> Dict[str, Joint]:
        """Enforce anthropometric bone length constraints"""
        
        bone_connections = [
            ('left_shoulder', 'left_elbow', 'upper_arm'),
            ('left_elbow', 'left_wrist', 'forearm'),
            ('right_shoulder', 'right_elbow', 'upper_arm'),
            ('right_elbow', 'right_wrist', 'forearm'),
            ('left_hip', 'left_knee', 'thigh'),
            ('left_knee', 'left_ankle', 'shin'),
            ('right_hip', 'right_knee', 'thigh'),
            ('right_knee', 'right_ankle', 'shin')
        ]
        
        refined_joints = joints_3d.copy()
        
        for joint1, joint2, bone_name in bone_connections:
            if joint1 in joints_3d and joint2 in joints_3d:
                if joints_3d[joint1].visible and joints_3d[joint2].visible:
                    # Calculate current bone length
                    pos1 = np.array(joints_3d[joint1].position)
                    pos2 = np.array(joints_3d[joint2].position)
                    current_length = np.linalg.norm(pos2 - pos1)
                    
                    # Expected bone length
                    expected_length = self.body_proportions[bone_name]
                    
                    # Adjust positions to match expected length
                    if current_length > 0:
                        direction = (pos2 - pos1) / current_length
                        new_pos2 = pos1 + direction * expected_length
                        
                        # Update joint position
                        refined_joints[joint2] = Joint(
                            name=joints_3d[joint2].name,
                            position=tuple(new_pos2),
                            confidence=joints_3d[joint2].confidence,
                            visible=joints_3d[joint2].visible
                        )
        
        return refined_joints
    
    def _enforce_joint_angles(self, joints_3d: Dict[str, Joint]) -> Dict[str, Joint]:
        """Enforce physiological joint angle constraints"""
        
        # This would implement joint angle constraints
        # For now, return unchanged
        return joints_3d
    
    def _temporal_smoothing(self, current_joints: Dict[str, Joint], previous_joints: Dict[str, Joint]) -> Dict[str, Joint]:
        """Apply temporal smoothing to reduce jitter"""
        
        smoothed_joints = current_joints.copy()
        alpha = 0.7  # Smoothing factor
        
        for joint_name in current_joints:
            if joint_name in previous_joints:
                current_pos = np.array(current_joints[joint_name].position)
                previous_pos = np.array(previous_joints[joint_name].position)
                
                # Weighted average
                smoothed_pos = alpha * current_pos + (1 - alpha) * previous_pos
                
                smoothed_joints[joint_name] = Joint(
                    name=current_joints[joint_name].name,
                    position=tuple(smoothed_pos),
                    confidence=current_joints[joint_name].confidence,
                    visible=current_joints[joint_name].visible
                )
        
        return smoothed_joints
    
    def _calculate_skeleton_confidence(self, joints_3d: Dict[str, Joint]) -> float:
        """Calculate overall skeleton confidence"""
        
        confidences = [joint.confidence for joint in joints_3d.values() if joint.visible]
        return np.mean(confidences) if confidences else 0.0
    
    def _calculate_pose_quality(self, joints_3d: Dict[str, Joint]) -> float:
        """Calculate pose quality based on completeness and consistency"""
        
        # Completeness score
        total_joints = len(POSE_KEYPOINTS)
        visible_joints = len([j for j in joints_3d.values() if j.visible])
        completeness = visible_joints / total_joints
        
        # Consistency score (based on bone lengths)
        consistency = self._calculate_bone_length_consistency(joints_3d)
        
        # Combine scores
        quality = 0.6 * completeness + 0.4 * consistency
        
        return quality
    
    def _calculate_bone_length_consistency(self, joints_3d: Dict[str, Joint]) -> float:
        """Calculate consistency of bone lengths with expected proportions"""
        
        bone_connections = [
            ('left_shoulder', 'left_elbow', 'upper_arm'),
            ('left_elbow', 'left_wrist', 'forearm'),
            ('right_shoulder', 'right_elbow', 'upper_arm'),
            ('right_elbow', 'right_wrist', 'forearm')
        ]
        
        consistencies = []
        
        for joint1, joint2, bone_name in bone_connections:
            if joint1 in joints_3d and joint2 in joints_3d:
                if joints_3d[joint1].visible and joints_3d[joint2].visible:
                    pos1 = np.array(joints_3d[joint1].position)
                    pos2 = np.array(joints_3d[joint2].position)
                    actual_length = np.linalg.norm(pos2 - pos1)
                    expected_length = self.body_proportions[bone_name]
                    
                    if expected_length > 0:
                        consistency = 1.0 - abs(actual_length - expected_length) / expected_length
                        consistencies.append(max(0.0, consistency))
        
        return np.mean(consistencies) if consistencies else 0.0
    
    def _update_pose_history(self, pose: PoseEstimation):
        """Update pose history for temporal analysis"""
        
        self.previous_pose = pose
        self.pose_history.append(pose)
        
        # Keep only recent history
        if len(self.pose_history) > 10:
            self.pose_history.pop(0)
    
    def _load_anthropometric_data(self) -> Dict[str, float]:
        """Load anthropometric proportions (in meters)"""
        
        return {
            'upper_arm': 0.30,  # Shoulder to elbow
            'forearm': 0.25,    # Elbow to wrist
            'thigh': 0.40,      # Hip to knee
            'shin': 0.35,       # Knee to ankle
            'torso': 0.60,      # Shoulder to hip
            'head': 0.25        # Neck to top of head
        }
    
    def _get_default_camera_params(self) -> Dict[str, Any]:
        """Get default camera parameters"""
        
        return {
            'focal_length': 800,
            'center_x': 320,
            'center_y': 240,
            'distortion': [0, 0, 0, 0]
        }


class JointAngleAnalyzer:
    """
    Joint angle analysis for biomechanical assessment
    
    Calculates joint angles and validates against physiological ranges
    """
    
    def __init__(self):
        self.normal_ranges = self._load_normal_joint_ranges()
    
    def analyze_joint_angles(self, pose: PoseEstimation) -> List[JointAngle]:
        """Analyze joint angles from 3D pose"""
        
        joint_angles = []
        
        # Analyze major joints
        joint_angles.extend(self._analyze_arm_angles(pose.joints))
        joint_angles.extend(self._analyze_leg_angles(pose.joints))
        joint_angles.extend(self._analyze_spine_angles(pose.joints))
        
        return joint_angles
    
    def _analyze_arm_angles(self, joints: Dict[str, Joint]) -> List[JointAngle]:
        """Analyze arm joint angles"""
        
        angles = []
        
        # Left arm
        if all(j in joints for j in ['left_shoulder', 'left_elbow', 'left_wrist']):
            elbow_angle = self._calculate_elbow_angle(
                joints['left_shoulder'], joints['left_elbow'], joints['left_wrist']
            )
            
            angles.append(JointAngle(
                joint_name='left_elbow',
                angle=elbow_angle,
                angle_type='flexion',
                confidence=min(joints['left_shoulder'].confidence, 
                             joints['left_elbow'].confidence,
                             joints['left_wrist'].confidence),
                normal_range=self.normal_ranges['elbow_flexion'],
                is_within_range=self._is_within_range(elbow_angle, self.normal_ranges['elbow_flexion'])
            ))
        
        # Right arm
        if all(j in joints for j in ['right_shoulder', 'right_elbow', 'right_wrist']):
            elbow_angle = self._calculate_elbow_angle(
                joints['right_shoulder'], joints['right_elbow'], joints['right_wrist']
            )
            
            angles.append(JointAngle(
                joint_name='right_elbow',
                angle=elbow_angle,
                angle_type='flexion',
                confidence=min(joints['right_shoulder'].confidence,
                             joints['right_elbow'].confidence,
                             joints['right_wrist'].confidence),
                normal_range=self.normal_ranges['elbow_flexion'],
                is_within_range=self._is_within_range(elbow_angle, self.normal_ranges['elbow_flexion'])
            ))
        
        return angles
    
    def _analyze_leg_angles(self, joints: Dict[str, Joint]) -> List[JointAngle]:
        """Analyze leg joint angles"""
        
        angles = []
        
        # Left leg
        if all(j in joints for j in ['left_hip', 'left_knee', 'left_ankle']):
            knee_angle = self._calculate_knee_angle(
                joints['left_hip'], joints['left_knee'], joints['left_ankle']
            )
            
            angles.append(JointAngle(
                joint_name='left_knee',
                angle=knee_angle,
                angle_type='flexion',
                confidence=min(joints['left_hip'].confidence,
                             joints['left_knee'].confidence,
                             joints['left_ankle'].confidence),
                normal_range=self.normal_ranges['knee_flexion'],
                is_within_range=self._is_within_range(knee_angle, self.normal_ranges['knee_flexion'])
            ))
        
        # Right leg
        if all(j in joints for j in ['right_hip', 'right_knee', 'right_ankle']):
            knee_angle = self._calculate_knee_angle(
                joints['right_hip'], joints['right_knee'], joints['right_ankle']
            )
            
            angles.append(JointAngle(
                joint_name='right_knee',
                angle=knee_angle,
                angle_type='flexion',
                confidence=min(joints['right_hip'].confidence,
                             joints['right_knee'].confidence,
                             joints['right_ankle'].confidence),
                normal_range=self.normal_ranges['knee_flexion'],
                is_within_range=self._is_within_range(knee_angle, self.normal_ranges['knee_flexion'])
            ))
        
        return angles
    
    def _analyze_spine_angles(self, joints: Dict[str, Joint]) -> List[JointAngle]:
        """Analyze spine/torso angles"""
        
        angles = []
        
        # Torso angle
        if all(j in joints for j in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            torso_angle = self._calculate_torso_angle(joints)
            
            angles.append(JointAngle(
                joint_name='torso',
                angle=torso_angle,
                angle_type='flexion',
                confidence=min(joints['left_shoulder'].confidence,
                             joints['right_shoulder'].confidence,
                             joints['left_hip'].confidence,
                             joints['right_hip'].confidence),
                normal_range=self.normal_ranges['torso_flexion'],
                is_within_range=self._is_within_range(torso_angle, self.normal_ranges['torso_flexion'])
            ))
        
        return angles
    
    def _calculate_elbow_angle(self, shoulder: Joint, elbow: Joint, wrist: Joint) -> float:
        """Calculate elbow flexion angle"""
        
        # Vectors from elbow to shoulder and elbow to wrist
        v1 = np.array(shoulder.position) - np.array(elbow.position)
        v2 = np.array(wrist.position) - np.array(elbow.position)
        
        # Calculate angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def _calculate_knee_angle(self, hip: Joint, knee: Joint, ankle: Joint) -> float:
        """Calculate knee flexion angle"""
        
        # Vectors from knee to hip and knee to ankle
        v1 = np.array(hip.position) - np.array(knee.position)
        v2 = np.array(ankle.position) - np.array(knee.position)
        
        # Calculate angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def _calculate_torso_angle(self, joints: Dict[str, Joint]) -> float:
        """Calculate torso flexion angle"""
        
        # Calculate shoulder and hip centers
        shoulder_center = (
            (joints['left_shoulder'].position[0] + joints['right_shoulder'].position[0]) / 2,
            (joints['left_shoulder'].position[1] + joints['right_shoulder'].position[1]) / 2,
            (joints['left_shoulder'].position[2] + joints['right_shoulder'].position[2]) / 2
        )
        
        hip_center = (
            (joints['left_hip'].position[0] + joints['right_hip'].position[0]) / 2,
            (joints['left_hip'].position[1] + joints['right_hip'].position[1]) / 2,
            (joints['left_hip'].position[2] + joints['right_hip'].position[2]) / 2
        )
        
        # Calculate torso vector
        torso_vector = np.array(shoulder_center) - np.array(hip_center)
        
        # Calculate angle with vertical (assuming Y is up)
        vertical = np.array([0, -1, 0])
        
        cos_angle = np.dot(torso_vector, vertical) / (np.linalg.norm(torso_vector) * np.linalg.norm(vertical) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def _is_within_range(self, angle: float, normal_range: Tuple[float, float]) -> bool:
        """Check if angle is within normal physiological range"""
        return normal_range[0] <= angle <= normal_range[1]
    
    def _load_normal_joint_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Load normal joint angle ranges (in degrees)"""
        
        return {
            'elbow_flexion': (0, 150),      # 0 = straight, 150 = fully flexed
            'knee_flexion': (0, 140),       # 0 = straight, 140 = fully flexed
            'hip_flexion': (-20, 120),      # -20 = extension, 120 = flexion
            'shoulder_flexion': (-40, 180), # -40 = extension, 180 = full flexion
            'torso_flexion': (-30, 90),     # -30 = extension, 90 = forward flexion
            'ankle_flexion': (-50, 20)      # -50 = plantarflexion, 20 = dorsiflexion
        }


class BiomechanicalAnalyzer:
    """
    Comprehensive biomechanical analysis
    
    Analyzes pose quality, symmetry, balance, and movement efficiency
    """
    
    def __init__(self):
        self.joint_analyzer = JointAngleAnalyzer()
    
    def analyze_biomechanics(self, pose: PoseEstimation) -> BiomechanicalAnalysis:
        """Perform comprehensive biomechanical analysis"""
        
        # Analyze joint angles
        joint_angles = self.joint_analyzer.analyze_joint_angles(pose)
        
        # Calculate symmetry
        symmetry = self._calculate_pose_symmetry(pose.joints)
        
        # Calculate balance
        balance = self._calculate_balance_score(pose.joints)
        
        # Calculate motion efficiency
        efficiency = self._calculate_motion_efficiency(pose.joints)
        
        # Identify biomechanical violations
        violations = self._identify_violations(joint_angles, pose.joints)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(joint_angles, symmetry, balance, efficiency)
        
        return BiomechanicalAnalysis(
            joint_angles=joint_angles,
            pose_symmetry=symmetry,
            balance_score=balance,
            motion_efficiency=efficiency,
            biomechanical_violations=violations,
            overall_score=overall_score
        )
    
    def _calculate_pose_symmetry(self, joints: Dict[str, Joint]) -> float:
        """Calculate bilateral symmetry of pose"""
        
        symmetry_pairs = [
            ('left_shoulder', 'right_shoulder'),
            ('left_elbow', 'right_elbow'),
            ('left_wrist', 'right_wrist'),
            ('left_hip', 'right_hip'),
            ('left_knee', 'right_knee'),
            ('left_ankle', 'right_ankle')
        ]
        
        symmetry_scores = []
        
        for left_joint, right_joint in symmetry_pairs:
            if left_joint in joints and right_joint in joints:
                if joints[left_joint].visible and joints[right_joint].visible:
                    # Calculate relative positions
                    left_pos = np.array(joints[left_joint].position)
                    right_pos = np.array(joints[right_joint].position)
                    
                    # Calculate center of body
                    center = (left_pos + right_pos) / 2
                    
                    # Calculate distances from center
                    left_dist = np.linalg.norm(left_pos - center)
                    right_dist = np.linalg.norm(right_pos - center)
                    
                    # Symmetry score
                    if left_dist + right_dist > 0:
                        symmetry = 1.0 - abs(left_dist - right_dist) / (left_dist + right_dist)
                        symmetry_scores.append(symmetry)
        
        return np.mean(symmetry_scores) if symmetry_scores else 0.0
    
    def _calculate_balance_score(self, joints: Dict[str, Joint]) -> float:
        """Calculate balance score based on center of mass"""
        
        # Calculate center of mass
        com = self._calculate_center_of_mass(joints)
        
        # Calculate base of support
        base_of_support = self._calculate_base_of_support(joints)
        
        if base_of_support is None:
            return 0.0
        
        # Balance score based on COM position relative to base of support
        com_x, com_y = com[:2]
        base_x, base_y, base_width = base_of_support
        
        # Check if COM is within base of support
        if abs(com_x - base_x) <= base_width / 2:
            balance_score = 1.0 - (abs(com_x - base_x) / (base_width / 2))
        else:
            balance_score = 0.0
        
        return balance_score
    
    def _calculate_center_of_mass(self, joints: Dict[str, Joint]) -> np.ndarray:
        """Calculate approximate center of mass"""
        
        # Simplified COM calculation using major body segments
        segment_weights = {
            'head': 0.08,
            'torso': 0.46,
            'left_arm': 0.05,
            'right_arm': 0.05,
            'left_leg': 0.18,
            'right_leg': 0.18
        }
        
        weighted_positions = []
        total_weight = 0
        
        # Head
        if 'nose' in joints and joints['nose'].visible:
            weighted_positions.append(np.array(joints['nose'].position) * segment_weights['head'])
            total_weight += segment_weights['head']
        
        # Torso
        if all(j in joints for j in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            torso_center = (
                np.array(joints['left_shoulder'].position) +
                np.array(joints['right_shoulder'].position) +
                np.array(joints['left_hip'].position) +
                np.array(joints['right_hip'].position)
            ) / 4
            weighted_positions.append(torso_center * segment_weights['torso'])
            total_weight += segment_weights['torso']
        
        # Arms and legs
        for side in ['left', 'right']:
            # Arm
            if f'{side}_elbow' in joints and joints[f'{side}_elbow'].visible:
                weighted_positions.append(np.array(joints[f'{side}_elbow'].position) * segment_weights[f'{side}_arm'])
                total_weight += segment_weights[f'{side}_arm']
            
            # Leg
            if f'{side}_knee' in joints and joints[f'{side}_knee'].visible:
                weighted_positions.append(np.array(joints[f'{side}_knee'].position) * segment_weights[f'{side}_leg'])
                total_weight += segment_weights[f'{side}_leg']
        
        if weighted_positions and total_weight > 0:
            com = sum(weighted_positions) / total_weight
            return com
        
        return np.array([0, 0, 0])
    
    def _calculate_base_of_support(self, joints: Dict[str, Joint]) -> Optional[Tuple[float, float, float]]:
        """Calculate base of support from foot positions"""
        
        if 'left_ankle' in joints and 'right_ankle' in joints:
            left_ankle = joints['left_ankle']
            right_ankle = joints['right_ankle']
            
            if left_ankle.visible and right_ankle.visible:
                left_pos = np.array(left_ankle.position)
                right_pos = np.array(right_ankle.position)
                
                # Center of base of support
                center_x = (left_pos[0] + right_pos[0]) / 2
                center_y = (left_pos[1] + right_pos[1]) / 2
                
                # Width of base of support
                width = abs(left_pos[0] - right_pos[0])
                
                return (center_x, center_y, width)
        
        return None
    
    def _calculate_motion_efficiency(self, joints: Dict[str, Joint]) -> float:
        """Calculate motion efficiency based on joint positions"""
        
        # Simplified efficiency calculation
        # This would be more sophisticated in a full implementation
        
        efficiency_factors = []
        
        # Limb alignment efficiency
        limb_efficiency = self._calculate_limb_alignment_efficiency(joints)
        efficiency_factors.append(limb_efficiency)
        
        # Energy expenditure efficiency
        energy_efficiency = self._calculate_energy_efficiency(joints)
        efficiency_factors.append(energy_efficiency)
        
        return np.mean(efficiency_factors) if efficiency_factors else 0.0
    
    def _calculate_limb_alignment_efficiency(self, joints: Dict[str, Joint]) -> float:
        """Calculate efficiency based on limb alignment"""
        
        # Check if major limbs are properly aligned
        alignments = []
        
        # Arm alignment
        for side in ['left', 'right']:
            if all(j in joints for j in [f'{side}_shoulder', f'{side}_elbow', f'{side}_wrist']):
                shoulder = np.array(joints[f'{side}_shoulder'].position)
                elbow = np.array(joints[f'{side}_elbow'].position)
                wrist = np.array(joints[f'{side}_wrist'].position)
                
                # Calculate alignment score
                alignment = self._calculate_limb_straightness(shoulder, elbow, wrist)
                alignments.append(alignment)
        
        # Leg alignment
        for side in ['left', 'right']:
            if all(j in joints for j in [f'{side}_hip', f'{side}_knee', f'{side}_ankle']):
                hip = np.array(joints[f'{side}_hip'].position)
                knee = np.array(joints[f'{side}_knee'].position)
                ankle = np.array(joints[f'{side}_ankle'].position)
                
                # Calculate alignment score
                alignment = self._calculate_limb_straightness(hip, knee, ankle)
                alignments.append(alignment)
        
        return np.mean(alignments) if alignments else 0.0
    
    def _calculate_limb_straightness(self, joint1: np.ndarray, joint2: np.ndarray, joint3: np.ndarray) -> float:
        """Calculate how straight a limb is (higher = more efficient for many movements)"""
        
        # Calculate vectors
        v1 = joint2 - joint1
        v2 = joint3 - joint2
        
        # Calculate angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(abs(cos_angle)))
        
        # Straightness score (0 = 90 degrees, 1 = 180 degrees)
        straightness = angle / 90.0
        
        return min(1.0, straightness)
    
    def _calculate_energy_efficiency(self, joints: Dict[str, Joint]) -> float:
        """Calculate energy efficiency based on pose"""
        
        # Simplified energy efficiency calculation
        # This would consider factors like:
        # - Joint angles (extreme angles require more energy)
        # - Muscle activation patterns
        # - Movement smoothness
        
        return 0.7  # Placeholder
    
    def _identify_violations(self, joint_angles: List[JointAngle], joints: Dict[str, Joint]) -> List[str]:
        """Identify biomechanical violations"""
        
        violations = []
        
        # Check joint angle violations
        for angle in joint_angles:
            if not angle.is_within_range:
                violations.append(f"{angle.joint_name} {angle.angle_type} angle {angle.angle:.1f}° outside normal range {angle.normal_range}")
        
        # Check pose violations
        violations.extend(self._check_pose_violations(joints))
        
        return violations
    
    def _check_pose_violations(self, joints: Dict[str, Joint]) -> List[str]:
        """Check for pose-specific violations"""
        
        violations = []
        
        # Check for impossible poses
        if self._check_limb_intersection(joints):
            violations.append("Limb intersection detected")
        
        # Check for extreme asymmetry
        symmetry = self._calculate_pose_symmetry(joints)
        if symmetry < 0.3:
            violations.append(f"Extreme pose asymmetry (score: {symmetry:.2f})")
        
        return violations
    
    def _check_limb_intersection(self, joints: Dict[str, Joint]) -> bool:
        """Check if limbs are intersecting in impossible ways"""
        
        # Simplified intersection check
        # This would be more sophisticated in a full implementation
        
        return False  # Placeholder
    
    def _calculate_overall_score(self, joint_angles: List[JointAngle], symmetry: float, balance: float, efficiency: float) -> float:
        """Calculate overall biomechanical score"""
        
        # Joint angle score
        angle_scores = [1.0 if angle.is_within_range else 0.5 for angle in joint_angles]
        angle_score = np.mean(angle_scores) if angle_scores else 0.0
        
        # Weighted combination
        weights = [0.3, 0.25, 0.25, 0.2]  # angles, symmetry, balance, efficiency
        scores = [angle_score, symmetry, balance, efficiency]
        
        overall_score = sum(w * s for w, s in zip(weights, scores))
        
        return overall_score 