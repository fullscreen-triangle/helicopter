"""
Template-Based Reverse Helicopter

This implements the user's brilliant insight: instead of generating full expectations,
users annotate/draw on a template frame to define what to track, then apply this
template across video sequences for efficient deviation detection.
"""

import torch
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from PIL import Image, ImageDraw
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TemplateElement:
    """Represents an annotated element in the template"""
    id: str
    name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    template_patch: np.ndarray
    track_type: str  # 'object', 'pose_point', 'region', 'measurement'
    expected_properties: Dict[str, Any]
    tolerance: float = 0.1


@dataclass
class TrackingResult:
    """Result of tracking a template element in a frame"""
    element_id: str
    found: bool
    location: Optional[Tuple[int, int, int, int]]
    confidence: float
    deviation_score: float
    properties: Dict[str, Any]
    deviation_description: str


class TemplateAnnotator:
    """Interactive tool for users to annotate template frames"""
    
    def __init__(self):
        self.template_image = None
        self.elements = []
        self.current_drawing = False
        self.current_element = None
    
    def load_template(self, image_path: str) -> Image.Image:
        """Load template image for annotation"""
        self.template_image = Image.open(image_path)
        return self.template_image
    
    def add_element_annotation(
        self,
        name: str,
        bbox: Tuple[int, int, int, int],
        track_type: str = 'object',
        expected_properties: Optional[Dict[str, Any]] = None,
        tolerance: float = 0.1
    ) -> str:
        """Add an annotated element to track"""
        
        element_id = f"element_{len(self.elements) + 1}"
        
        # Extract template patch
        x1, y1, x2, y2 = bbox
        template_patch = np.array(self.template_image.crop(bbox))
        
        # Calculate center
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        element = TemplateElement(
            id=element_id,
            name=name,
            bbox=bbox,
            center=center,
            template_patch=template_patch,
            track_type=track_type,
            expected_properties=expected_properties or {},
            tolerance=tolerance
        )
        
        self.elements.append(element)
        logger.info(f"Added template element: {name} ({track_type})")
        
        return element_id
    
    def add_pose_points(
        self,
        pose_points: Dict[str, Tuple[int, int]],
        joint_connections: Optional[List[Tuple[str, str]]] = None
    ):
        """Add human pose points to track"""
        
        for joint_name, (x, y) in pose_points.items():
            # Create small bounding box around point
            bbox = (x-10, y-10, x+10, y+10)
            
            self.add_element_annotation(
                name=joint_name,
                bbox=bbox,
                track_type='pose_point',
                expected_properties={
                    'joint_type': joint_name,
                    'connections': joint_connections or []
                }
            )
    
    def add_measurement_line(
        self,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
        measurement_name: str,
        expected_length: Optional[float] = None
    ):
        """Add measurement line to track"""
        
        x1, y1 = start_point
        x2, y2 = end_point
        
        # Create bounding box encompassing the line
        bbox = (
            min(x1, x2) - 5,
            min(y1, y2) - 5,
            max(x1, x2) + 5,
            max(y1, y2) + 5
        )
        
        self.add_element_annotation(
            name=measurement_name,
            bbox=bbox,
            track_type='measurement',
            expected_properties={
                'start_point': start_point,
                'end_point': end_point,
                'expected_length': expected_length
            }
        )
    
    def save_template(self, template_path: str):
        """Save annotated template"""
        template_data = {
            'image_path': template_path,
            'elements': [
                {
                    'id': elem.id,
                    'name': elem.name,
                    'bbox': elem.bbox,
                    'center': elem.center,
                    'track_type': elem.track_type,
                    'expected_properties': elem.expected_properties,
                    'tolerance': elem.tolerance
                }
                for elem in self.elements
            ]
        }
        
        import json
        with open(template_path.replace('.jpg', '_template.json'), 'w') as f:
            json.dump(template_data, f, indent=2)
        
        logger.info(f"Saved template with {len(self.elements)} elements")


class TemplateTracker:
    """Tracks template elements across video frames"""
    
    def __init__(self, template_annotator: TemplateAnnotator):
        self.template = template_annotator
        self.tracking_methods = {
            'object': self._track_object,
            'pose_point': self._track_pose_point,
            'region': self._track_region,
            'measurement': self._track_measurement
        }
    
    def track_frame(self, frame: np.ndarray) -> List[TrackingResult]:
        """Track all template elements in a single frame"""
        
        results = []
        
        for element in self.template.elements:
            result = self.tracking_methods[element.track_type](element, frame)
            results.append(result)
        
        return results
    
    def track_video(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> List[List[TrackingResult]]:
        """Track template elements across entire video"""
        
        cap = cv2.VideoCapture(video_path)
        all_results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track elements in current frame
            frame_results = self.track_frame(frame)
            all_results.append(frame_results)
            
            # Optionally visualize results
            if output_path:
                annotated_frame = self._visualize_tracking(frame, frame_results)
                # Save or display annotated frame
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        cap.release()
        logger.info(f"Completed tracking on {frame_count} frames")
        
        return all_results
    
    def _track_object(self, element: TemplateElement, frame: np.ndarray) -> TrackingResult:
        """Track object using template matching"""
        
        # Convert template patch to grayscale for matching
        template_gray = cv2.cvtColor(element.template_patch, cv2.COLOR_RGB2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Template matching
        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # Check if match is good enough
        found = max_val > (1.0 - element.tolerance)
        
        if found:
            # Calculate new bounding box
            h, w = template_gray.shape
            x, y = max_loc
            new_bbox = (x, y, x + w, y + h)
            
            # Calculate deviation from expected position
            expected_center = element.center
            actual_center = (x + w//2, y + h//2)
            deviation = np.sqrt(
                (expected_center[0] - actual_center[0])**2 + 
                (expected_center[1] - actual_center[1])**2
            )
            
            deviation_score = min(deviation / 100.0, 1.0)  # Normalize
            
            return TrackingResult(
                element_id=element.id,
                found=True,
                location=new_bbox,
                confidence=max_val,
                deviation_score=deviation_score,
                properties={'displacement': deviation},
                deviation_description=f"{element.name} moved {deviation:.1f} pixels from expected position"
            )
        
        else:
            return TrackingResult(
                element_id=element.id,
                found=False,
                location=None,
                confidence=max_val,
                deviation_score=1.0,
                properties={},
                deviation_description=f"{element.name} not found in frame (confidence: {max_val:.3f})"
            )
    
    def _track_pose_point(self, element: TemplateElement, frame: np.ndarray) -> TrackingResult:
        """Track human pose points"""
        
        # For pose points, we might use a more sophisticated approach
        # like optical flow or pose estimation
        
        # Simplified: use template matching for now
        return self._track_object(element, frame)
    
    def _track_region(self, element: TemplateElement, frame: np.ndarray) -> TrackingResult:
        """Track region properties (color, texture, etc.)"""
        
        # Extract region from current frame at expected location
        x1, y1, x2, y2 = element.bbox
        current_region = frame[y1:y2, x1:x2]
        
        # Compare with template region
        template_region = element.template_patch
        
        # Calculate difference metrics
        if current_region.shape == template_region.shape:
            # Color difference
            color_diff = np.mean(np.abs(current_region.astype(float) - template_region.astype(float)))
            
            # Structural similarity
            from skimage.metrics import structural_similarity
            ssim = structural_similarity(
                cv2.cvtColor(current_region, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(template_region, cv2.COLOR_RGB2GRAY)
            )
            
            deviation_score = 1.0 - ssim
            
            return TrackingResult(
                element_id=element.id,
                found=True,
                location=element.bbox,
                confidence=ssim,
                deviation_score=deviation_score,
                properties={
                    'color_difference': color_diff,
                    'structural_similarity': ssim
                },
                deviation_description=f"{element.name} region changed (SSIM: {ssim:.3f})"
            )
        
        else:
            return TrackingResult(
                element_id=element.id,
                found=False,
                location=None,
                confidence=0.0,
                deviation_score=1.0,
                properties={},
                deviation_description=f"{element.name} region not accessible"
            )
    
    def _track_measurement(self, element: TemplateElement, frame: np.ndarray) -> TrackingResult:
        """Track measurement lines/distances"""
        
        expected_start = element.expected_properties['start_point']
        expected_end = element.expected_properties['end_point']
        expected_length = element.expected_properties.get('expected_length')
        
        # For measurements, we need to detect the actual line/distance
        # This is simplified - in practice, you'd use edge detection, etc.
        
        # Calculate expected length if not provided
        if expected_length is None:
            expected_length = np.sqrt(
                (expected_end[0] - expected_start[0])**2 + 
                (expected_end[1] - expected_start[1])**2
            )
        
        # For now, assume we can detect the actual measurement
        # In practice, this would involve computer vision techniques
        actual_length = expected_length  # Placeholder
        
        deviation_score = abs(actual_length - expected_length) / expected_length
        
        return TrackingResult(
            element_id=element.id,
            found=True,
            location=element.bbox,
            confidence=0.8,  # Placeholder
            deviation_score=deviation_score,
            properties={
                'expected_length': expected_length,
                'actual_length': actual_length
            },
            deviation_description=f"{element.name}: {actual_length:.1f} vs expected {expected_length:.1f}"
        )
    
    def _visualize_tracking(self, frame: np.ndarray, results: List[TrackingResult]) -> np.ndarray:
        """Visualize tracking results on frame"""
        
        annotated = frame.copy()
        
        for result in results:
            if result.found and result.location:
                x1, y1, x2, y2 = result.location
                
                # Color based on deviation score
                if result.deviation_score < 0.1:
                    color = (0, 255, 0)  # Green - good
                elif result.deviation_score < 0.3:
                    color = (0, 255, 255)  # Yellow - moderate
                else:
                    color = (0, 0, 255)  # Red - significant deviation
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Add text label
                label = f"{result.element_id}: {result.deviation_score:.2f}"
                cv2.putText(annotated, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return annotated


# Convenience functions for common use cases

def track_sports_technique(video_path: str, template_frame_path: str) -> Dict[str, Any]:
    """Track sports technique across video using template"""
    
    # Load template and set up common sports tracking points
    annotator = TemplateAnnotator()
    annotator.load_template(template_frame_path)
    
    # Add common pose points for sports analysis
    # User would normally annotate these interactively
    pose_points = {
        'head': (200, 100),
        'shoulder_left': (150, 200),
        'shoulder_right': (250, 200),
        'elbow_left': (120, 300),
        'elbow_right': (280, 300),
        'hip_center': (200, 400),
        'knee_left': (170, 500),
        'knee_right': (230, 500)
    }
    
    annotator.add_pose_points(pose_points)
    
    # Track across video
    tracker = TemplateTracker(annotator)
    results = tracker.track_video(video_path)
    
    # Analyze deviations
    analysis = analyze_tracking_results(results)
    
    return {
        'tracking_results': results,
        'analysis': analysis,
        'summary': generate_technique_summary(results)
    }


def analyze_tracking_results(results: List[List[TrackingResult]]) -> Dict[str, Any]:
    """Analyze tracking results to identify patterns and deviations"""
    
    analysis = {
        'total_frames': len(results),
        'element_analysis': {},
        'temporal_patterns': {},
        'deviation_summary': {}
    }
    
    # Analyze each element
    all_elements = set()
    for frame_results in results:
        for result in frame_results:
            all_elements.add(result.element_id)
    
    for element_id in all_elements:
        element_data = []
        for frame_results in results:
            element_result = next((r for r in frame_results if r.element_id == element_id), None)
            if element_result:
                element_data.append(element_result)
        
        # Calculate statistics
        deviation_scores = [r.deviation_score for r in element_data if r.found]
        confidences = [r.confidence for r in element_data if r.found]
        
        analysis['element_analysis'][element_id] = {
            'detection_rate': len([r for r in element_data if r.found]) / len(element_data),
            'avg_deviation': np.mean(deviation_scores) if deviation_scores else 1.0,
            'max_deviation': max(deviation_scores) if deviation_scores else 1.0,
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'stability': 1.0 - np.std(deviation_scores) if len(deviation_scores) > 1 else 0.0
        }
    
    return analysis


def generate_technique_summary(results: List[List[TrackingResult]]) -> str:
    """Generate human-readable summary of technique analysis"""
    
    analysis = analyze_tracking_results(results)
    
    summary_parts = []
    summary_parts.append(f"Analyzed {analysis['total_frames']} frames")
    
    # Identify most problematic elements
    problematic_elements = []
    stable_elements = []
    
    for element_id, stats in analysis['element_analysis'].items():
        if stats['avg_deviation'] > 0.3:
            problematic_elements.append((element_id, stats['avg_deviation']))
        elif stats['stability'] > 0.8:
            stable_elements.append((element_id, stats['stability']))
    
    if problematic_elements:
        problematic_elements.sort(key=lambda x: x[1], reverse=True)
        summary_parts.append("\nAreas needing improvement:")
        for element_id, deviation in problematic_elements[:3]:
            summary_parts.append(f"- {element_id}: high variation (deviation: {deviation:.2f})")
    
    if stable_elements:
        stable_elements.sort(key=lambda x: x[1], reverse=True)
        summary_parts.append("\nConsistent technique elements:")
        for element_id, stability in stable_elements[:3]:
            summary_parts.append(f"- {element_id}: stable form (stability: {stability:.2f})")
    
    return "\n".join(summary_parts) 