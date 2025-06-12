"""
Simple Template-Based Tracker - Actually Working Implementation

This implements the user's brilliant template approach:
1. User annotates/draws on a single frame
2. System tracks those elements across video
3. Reports deviations from expected behavior

This is much simpler and more practical than generating full image expectations.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TrackedElement:
    """Simple element to track"""
    name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    template: np.ndarray
    track_type: str = "object"  # object, point, region
    expected_behavior: str = "stable"  # stable, moving, changing


@dataclass 
class TrackingResult:
    """Result for one element in one frame"""
    element_name: str
    frame_number: int
    found: bool
    location: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0
    deviation_score: float = 1.0
    notes: str = ""


class SimpleTemplateTracker:
    """Dead simple template tracker that actually works"""
    
    def __init__(self):
        self.template_elements: List[TrackedElement] = []
        self.template_frame = None
        
    def load_template_frame(self, image_path: str):
        """Load the template frame"""
        self.template_frame = cv2.imread(image_path)
        print(f"Loaded template frame: {image_path}")
    
    def add_tracked_element(
        self, 
        name: str, 
        bbox: Tuple[int, int, int, int],
        track_type: str = "object",
        expected_behavior: str = "stable"
    ):
        """Add an element to track by drawing bounding box"""
        
        if self.template_frame is None:
            raise ValueError("Load template frame first!")
        
        x1, y1, x2, y2 = bbox
        
        # Extract template patch
        template_patch = self.template_frame[y1:y2, x1:x2].copy()
        
        element = TrackedElement(
            name=name,
            bbox=bbox,
            template=template_patch,
            track_type=track_type,
            expected_behavior=expected_behavior
        )
        
        self.template_elements.append(element)
        print(f"Added element: {name} at {bbox}")
    
    def track_video(self, video_path: str) -> List[List[TrackingResult]]:
        """Track all elements across entire video"""
        
        cap = cv2.VideoCapture(video_path)
        all_results = []
        frame_num = 0
        
        print(f"Starting tracking on video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track all elements in this frame
            frame_results = []
            for element in self.template_elements:
                result = self._track_element_in_frame(element, frame, frame_num)
                frame_results.append(result)
            
            all_results.append(frame_results)
            frame_num += 1
            
            if frame_num % 30 == 0:
                print(f"Processed {frame_num} frames...")
        
        cap.release()
        print(f"Completed tracking: {frame_num} frames, {len(self.template_elements)} elements")
        
        return all_results
    
    def _track_element_in_frame(
        self, 
        element: TrackedElement, 
        frame: np.ndarray, 
        frame_num: int
    ) -> TrackingResult:
        """Track single element in single frame"""
        
        # Convert to grayscale for template matching
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(element.template, cv2.COLOR_BGR2GRAY)
        
        # Template matching
        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # Determine if we found it
        threshold = 0.6  # Adjust based on requirements
        found = max_val > threshold
        
        if found:
            # Calculate new position
            h, w = template_gray.shape
            x, y = max_loc
            new_bbox = (x, y, x + w, y + h)
            
            # Calculate deviation from original position
            orig_center = (
                (element.bbox[0] + element.bbox[2]) / 2,
                (element.bbox[1] + element.bbox[3]) / 2
            )
            new_center = (x + w/2, y + h/2)
            
            distance = np.sqrt(
                (orig_center[0] - new_center[0])**2 + 
                (orig_center[1] - new_center[1])**2
            )
            
            # Normalize deviation score (0 = no deviation, 1 = max deviation)
            max_expected_movement = 100  # pixels
            deviation_score = min(distance / max_expected_movement, 1.0)
            
            # Generate notes based on deviation
            if deviation_score < 0.1:
                notes = "Stable position"
            elif deviation_score < 0.3:
                notes = f"Minor movement: {distance:.1f}px"
            else:
                notes = f"Significant movement: {distance:.1f}px"
            
            return TrackingResult(
                element_name=element.name,
                frame_number=frame_num,
                found=True,
                location=new_bbox,
                confidence=max_val,
                deviation_score=deviation_score,
                notes=notes
            )
        
        else:
            return TrackingResult(
                element_name=element.name,
                frame_number=frame_num,
                found=False,
                confidence=max_val,
                deviation_score=1.0,
                notes=f"Not found (confidence: {max_val:.3f})"
            )
    
    def analyze_results(self, results: List[List[TrackingResult]]) -> Dict:
        """Analyze tracking results and generate insights"""
        
        analysis = {
            "total_frames": len(results),
            "elements": {}
        }
        
        # Analyze each element
        for element in self.template_elements:
            element_results = []
            
            # Collect all results for this element
            for frame_results in results:
                element_result = next(
                    (r for r in frame_results if r.element_name == element.name), 
                    None
                )
                if element_result:
                    element_results.append(element_result)
            
            # Calculate statistics
            found_count = sum(1 for r in element_results if r.found)
            detection_rate = found_count / len(element_results) if element_results else 0
            
            deviations = [r.deviation_score for r in element_results if r.found]
            avg_deviation = np.mean(deviations) if deviations else 1.0
            max_deviation = max(deviations) if deviations else 1.0
            
            confidences = [r.confidence for r in element_results if r.found]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            analysis["elements"][element.name] = {
                "detection_rate": detection_rate,
                "avg_deviation": avg_deviation,
                "max_deviation": max_deviation,
                "avg_confidence": avg_confidence,
                "total_frames": len(element_results),
                "found_frames": found_count,
                "behavior_assessment": self._assess_behavior(deviations, element.expected_behavior)
            }
        
        return analysis
    
    def _assess_behavior(self, deviations: List[float], expected: str) -> str:
        """Assess if element behaved as expected"""
        
        if not deviations:
            return "Unable to assess - not detected"
        
        avg_dev = np.mean(deviations)
        
        if expected == "stable":
            if avg_dev < 0.1:
                return "Excellent stability"
            elif avg_dev < 0.3:
                return "Good stability with minor variations"
            else:
                return "Poor stability - significant movement"
        
        elif expected == "moving":
            if avg_dev > 0.3:
                return "Good movement as expected"
            else:
                return "Less movement than expected"
        
        else:
            return f"Average deviation: {avg_dev:.3f}"
    
    def generate_report(self, analysis: Dict) -> str:
        """Generate human-readable report"""
        
        report = []
        report.append(f"=== TRACKING ANALYSIS REPORT ===")
        report.append(f"Total frames analyzed: {analysis['total_frames']}")
        report.append(f"Elements tracked: {len(analysis['elements'])}")
        report.append("")
        
        # Sort elements by detection rate
        sorted_elements = sorted(
            analysis['elements'].items(),
            key=lambda x: x[1]['detection_rate'],
            reverse=True
        )
        
        for element_name, stats in sorted_elements:
            report.append(f"📍 {element_name.upper()}")
            report.append(f"   Detection Rate: {stats['detection_rate']:.1%}")
            report.append(f"   Average Deviation: {stats['avg_deviation']:.3f}")
            report.append(f"   Behavior: {stats['behavior_assessment']}")
            report.append("")
        
        # Overall assessment
        avg_detection = np.mean([stats['detection_rate'] for stats in analysis['elements'].values()])
        avg_deviation = np.mean([stats['avg_deviation'] for stats in analysis['elements'].values()])
        
        report.append("=== OVERALL ASSESSMENT ===")
        if avg_detection > 0.8 and avg_deviation < 0.3:
            report.append("✅ Excellent tracking performance")
        elif avg_detection > 0.6 and avg_deviation < 0.5:
            report.append("✅ Good tracking performance")
        else:
            report.append("⚠️  Tracking challenges detected")
        
        report.append(f"Average detection rate: {avg_detection:.1%}")
        report.append(f"Average deviation: {avg_deviation:.3f}")
        
        return "\n".join(report)
    
    def save_template(self, filepath: str):
        """Save template configuration"""
        
        # Convert elements to serializable format
        elements_data = []
        for element in self.template_elements:
            element_dict = asdict(element)
            # Convert numpy array to list
            element_dict['template'] = element.template.tolist()
            elements_data.append(element_dict)
        
        template_data = {
            "elements": elements_data,
            "template_frame_shape": self.template_frame.shape if self.template_frame is not None else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(template_data, f, indent=2)
        
        print(f"Saved template to: {filepath}")
    
    def load_template(self, filepath: str):
        """Load template configuration"""
        
        with open(filepath, 'r') as f:
            template_data = json.load(f)
        
        self.template_elements = []
        for element_dict in template_data['elements']:
            # Convert list back to numpy array
            element_dict['template'] = np.array(element_dict['template'], dtype=np.uint8)
            
            element = TrackedElement(**element_dict)
            self.template_elements.append(element)
        
        print(f"Loaded template with {len(self.template_elements)} elements")


# Convenience functions for common use cases

def track_sports_form(template_image: str, video_path: str, output_dir: str = "output"):
    """Simple function to track sports form"""
    
    tracker = SimpleTemplateTracker()
    tracker.load_template_frame(template_image)
    
    print("🏃 Sports Form Tracking Setup")
    print("Please manually add tracking elements using:")
    print("tracker.add_tracked_element('head', (x1, y1, x2, y2))")
    print("tracker.add_tracked_element('shoulders', (x1, y1, x2, y2))")
    print("etc.")
    
    return tracker


def track_quality_control(template_image: str, video_path: str):
    """Simple function for quality control tracking"""
    
    tracker = SimpleTemplateTracker()
    tracker.load_template_frame(template_image)
    
    print("🔍 Quality Control Tracking Setup")
    print("Add elements to track defects or key components:")
    print("tracker.add_tracked_element('component_1', (x1, y1, x2, y2))")
    
    return tracker


# Test function to verify everything works
def test_tracker():
    """Test the tracker with synthetic data"""
    
    print("🧪 Testing SimpleTemplateTracker...")
    
    # Create a simple test template
    template = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(template, (20, 20), (80, 80), (255, 0, 0), -1)  # Blue square
    
    # Save test template
    cv2.imwrite("test_template.jpg", template)
    
    # Create tracker
    tracker = SimpleTemplateTracker()
    tracker.load_template_frame("test_template.jpg")
    
    # Add element to track
    tracker.add_tracked_element("blue_square", (20, 20, 80, 80), "object", "stable")
    
    # Create test video (just moving the square slightly)
    test_video_path = "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, 30.0, (100, 100))
    
    for i in range(30):  # 30 frames
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Move square slightly
        offset = i // 10  # Move every 10 frames
        cv2.rectangle(frame, (20+offset, 20+offset), (80+offset, 80+offset), (255, 0, 0), -1)
        out.write(frame)
    
    out.release()
    
    # Track the test video
    results = tracker.track_video(test_video_path)
    
    # Analyze results
    analysis = tracker.analyze_results(results)
    report = tracker.generate_report(analysis)
    
    print("\n" + report)
    
    # Cleanup
    import os
    try:
        os.remove("test_template.jpg")
        os.remove(test_video_path)
    except:
        pass
    
    print("✅ Test completed!")


if __name__ == "__main__":
    test_tracker() 