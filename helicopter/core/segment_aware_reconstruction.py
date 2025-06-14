"""
Segment-Aware Reconstruction Engine

Addresses the critical insight: AI changes everything when modifying anything, and pixels 
mean nothing semantically to AI. Therefore, reconstruction must happen in isolated segments,
with each segment getting its own iteration cycles based on complexity and requirements.

Core Innovation:
- Segment the image into semantic regions
- Each segment gets independent reconstruction iterations
- Different segments require different numbers of iterations
- Prevents AI from changing unrelated parts when reconstructing specific areas
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SegmentType(Enum):
    """Different types of image segments requiring different reconstruction approaches."""
    TEXT_REGION = "text_region"           # Text, signs, writing - needs high precision
    FACE_REGION = "face_region"           # Faces - needs identity preservation
    OBJECT_REGION = "object_region"       # Distinct objects - needs shape/form preservation
    BACKGROUND_REGION = "background"      # Background areas - can be more flexible
    TEXTURE_REGION = "texture_region"     # Textures, patterns - needs consistency
    EDGE_REGION = "edge_region"           # Boundaries, edges - needs sharpness
    DETAIL_REGION = "detail_region"       # Fine details - needs high iteration count
    SIMPLE_REGION = "simple_region"       # Simple areas - needs few iterations


@dataclass
class ImageSegment:
    """Represents a semantic segment of the image with its own reconstruction requirements."""
    
    segment_id: str
    segment_type: SegmentType
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    mask: np.ndarray
    pixels: np.ndarray
    
    # Reconstruction parameters specific to this segment
    max_iterations: int = 5
    quality_threshold: float = 0.85
    change_tolerance: float = 0.1  # How much change is acceptable
    priority: int = 1  # Higher priority segments get processed first
    
    # State tracking
    current_iteration: int = 0
    reconstruction_history: List[np.ndarray] = field(default_factory=list)
    quality_history: List[float] = field(default_factory=list)
    converged: bool = False
    final_quality: float = 0.0
    
    # Dependencies - segments that must be stable before this one can be processed
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)


@dataclass
class ReconstructionPlan:
    """Plan for reconstructing segments in the correct order with appropriate iterations."""
    
    plan_id: str
    segments: List[ImageSegment]
    execution_order: List[str]  # Segment IDs in execution order
    total_estimated_iterations: int
    complexity_score: float
    
    # Execution state
    current_segment_idx: int = 0
    completed_segments: Set[str] = field(default_factory=set)
    failed_segments: Set[str] = field(default_factory=set)
    execution_log: List[Dict[str, Any]] = field(default_factory=list)


class ImageSegmenter:
    """Segments images into semantic regions for independent reconstruction."""
    
    def __init__(self):
        self.segmentation_cache = {}
    
    def segment_image(self, image: np.ndarray, description: str = "") -> List[ImageSegment]:
        """
        Segment image into semantic regions.
        
        This is a simplified implementation - in practice, you'd use more sophisticated
        segmentation methods like SAM, semantic segmentation models, etc.
        """
        
        h, w = image.shape[:2]
        segments = []
        
        # Basic segmentation strategies
        segments.extend(self._detect_text_regions(image))
        segments.extend(self._detect_face_regions(image))
        segments.extend(self._detect_edge_regions(image))
        segments.extend(self._detect_texture_regions(image))
        segments.extend(self._create_grid_segments(image))
        
        # Remove overlapping segments and merge similar ones
        segments = self._merge_overlapping_segments(segments)
        
        # Set dependencies based on spatial relationships
        self._set_segment_dependencies(segments)
        
        # Assign reconstruction parameters based on segment type
        self._assign_reconstruction_parameters(segments, description)
        
        return segments
    
    def _detect_text_regions(self, image: np.ndarray) -> List[ImageSegment]:
        """Detect regions likely to contain text."""
        
        segments = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use edge detection to find text-like regions
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours that might be text
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Text-like size
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (text is usually wider than tall)
                aspect_ratio = w / h
                if 1.5 < aspect_ratio < 10:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [contour], 255)
                    
                    segment = ImageSegment(
                        segment_id=f"text_{i}",
                        segment_type=SegmentType.TEXT_REGION,
                        bbox=(x, y, w, h),
                        mask=mask,
                        pixels=image[y:y+h, x:x+w].copy()
                    )
                    segments.append(segment)
        
        return segments
    
    def _detect_face_regions(self, image: np.ndarray) -> List[ImageSegment]:
        """Detect face regions using simple methods."""
        
        segments = []
        
        # This is a placeholder - in practice, you'd use face detection models
        # For now, we'll use Haar cascades as a simple example
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for i, (x, y, w, h) in enumerate(faces):
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[y:y+h, x:x+w] = 255
                
                segment = ImageSegment(
                    segment_id=f"face_{i}",
                    segment_type=SegmentType.FACE_REGION,
                    bbox=(x, y, w, h),
                    mask=mask,
                    pixels=image[y:y+h, x:x+w].copy()
                )
                segments.append(segment)
        
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
        
        return segments
    
    def _detect_edge_regions(self, image: np.ndarray) -> List[ImageSegment]:
        """Detect regions with strong edges that need careful reconstruction."""
        
        segments = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect strong edges
        edges = cv2.Canny(gray, 100, 200)
        
        # Dilate to create edge regions
        kernel = np.ones((5, 5), np.uint8)
        edge_regions = cv2.dilate(edges, kernel, iterations=2)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(edge_regions)
        
        for label in range(1, num_labels):
            mask = (labels == label).astype(np.uint8) * 255
            
            # Get bounding box
            coords = np.where(mask > 0)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                w, h = x_max - x_min, y_max - y_min
                
                if w > 20 and h > 20:  # Minimum size
                    segment = ImageSegment(
                        segment_id=f"edge_{label}",
                        segment_type=SegmentType.EDGE_REGION,
                        bbox=(x_min, y_min, w, h),
                        mask=mask,
                        pixels=image[y_min:y_max+1, x_min:x_max+1].copy()
                    )
                    segments.append(segment)
        
        return segments
    
    def _detect_texture_regions(self, image: np.ndarray) -> List[ImageSegment]:
        """Detect regions with consistent textures."""
        
        segments = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use local binary patterns or similar for texture detection
        # This is a simplified version using variance
        
        # Divide image into blocks and analyze texture
        block_size = 32
        h, w = gray.shape
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                
                # Calculate texture measures
                variance = np.var(block)
                
                # High variance indicates texture
                if variance > 500:  # Threshold for texture
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    mask[y:y+block_size, x:x+block_size] = 255
                    
                    segment = ImageSegment(
                        segment_id=f"texture_{y}_{x}",
                        segment_type=SegmentType.TEXTURE_REGION,
                        bbox=(x, y, block_size, block_size),
                        mask=mask,
                        pixels=image[y:y+block_size, x:x+block_size].copy()
                    )
                    segments.append(segment)
        
        return segments
    
    def _create_grid_segments(self, image: np.ndarray) -> List[ImageSegment]:
        """Create grid-based segments for areas not covered by other methods."""
        
        segments = []
        h, w = image.shape[:2]
        
        # Create a coarse grid for remaining areas
        grid_size = 64
        
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                actual_h = min(grid_size, h - y)
                actual_w = min(grid_size, w - x)
                
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[y:y+actual_h, x:x+actual_w] = 255
                
                # Determine segment type based on content
                block = image[y:y+actual_h, x:x+actual_w]
                segment_type = self._classify_block_type(block)
                
                segment = ImageSegment(
                    segment_id=f"grid_{y}_{x}",
                    segment_type=segment_type,
                    bbox=(x, y, actual_w, actual_h),
                    mask=mask,
                    pixels=block.copy()
                )
                segments.append(segment)
        
        return segments
    
    def _classify_block_type(self, block: np.ndarray) -> SegmentType:
        """Classify a block into a segment type based on its content."""
        
        gray = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)
        
        # Calculate various measures
        variance = np.var(gray)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Simple classification rules
        if edge_density > 0.1:
            return SegmentType.EDGE_REGION
        elif variance > 1000:
            return SegmentType.TEXTURE_REGION
        elif variance > 500:
            return SegmentType.DETAIL_REGION
        else:
            return SegmentType.SIMPLE_REGION
    
    def _merge_overlapping_segments(self, segments: List[ImageSegment]) -> List[ImageSegment]:
        """Merge overlapping segments to avoid conflicts."""
        
        # This is a simplified version - in practice, you'd use more sophisticated merging
        merged_segments = []
        used_indices = set()
        
        for i, segment1 in enumerate(segments):
            if i in used_indices:
                continue
            
            # Check for overlaps with remaining segments
            overlapping = [i]
            
            for j, segment2 in enumerate(segments[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Check if segments overlap significantly
                if self._segments_overlap(segment1, segment2):
                    overlapping.append(j)
            
            if len(overlapping) == 1:
                merged_segments.append(segment1)
            else:
                # Merge overlapping segments
                merged_segment = self._merge_segments([segments[idx] for idx in overlapping])
                merged_segments.append(merged_segment)
            
            used_indices.update(overlapping)
        
        return merged_segments
    
    def _segments_overlap(self, seg1: ImageSegment, seg2: ImageSegment) -> bool:
        """Check if two segments overlap significantly."""
        
        x1, y1, w1, h1 = seg1.bbox
        x2, y2, w2, h2 = seg2.bbox
        
        # Calculate intersection
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        
        intersection_area = x_overlap * y_overlap
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area
        
        # Check if overlap is significant (IoU > 0.3)
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou > 0.3
    
    def _merge_segments(self, segments: List[ImageSegment]) -> ImageSegment:
        """Merge multiple segments into one."""
        
        # Find bounding box that contains all segments
        min_x = min(seg.bbox[0] for seg in segments)
        min_y = min(seg.bbox[1] for seg in segments)
        max_x = max(seg.bbox[0] + seg.bbox[2] for seg in segments)
        max_y = max(seg.bbox[1] + seg.bbox[3] for seg in segments)
        
        merged_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
        
        # Create merged mask
        merged_mask = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        for segment in segments:
            x, y, w, h = segment.bbox
            rel_x, rel_y = x - min_x, y - min_y
            segment_mask = segment.mask[y:y+h, x:x+w]
            merged_mask[rel_y:rel_y+h, rel_x:rel_x+w] = np.maximum(
                merged_mask[rel_y:rel_y+h, rel_x:rel_x+w], 
                segment_mask
            )
        
        # Determine merged segment type (use the most complex type)
        type_priority = {
            SegmentType.TEXT_REGION: 6,
            SegmentType.FACE_REGION: 5,
            SegmentType.DETAIL_REGION: 4,
            SegmentType.EDGE_REGION: 3,
            SegmentType.TEXTURE_REGION: 2,
            SegmentType.OBJECT_REGION: 1,
            SegmentType.SIMPLE_REGION: 0,
            SegmentType.BACKGROUND_REGION: 0
        }
        
        merged_type = max(segments, key=lambda s: type_priority[s.segment_type]).segment_type
        
        # Create merged segment
        merged_segment = ImageSegment(
            segment_id=f"merged_{'_'.join(s.segment_id for s in segments)}",
            segment_type=merged_type,
            bbox=merged_bbox,
            mask=merged_mask,
            pixels=np.zeros((max_y - min_y, max_x - min_x, 3), dtype=np.uint8)  # Will be filled later
        )
        
        return merged_segment
    
    def _set_segment_dependencies(self, segments: List[ImageSegment]):
        """Set dependencies between segments based on spatial relationships."""
        
        for i, segment1 in enumerate(segments):
            for j, segment2 in enumerate(segments):
                if i == j:
                    continue
                
                # Text regions should be processed after background is stable
                if (segment1.segment_type == SegmentType.TEXT_REGION and 
                    segment2.segment_type == SegmentType.BACKGROUND_REGION):
                    if self._segments_adjacent(segment1, segment2):
                        segment1.dependencies.add(segment2.segment_id)
                        segment2.dependents.add(segment1.segment_id)
                
                # Face regions should be processed after background
                if (segment1.segment_type == SegmentType.FACE_REGION and 
                    segment2.segment_type == SegmentType.BACKGROUND_REGION):
                    if self._segments_adjacent(segment1, segment2):
                        segment1.dependencies.add(segment2.segment_id)
                        segment2.dependents.add(segment1.segment_id)
    
    def _segments_adjacent(self, seg1: ImageSegment, seg2: ImageSegment) -> bool:
        """Check if two segments are adjacent."""
        
        x1, y1, w1, h1 = seg1.bbox
        x2, y2, w2, h2 = seg2.bbox
        
        # Check if segments are close to each other
        distance = min(
            abs(x1 - (x2 + w2)),  # Left edge to right edge
            abs((x1 + w1) - x2),  # Right edge to left edge
            abs(y1 - (y2 + h2)),  # Top edge to bottom edge
            abs((y1 + h1) - y2)   # Bottom edge to top edge
        )
        
        return distance < 10  # Adjacent if within 10 pixels
    
    def _assign_reconstruction_parameters(self, segments: List[ImageSegment], description: str):
        """Assign reconstruction parameters based on segment type and context."""
        
        for segment in segments:
            if segment.segment_type == SegmentType.TEXT_REGION:
                segment.max_iterations = 10  # Text needs high precision
                segment.quality_threshold = 0.95
                segment.change_tolerance = 0.05
                segment.priority = 5
            
            elif segment.segment_type == SegmentType.FACE_REGION:
                segment.max_iterations = 8
                segment.quality_threshold = 0.90
                segment.change_tolerance = 0.1
                segment.priority = 4
            
            elif segment.segment_type == SegmentType.DETAIL_REGION:
                segment.max_iterations = 6
                segment.quality_threshold = 0.85
                segment.change_tolerance = 0.15
                segment.priority = 3
            
            elif segment.segment_type == SegmentType.EDGE_REGION:
                segment.max_iterations = 5
                segment.quality_threshold = 0.80
                segment.change_tolerance = 0.2
                segment.priority = 2
            
            elif segment.segment_type == SegmentType.TEXTURE_REGION:
                segment.max_iterations = 4
                segment.quality_threshold = 0.75
                segment.change_tolerance = 0.25
                segment.priority = 2
            
            else:  # SIMPLE_REGION, BACKGROUND_REGION, OBJECT_REGION
                segment.max_iterations = 3
                segment.quality_threshold = 0.70
                segment.change_tolerance = 0.3
                segment.priority = 1


class SegmentAwareReconstructionEngine:
    """
    Main engine for segment-aware reconstruction.
    
    Addresses the core problem: AI changes everything when modifying anything.
    Solution: Reconstruct each segment independently with its own iteration cycles.
    """
    
    def __init__(self, api_key: str = None):
        from .pakati_inspired_reconstruction import HuggingFaceAPI
        
        self.api = HuggingFaceAPI(api_key) if api_key else None
        self.segmenter = ImageSegmenter()
        self.reconstruction_history = []
        
    def segment_aware_reconstruction(self, image: np.ndarray, 
                                   description: str = "") -> Dict[str, Any]:
        """
        Perform segment-aware reconstruction with independent iteration cycles.
        
        This addresses the key insight that different regions need different amounts
        of iteration and AI tends to change everything when modifying anything.
        """
        
        logger.info(f"Starting segment-aware reconstruction: {description}")
        
        # Step 1: Segment the image
        segments = self.segmenter.segment_image(image, description)
        logger.info(f"Created {len(segments)} segments")
        
        # Step 2: Create reconstruction plan
        plan = self._create_reconstruction_plan(segments)
        logger.info(f"Created reconstruction plan with {plan.total_estimated_iterations} total iterations")
        
        # Step 3: Execute reconstruction plan
        results = self._execute_reconstruction_plan(image, plan)
        
        # Step 4: Combine results and assess overall quality
        final_results = self._combine_segment_results(image, results, plan)
        
        return final_results
    
    def _create_reconstruction_plan(self, segments: List[ImageSegment]) -> ReconstructionPlan:
        """Create an execution plan for segment reconstruction."""
        
        # Sort segments by priority and dependencies
        execution_order = self._determine_execution_order(segments)
        
        # Calculate total estimated iterations
        total_iterations = sum(seg.max_iterations for seg in segments)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(segments)
        
        plan = ReconstructionPlan(
            plan_id=f"plan_{int(time.time())}",
            segments=segments,
            execution_order=execution_order,
            total_estimated_iterations=total_iterations,
            complexity_score=complexity_score
        )
        
        return plan
    
    def _determine_execution_order(self, segments: List[ImageSegment]) -> List[str]:
        """Determine the order in which segments should be processed."""
        
        # Create a dependency graph
        segment_dict = {seg.segment_id: seg for seg in segments}
        
        # Topological sort considering dependencies and priorities
        visited = set()
        temp_visited = set()
        execution_order = []
        
        def visit(segment_id: str):
            if segment_id in temp_visited:
                # Circular dependency - break it by priority
                return
            
            if segment_id in visited:
                return
            
            temp_visited.add(segment_id)
            segment = segment_dict[segment_id]
            
            # Visit dependencies first
            for dep_id in segment.dependencies:
                if dep_id in segment_dict:
                    visit(dep_id)
            
            temp_visited.remove(segment_id)
            visited.add(segment_id)
            execution_order.append(segment_id)
        
        # Sort segments by priority first, then process
        sorted_segments = sorted(segments, key=lambda s: (-s.priority, s.segment_id))
        
        for segment in sorted_segments:
            if segment.segment_id not in visited:
                visit(segment.segment_id)
        
        return execution_order
    
    def _calculate_complexity_score(self, segments: List[ImageSegment]) -> float:
        """Calculate overall complexity score for the reconstruction task."""
        
        complexity = 0.0
        
        for segment in segments:
            # Base complexity from segment type
            type_complexity = {
                SegmentType.TEXT_REGION: 1.0,
                SegmentType.FACE_REGION: 0.9,
                SegmentType.DETAIL_REGION: 0.8,
                SegmentType.EDGE_REGION: 0.7,
                SegmentType.TEXTURE_REGION: 0.6,
                SegmentType.OBJECT_REGION: 0.5,
                SegmentType.SIMPLE_REGION: 0.3,
                SegmentType.BACKGROUND_REGION: 0.2
            }
            
            segment_complexity = type_complexity.get(segment.segment_type, 0.5)
            
            # Adjust for size
            area = segment.bbox[2] * segment.bbox[3]
            size_factor = min(1.0, area / 10000)  # Normalize to reasonable range
            
            complexity += segment_complexity * size_factor
        
        return complexity / len(segments) if segments else 0.0
    
    def _execute_reconstruction_plan(self, original_image: np.ndarray, 
                                   plan: ReconstructionPlan) -> Dict[str, Any]:
        """Execute the reconstruction plan segment by segment."""
        
        results = {
            'plan_id': plan.plan_id,
            'segment_results': {},
            'execution_log': [],
            'total_iterations_performed': 0,
            'successful_segments': 0,
            'failed_segments': 0
        }
        
        # Create working image
        working_image = original_image.copy()
        segment_dict = {seg.segment_id: seg for seg in plan.segments}
        
        # Process segments in execution order
        for segment_id in plan.execution_order:
            segment = segment_dict[segment_id]
            
            logger.info(f"Processing segment {segment_id} ({segment.segment_type.value})")
            
            # Check if dependencies are satisfied
            if not self._dependencies_satisfied(segment, plan.completed_segments):
                logger.warning(f"Dependencies not satisfied for {segment_id}, skipping")
                plan.failed_segments.add(segment_id)
                continue
            
            # Reconstruct this segment with its own iteration cycles
            segment_result = self._reconstruct_segment(
                working_image, segment, original_image
            )
            
            results['segment_results'][segment_id] = segment_result
            results['total_iterations_performed'] += segment_result['iterations_performed']
            
            if segment_result['success']:
                # Update working image with reconstructed segment
                working_image = self._apply_segment_reconstruction(
                    working_image, segment, segment_result['final_reconstruction']
                )
                plan.completed_segments.add(segment_id)
                results['successful_segments'] += 1
            else:
                plan.failed_segments.add(segment_id)
                results['failed_segments'] += 1
            
            # Log execution
            log_entry = {
                'segment_id': segment_id,
                'segment_type': segment.segment_type.value,
                'iterations': segment_result['iterations_performed'],
                'final_quality': segment_result['final_quality'],
                'success': segment_result['success'],
                'timestamp': time.time()
            }
            results['execution_log'].append(log_entry)
        
        results['final_image'] = working_image
        
        return results
    
    def _dependencies_satisfied(self, segment: ImageSegment, 
                              completed_segments: Set[str]) -> bool:
        """Check if all dependencies for a segment are satisfied."""
        
        return segment.dependencies.issubset(completed_segments)
    
    def _reconstruct_segment(self, working_image: np.ndarray, 
                           segment: ImageSegment,
                           original_image: np.ndarray) -> Dict[str, Any]:
        """Reconstruct a single segment with its own iteration cycles."""
        
        if not self.api:
            return self._local_segment_reconstruction(working_image, segment, original_image)
        
        segment_result = {
            'segment_id': segment.segment_id,
            'segment_type': segment.segment_type.value,
            'iterations_performed': 0,
            'quality_progression': [],
            'reconstruction_history': [],
            'final_quality': 0.0,
            'final_reconstruction': None,
            'success': False,
            'convergence_reason': 'max_iterations'
        }
        
        # Extract segment region from working image
        x, y, w, h = segment.bbox
        current_segment = working_image[y:y+h, x:x+w].copy()
        original_segment = original_image[y:y+h, x:x+w].copy()
        
        # Create mask for this segment only
        segment_mask = segment.mask[y:y+h, x:x+w]
        
        # Iterative reconstruction for this segment
        for iteration in range(segment.max_iterations):
            logger.debug(f"Segment {segment.segment_id} iteration {iteration + 1}")
            
            try:
                # Create masked version for reconstruction
                masked_segment = current_segment.copy()
                reconstruction_mask = (segment_mask == 0).astype(np.uint8)
                
                if np.sum(reconstruction_mask) == 0:
                    # Nothing to reconstruct
                    segment_result['final_reconstruction'] = current_segment
                    segment_result['final_quality'] = 1.0
                    segment_result['success'] = True
                    segment_result['convergence_reason'] = 'nothing_to_reconstruct'
                    break
                
                # Perform reconstruction using API
                reconstructed_segment = self.api.inpaint(
                    masked_segment, 
                    reconstruction_mask * 255,
                    self._generate_segment_prompt(segment, working_image)
                )
                
                # Calculate quality
                quality = self._calculate_segment_quality(
                    original_segment, reconstructed_segment, reconstruction_mask
                )
                
                segment_result['quality_progression'].append(quality)
                segment_result['reconstruction_history'].append(reconstructed_segment.copy())
                segment_result['iterations_performed'] += 1
                
                # Check for convergence
                if quality >= segment.quality_threshold:
                    segment_result['final_reconstruction'] = reconstructed_segment
                    segment_result['final_quality'] = quality
                    segment_result['success'] = True
                    segment_result['convergence_reason'] = 'quality_threshold_reached'
                    break
                
                # Check for stability (not changing much between iterations)
                if (len(segment_result['quality_progression']) > 1 and
                    abs(segment_result['quality_progression'][-1] - 
                        segment_result['quality_progression'][-2]) < segment.change_tolerance):
                    segment_result['final_reconstruction'] = reconstructed_segment
                    segment_result['final_quality'] = quality
                    segment_result['success'] = quality >= 0.6  # Minimum acceptable quality
                    segment_result['convergence_reason'] = 'stability_reached'
                    break
                
                # Update current segment for next iteration
                current_segment = reconstructed_segment
                
            except Exception as e:
                logger.error(f"Error in segment {segment.segment_id} iteration {iteration}: {e}")
                break
        
        # Set final results if not already set
        if segment_result['final_reconstruction'] is None:
            segment_result['final_reconstruction'] = current_segment
            segment_result['final_quality'] = segment_result['quality_progression'][-1] if segment_result['quality_progression'] else 0.0
            segment_result['success'] = segment_result['final_quality'] >= 0.6
        
        return segment_result
    
    def _local_segment_reconstruction(self, working_image: np.ndarray,
                                    segment: ImageSegment,
                                    original_image: np.ndarray) -> Dict[str, Any]:
        """Fallback local reconstruction when API is not available."""
        
        # This is a placeholder for local reconstruction
        # In practice, you'd implement local neural network-based reconstruction
        
        return {
            'segment_id': segment.segment_id,
            'segment_type': segment.segment_type.value,
            'iterations_performed': 1,
            'quality_progression': [0.5],
            'reconstruction_history': [],
            'final_quality': 0.5,
            'final_reconstruction': segment.pixels,
            'success': False,
            'convergence_reason': 'local_fallback'
        }
    
    def _generate_segment_prompt(self, segment: ImageSegment, 
                               working_image: np.ndarray) -> str:
        """Generate appropriate prompt for segment reconstruction."""
        
        base_prompts = {
            SegmentType.TEXT_REGION: "clear readable text, sharp letters, correct spelling",
            SegmentType.FACE_REGION: "natural human face, correct proportions, clear features",
            SegmentType.DETAIL_REGION: "fine details, high resolution, sharp focus",
            SegmentType.EDGE_REGION: "sharp edges, clear boundaries, precise lines",
            SegmentType.TEXTURE_REGION: "consistent texture, natural pattern, seamless",
            SegmentType.OBJECT_REGION: "complete object, correct shape, natural appearance",
            SegmentType.SIMPLE_REGION: "smooth consistent area, natural colors",
            SegmentType.BACKGROUND_REGION: "natural background, consistent lighting"
        }
        
        base_prompt = base_prompts.get(segment.segment_type, "high quality reconstruction")
        
        return f"{base_prompt}, photorealistic, seamless integration"
    
    def _calculate_segment_quality(self, original: np.ndarray, 
                                 reconstructed: np.ndarray,
                                 mask: np.ndarray) -> float:
        """Calculate quality score for segment reconstruction."""
        
        if mask.sum() == 0:
            return 1.0
        
        # Focus on reconstructed regions
        reconstruction_regions = mask > 0
        
        if not np.any(reconstruction_regions):
            return 1.0
        
        # Calculate MSE in reconstructed regions
        original_region = original[reconstruction_regions]
        reconstructed_region = reconstructed[reconstruction_regions]
        
        mse = np.mean((original_region.astype(float) - reconstructed_region.astype(float)) ** 2)
        
        # Convert to quality score
        quality = 1.0 - (mse / (255 ** 2))
        
        return max(0.0, min(1.0, quality))
    
    def _apply_segment_reconstruction(self, working_image: np.ndarray,
                                    segment: ImageSegment,
                                    reconstructed_segment: np.ndarray) -> np.ndarray:
        """Apply reconstructed segment to the working image."""
        
        result_image = working_image.copy()
        x, y, w, h = segment.bbox
        
        # Apply only the reconstructed parts (where mask is 0)
        segment_mask = segment.mask[y:y+h, x:x+w]
        reconstruction_mask = segment_mask == 0
        
        if reconstructed_segment.shape[:2] == (h, w):
            result_image[y:y+h, x:x+w][reconstruction_mask] = reconstructed_segment[reconstruction_mask]
        
        return result_image
    
    def _combine_segment_results(self, original_image: np.ndarray,
                               results: Dict[str, Any],
                               plan: ReconstructionPlan) -> Dict[str, Any]:
        """Combine individual segment results into final assessment."""
        
        final_results = {
            'original_image_shape': original_image.shape,
            'total_segments': len(plan.segments),
            'successful_segments': results['successful_segments'],
            'failed_segments': results['failed_segments'],
            'total_iterations': results['total_iterations_performed'],
            'final_reconstructed_image': results['final_image'],
            'segment_details': results['segment_results'],
            'execution_log': results['execution_log'],
            'overall_quality': 0.0,
            'understanding_level': 'unknown',
            'insights': []
        }
        
        # Calculate overall quality
        if results['segment_results']:
            segment_qualities = [
                result['final_quality'] for result in results['segment_results'].values()
            ]
            final_results['overall_quality'] = np.mean(segment_qualities)
        
        # Determine understanding level
        if final_results['overall_quality'] >= 0.9:
            final_results['understanding_level'] = 'excellent'
        elif final_results['overall_quality'] >= 0.8:
            final_results['understanding_level'] = 'good'
        elif final_results['overall_quality'] >= 0.6:
            final_results['understanding_level'] = 'moderate'
        else:
            final_results['understanding_level'] = 'limited'
        
        # Generate insights
        final_results['insights'] = self._generate_segment_insights(results, plan)
        
        return final_results
    
    def _generate_segment_insights(self, results: Dict[str, Any],
                                 plan: ReconstructionPlan) -> List[str]:
        """Generate insights from segment-aware reconstruction."""
        
        insights = []
        
        # Overall performance
        success_rate = results['successful_segments'] / len(plan.segments)
        insights.append(f"Successfully reconstructed {success_rate:.1%} of segments")
        
        # Segment type analysis
        type_performance = {}
        for segment_id, result in results['segment_results'].items():
            seg_type = result['segment_type']
            if seg_type not in type_performance:
                type_performance[seg_type] = []
            type_performance[seg_type].append(result['final_quality'])
        
        for seg_type, qualities in type_performance.items():
            avg_quality = np.mean(qualities)
            insights.append(f"{seg_type}: {avg_quality:.3f} average quality")
        
        # Iteration efficiency
        avg_iterations = results['total_iterations_performed'] / len(plan.segments)
        insights.append(f"Average {avg_iterations:.1f} iterations per segment")
        
        # Identify challenging segments
        challenging_segments = [
            result for result in results['segment_results'].values()
            if result['final_quality'] < 0.7
        ]
        
        if challenging_segments:
            insights.append(f"{len(challenging_segments)} segments required additional attention")
        
        return insights


# Example usage
if __name__ == "__main__":
    # Initialize segment-aware reconstruction engine
    engine = SegmentAwareReconstructionEngine()
    
    # Test with sample image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Perform segment-aware reconstruction
    results = engine.segment_aware_reconstruction(
        test_image, 
        "complex image with text, faces, and various objects"
    )
    
    print(f"Segment-Aware Reconstruction Results:")
    print(f"Understanding Level: {results['understanding_level']}")
    print(f"Overall Quality: {results['overall_quality']:.3f}")
    print(f"Successful Segments: {results['successful_segments']}/{results['total_segments']}")
    print(f"Total Iterations: {results['total_iterations']}")
    
    for insight in results['insights']:
        print(f"• {insight}") 