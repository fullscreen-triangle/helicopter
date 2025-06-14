"""
Zengeza Noise Detection Module

Calculates the probable amount of "noise" or garbage per segment per iteration.

Core Innovation:
- Not every part of an image is important for understanding
- Much of image content is "noise" that doesn't contribute to comprehension
- This noise is not necessarily "clear" - it can be subtle and context-dependent
- Zengeza quantifies noise probability per segment to focus on important regions
- Helps reconstruction engines prioritize meaningful content over garbage

The name "Zengeza" reflects the process of separating wheat from chaff,
identifying what matters vs what's just noise in visual understanding.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
from collections import defaultdict
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class NoiseType(Enum):
    """Types of noise that can be detected in image segments."""
    VISUAL_NOISE = "visual_noise"           # Random pixels, artifacts, compression noise
    SEMANTIC_NOISE = "semantic_noise"       # Irrelevant objects/content
    STRUCTURAL_NOISE = "structural_noise"   # Meaningless patterns, textures
    CONTEXTUAL_NOISE = "contextual_noise"   # Content that doesn't fit context
    RECONSTRUCTION_NOISE = "reconstruction_noise"  # Artifacts from reconstruction
    ATTENTION_NOISE = "attention_noise"     # Content that distracts from main subject
    REDUNDANT_NOISE = "redundant_noise"     # Repetitive, non-informative content


class NoiseLevel(Enum):
    """Levels of noise probability."""
    MINIMAL = "minimal"         # 0.0 - 0.2: Very low noise, high importance
    LOW = "low"                # 0.2 - 0.4: Some noise, mostly important
    MODERATE = "moderate"       # 0.4 - 0.6: Moderate noise, mixed importance
    HIGH = "high"              # 0.6 - 0.8: High noise, low importance
    CRITICAL = "critical"       # 0.8 - 1.0: Mostly noise, very low importance


@dataclass
class NoiseMetrics:
    """Comprehensive noise metrics for a segment."""
    
    # Basic noise measures
    visual_noise_score: float = 0.0        # Pixel-level noise
    semantic_noise_score: float = 0.0      # Content relevance noise
    structural_noise_score: float = 0.0    # Pattern/structure noise
    contextual_noise_score: float = 0.0    # Context mismatch noise
    
    # Composite measures
    overall_noise_probability: float = 0.0  # Combined noise probability
    importance_score: float = 0.0           # Inverse of noise (how important)
    confidence: float = 0.0                 # Confidence in noise assessment
    
    # Detailed analysis
    noise_types_detected: List[NoiseType] = field(default_factory=list)
    noise_level: NoiseLevel = NoiseLevel.MODERATE
    
    # Temporal tracking
    noise_history: List[float] = field(default_factory=list)
    noise_trend: str = "stable"  # "increasing", "decreasing", "stable"
    
    # Segment characteristics
    segment_complexity: float = 0.0
    information_density: float = 0.0
    reconstruction_contribution: float = 0.0


@dataclass
class ZengezaSegment:
    """A segment with noise analysis."""
    
    segment_id: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    pixels: np.ndarray
    mask: Optional[np.ndarray] = None
    
    # Noise analysis
    noise_metrics: NoiseMetrics = field(default_factory=NoiseMetrics)
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Analysis state
    last_analyzed: float = 0.0
    analysis_count: int = 0
    stable_noise_estimate: bool = False


class ZengezaNoiseAnalyzer:
    """Core noise analysis engine that calculates noise probabilities."""
    
    def __init__(self):
        self.noise_thresholds = {
            'visual_noise': {'low': 0.1, 'moderate': 0.3, 'high': 0.6, 'critical': 0.8},
            'semantic_noise': {'low': 0.15, 'moderate': 0.35, 'high': 0.65, 'critical': 0.85},
            'structural_noise': {'low': 0.2, 'moderate': 0.4, 'high': 0.7, 'critical': 0.9}
        }
        
        logger.info("Zengeza noise analyzer initialized")
    
    def analyze_segment_noise(self, segment: ZengezaSegment, 
                            context: Dict[str, Any],
                            iteration: int) -> NoiseMetrics:
        """Analyze noise in a segment for a specific iteration."""
        
        logger.debug(f"Analyzing noise for segment {segment.segment_id}, iteration {iteration}")
        
        # Calculate different types of noise
        visual_noise = self._calculate_visual_noise(segment.pixels)
        semantic_noise = self._calculate_semantic_noise(segment.pixels, context)
        structural_noise = self._calculate_structural_noise(segment.pixels)
        contextual_noise = self._calculate_contextual_noise(segment, context)
        
        # Calculate composite noise probability
        overall_noise = self._calculate_overall_noise_probability(
            visual_noise, semantic_noise, structural_noise, contextual_noise
        )
        
        # Calculate importance (inverse of noise)
        importance = 1.0 - overall_noise
        
        # Calculate confidence in assessment
        confidence = self._calculate_noise_confidence(segment, iteration)
        
        # Detect specific noise types
        noise_types = self._detect_noise_types(
            visual_noise, semantic_noise, structural_noise, contextual_noise
        )
        
        # Determine noise level
        noise_level = self._classify_noise_level(overall_noise)
        
        # Calculate additional metrics
        complexity = self._calculate_segment_complexity(segment.pixels)
        info_density = self._calculate_information_density(segment.pixels)
        recon_contribution = self._estimate_reconstruction_contribution(segment, context)
        
        # Create noise metrics
        metrics = NoiseMetrics(
            visual_noise_score=visual_noise,
            semantic_noise_score=semantic_noise,
            structural_noise_score=structural_noise,
            contextual_noise_score=contextual_noise,
            overall_noise_probability=overall_noise,
            importance_score=importance,
            confidence=confidence,
            noise_types_detected=noise_types,
            noise_level=noise_level,
            segment_complexity=complexity,
            information_density=info_density,
            reconstruction_contribution=recon_contribution
        )
        
        # Update noise history and trend
        self._update_noise_history(segment, metrics, iteration)
        
        return metrics
    
    def _calculate_visual_noise(self, pixels: np.ndarray) -> float:
        """Calculate visual/pixel-level noise score."""
        
        if pixels.size == 0:
            return 1.0
        
        # Convert to grayscale for analysis
        if len(pixels.shape) == 3:
            gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        else:
            gray = pixels.copy()
        
        # Calculate various noise indicators
        noise_indicators = []
        
        # 1. High frequency noise (using Laplacian)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_indicators.append(min(1.0, laplacian_var / 1000.0))
        
        # 2. Local variance (inconsistent pixel values)
        kernel = np.ones((3, 3), np.float32) / 9
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        avg_local_var = np.mean(local_var)
        noise_indicators.append(min(1.0, avg_local_var / 500.0))
        
        # 3. Edge density (too many edges can indicate noise)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density > 0.3:  # Too many edges
            noise_indicators.append(edge_density - 0.3)
        else:
            noise_indicators.append(0.0)
        
        # 4. Histogram analysis (uniform distribution indicates noise)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist / np.sum(hist)
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
        # High entropy can indicate noise
        noise_indicators.append(min(1.0, max(0.0, (entropy - 6.0) / 2.0)))
        
        # Combine indicators
        visual_noise_score = np.mean(noise_indicators)
        
        return min(1.0, max(0.0, visual_noise_score))
    
    def _calculate_semantic_noise(self, pixels: np.ndarray, context: Dict[str, Any]) -> float:
        """Calculate semantic noise - content that doesn't contribute to understanding."""
        
        semantic_indicators = []
        
        # 1. Color coherence - random colors indicate semantic noise
        if len(pixels.shape) == 3:
            # Calculate color variance
            color_std = np.std(pixels, axis=(0, 1))
            color_coherence = 1.0 - (np.mean(color_std) / 255.0)
            semantic_indicators.append(1.0 - color_coherence)
        
        # 2. Texture consistency
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) if len(pixels.shape) == 3 else pixels
        
        # Local Binary Pattern for texture analysis
        texture_variance = self._calculate_texture_variance(gray)
        semantic_indicators.append(min(1.0, texture_variance / 100.0))
        
        # 3. Context mismatch (simplified)
        expected_complexity = context.get('expected_complexity', 0.5)
        actual_complexity = self._calculate_segment_complexity(pixels)
        complexity_mismatch = abs(expected_complexity - actual_complexity)
        semantic_indicators.append(complexity_mismatch)
        
        # 4. Reconstruction relevance
        reconstruction_quality = context.get('reconstruction_quality', 0.5)
        # If reconstruction quality is low, this segment might be noise
        if reconstruction_quality < 0.3:
            semantic_indicators.append(0.7)
        else:
            semantic_indicators.append(0.0)
        
        semantic_noise_score = np.mean(semantic_indicators)
        
        return min(1.0, max(0.0, semantic_noise_score))
    
    def _calculate_structural_noise(self, pixels: np.ndarray) -> float:
        """Calculate structural noise - meaningless patterns and structures."""
        
        if pixels.size == 0:
            return 1.0
        
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) if len(pixels.shape) == 3 else pixels
        
        structural_indicators = []
        
        # 1. Repetitive patterns (can be noise)
        # Use autocorrelation to detect repetition
        if gray.shape[0] > 4 and gray.shape[1] > 4:
            template = gray[:gray.shape[0]//2, :gray.shape[1]//2]
            autocorr = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            max_autocorr = np.max(autocorr)
            if max_autocorr > 0.8:  # High repetition
                structural_indicators.append(max_autocorr - 0.5)
            else:
                structural_indicators.append(0.0)
        else:
            structural_indicators.append(0.3)
        
        # 2. Lack of meaningful structure
        # Use Hough lines to detect structured content
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=max(10, min(gray.shape)//4))
        
        if lines is None:
            structural_indicators.append(0.3)  # No structure found
        else:
            # Too many or too few lines can indicate noise
            line_count = len(lines)
            if line_count < 2:
                structural_indicators.append(0.4)
            elif line_count > 20:
                structural_indicators.append(0.6)
            else:
                structural_indicators.append(0.0)
        
        # 3. Gradient coherence
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_coherence = np.std(gradient_magnitude) / (np.mean(gradient_magnitude) + 1e-10)
        
        if gradient_coherence > 2.0:  # Incoherent gradients
            structural_indicators.append(min(1.0, gradient_coherence / 5.0))
        else:
            structural_indicators.append(0.0)
        
        structural_noise_score = np.mean(structural_indicators)
        
        return min(1.0, max(0.0, structural_noise_score))
    
    def _calculate_contextual_noise(self, segment: ZengezaSegment, context: Dict[str, Any]) -> float:
        """Calculate contextual noise - content that doesn't fit the overall context."""
        
        contextual_indicators = []
        
        # 1. Size appropriateness
        segment_area = segment.bbox[2] * segment.bbox[3]
        total_area = context.get('total_image_area', segment_area * 10)
        size_ratio = segment_area / total_area
        
        # Very small or very large segments might be noise
        if size_ratio < 0.01 or size_ratio > 0.5:
            contextual_indicators.append(0.4)
        else:
            contextual_indicators.append(0.0)
        
        # 2. Position appropriateness
        x, y, w, h = segment.bbox
        image_center_x = context.get('image_width', w * 2) / 2
        image_center_y = context.get('image_height', h * 2) / 2
        
        segment_center_x = x + w / 2
        segment_center_y = y + h / 2
        
        # Distance from image center (normalized)
        distance_from_center = np.sqrt(
            ((segment_center_x - image_center_x) / image_center_x) ** 2 +
            ((segment_center_y - image_center_y) / image_center_y) ** 2
        )
        
        # Segments very far from center might be less important
        if distance_from_center > 1.5:
            contextual_indicators.append(min(0.5, distance_from_center - 1.0))
        else:
            contextual_indicators.append(0.0)
        
        # 3. Consistency with neighboring segments
        neighbor_consistency = context.get('neighbor_consistency', 0.5)
        if neighbor_consistency < 0.3:
            contextual_indicators.append(0.6)
        else:
            contextual_indicators.append(0.0)
        
        # 4. Reconstruction importance
        reconstruction_contribution = context.get('reconstruction_contribution', 0.5)
        if reconstruction_contribution < 0.2:
            contextual_indicators.append(0.7)
        else:
            contextual_indicators.append(0.0)
        
        contextual_noise_score = np.mean(contextual_indicators)
        
        return min(1.0, max(0.0, contextual_noise_score))
    
    def _calculate_overall_noise_probability(self, visual: float, semantic: float, 
                                           structural: float, contextual: float) -> float:
        """Calculate overall noise probability from individual components."""
        
        # Weighted combination of noise types
        weights = {
            'visual': 0.2,      # Visual noise is often obvious
            'semantic': 0.4,    # Semantic noise is very important
            'structural': 0.25, # Structural noise affects understanding
            'contextual': 0.15  # Contextual noise is secondary
        }
        
        overall_noise = (
            weights['visual'] * visual +
            weights['semantic'] * semantic +
            weights['structural'] * structural +
            weights['contextual'] * contextual
        )
        
        return min(1.0, max(0.0, overall_noise))
    
    def _calculate_noise_confidence(self, segment: ZengezaSegment, iteration: int) -> float:
        """Calculate confidence in the noise assessment."""
        
        # Base confidence
        confidence = 0.5
        
        # Increase confidence with more iterations
        confidence += min(0.3, iteration * 0.05)
        
        # Increase confidence with larger segments
        segment_area = segment.bbox[2] * segment.bbox[3]
        if segment_area > 1000:
            confidence += 0.1
        elif segment_area < 100:
            confidence -= 0.1
        
        # Increase confidence if noise estimate is stable
        if segment.stable_noise_estimate:
            confidence += 0.2
        
        return min(1.0, max(0.0, confidence))
    
    def _detect_noise_types(self, visual: float, semantic: float, 
                          structural: float, contextual: float) -> List[NoiseType]:
        """Detect specific types of noise present."""
        
        noise_types = []
        
        if visual > 0.4:
            noise_types.append(NoiseType.VISUAL_NOISE)
        
        if semantic > 0.4:
            noise_types.append(NoiseType.SEMANTIC_NOISE)
        
        if structural > 0.4:
            noise_types.append(NoiseType.STRUCTURAL_NOISE)
        
        if contextual > 0.4:
            noise_types.append(NoiseType.CONTEXTUAL_NOISE)
        
        # Additional noise types based on combinations
        if visual > 0.3 and structural > 0.3:
            noise_types.append(NoiseType.RECONSTRUCTION_NOISE)
        
        if semantic > 0.5 and contextual > 0.3:
            noise_types.append(NoiseType.ATTENTION_NOISE)
        
        return noise_types
    
    def _classify_noise_level(self, overall_noise: float) -> NoiseLevel:
        """Classify the overall noise level."""
        
        if overall_noise < 0.2:
            return NoiseLevel.MINIMAL
        elif overall_noise < 0.4:
            return NoiseLevel.LOW
        elif overall_noise < 0.6:
            return NoiseLevel.MODERATE
        elif overall_noise < 0.8:
            return NoiseLevel.HIGH
        else:
            return NoiseLevel.CRITICAL
    
    def _calculate_segment_complexity(self, pixels: np.ndarray) -> float:
        """Calculate complexity of a segment."""
        
        if pixels.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) if len(pixels.shape) == 3 else pixels
        
        # Use edge density as complexity measure
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Use entropy as additional complexity measure
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist / np.sum(hist)
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
        
        # Combine measures
        complexity = (edge_density + entropy / 8.0) / 2.0
        
        return min(1.0, max(0.0, complexity))
    
    def _calculate_information_density(self, pixels: np.ndarray) -> float:
        """Calculate information density of a segment."""
        
        if pixels.size == 0:
            return 0.0
        
        # Use compression ratio as information density measure
        # More compressible = less information
        
        # Encode as JPEG and measure compression ratio
        try:
            _, encoded = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, 50])
            compression_ratio = len(encoded) / pixels.nbytes
            
            # Higher compression ratio = more information
            info_density = min(1.0, compression_ratio * 2.0)
            
        except:
            # Fallback to variance-based measure
            variance = np.var(pixels)
            info_density = min(1.0, variance / 10000.0)
        
        return info_density
    
    def _estimate_reconstruction_contribution(self, segment: ZengezaSegment, 
                                           context: Dict[str, Any]) -> float:
        """Estimate how much this segment contributes to reconstruction quality."""
        
        contribution_factors = []
        
        # 1. Size factor - larger segments typically contribute more
        segment_area = segment.bbox[2] * segment.bbox[3]
        total_area = context.get('total_image_area', segment_area * 10)
        size_factor = min(1.0, segment_area / (total_area * 0.1))
        contribution_factors.append(size_factor)
        
        # 2. Complexity factor - more complex segments contribute more
        complexity = self._calculate_segment_complexity(segment.pixels)
        contribution_factors.append(complexity)
        
        # 3. Position factor - central segments typically contribute more
        x, y, w, h = segment.bbox
        center_x = x + w / 2
        center_y = y + h / 2
        
        image_center_x = context.get('image_width', w * 2) / 2
        image_center_y = context.get('image_height', h * 2) / 2
        
        distance_from_center = np.sqrt(
            ((center_x - image_center_x) / image_center_x) ** 2 +
            ((center_y - image_center_y) / image_center_y) ** 2
        )
        
        position_factor = max(0.0, 1.0 - distance_from_center)
        contribution_factors.append(position_factor)
        
        # 4. Information density factor
        info_density = self._calculate_information_density(segment.pixels)
        contribution_factors.append(info_density)
        
        reconstruction_contribution = np.mean(contribution_factors)
        
        return min(1.0, max(0.0, reconstruction_contribution))
    
    def _update_noise_history(self, segment: ZengezaSegment, 
                            metrics: NoiseMetrics, iteration: int):
        """Update noise history and detect trends."""
        
        # Add to history
        metrics.noise_history.append(metrics.overall_noise_probability)
        
        # Keep only recent history
        if len(metrics.noise_history) > 10:
            metrics.noise_history = metrics.noise_history[-10:]
        
        # Detect trend
        if len(metrics.noise_history) >= 3:
            recent_values = metrics.noise_history[-3:]
            
            if all(recent_values[i] > recent_values[i-1] for i in range(1, len(recent_values))):
                metrics.noise_trend = "increasing"
            elif all(recent_values[i] < recent_values[i-1] for i in range(1, len(recent_values))):
                metrics.noise_trend = "decreasing"
            else:
                metrics.noise_trend = "stable"
        
        # Check if noise estimate is stable
        if len(metrics.noise_history) >= 5:
            recent_std = np.std(metrics.noise_history[-5:])
            if recent_std < 0.05:  # Very stable
                segment.stable_noise_estimate = True
        
        # Update segment analysis state
        segment.last_analyzed = time.time()
        segment.analysis_count += 1
        
        # Store iteration data
        iteration_data = {
            'iteration': iteration,
            'noise_probability': metrics.overall_noise_probability,
            'importance_score': metrics.importance_score,
            'confidence': metrics.confidence,
            'noise_types': [nt.value for nt in metrics.noise_types_detected],
            'timestamp': time.time()
        }
        
        segment.iteration_history.append(iteration_data)
    
    def _calculate_texture_variance(self, gray: np.ndarray) -> float:
        """Calculate texture variance using local patterns."""
        
        if gray.size == 0:
            return 0.0
        
        # Simple texture measure using local standard deviation
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        texture_variance = np.mean(local_variance)
        
        return texture_variance


class ZengezaEngine:
    """
    Main Zengeza engine for noise detection and quantification.
    
    Calculates probable amount of noise/garbage per segment per iteration
    to help focus reconstruction on important content.
    """
    
    def __init__(self):
        self.noise_analyzer = ZengezaNoiseAnalyzer()
        self.segments: Dict[str, ZengezaSegment] = {}
        self.global_noise_stats = {
            'average_noise_level': 0.5,
            'noise_distribution': {},
            'iteration_trends': []
        }
        
        logger.info("🗑️ Zengeza noise detection engine initialized")
    
    def analyze_image_noise(self, image: np.ndarray, 
                          segments: List[Dict[str, Any]],
                          context: Dict[str, Any],
                          iteration: int = 0) -> Dict[str, Any]:
        """
        Analyze noise across all segments in an image for a specific iteration.
        """
        
        logger.info(f"🗑️ Analyzing image noise for iteration {iteration}")
        
        # Convert segments to ZengezaSegments if needed
        zengeza_segments = []
        
        for i, seg_data in enumerate(segments):
            segment_id = seg_data.get('segment_id', f"segment_{i}")
            
            if segment_id not in self.segments:
                # Create new ZengezaSegment
                bbox = seg_data.get('bbox', (0, 0, 32, 32))
                x, y, w, h = bbox
                
                pixels = seg_data.get('pixels')
                if pixels is None:
                    pixels = image[y:y+h, x:x+w] if y+h <= image.shape[0] and x+w <= image.shape[1] else np.zeros((h, w, 3), dtype=np.uint8)
                
                mask = seg_data.get('mask')
                
                zengeza_segment = ZengezaSegment(
                    segment_id=segment_id,
                    bbox=bbox,
                    pixels=pixels,
                    mask=mask
                )
                
                self.segments[segment_id] = zengeza_segment
            
            zengeza_segments.append(self.segments[segment_id])
        
        # Analyze noise for each segment
        segment_results = {}
        noise_levels = []
        importance_scores = []
        
        for segment in zengeza_segments:
            # Create segment-specific context
            segment_context = context.copy()
            segment_context.update({
                'total_image_area': image.shape[0] * image.shape[1],
                'image_width': image.shape[1],
                'image_height': image.shape[0]
            })
            
            # Analyze noise
            noise_metrics = self.noise_analyzer.analyze_segment_noise(
                segment, segment_context, iteration
            )
            
            # Store results
            segment_results[segment.segment_id] = {
                'noise_metrics': noise_metrics,
                'bbox': segment.bbox,
                'noise_probability': noise_metrics.overall_noise_probability,
                'importance_score': noise_metrics.importance_score,
                'noise_level': noise_metrics.noise_level.value,
                'noise_types': [nt.value for nt in noise_metrics.noise_types_detected],
                'confidence': noise_metrics.confidence,
                'reconstruction_contribution': noise_metrics.reconstruction_contribution
            }
            
            noise_levels.append(noise_metrics.overall_noise_probability)
            importance_scores.append(noise_metrics.importance_score)
        
        # Calculate global statistics
        global_stats = self._calculate_global_noise_statistics(
            noise_levels, importance_scores, iteration
        )
        
        # Generate noise insights
        insights = self._generate_noise_insights(segment_results, global_stats)
        
        # Create prioritized segment list
        prioritized_segments = self._prioritize_segments_by_importance(segment_results)
        
        results = {
            'iteration': iteration,
            'total_segments': len(segment_results),
            'segment_noise_analysis': segment_results,
            'global_noise_statistics': global_stats,
            'prioritized_segments': prioritized_segments,
            'noise_insights': insights,
            'analysis_method': 'zengeza_noise_detection',
            'high_noise_segments': [
                seg_id for seg_id, data in segment_results.items()
                if data['noise_probability'] > 0.6
            ],
            'high_importance_segments': [
                seg_id for seg_id, data in segment_results.items()
                if data['importance_score'] > 0.7
            ]
        }
        
        logger.info(f"🗑️ Noise analysis complete: {len(results['high_noise_segments'])} high-noise segments, "
                   f"{len(results['high_importance_segments'])} high-importance segments")
        
        return results
    
    def _calculate_global_noise_statistics(self, noise_levels: List[float], 
                                         importance_scores: List[float],
                                         iteration: int) -> Dict[str, Any]:
        """Calculate global noise statistics across all segments."""
        
        if not noise_levels:
            return {'average_noise_level': 0.5, 'average_importance': 0.5}
        
        stats = {
            'average_noise_level': np.mean(noise_levels),
            'noise_std': np.std(noise_levels),
            'average_importance': np.mean(importance_scores),
            'importance_std': np.std(importance_scores),
            'min_noise': np.min(noise_levels),
            'max_noise': np.max(noise_levels),
            'noise_distribution': {
                'minimal': sum(1 for n in noise_levels if n < 0.2) / len(noise_levels),
                'low': sum(1 for n in noise_levels if 0.2 <= n < 0.4) / len(noise_levels),
                'moderate': sum(1 for n in noise_levels if 0.4 <= n < 0.6) / len(noise_levels),
                'high': sum(1 for n in noise_levels if 0.6 <= n < 0.8) / len(noise_levels),
                'critical': sum(1 for n in noise_levels if n >= 0.8) / len(noise_levels)
            },
            'iteration': iteration
        }
        
        # Update global tracking
        self.global_noise_stats['average_noise_level'] = stats['average_noise_level']
        self.global_noise_stats['noise_distribution'] = stats['noise_distribution']
        self.global_noise_stats['iteration_trends'].append({
            'iteration': iteration,
            'average_noise': stats['average_noise_level'],
            'average_importance': stats['average_importance']
        })
        
        # Keep only recent trends
        if len(self.global_noise_stats['iteration_trends']) > 20:
            self.global_noise_stats['iteration_trends'] = self.global_noise_stats['iteration_trends'][-20:]
        
        return stats
    
    def _prioritize_segments_by_importance(self, segment_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a prioritized list of segments based on importance (inverse of noise)."""
        
        segments_with_priority = []
        
        for segment_id, data in segment_results.items():
            priority_score = (
                data['importance_score'] * 0.4 +
                data['reconstruction_contribution'] * 0.3 +
                data['confidence'] * 0.2 +
                (1.0 - data['noise_probability']) * 0.1
            )
            
            segments_with_priority.append({
                'segment_id': segment_id,
                'priority_score': priority_score,
                'importance_score': data['importance_score'],
                'noise_probability': data['noise_probability'],
                'noise_level': data['noise_level'],
                'reconstruction_contribution': data['reconstruction_contribution']
            })
        
        # Sort by priority score (highest first)
        segments_with_priority.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return segments_with_priority
    
    def _generate_noise_insights(self, segment_results: Dict[str, Any], 
                               global_stats: Dict[str, Any]) -> List[str]:
        """Generate insights about noise patterns and recommendations."""
        
        insights = []
        
        # Global noise level insights
        avg_noise = global_stats['average_noise_level']
        if avg_noise < 0.3:
            insights.append("Low overall noise level - image has high information content")
        elif avg_noise > 0.7:
            insights.append("High overall noise level - significant garbage content detected")
        else:
            insights.append("Moderate noise level - mixed information and garbage content")
        
        # Noise distribution insights
        noise_dist = global_stats['noise_distribution']
        if noise_dist['critical'] > 0.2:
            insights.append(f"{noise_dist['critical']:.1%} of segments have critical noise levels")
        
        if noise_dist['minimal'] > 0.3:
            insights.append(f"{noise_dist['minimal']:.1%} of segments are highly important")
        
        # Segment-specific insights
        high_noise_count = sum(1 for data in segment_results.values() 
                              if data['noise_probability'] > 0.6)
        
        if high_noise_count > 0:
            insights.append(f"{high_noise_count} segments identified as mostly noise")
        
        # Reconstruction recommendations
        high_importance_count = sum(1 for data in segment_results.values() 
                                  if data['importance_score'] > 0.7)
        
        insights.append(f"Focus reconstruction efforts on {high_importance_count} high-importance segments")
        
        # Noise type analysis
        all_noise_types = []
        for data in segment_results.values():
            all_noise_types.extend(data['noise_types'])
        
        if all_noise_types:
            from collections import Counter
            noise_type_counts = Counter(all_noise_types)
            most_common_noise = noise_type_counts.most_common(1)[0]
            insights.append(f"Most common noise type: {most_common_noise[0]} ({most_common_noise[1]} segments)")
        
        return insights
    
    def get_noise_report(self) -> Dict[str, Any]:
        """Get comprehensive noise analysis report."""
        
        return {
            'total_segments_analyzed': len(self.segments),
            'global_noise_statistics': self.global_noise_stats,
            'segment_summaries': {
                seg_id: {
                    'analysis_count': seg.analysis_count,
                    'stable_estimate': seg.stable_noise_estimate,
                    'last_noise_level': seg.noise_metrics.noise_level.value if hasattr(seg, 'noise_metrics') else 'unknown'
                }
                for seg_id, seg in self.segments.items()
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize Zengeza engine
    zengeza = ZengezaEngine()
    
    # Create test image and segments
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    test_segments = [
        {
            'segment_id': 'segment_0',
            'bbox': (0, 0, 64, 64),
            'pixels': test_image[0:64, 0:64]
        },
        {
            'segment_id': 'segment_1', 
            'bbox': (64, 64, 64, 64),
            'pixels': test_image[64:128, 64:128]
        }
    ]
    
    context = {
        'reconstruction_quality': 0.75,
        'expected_complexity': 0.6
    }
    
    # Analyze noise
    results = zengeza.analyze_image_noise(test_image, test_segments, context, iteration=1)
    
    print(f"🗑️ Zengeza Noise Analysis Results:")
    print(f"Total segments: {results['total_segments']}")
    print(f"High-noise segments: {len(results['high_noise_segments'])}")
    print(f"High-importance segments: {len(results['high_importance_segments'])}")
    print(f"Average noise level: {results['global_noise_statistics']['average_noise_level']:.3f}")
    
    print(f"\nInsights:")
    for insight in results['noise_insights']:
        print(f"  • {insight}")
    
    print(f"\nPrioritized segments:")
    for i, seg in enumerate(results['prioritized_segments'][:3]):
        print(f"  {i+1}. {seg['segment_id']}: priority {seg['priority_score']:.3f}, "
              f"noise {seg['noise_probability']:.3f}") 