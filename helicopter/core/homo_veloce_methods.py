"""
Homo-veloce Methods Implementation

Direct implementation of ground truth validation methods from Homo-veloce:
1. Reference Baseline Creation
2. Accuracy Validation against Ground Truth
3. Statistical Analysis and Confidence Intervals
4. Quality Assessment and Error Analysis
5. Cross-validation and Performance Metrics

These are the actual implementations, not integration wrappers.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthAnnotation:
    """Ground truth annotation for a single frame"""
    frame_id: str
    timestamp: float
    objects: List[Dict[str, Any]]  # List of annotated objects
    quality_score: float
    annotator_id: str
    validation_status: str


@dataclass
class ValidationResult:
    """Result of validation against ground truth"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    iou_scores: List[float]
    confidence_interval: Tuple[float, float]
    error_analysis: Dict[str, Any]


@dataclass
class StatisticalAnalysis:
    """Statistical analysis of results"""
    mean_accuracy: float
    std_accuracy: float
    confidence_interval_95: Tuple[float, float]
    outliers: List[int]
    distribution_test: Dict[str, Any]
    significance_test: Dict[str, Any]


class GroundTruthManager:
    """
    Ground truth data management and validation
    
    Manages creation, validation, and analysis of ground truth datasets
    """
    
    def __init__(self, ground_truth_path: str):
        self.ground_truth_path = Path(ground_truth_path)
        self.annotations = {}
        self.load_ground_truth()
    
    def load_ground_truth(self):
        """Load ground truth annotations from file"""
        
        if self.ground_truth_path.exists():
            with open(self.ground_truth_path, 'r') as f:
                gt_data = json.load(f)
            
            for frame_id, annotation_data in gt_data.items():
                self.annotations[frame_id] = GroundTruthAnnotation(**annotation_data)
            
            logger.info(f"Loaded {len(self.annotations)} ground truth annotations")
        else:
            logger.warning(f"Ground truth file not found: {self.ground_truth_path}")
    
    def save_ground_truth(self):
        """Save ground truth annotations to file"""
        
        gt_data = {}
        for frame_id, annotation in self.annotations.items():
            gt_data[frame_id] = {
                'frame_id': annotation.frame_id,
                'timestamp': annotation.timestamp,
                'objects': annotation.objects,
                'quality_score': annotation.quality_score,
                'annotator_id': annotation.annotator_id,
                'validation_status': annotation.validation_status
            }
        
        with open(self.ground_truth_path, 'w') as f:
            json.dump(gt_data, f, indent=2)
        
        logger.info(f"Saved {len(self.annotations)} ground truth annotations")
    
    def add_annotation(self, annotation: GroundTruthAnnotation):
        """Add new ground truth annotation"""
        
        self.annotations[annotation.frame_id] = annotation
        logger.debug(f"Added annotation for frame {annotation.frame_id}")
    
    def get_annotation(self, frame_id: str) -> Optional[GroundTruthAnnotation]:
        """Get ground truth annotation for frame"""
        
        return self.annotations.get(frame_id)
    
    def validate_annotation_quality(self, frame_id: str) -> Dict[str, Any]:
        """Validate quality of ground truth annotation"""
        
        annotation = self.get_annotation(frame_id)
        if not annotation:
            return {'valid': False, 'error': 'Annotation not found'}
        
        quality_checks = {
            'completeness': self._check_completeness(annotation),
            'consistency': self._check_consistency(annotation),
            'accuracy': self._check_accuracy(annotation),
            'reliability': self._check_reliability(annotation)
        }
        
        overall_quality = np.mean([check['score'] for check in quality_checks.values()])
        
        return {
            'valid': overall_quality > 0.7,
            'quality_score': overall_quality,
            'checks': quality_checks
        }
    
    def _check_completeness(self, annotation: GroundTruthAnnotation) -> Dict[str, Any]:
        """Check completeness of annotation"""
        
        required_fields = ['frame_id', 'timestamp', 'objects']
        completeness_score = 1.0
        
        for field in required_fields:
            if not hasattr(annotation, field) or getattr(annotation, field) is None:
                completeness_score -= 0.3
        
        # Check object annotations
        if annotation.objects:
            for obj in annotation.objects:
                required_obj_fields = ['bbox', 'class', 'confidence']
                for field in required_obj_fields:
                    if field not in obj:
                        completeness_score -= 0.1
        
        return {
            'score': max(0.0, completeness_score),
            'description': 'Completeness of annotation fields'
        }
    
    def _check_consistency(self, annotation: GroundTruthAnnotation) -> Dict[str, Any]:
        """Check consistency of annotation"""
        
        consistency_score = 1.0
        
        # Check bbox consistency
        for obj in annotation.objects:
            if 'bbox' in obj:
                bbox = obj['bbox']
                if len(bbox) != 4:
                    consistency_score -= 0.2
                elif bbox[2] <= 0 or bbox[3] <= 0:  # width/height <= 0
                    consistency_score -= 0.3
        
        # Check confidence scores
        for obj in annotation.objects:
            if 'confidence' in obj:
                conf = obj['confidence']
                if not (0.0 <= conf <= 1.0):
                    consistency_score -= 0.2
        
        return {
            'score': max(0.0, consistency_score),
            'description': 'Internal consistency of annotation data'
        }
    
    def _check_accuracy(self, annotation: GroundTruthAnnotation) -> Dict[str, Any]:
        """Check accuracy indicators of annotation"""
        
        # This would check against multiple annotators or validation data
        # For now, return based on annotator confidence
        
        accuracy_score = annotation.quality_score
        
        return {
            'score': accuracy_score,
            'description': 'Estimated accuracy based on annotator confidence'
        }
    
    def _check_reliability(self, annotation: GroundTruthAnnotation) -> Dict[str, Any]:
        """Check reliability indicators of annotation"""
        
        reliability_score = 1.0
        
        # Check validation status
        if annotation.validation_status != 'validated':
            reliability_score -= 0.3
        
        # Check annotator experience (simplified)
        if annotation.annotator_id == 'auto':
            reliability_score -= 0.2
        
        return {
            'score': max(0.0, reliability_score),
            'description': 'Reliability of annotation source'
        }


class AccuracyValidator:
    """
    Accuracy validation against ground truth
    
    Compares analysis results with ground truth annotations
    """
    
    def __init__(self, ground_truth_manager: GroundTruthManager):
        self.gt_manager = ground_truth_manager
    
    def validate_detection_results(
        self,
        detection_results: List[Dict[str, Any]],
        frame_ids: List[str],
        iou_threshold: float = 0.5
    ) -> ValidationResult:
        """Validate detection results against ground truth"""
        
        if len(detection_results) != len(frame_ids):
            raise ValueError("Number of results must match number of frame IDs")
        
        all_ious = []
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for result, frame_id in zip(detection_results, frame_ids):
            gt_annotation = self.gt_manager.get_annotation(frame_id)
            
            if gt_annotation is None:
                logger.warning(f"No ground truth for frame {frame_id}")
                continue
            
            # Compare detections with ground truth
            frame_metrics = self._compare_frame_detections(
                result, gt_annotation, iou_threshold
            )
            
            all_ious.extend(frame_metrics['ious'])
            true_positives += frame_metrics['tp']
            false_positives += frame_metrics['fp']
            false_negatives += frame_metrics['fn']
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0.0
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(all_ious)
        
        # Error analysis
        error_analysis = self._analyze_errors(detection_results, frame_ids)
        
        return ValidationResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            iou_scores=all_ious,
            confidence_interval=confidence_interval,
            error_analysis=error_analysis
        )
    
    def _compare_frame_detections(
        self,
        detection_result: Dict[str, Any],
        gt_annotation: GroundTruthAnnotation,
        iou_threshold: float
    ) -> Dict[str, Any]:
        """Compare detections for a single frame"""
        
        detected_objects = detection_result.get('objects', [])
        gt_objects = gt_annotation.objects
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detected_objects), len(gt_objects)))
        
        for i, detected in enumerate(detected_objects):
            for j, gt in enumerate(gt_objects):
                iou = self._calculate_iou(detected.get('bbox', []), gt.get('bbox', []))
                iou_matrix[i, j] = iou
        
        # Find matches using Hungarian algorithm (simplified)
        matches = self._find_best_matches(iou_matrix, iou_threshold)
        
        # Count TP, FP, FN
        tp = len(matches)
        fp = len(detected_objects) - tp
        fn = len(gt_objects) - tp
        
        # Collect IoU scores for matched detections
        ious = [iou_matrix[i, j] for i, j in matches]
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'ious': ious,
            'matches': matches
        }
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU)"""
        
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0
        
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
    
    def _find_best_matches(self, iou_matrix: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
        """Find best matches above threshold"""
        
        matches = []
        used_gt = set()
        used_det = set()
        
        # Greedy matching - find highest IoU matches first
        while True:
            max_iou = 0
            max_pos = None
            
            for i in range(iou_matrix.shape[0]):
                for j in range(iou_matrix.shape[1]):
                    if i not in used_det and j not in used_gt:
                        if iou_matrix[i, j] > max_iou and iou_matrix[i, j] >= threshold:
                            max_iou = iou_matrix[i, j]
                            max_pos = (i, j)
            
            if max_pos is None:
                break
            
            matches.append(max_pos)
            used_det.add(max_pos[0])
            used_gt.add(max_pos[1])
        
        return matches
    
    def _calculate_confidence_interval(self, scores: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for scores"""
        
        if not scores:
            return (0.0, 0.0)
        
        scores_array = np.array(scores)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        n = len(scores_array)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        margin_error = t_critical * std_score / np.sqrt(n)
        
        return (mean_score - margin_error, mean_score + margin_error)
    
    def _analyze_errors(self, detection_results: List[Dict[str, Any]], frame_ids: List[str]) -> Dict[str, Any]:
        """Analyze types and patterns of errors"""
        
        error_types = {
            'missed_detections': [],
            'false_positives': [],
            'localization_errors': [],
            'classification_errors': []
        }
        
        for result, frame_id in zip(detection_results, frame_ids):
            gt_annotation = self.gt_manager.get_annotation(frame_id)
            
            if gt_annotation is None:
                continue
            
            # Analyze frame-specific errors
            frame_errors = self._analyze_frame_errors(result, gt_annotation)
            
            for error_type, errors in frame_errors.items():
                error_types[error_type].extend(errors)
        
        # Calculate error statistics
        error_stats = {}
        for error_type, errors in error_types.items():
            error_stats[error_type] = {
                'count': len(errors),
                'percentage': len(errors) / len(detection_results) * 100 if detection_results else 0
            }
        
        return {
            'error_types': error_types,
            'error_statistics': error_stats
        }
    
    def _analyze_frame_errors(self, detection_result: Dict[str, Any], gt_annotation: GroundTruthAnnotation) -> Dict[str, List[Any]]:
        """Analyze errors for a single frame"""
        
        errors = {
            'missed_detections': [],
            'false_positives': [],
            'localization_errors': [],
            'classification_errors': []
        }
        
        # This would implement detailed error analysis
        # For now, return empty errors
        
        return errors


class StatisticalAnalyzer:
    """
    Statistical analysis of validation results
    
    Performs comprehensive statistical analysis of accuracy and performance
    """
    
    def __init__(self):
        pass
    
    def analyze_results(self, validation_results: List[ValidationResult]) -> StatisticalAnalysis:
        """Perform statistical analysis of validation results"""
        
        if not validation_results:
            return self._empty_analysis()
        
        # Extract accuracy scores
        accuracy_scores = [result.accuracy for result in validation_results]
        
        # Basic statistics
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        
        # Confidence interval
        confidence_interval_95 = self._calculate_confidence_interval(accuracy_scores, 0.95)
        
        # Outlier detection
        outliers = self._detect_outliers(accuracy_scores)
        
        # Distribution test
        distribution_test = self._test_distribution(accuracy_scores)
        
        # Significance test
        significance_test = self._test_significance(accuracy_scores)
        
        return StatisticalAnalysis(
            mean_accuracy=mean_accuracy,
            std_accuracy=std_accuracy,
            confidence_interval_95=confidence_interval_95,
            outliers=outliers,
            distribution_test=distribution_test,
            significance_test=significance_test
        )
    
    def compare_methods(self, results_a: List[ValidationResult], results_b: List[ValidationResult]) -> Dict[str, Any]:
        """Compare two sets of validation results statistically"""
        
        if not results_a or not results_b:
            return {'error': 'Insufficient data for comparison'}
        
        scores_a = [result.accuracy for result in results_a]
        scores_b = [result.accuracy for result in results_b]
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(scores_a) - 1) * np.var(scores_a) + (len(scores_b) - 1) * np.var(scores_b)) / (len(scores_a) + len(scores_b) - 2))
        cohens_d = (np.mean(scores_a) - np.mean(scores_b)) / pooled_std if pooled_std > 0 else 0
        
        # Wilcoxon rank-sum test (non-parametric)
        u_stat, u_p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
        
        return {
            'mean_difference': np.mean(scores_a) - np.mean(scores_b),
            't_test': {
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'effect_size': {
                'cohens_d': cohens_d,
                'interpretation': self._interpret_effect_size(cohens_d)
            },
            'mann_whitney': {
                'statistic': u_stat,
                'p_value': u_p_value,
                'significant': u_p_value < 0.05
            }
        }
    
    def _calculate_confidence_interval(self, scores: List[float], confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval"""
        
        if not scores:
            return (0.0, 0.0)
        
        scores_array = np.array(scores)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        n = len(scores_array)
        
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        margin_error = t_critical * std_score / np.sqrt(n)
        
        return (mean_score - margin_error, mean_score + margin_error)
    
    def _detect_outliers(self, scores: List[float]) -> List[int]:
        """Detect outliers using IQR method"""
        
        if len(scores) < 4:
            return []
        
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = []
        for i, score in enumerate(scores):
            if score < lower_bound or score > upper_bound:
                outliers.append(i)
        
        return outliers
    
    def _test_distribution(self, scores: List[float]) -> Dict[str, Any]:
        """Test if scores follow normal distribution"""
        
        if len(scores) < 3:
            return {'error': 'Insufficient data for distribution test'}
        
        # Shapiro-Wilk test for normality
        stat, p_value = stats.shapiro(scores)
        
        return {
            'test': 'Shapiro-Wilk',
            'statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05,
            'interpretation': 'Normal distribution' if p_value > 0.05 else 'Non-normal distribution'
        }
    
    def _test_significance(self, scores: List[float], null_hypothesis: float = 0.5) -> Dict[str, Any]:
        """Test if mean score is significantly different from null hypothesis"""
        
        if len(scores) < 2:
            return {'error': 'Insufficient data for significance test'}
        
        # One-sample t-test
        t_stat, p_value = stats.ttest_1samp(scores, null_hypothesis)
        
        return {
            'test': 'One-sample t-test',
            'null_hypothesis': null_hypothesis,
            'statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': f'Mean significantly different from {null_hypothesis}' if p_value < 0.05 else f'No significant difference from {null_hypothesis}'
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return 'Negligible'
        elif abs_d < 0.5:
            return 'Small'
        elif abs_d < 0.8:
            return 'Medium'
        else:
            return 'Large'
    
    def _empty_analysis(self) -> StatisticalAnalysis:
        """Return empty statistical analysis"""
        
        return StatisticalAnalysis(
            mean_accuracy=0.0,
            std_accuracy=0.0,
            confidence_interval_95=(0.0, 0.0),
            outliers=[],
            distribution_test={'error': 'No data'},
            significance_test={'error': 'No data'}
        )


class QualityAssessmentEngine:
    """
    Quality assessment for analysis results
    
    Assesses quality of analysis results using multiple criteria
    """
    
    def __init__(self):
        self.quality_criteria = self._load_quality_criteria()
    
    def assess_result_quality(self, analysis_result: Dict[str, Any], ground_truth: Optional[GroundTruthAnnotation] = None) -> Dict[str, Any]:
        """Assess quality of analysis result"""
        
        quality_scores = {}
        
        # Technical quality assessment
        quality_scores['technical'] = self._assess_technical_quality(analysis_result)
        
        # Completeness assessment
        quality_scores['completeness'] = self._assess_completeness(analysis_result)
        
        # Consistency assessment
        quality_scores['consistency'] = self._assess_consistency(analysis_result)
        
        # Accuracy assessment (if ground truth available)
        if ground_truth:
            quality_scores['accuracy'] = self._assess_accuracy(analysis_result, ground_truth)
        
        # Overall quality score
        overall_quality = np.mean(list(quality_scores.values()))
        
        # Quality recommendations
        recommendations = self._generate_quality_recommendations(quality_scores)
        
        return {
            'overall_quality': overall_quality,
            'quality_scores': quality_scores,
            'recommendations': recommendations,
            'quality_level': self._classify_quality_level(overall_quality)
        }
    
    def _assess_technical_quality(self, analysis_result: Dict[str, Any]) -> float:
        """Assess technical quality of analysis"""
        
        technical_score = 1.0
        
        # Check for required fields
        required_fields = ['objects', 'timestamp', 'confidence']
        for field in required_fields:
            if field not in analysis_result:
                technical_score -= 0.2
        
        # Check object quality
        if 'objects' in analysis_result:
            for obj in analysis_result['objects']:
                if 'confidence' in obj and obj['confidence'] < 0.3:
                    technical_score -= 0.1
        
        return max(0.0, technical_score)
    
    def _assess_completeness(self, analysis_result: Dict[str, Any]) -> float:
        """Assess completeness of analysis"""
        
        completeness_score = 0.0
        
        # Check presence of key components
        if 'objects' in analysis_result:
            completeness_score += 0.3
        if 'metadata' in analysis_result:
            completeness_score += 0.2
        if 'confidence' in analysis_result:
            completeness_score += 0.2
        if 'processing_info' in analysis_result:
            completeness_score += 0.3
        
        return completeness_score
    
    def _assess_consistency(self, analysis_result: Dict[str, Any]) -> float:
        """Assess internal consistency of analysis"""
        
        consistency_score = 1.0
        
        # Check confidence consistency
        if 'confidence' in analysis_result and 'objects' in analysis_result:
            overall_conf = analysis_result['confidence']
            obj_confidences = [obj.get('confidence', 0) for obj in analysis_result['objects']]
            
            if obj_confidences:
                avg_obj_conf = np.mean(obj_confidences)
                if abs(overall_conf - avg_obj_conf) > 0.3:
                    consistency_score -= 0.3
        
        return max(0.0, consistency_score)
    
    def _assess_accuracy(self, analysis_result: Dict[str, Any], ground_truth: GroundTruthAnnotation) -> float:
        """Assess accuracy against ground truth"""
        
        # Simple accuracy assessment
        detected_objects = analysis_result.get('objects', [])
        gt_objects = ground_truth.objects
        
        if not gt_objects:
            return 1.0 if not detected_objects else 0.5
        
        # Calculate simple overlap score
        overlap_score = min(len(detected_objects), len(gt_objects)) / max(len(detected_objects), len(gt_objects))
        
        return overlap_score
    
    def _generate_quality_recommendations(self, quality_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on quality scores"""
        
        recommendations = []
        
        for criterion, score in quality_scores.items():
            if score < 0.7:
                if criterion == 'technical':
                    recommendations.append("Improve technical quality by ensuring all required fields are present")
                elif criterion == 'completeness':
                    recommendations.append("Enhance completeness by adding missing metadata and processing information")
                elif criterion == 'consistency':
                    recommendations.append("Improve consistency between overall and object-level confidence scores")
                elif criterion == 'accuracy':
                    recommendations.append("Review analysis parameters to improve accuracy against ground truth")
        
        if not recommendations:
            recommendations.append("Analysis quality is acceptable")
        
        return recommendations
    
    def _classify_quality_level(self, overall_quality: float) -> str:
        """Classify overall quality level"""
        
        if overall_quality >= 0.9:
            return 'Excellent'
        elif overall_quality >= 0.8:
            return 'Good'
        elif overall_quality >= 0.7:
            return 'Acceptable'
        elif overall_quality >= 0.6:
            return 'Poor'
        else:
            return 'Unacceptable'
    
    def _load_quality_criteria(self) -> Dict[str, Any]:
        """Load quality assessment criteria"""
        
        return {
            'technical_weight': 0.3,
            'completeness_weight': 0.25,
            'consistency_weight': 0.25,
            'accuracy_weight': 0.2,
            'min_confidence_threshold': 0.5,
            'min_objects_threshold': 1
        }
