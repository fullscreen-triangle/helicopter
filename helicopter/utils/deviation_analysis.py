"""
Deviation Analysis Utilities

This module provides utilities for extracting and analyzing meaningful deviations
between actual images and expected baselines in the Reverse Helicopter approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image
import cv2
import logging
from scipy import ndimage
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


class DeviationExtractor:
    """
    Extract meaningful deviations from multi-scale difference maps
    
    This class combines pixel-level, feature-level, and semantic-level
    differences to identify clinically/practically significant deviations.
    """
    
    def __init__(self, domain: str = "general"):
        self.domain = domain
        
        # Domain-specific parameters
        self.deviation_thresholds = self._get_domain_thresholds(domain)
        self.spatial_clustering_params = self._get_clustering_params(domain)
        
        logger.info(f"Initialized DeviationExtractor for domain: {domain}")
    
    def combine_deviation_maps(
        self,
        pixel_diff: torch.Tensor,
        feature_diff: torch.Tensor,
        semantic_diff: torch.Tensor,
        weights: Optional[Tuple[float, float, float]] = None
    ) -> torch.Tensor:
        """
        Combine multiple deviation maps into unified deviation map
        
        Args:
            pixel_diff: Pixel-level differences
            feature_diff: Feature-level differences  
            semantic_diff: Semantic-level differences
            weights: Weights for combining maps (pixel, feature, semantic)
            
        Returns:
            Combined deviation map
        """
        if weights is None:
            weights = self._get_default_weights()
        
        # Normalize all maps to [0, 1]
        pixel_norm = self._normalize_deviation_map(pixel_diff)
        feature_norm = self._normalize_deviation_map(feature_diff)
        semantic_norm = self._normalize_deviation_map(semantic_diff)
        
        # Weighted combination
        combined = (
            weights[0] * pixel_norm +
            weights[1] * feature_norm +
            weights[2] * semantic_norm
        )
        
        # Apply domain-specific post-processing
        combined = self._apply_domain_filtering(combined)
        
        return combined
    
    def extract_deviation_instances(
        self,
        deviation_map: torch.Tensor,
        focus_regions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract individual deviation instances from combined map
        
        Args:
            deviation_map: Combined deviation map
            focus_regions: Regions to focus analysis on
            
        Returns:
            List of deviation instances with metadata
        """
        deviation_instances = []
        
        # Convert to numpy for processing
        if isinstance(deviation_map, torch.Tensor):
            deviation_np = deviation_map.cpu().numpy()
        else:
            deviation_np = deviation_map
        
        # Threshold to find significant deviations
        threshold = self.deviation_thresholds['significance']
        binary_mask = deviation_np > threshold
        
        # Find connected components (deviation clusters)
        labeled_deviations, num_deviations = ndimage.label(binary_mask)
        
        for deviation_id in range(1, num_deviations + 1):
            deviation_mask = labeled_deviations == deviation_id
            
            # Extract deviation properties
            deviation_data = self._extract_deviation_properties(
                deviation_mask, deviation_np, deviation_id
            )
            
            # Filter by region relevance
            if self._is_relevant_deviation(deviation_data, focus_regions):
                deviation_instances.append(deviation_data)
        
        # Sort by severity
        deviation_instances.sort(key=lambda x: x['severity'], reverse=True)
        
        return deviation_instances
    
    def _extract_deviation_properties(
        self,
        mask: np.ndarray,
        deviation_map: np.ndarray,
        deviation_id: int
    ) -> Dict[str, Any]:
        """Extract properties of a single deviation"""
        
        # Find bounding box
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return {}
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        bbox = (x_min, y_min, x_max, y_max)
        
        # Calculate severity metrics
        deviation_values = deviation_map[mask]
        severity = float(np.mean(deviation_values))
        max_severity = float(np.max(deviation_values))
        area = int(np.sum(mask))
        
        # Determine deviation type
        deviation_type = self._classify_deviation_type(
            mask, deviation_map, bbox
        )
        
        # Generate description
        description = self._generate_deviation_description(
            deviation_type, severity, area, bbox
        )
        
        # Create spatial embedding (placeholder)
        spatial_embedding = self._create_spatial_embedding(mask, bbox)
        
        # Extract visual features (placeholder)
        visual_features = self._extract_visual_features(mask, deviation_map)
        
        return {
            'id': deviation_id,
            'type': deviation_type,
            'severity': severity,
            'max_severity': max_severity,
            'area': area,
            'bbox': bbox,
            'description': description,
            'region': self._determine_region(bbox),
            'spatial_embedding': spatial_embedding,
            'visual_features': visual_features,
            'mask': mask
        }
    
    def _classify_deviation_type(
        self,
        mask: np.ndarray,
        deviation_map: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> str:
        """Classify the type of deviation"""
        
        # Analyze shape characteristics
        area = np.sum(mask)
        perimeter = self._calculate_perimeter(mask)
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        # Analyze intensity characteristics
        deviation_values = deviation_map[mask]
        intensity_std = np.std(deviation_values)
        
        # Domain-specific classification
        if self.domain == "medical_imaging":
            return self._classify_medical_deviation(
                area, circularity, intensity_std
            )
        elif self.domain == "quality_control":
            return self._classify_quality_deviation(
                area, circularity, intensity_std
            )
        else:
            return self._classify_general_deviation(
                area, circularity, intensity_std
            )
    
    def _classify_medical_deviation(
        self,
        area: int,
        circularity: float,
        intensity_std: float
    ) -> str:
        """Classify medical imaging deviations"""
        
        if area < 100:
            return "small_nodule" if circularity > 0.7 else "artifact"
        elif area < 1000:
            if circularity > 0.6:
                return "mass" if intensity_std > 0.1 else "cyst"
            else:
                return "infiltrate" if intensity_std > 0.2 else "atelectasis"
        else:
            return "consolidation" if intensity_std > 0.15 else "pneumothorax"
    
    def _classify_quality_deviation(
        self,
        area: int,
        circularity: float,
        intensity_std: float
    ) -> str:
        """Classify quality control deviations"""
        
        if area < 50:
            return "scratch" if circularity < 0.3 else "pit"
        elif area < 500:
            return "dent" if circularity > 0.5 else "crack"
        else:
            return "major_defect"
    
    def _classify_general_deviation(
        self,
        area: int,
        circularity: float,
        intensity_std: float
    ) -> str:
        """Classify general deviations"""
        
        if area < 100:
            return "minor_variation"
        elif area < 1000:
            return "moderate_difference"
        else:
            return "major_difference"
    
    def _generate_deviation_description(
        self,
        deviation_type: str,
        severity: float,
        area: int,
        bbox: Tuple[int, int, int, int]
    ) -> str:
        """Generate human-readable description of deviation"""
        
        # Severity qualifiers
        if severity > 0.8:
            severity_desc = "significant"
        elif severity > 0.5:
            severity_desc = "moderate"
        else:
            severity_desc = "mild"
        
        # Size qualifiers
        if area > 1000:
            size_desc = "large"
        elif area > 100:
            size_desc = "medium-sized"
        else:
            size_desc = "small"
        
        # Location description
        x_center = (bbox[0] + bbox[2]) // 2
        y_center = (bbox[1] + bbox[3]) // 2
        location_desc = self._describe_location(x_center, y_center)
        
        # Domain-specific descriptions
        type_descriptions = {
            "medical_imaging": {
                "mass": f"{severity_desc} {size_desc} mass",
                "nodule": f"{severity_desc} {size_desc} nodule",
                "infiltrate": f"{severity_desc} infiltrative changes",
                "consolidation": f"{severity_desc} consolidation",
                "pneumothorax": f"{severity_desc} pneumothorax"
            },
            "quality_control": {
                "scratch": f"{severity_desc} surface scratch",
                "dent": f"{severity_desc} dent or depression",
                "crack": f"{severity_desc} crack or fissure",
                "major_defect": f"{severity_desc} structural defect"
            }
        }
        
        domain_desc = type_descriptions.get(self.domain, {})
        base_desc = domain_desc.get(deviation_type, f"{severity_desc} {deviation_type}")
        
        return f"{base_desc} in {location_desc}"
    
    def _determine_region(self, bbox: Tuple[int, int, int, int]) -> str:
        """Determine which anatomical/functional region the deviation is in"""
        
        x_center = (bbox[0] + bbox[2]) // 2
        y_center = (bbox[1] + bbox[3]) // 2
        
        # Domain-specific region mapping (simplified)
        if self.domain == "medical_imaging":
            # Assume chest X-ray layout
            if y_center < 200:
                return "upper_fields"
            elif y_center > 400:
                return "lower_fields"
            elif x_center < 300:
                return "left_lung"
            else:
                return "right_lung"
        else:
            return "central_region"
    
    def _describe_location(self, x: int, y: int) -> str:
        """Describe location in human terms"""
        
        # Simple quadrant description
        if x < 256 and y < 256:
            return "upper left region"
        elif x >= 256 and y < 256:
            return "upper right region"
        elif x < 256 and y >= 256:
            return "lower left region"
        else:
            return "lower right region"
    
    def _calculate_perimeter(self, mask: np.ndarray) -> int:
        """Calculate perimeter of binary mask"""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return sum(cv2.arcLength(contour, True) for contour in contours)
    
    def _create_spatial_embedding(
        self,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> torch.Tensor:
        """Create spatial embedding for deviation (placeholder)"""
        
        # Simple spatial features
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        spatial_features = torch.tensor([
            center_x / 512,  # Normalized center x
            center_y / 512,  # Normalized center y
            width / 512,     # Normalized width
            height / 512,    # Normalized height
            np.sum(mask) / (512 * 512)  # Normalized area
        ], dtype=torch.float32)
        
        return spatial_features
    
    def _extract_visual_features(
        self,
        mask: np.ndarray,
        deviation_map: np.ndarray
    ) -> torch.Tensor:
        """Extract visual features from deviation (placeholder)"""
        
        deviation_values = deviation_map[mask]
        
        visual_features = torch.tensor([
            float(np.mean(deviation_values)),
            float(np.std(deviation_values)),
            float(np.min(deviation_values)),
            float(np.max(deviation_values)),
            float(np.median(deviation_values))
        ], dtype=torch.float32)
        
        return visual_features
    
    def _normalize_deviation_map(self, deviation_map: torch.Tensor) -> torch.Tensor:
        """Normalize deviation map to [0, 1] range"""
        min_val = torch.min(deviation_map)
        max_val = torch.max(deviation_map)
        
        if max_val > min_val:
            return (deviation_map - min_val) / (max_val - min_val)
        else:
            return torch.zeros_like(deviation_map)
    
    def _apply_domain_filtering(self, deviation_map: torch.Tensor) -> torch.Tensor:
        """Apply domain-specific filtering to deviation map"""
        
        # Simple Gaussian smoothing for now
        if len(deviation_map.shape) == 2:
            deviation_map = deviation_map.unsqueeze(0).unsqueeze(0)
        
        # Apply Gaussian filter
        kernel_size = 5
        sigma = 1.0
        kernel = self._gaussian_kernel(kernel_size, sigma).to(deviation_map.device)
        
        filtered = F.conv2d(
            deviation_map,
            kernel.unsqueeze(0).unsqueeze(0),
            padding=kernel_size // 2
        )
        
        return filtered.squeeze()
    
    def _gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian kernel for smoothing"""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        
        kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        
        return kernel.outer(kernel)
    
    def _is_relevant_deviation(
        self,
        deviation_data: Dict[str, Any],
        focus_regions: List[str]
    ) -> bool:
        """Check if deviation is relevant for analysis"""
        
        # Check minimum area
        if deviation_data['area'] < self.deviation_thresholds['min_area']:
            return False
        
        # Check minimum severity
        if deviation_data['severity'] < self.deviation_thresholds['min_severity']:
            return False
        
        # Check region relevance
        if focus_regions and deviation_data['region'] not in focus_regions:
            return False
        
        return True
    
    def _get_domain_thresholds(self, domain: str) -> Dict[str, float]:
        """Get domain-specific thresholds"""
        
        thresholds = {
            "medical_imaging": {
                "significance": 0.3,
                "min_area": 25,
                "min_severity": 0.2
            },
            "quality_control": {
                "significance": 0.4,
                "min_area": 10,
                "min_severity": 0.3
            },
            "sports_analysis": {
                "significance": 0.2,
                "min_area": 50,
                "min_severity": 0.15
            },
            "general": {
                "significance": 0.5,
                "min_area": 20,
                "min_severity": 0.25
            }
        }
        
        return thresholds.get(domain, thresholds["general"])
    
    def _get_clustering_params(self, domain: str) -> Dict[str, Any]:
        """Get clustering parameters for spatial analysis"""
        
        params = {
            "medical_imaging": {"eps": 50, "min_samples": 3},
            "quality_control": {"eps": 20, "min_samples": 2},
            "sports_analysis": {"eps": 100, "min_samples": 5},
            "general": {"eps": 30, "min_samples": 3}
        }
        
        return params.get(domain, params["general"])
    
    def _get_default_weights(self) -> Tuple[float, float, float]:
        """Get default weights for combining deviation maps"""
        
        weights = {
            "medical_imaging": (0.2, 0.5, 0.3),  # Favor feature-level
            "quality_control": (0.4, 0.4, 0.2),  # Balance pixel and feature
            "sports_analysis": (0.3, 0.4, 0.3),  # Balanced approach
            "general": (0.33, 0.33, 0.34)        # Equal weights
        }
        
        return weights.get(self.domain, weights["general"]) 