"""
Reverse Helicopter: Differential Visual Analysis Engine

This module implements the breakthrough "Reverse Helicopter" approach that extracts
meaningful knowledge by comparing actual images against domain expectations, rather
than describing everything from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image
import logging
from dataclasses import dataclass

from ..models.pakati_integration import PakatiGenerator
from ..models.diffusion_models import DiffusionComparator
from ..utils.image_processing import ImageProcessor
from ..utils.deviation_analysis import DeviationExtractor

logger = logging.getLogger(__name__)


@dataclass
class DeviationToken:
    """Represents a meaningful deviation from expectation"""
    region: str
    deviation_type: str
    severity: float
    location: Tuple[int, int, int, int]  # bbox
    description: str
    clinical_significance: float
    spatial_embedding: torch.Tensor
    visual_features: torch.Tensor


@dataclass
class ExpectationBaseline:
    """Contains the expected image and associated metadata"""
    ideal_image: Image.Image
    regional_annotations: Dict[str, Any]
    generation_params: Dict[str, Any]
    confidence_map: torch.Tensor


class ReverseHelicopter:
    """
    Main Reverse Helicopter engine that performs differential analysis
    
    Instead of describing everything in an image, this approach:
    1. Generates expected "ideal" image from domain description
    2. Compares actual image with expected baseline
    3. Extracts only meaningful deviations
    4. Converts deviations to expert-level knowledge tokens
    """
    
    def __init__(
        self,
        pakati_model: str,
        diffusion_model: str = "stabilityai/stable-diffusion-2-1",
        domain: str = "general",
        device: str = "auto"
    ):
        self.device = self._setup_device(device)
        self.domain = domain
        
        # Initialize components
        self.pakati_generator = PakatiGenerator(pakati_model, device=self.device)
        self.diffusion_comparator = DiffusionComparator(diffusion_model, device=self.device)
        self.image_processor = ImageProcessor()
        self.deviation_extractor = DeviationExtractor(domain=domain)
        
        # Domain-specific thresholds
        self.significance_threshold = self._get_domain_threshold(domain)
        
        logger.info(f"Initialized Reverse Helicopter for domain: {domain}")
    
    def extract_deviations(
        self,
        actual_image: str | Image.Image,
        expected_description: str,
        focus_regions: Optional[List[str]] = None,
        return_baseline: bool = False
    ) -> List[DeviationToken] | Tuple[List[DeviationToken], ExpectationBaseline]:
        """
        Core method: Extract meaningful deviations from actual vs expected
        
        Args:
            actual_image: Path to actual image or PIL Image
            expected_description: Domain description of what should be in image
            focus_regions: Specific regions to analyze (optional)
            return_baseline: Whether to return the generated baseline
            
        Returns:
            List of deviation tokens (and optionally the expectation baseline)
        """
        logger.info(f"Starting differential analysis for: {expected_description}")
        
        # Step 1: Load and preprocess actual image
        if isinstance(actual_image, str):
            actual_image = Image.open(actual_image)
        actual_tensor = self.image_processor.preprocess_image(actual_image)
        
        # Step 2: Generate expected baseline using Pakati
        baseline = self._generate_expectation_baseline(expected_description, focus_regions)
        
        # Step 3: Extract meaningful deviations
        deviations = self._extract_meaningful_deviations(
            actual_tensor, baseline, focus_regions
        )
        
        # Step 4: Filter by significance and convert to tokens
        significant_deviations = self._filter_significant_deviations(deviations)
        deviation_tokens = self._convert_to_tokens(significant_deviations)
        
        logger.info(f"Extracted {len(deviation_tokens)} significant deviations")
        
        if return_baseline:
            return deviation_tokens, baseline
        return deviation_tokens
    
    def generate_expert_analysis(
        self,
        deviation_tokens: List[DeviationToken],
        context: Optional[str] = None
    ) -> str:
        """
        Generate expert-level textual analysis from deviation tokens
        
        Args:
            deviation_tokens: List of extracted deviations
            context: Additional context for analysis
            
        Returns:
            Expert-level analysis string
        """
        if not deviation_tokens:
            return f"Image appears normal for {self.domain} domain expectations."
        
        # Sort by clinical significance
        sorted_tokens = sorted(
            deviation_tokens, 
            key=lambda x: x.clinical_significance, 
            reverse=True
        )
        
        analysis_parts = []
        
        # High significance findings
        high_sig = [t for t in sorted_tokens if t.clinical_significance > 0.7]
        if high_sig:
            analysis_parts.append("Key findings:")
            for token in high_sig:
                analysis_parts.append(f"- {token.description} (severity: {token.severity:.2f})")
        
        # Medium significance findings
        med_sig = [t for t in sorted_tokens if 0.3 < t.clinical_significance <= 0.7]
        if med_sig:
            analysis_parts.append("\nAdditional observations:")
            for token in med_sig:
                analysis_parts.append(f"- {token.description}")
        
        # Spatial relationships
        spatial_analysis = self._analyze_spatial_relationships(deviation_tokens)
        if spatial_analysis:
            analysis_parts.append(f"\nSpatial analysis: {spatial_analysis}")
        
        return "\n".join(analysis_parts)
    
    def _generate_expectation_baseline(
        self,
        description: str,
        focus_regions: Optional[List[str]] = None
    ) -> ExpectationBaseline:
        """Generate the expected baseline image using Pakati"""
        
        # Enhance description with domain-specific context
        enhanced_description = self._enhance_description_for_domain(description)
        
        # Generate ideal image using Pakati
        ideal_image = self.pakati_generator.generate_image(
            prompt=enhanced_description,
            regions=focus_regions,
            domain=self.domain
        )
        
        # Extract regional annotations
        regional_annotations = self.pakati_generator.get_regional_annotations()
        
        # Generate confidence map
        confidence_map = self._generate_confidence_map(ideal_image, regional_annotations)
        
        return ExpectationBaseline(
            ideal_image=ideal_image,
            regional_annotations=regional_annotations,
            generation_params={"prompt": enhanced_description, "domain": self.domain},
            confidence_map=confidence_map
        )
    
    def _extract_meaningful_deviations(
        self,
        actual_tensor: torch.Tensor,
        baseline: ExpectationBaseline,
        focus_regions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Extract deviations using advanced perceptual comparison"""
        
        baseline_tensor = self.image_processor.preprocess_image(baseline.ideal_image)
        
        # Multi-scale perceptual comparison
        deviations = []
        
        # Pixel-level differences
        pixel_diff = self._compute_pixel_differences(actual_tensor, baseline_tensor)
        
        # Feature-level differences using diffusion model embeddings
        feature_diff = self.diffusion_comparator.compute_feature_differences(
            actual_tensor, baseline_tensor
        )
        
        # Semantic-level differences
        semantic_diff = self._compute_semantic_differences(
            actual_tensor, baseline_tensor, baseline.regional_annotations
        )
        
        # Combine and localize deviations
        combined_deviations = self.deviation_extractor.combine_deviation_maps(
            pixel_diff, feature_diff, semantic_diff
        )
        
        # Extract specific deviation instances
        deviation_instances = self.deviation_extractor.extract_deviation_instances(
            combined_deviations, 
            focus_regions or list(baseline.regional_annotations.keys())
        )
        
        return deviation_instances
    
    def _filter_significant_deviations(
        self,
        deviations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter deviations by clinical/practical significance"""
        
        significant = []
        for deviation in deviations:
            # Compute significance based on domain-specific criteria
            significance = self._compute_clinical_significance(deviation)
            
            if significance > self.significance_threshold:
                deviation['clinical_significance'] = significance
                significant.append(deviation)
        
        return significant
    
    def _convert_to_tokens(
        self,
        deviations: List[Dict[str, Any]]
    ) -> List[DeviationToken]:
        """Convert deviation data to structured tokens"""
        
        tokens = []
        for deviation in deviations:
            token = DeviationToken(
                region=deviation['region'],
                deviation_type=deviation['type'],
                severity=deviation['severity'],
                location=deviation['bbox'],
                description=deviation['description'],
                clinical_significance=deviation['clinical_significance'],
                spatial_embedding=deviation['spatial_embedding'],
                visual_features=deviation['visual_features']
            )
            tokens.append(token)
        
        return tokens
    
    def _setup_device(self, device: str) -> str:
        """Setup computing device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _get_domain_threshold(self, domain: str) -> float:
        """Get significance threshold for domain"""
        thresholds = {
            "medical_imaging": 0.3,
            "quality_control": 0.4,
            "sports_analysis": 0.2,
            "general": 0.5
        }
        return thresholds.get(domain, 0.5)
    
    def _enhance_description_for_domain(self, description: str) -> str:
        """Add domain-specific context to description"""
        domain_enhancements = {
            "medical_imaging": "high quality medical imaging, diagnostic clarity, anatomically correct, professional medical photography",
            "quality_control": "perfect manufacturing quality, no defects, industrial standards",
            "sports_analysis": "optimal athletic form, proper biomechanics, professional sports photography"
        }
        
        enhancement = domain_enhancements.get(self.domain, "high quality, professional")
        return f"{description}, {enhancement}"
    
    def _generate_confidence_map(
        self,
        image: Image.Image,
        annotations: Dict[str, Any]
    ) -> torch.Tensor:
        """Generate confidence map for expectation baseline"""
        # Placeholder implementation
        h, w = image.size
        return torch.ones((1, h, w))
    
    def _compute_pixel_differences(
        self,
        actual: torch.Tensor,
        baseline: torch.Tensor
    ) -> torch.Tensor:
        """Compute pixel-level differences"""
        return torch.abs(actual - baseline)
    
    def _compute_semantic_differences(
        self,
        actual: torch.Tensor,
        baseline: torch.Tensor,
        annotations: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute semantic-level differences"""
        # Placeholder for semantic comparison
        return torch.abs(actual - baseline)
    
    def _compute_clinical_significance(self, deviation: Dict[str, Any]) -> float:
        """Compute clinical/practical significance of deviation"""
        # Domain-specific significance calculation
        base_significance = deviation['severity']
        
        # Adjust based on region importance
        region_weights = self._get_region_importance_weights()
        region_weight = region_weights.get(deviation['region'], 1.0)
        
        return base_significance * region_weight
    
    def _get_region_importance_weights(self) -> Dict[str, float]:
        """Get importance weights for different regions by domain"""
        weights = {
            "medical_imaging": {
                "lung_fields": 1.0,
                "heart": 0.9,
                "bones": 0.7,
                "soft_tissue": 0.6
            },
            "quality_control": {
                "critical_components": 1.0,
                "surface_finish": 0.8,
                "dimensions": 0.9
            }
        }
        return weights.get(self.domain, {})
    
    def _analyze_spatial_relationships(
        self,
        tokens: List[DeviationToken]
    ) -> str:
        """Analyze spatial relationships between deviations"""
        if len(tokens) < 2:
            return ""
        
        # Simple spatial relationship analysis
        relationships = []
        for i, token1 in enumerate(tokens):
            for token2 in tokens[i+1:]:
                if self._are_spatially_related(token1, token2):
                    relationships.append(f"{token1.region} and {token2.region} show correlated findings")
        
        return "; ".join(relationships) if relationships else ""
    
    def _are_spatially_related(
        self,
        token1: DeviationToken,
        token2: DeviationToken
    ) -> bool:
        """Check if two deviations are spatially related"""
        # Simple distance-based relationship
        x1, y1 = (token1.location[0] + token1.location[2]) // 2, (token1.location[1] + token1.location[3]) // 2
        x2, y2 = (token2.location[0] + token2.location[2]) // 2, (token2.location[1] + token2.location[3]) // 2
        
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return distance < 100  # pixels 