"""
Pakati Methods Implementation

Direct implementation of Pakati reverse analysis methods:
1. Regional Control Reversal - Extract control vectors from generated images
2. Diffusion Model Analysis - Reverse diffusion process to extract semantics
3. Semantic Extraction - Extract semantic features and representations
4. Visual Token Generation - Generate tokens from visual features
5. Text-to-Image Reversal - Reverse the text-to-image generation process

These are the actual implementations, not integration wrappers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from PIL import Image
import cv2
from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline
import json

logger = logging.getLogger(__name__)


@dataclass
class RegionalControl:
    """Regional control information"""
    region_id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    control_vector: np.ndarray
    semantic_label: str
    confidence: float
    generation_parameters: Dict[str, Any]


@dataclass
class SemanticFeature:
    """Semantic feature extracted from image"""
    feature_type: str
    feature_vector: np.ndarray
    spatial_location: Tuple[int, int]
    semantic_description: str
    confidence: float


@dataclass
class VisualToken:
    """Visual token representation"""
    token_id: int
    embedding: np.ndarray
    spatial_attention: np.ndarray
    semantic_meaning: str
    visual_concept: str
    generation_strength: float


@dataclass
class PakatiReverseResult:
    """Result from Pakati reverse analysis"""
    regional_controls: List[RegionalControl]
    semantic_features: List[SemanticFeature]
    visual_tokens: List[VisualToken]
    generation_text: str
    reverse_confidence: float
    analysis_metadata: Dict[str, Any]


class RegionalControlExtractor:
    """
    Regional Control Extractor
    
    Extracts regional control information by reversing the regional control process
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Move to device
        self.clip_model.to(self.device)
        
        logger.info(f"Initialized Regional Control Extractor on {self.device}")
    
    def extract_regional_controls(self, image: np.ndarray, num_regions: int = 8) -> List[RegionalControl]:
        """Extract regional control information from image"""
        
        # Convert image to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Segment image into regions
        regions = self._segment_image_regions(image_rgb, num_regions)
        
        regional_controls = []
        
        for i, region_info in enumerate(regions):
            # Extract control vector for this region
            control_vector = self._extract_region_control_vector(
                image_rgb, region_info['bbox'], region_info['mask']
            )
            
            # Extract semantic information
            semantic_label = self._analyze_region_semantics(
                image_rgb, region_info['bbox']
            )
            
            # Calculate confidence
            confidence = self._calculate_region_confidence(
                control_vector, region_info['mask']
            )
            
            # Estimate generation parameters
            generation_params = self._estimate_generation_parameters(
                control_vector, semantic_label
            )
            
            regional_control = RegionalControl(
                region_id=i,
                bbox=region_info['bbox'],
                control_vector=control_vector,
                semantic_label=semantic_label,
                confidence=confidence,
                generation_parameters=generation_params
            )
            
            regional_controls.append(regional_control)
        
        return regional_controls
    
    def _segment_image_regions(self, image: np.ndarray, num_regions: int) -> List[Dict[str, Any]]:
        """Segment image into regions for control extraction"""
        
        h, w = image.shape[:2]
        regions = []
        
        # Simple grid-based segmentation (could be replaced with more sophisticated methods)
        grid_size = int(np.sqrt(num_regions))
        region_h = h // grid_size
        region_w = w // grid_size
        
        region_id = 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                if region_id >= num_regions:
                    break
                
                # Calculate region bounds
                y1 = i * region_h
                y2 = min((i + 1) * region_h, h)
                x1 = j * region_w
                x2 = min((j + 1) * region_w, w)
                
                # Create mask for this region
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                
                # Apply adaptive segmentation within region
                region_mask = self._refine_region_mask(image[y1:y2, x1:x2], mask[y1:y2, x1:x2])
                mask[y1:y2, x1:x2] = region_mask
                
                regions.append({
                    'region_id': region_id,
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'mask': mask
                })
                
                region_id += 1
        
        return regions
    
    def _refine_region_mask(self, region_image: np.ndarray, initial_mask: np.ndarray) -> np.ndarray:
        """Refine region mask using image features"""
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(region_image, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel)
        
        # Incorporate edge information
        refined_mask = cv2.bitwise_and(refined_mask, cv2.bitwise_not(edges))
        
        return refined_mask
    
    def _extract_region_control_vector(self, image: np.ndarray, bbox: Tuple[int, int, int, int], mask: np.ndarray) -> np.ndarray:
        """Extract control vector for a specific region"""
        
        x, y, w, h = bbox
        
        # Extract region from image
        region_image = image[y:y+h, x:x+w]
        region_mask = mask[y:y+h, x:x+w]
        
        # Convert to PIL for CLIP processing
        pil_image = Image.fromarray(region_image)
        
        # Process with CLIP
        inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            control_vector = image_features.cpu().numpy().flatten()
        
        # Apply mask weighting
        mask_weight = np.sum(region_mask > 0) / (region_mask.shape[0] * region_mask.shape[1])
        control_vector = control_vector * mask_weight
        
        return control_vector
    
    def _analyze_region_semantics(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """Analyze semantic content of region"""
        
        x, y, w, h = bbox
        region_image = image[y:y+h, x:x+w]
        
        # Use CLIP to analyze semantic content
        pil_image = Image.fromarray(region_image)
        
        # Predefined semantic categories
        semantic_categories = [
            "person", "face", "body", "hands", "clothing",
            "background", "sky", "landscape", "building", "object",
            "animal", "vehicle", "furniture", "food", "nature"
        ]
        
        # Process with CLIP
        inputs = self.clip_processor(
            text=semantic_categories,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Get most likely semantic category
            max_idx = torch.argmax(probs, dim=1).item()
            semantic_label = semantic_categories[max_idx]
        
        return semantic_label
    
    def _calculate_region_confidence(self, control_vector: np.ndarray, mask: np.ndarray) -> float:
        """Calculate confidence for region control extraction"""
        
        # Confidence based on control vector magnitude and mask coverage
        vector_magnitude = np.linalg.norm(control_vector)
        mask_coverage = np.sum(mask > 0) / mask.size
        
        # Normalize and combine
        confidence = (vector_magnitude / 100.0) * mask_coverage
        
        return min(1.0, confidence)
    
    def _estimate_generation_parameters(self, control_vector: np.ndarray, semantic_label: str) -> Dict[str, Any]:
        """Estimate generation parameters from control vector"""
        
        # Simple parameter estimation based on vector characteristics
        vector_mean = np.mean(control_vector)
        vector_std = np.std(control_vector)
        
        # Estimate guidance scale
        guidance_scale = 7.5 + (vector_std * 5.0)
        
        # Estimate number of inference steps
        inference_steps = 20 + int(vector_mean * 30)
        
        # Estimate strength
        strength = min(1.0, max(0.1, np.abs(vector_mean)))
        
        return {
            'guidance_scale': guidance_scale,
            'num_inference_steps': inference_steps,
            'strength': strength,
            'semantic_focus': semantic_label
        }


class DiffusionReverseAnalyzer:
    """
    Diffusion Reverse Analyzer
    
    Analyzes images by reversing the diffusion process to extract generation semantics
    """
    
    def __init__(self, model_name: str = "stabilityai/stable-diffusion-2-1"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load diffusion model
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.pipeline.to(self.device)
        except Exception as e:
            logger.warning(f"Could not load diffusion model: {e}")
            self.pipeline = None
        
        logger.info("Initialized Diffusion Reverse Analyzer")
    
    def reverse_diffusion_process(self, image: np.ndarray, num_steps: int = 20) -> Dict[str, Any]:
        """Reverse the diffusion process to extract generation information"""
        
        if self.pipeline is None:
            return {'error': 'Diffusion model not available'}
        
        # Convert image to tensor
        image_tensor = self._prepare_image_tensor(image)
        
        # Perform reverse diffusion
        latent_trajectory = self._extract_latent_trajectory(image_tensor, num_steps)
        
        # Analyze noise patterns
        noise_analysis = self._analyze_noise_patterns(latent_trajectory)
        
        # Extract semantic information
        semantic_info = self._extract_semantic_information(latent_trajectory)
        
        # Estimate original prompt
        estimated_prompt = self._estimate_generation_prompt(semantic_info, noise_analysis)
        
        return {
            'latent_trajectory': latent_trajectory,
            'noise_analysis': noise_analysis,
            'semantic_info': semantic_info,
            'estimated_prompt': estimated_prompt,
            'reverse_confidence': self._calculate_reverse_confidence(noise_analysis)
        }
    
    def _prepare_image_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Prepare image as tensor for processing"""
        
        # Resize to model input size
        image_resized = cv2.resize(image, (512, 512))
        
        # Convert to RGB if needed
        if len(image_resized.shape) == 3 and image_resized.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_resized
        
        # Normalize to [-1, 1]
        image_normalized = (image_rgb.astype(np.float32) / 127.5) - 1.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
    
    def _extract_latent_trajectory(self, image_tensor: torch.Tensor, num_steps: int) -> List[torch.Tensor]:
        """Extract latent trajectory by reversing diffusion process"""
        
        trajectory = []
        
        # Encode image to latent space
        with torch.no_grad():
            latent = self.pipeline.vae.encode(image_tensor).latent_dist.sample()
            latent = latent * self.pipeline.vae.config.scaling_factor
        
        trajectory.append(latent.clone())
        
        # Reverse diffusion steps
        timesteps = self.pipeline.scheduler.timesteps[-num_steps:]
        
        for i, timestep in enumerate(timesteps):
            with torch.no_grad():
                # Add noise progressively
                noise = torch.randn_like(latent)
                noisy_latent = self.pipeline.scheduler.add_noise(latent, noise, timestep)
                trajectory.append(noisy_latent.clone())
        
        return trajectory
    
    def _analyze_noise_patterns(self, latent_trajectory: List[torch.Tensor]) -> Dict[str, Any]:
        """Analyze noise patterns in latent trajectory"""
        
        noise_characteristics = {
            'noise_levels': [],
            'frequency_analysis': [],
            'spatial_patterns': []
        }
        
        for i in range(1, len(latent_trajectory)):
            # Calculate noise level
            noise_diff = latent_trajectory[i] - latent_trajectory[i-1]
            noise_level = torch.std(noise_diff).item()
            noise_characteristics['noise_levels'].append(noise_level)
            
            # Frequency analysis
            fft = torch.fft.fft2(noise_diff.squeeze())
            freq_magnitude = torch.abs(fft).mean().item()
            noise_characteristics['frequency_analysis'].append(freq_magnitude)
            
            # Spatial pattern analysis
            spatial_var = torch.var(noise_diff, dim=[2, 3]).mean().item()
            noise_characteristics['spatial_patterns'].append(spatial_var)
        
        return noise_characteristics
    
    def _extract_semantic_information(self, latent_trajectory: List[torch.Tensor]) -> Dict[str, Any]:
        """Extract semantic information from latent trajectory"""
        
        semantic_info = {
            'content_evolution': [],
            'semantic_stability': 0.0,
            'feature_importance': []
        }
        
        # Analyze content evolution
        for latent in latent_trajectory:
            # Simple content measure
            content_measure = torch.mean(torch.abs(latent)).item()
            semantic_info['content_evolution'].append(content_measure)
        
        # Calculate semantic stability
        content_evolution = semantic_info['content_evolution']
        if len(content_evolution) > 1:
            stability = 1.0 - (np.std(content_evolution) / np.mean(content_evolution))
            semantic_info['semantic_stability'] = max(0.0, stability)
        
        # Feature importance analysis
        initial_latent = latent_trajectory[0]
        feature_importance = torch.std(initial_latent, dim=[0, 2, 3]).cpu().numpy()
        semantic_info['feature_importance'] = feature_importance.tolist()
        
        return semantic_info
    
    def _estimate_generation_prompt(self, semantic_info: Dict[str, Any], noise_analysis: Dict[str, Any]) -> str:
        """Estimate the generation prompt from analysis"""
        
        # This is a simplified estimation
        # In practice, this would use more sophisticated methods
        
        # Analyze content characteristics
        content_evolution = semantic_info['content_evolution']
        stability = semantic_info['semantic_stability']
        
        # Generate prompt components based on analysis
        prompt_components = []
        
        if stability > 0.7:
            prompt_components.append("detailed")
        
        if np.mean(content_evolution) > 0.5:
            prompt_components.append("high quality")
        
        # Analyze noise patterns
        noise_levels = noise_analysis['noise_levels']
        if np.mean(noise_levels) > 0.3:
            prompt_components.append("artistic")
        
        # Combine components
        if prompt_components:
            estimated_prompt = ", ".join(prompt_components)
        else:
            estimated_prompt = "image"
        
        return estimated_prompt
    
    def _calculate_reverse_confidence(self, noise_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in reverse analysis"""
        
        # Simple confidence calculation based on noise consistency
        noise_levels = noise_analysis['noise_levels']
        
        if not noise_levels:
            return 0.0
        
        # Confidence based on noise level consistency
        noise_std = np.std(noise_levels)
        noise_mean = np.mean(noise_levels)
        
        if noise_mean > 0:
            consistency = 1.0 - (noise_std / noise_mean)
            confidence = max(0.0, min(1.0, consistency))
        else:
            confidence = 0.0
        
        return confidence


class SemanticExtractor:
    """
    Semantic Feature Extractor
    
    Extracts semantic features from images using multiple analysis methods
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)
        
        logger.info("Initialized Semantic Extractor")
    
    def extract_semantic_features(self, image: np.ndarray, feature_resolution: int = 16) -> List[SemanticFeature]:
        """Extract semantic features from image"""
        
        # Extract features at multiple scales
        features = []
        
        # Global semantic features
        global_features = self._extract_global_semantic_features(image)
        features.extend(global_features)
        
        # Local semantic features
        local_features = self._extract_local_semantic_features(image, feature_resolution)
        features.extend(local_features)
        
        # Object-level semantic features
        object_features = self._extract_object_semantic_features(image)
        features.extend(object_features)
        
        return features
    
    def _extract_global_semantic_features(self, image: np.ndarray) -> List[SemanticFeature]:
        """Extract global semantic features from entire image"""
        
        # Convert to PIL
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb)
        
        # Process with CLIP
        inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            feature_vector = image_features.cpu().numpy().flatten()
        
        # Analyze semantic content
        semantic_description = self._analyze_global_semantics(image_rgb)
        
        global_feature = SemanticFeature(
            feature_type="global",
            feature_vector=feature_vector,
            spatial_location=(image.shape[1] // 2, image.shape[0] // 2),
            semantic_description=semantic_description,
            confidence=0.8  # High confidence for global features
        )
        
        return [global_feature]
    
    def _extract_local_semantic_features(self, image: np.ndarray, resolution: int) -> List[SemanticFeature]:
        """Extract local semantic features using patch-based analysis"""
        
        h, w = image.shape[:2]
        patch_h = h // resolution
        patch_w = w // resolution
        
        local_features = []
        
        for i in range(resolution):
            for j in range(resolution):
                # Extract patch
                y1 = i * patch_h
                y2 = min((i + 1) * patch_h, h)
                x1 = j * patch_w
                x2 = min((j + 1) * patch_w, w)
                
                patch = image[y1:y2, x1:x2]
                
                # Skip if patch is too small
                if patch.shape[0] < 32 or patch.shape[1] < 32:
                    continue
                
                # Extract features from patch
                patch_feature = self._extract_patch_features(patch, (x1 + x2) // 2, (y1 + y2) // 2)
                
                if patch_feature is not None:
                    local_features.append(patch_feature)
        
        return local_features
    
    def _extract_patch_features(self, patch: np.ndarray, center_x: int, center_y: int) -> Optional[SemanticFeature]:
        """Extract features from image patch"""
        
        # Convert to RGB if needed
        if len(patch.shape) == 3 and patch.shape[2] == 3:
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        else:
            patch_rgb = patch
        
        # Check if patch has enough content
        if np.std(patch_rgb) < 10:  # Too uniform
            return None
        
        # Process with CLIP
        pil_patch = Image.fromarray(patch_rgb)
        inputs = self.clip_processor(images=pil_patch, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            patch_features = self.clip_model.get_image_features(**inputs)
            feature_vector = patch_features.cpu().numpy().flatten()
        
        # Analyze patch semantics
        semantic_description = self._analyze_patch_semantics(patch_rgb)
        
        # Calculate confidence
        confidence = min(1.0, np.std(patch_rgb) / 50.0)
        
        return SemanticFeature(
            feature_type="local",
            feature_vector=feature_vector,
            spatial_location=(center_x, center_y),
            semantic_description=semantic_description,
            confidence=confidence
        )
    
    def _extract_object_semantic_features(self, image: np.ndarray) -> List[SemanticFeature]:
        """Extract object-level semantic features"""
        
        # Simple object detection using contours
        # In practice, this would use more sophisticated object detection
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        object_features = []
        
        for i, contour in enumerate(contours):
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < 1000:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract object region
            object_region = image[y:y+h, x:x+w]
            
            # Extract features
            object_feature = self._extract_patch_features(object_region, x + w//2, y + h//2)
            
            if object_feature is not None:
                object_feature.feature_type = "object"
                object_features.append(object_feature)
        
        return object_features
    
    def _analyze_global_semantics(self, image: np.ndarray) -> str:
        """Analyze global semantic content"""
        
        # Predefined global categories
        global_categories = [
            "portrait", "landscape", "urban", "indoor", "outdoor",
            "nature", "abstract", "artistic", "photographic", "realistic"
        ]
        
        pil_image = Image.fromarray(image)
        
        # Process with CLIP
        inputs = self.clip_processor(
            text=global_categories,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Get most likely category
            max_idx = torch.argmax(probs, dim=1).item()
            semantic_description = global_categories[max_idx]
        
        return semantic_description
    
    def _analyze_patch_semantics(self, patch: np.ndarray) -> str:
        """Analyze semantic content of patch"""
        
        # Predefined patch categories
        patch_categories = [
            "face", "eye", "hair", "skin", "clothing",
            "background", "texture", "object", "detail", "edge"
        ]
        
        pil_patch = Image.fromarray(patch)
        
        # Process with CLIP
        inputs = self.clip_processor(
            text=patch_categories,
            images=pil_patch,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Get most likely category
            max_idx = torch.argmax(probs, dim=1).item()
            semantic_description = patch_categories[max_idx]
        
        return semantic_description


class VisualTokenGenerator:
    """
    Visual Token Generator
    
    Generates visual tokens from semantic features and spatial information
    """
    
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.token_embeddings = {}
        self.spatial_encoder = self._create_spatial_encoder()
        
        logger.info(f"Initialized Visual Token Generator with vocab size {vocab_size}")
    
    def generate_visual_tokens(self, semantic_features: List[SemanticFeature], image_shape: Tuple[int, int]) -> List[VisualToken]:
        """Generate visual tokens from semantic features"""
        
        visual_tokens = []
        
        for i, feature in enumerate(semantic_features):
            # Generate token embedding
            token_embedding = self._generate_token_embedding(feature)
            
            # Generate spatial attention
            spatial_attention = self._generate_spatial_attention(feature, image_shape)
            
            # Determine visual concept
            visual_concept = self._determine_visual_concept(feature)
            
            # Calculate generation strength
            generation_strength = self._calculate_generation_strength(feature)
            
            visual_token = VisualToken(
                token_id=i,
                embedding=token_embedding,
                spatial_attention=spatial_attention,
                semantic_meaning=feature.semantic_description,
                visual_concept=visual_concept,
                generation_strength=generation_strength
            )
            
            visual_tokens.append(visual_token)
        
        return visual_tokens
    
    def _generate_token_embedding(self, feature: SemanticFeature) -> np.ndarray:
        """Generate token embedding from semantic feature"""
        
        # Use feature vector as base
        base_embedding = feature.feature_vector
        
        # Apply dimensionality reduction if needed
        if len(base_embedding) > 512:
            # Simple PCA-like reduction
            embedding = base_embedding[:512]
        else:
            # Pad if too short
            embedding = np.pad(base_embedding, (0, max(0, 512 - len(base_embedding))))
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def _generate_spatial_attention(self, feature: SemanticFeature, image_shape: Tuple[int, int]) -> np.ndarray:
        """Generate spatial attention map for feature"""
        
        h, w = image_shape
        attention_map = np.zeros((h, w), dtype=np.float32)
        
        # Center attention around feature location
        center_x, center_y = feature.spatial_location
        
        # Create gaussian attention
        y_indices, x_indices = np.ogrid[:h, :w]
        distance_sq = (x_indices - center_x)**2 + (y_indices - center_y)**2
        
        # Attention radius based on feature type
        if feature.feature_type == "global":
            radius = min(h, w) * 0.5
        elif feature.feature_type == "object":
            radius = min(h, w) * 0.2
        else:  # local
            radius = min(h, w) * 0.1
        
        attention_map = np.exp(-distance_sq / (2 * radius**2))
        
        # Weight by feature confidence
        attention_map *= feature.confidence
        
        return attention_map
    
    def _determine_visual_concept(self, feature: SemanticFeature) -> str:
        """Determine visual concept from feature"""
        
        # Map semantic descriptions to visual concepts
        concept_mapping = {
            "face": "human_face",
            "eye": "facial_feature",
            "hair": "texture",
            "skin": "surface",
            "clothing": "textile",
            "background": "environment",
            "landscape": "scenery",
            "urban": "architecture",
            "nature": "organic",
            "object": "artifact"
        }
        
        semantic_desc = feature.semantic_description.lower()
        
        for key, concept in concept_mapping.items():
            if key in semantic_desc:
                return concept
        
        return "generic"
    
    def _calculate_generation_strength(self, feature: SemanticFeature) -> float:
        """Calculate generation strength for feature"""
        
        # Base strength from confidence
        strength = feature.confidence
        
        # Adjust based on feature type
        if feature.feature_type == "global":
            strength *= 0.8  # Global features have lower individual strength
        elif feature.feature_type == "object":
            strength *= 1.2  # Object features have higher strength
        
        # Adjust based on feature vector magnitude
        vector_magnitude = np.linalg.norm(feature.feature_vector)
        normalized_magnitude = min(1.0, vector_magnitude / 100.0)
        strength *= normalized_magnitude
        
        return min(1.0, strength)
    
    def _create_spatial_encoder(self) -> nn.Module:
        """Create spatial encoder for positional information"""
        
        class SpatialEncoder(nn.Module):
            def __init__(self, d_model: int = 256):
                super().__init__()
                self.d_model = d_model
            
            def forward(self, positions: torch.Tensor) -> torch.Tensor:
                # Simple sinusoidal positional encoding
                pe = torch.zeros(positions.shape[0], self.d_model)
                
                for i in range(0, self.d_model, 2):
                    div_term = torch.exp(torch.arange(0, self.d_model, 2) * 
                                       -(math.log(10000.0) / self.d_model))
                    pe[:, i] = torch.sin(positions[:, 0] * div_term[i//2])
                    if i + 1 < self.d_model:
                        pe[:, i + 1] = torch.cos(positions[:, 0] * div_term[i//2])
                
                return pe
        
        return SpatialEncoder()


class PakatiReverseEngine:
    """
    Main Pakati Reverse Engine
    
    Orchestrates all reverse analysis components
    """
    
    def __init__(self):
        self.regional_extractor = RegionalControlExtractor()
        self.diffusion_analyzer = DiffusionReverseAnalyzer()
        self.semantic_extractor = SemanticExtractor()
        self.token_generator = VisualTokenGenerator()
        
        logger.info("Initialized Pakati Reverse Engine")
    
    def analyze_image(self, image: np.ndarray, analysis_config: Optional[Dict[str, Any]] = None) -> PakatiReverseResult:
        """Perform complete Pakati reverse analysis"""
        
        if analysis_config is None:
            analysis_config = self._get_default_config()
        
        logger.info("Starting Pakati reverse analysis")
        
        # Extract regional controls
        regional_controls = self.regional_extractor.extract_regional_controls(
            image, analysis_config.get('num_regions', 8)
        )
        
        # Perform diffusion reverse analysis
        diffusion_result = self.diffusion_analyzer.reverse_diffusion_process(
            image, analysis_config.get('diffusion_steps', 20)
        )
        
        # Extract semantic features
        semantic_features = self.semantic_extractor.extract_semantic_features(
            image, analysis_config.get('feature_resolution', 16)
        )
        
        # Generate visual tokens
        visual_tokens = self.token_generator.generate_visual_tokens(
            semantic_features, image.shape[:2]
        )
        
        # Estimate generation text
        generation_text = self._estimate_generation_text(
            regional_controls, semantic_features, diffusion_result
        )
        
        # Calculate overall confidence
        reverse_confidence = self._calculate_overall_confidence(
            regional_controls, diffusion_result, semantic_features
        )
        
        # Compile metadata
        analysis_metadata = {
            'image_shape': image.shape,
            'num_regional_controls': len(regional_controls),
            'num_semantic_features': len(semantic_features),
            'num_visual_tokens': len(visual_tokens),
            'analysis_config': analysis_config,
            'diffusion_available': self.diffusion_analyzer.pipeline is not None
        }
        
        return PakatiReverseResult(
            regional_controls=regional_controls,
            semantic_features=semantic_features,
            visual_tokens=visual_tokens,
            generation_text=generation_text,
            reverse_confidence=reverse_confidence,
            analysis_metadata=analysis_metadata
        )
    
    def _estimate_generation_text(
        self,
        regional_controls: List[RegionalControl],
        semantic_features: List[SemanticFeature],
        diffusion_result: Dict[str, Any]
    ) -> str:
        """Estimate the generation text from all analysis components"""
        
        text_components = []
        
        # From diffusion analysis
        if 'estimated_prompt' in diffusion_result:
            text_components.append(diffusion_result['estimated_prompt'])
        
        # From semantic features
        global_semantics = [f.semantic_description for f in semantic_features if f.feature_type == "global"]
        if global_semantics:
            text_components.extend(global_semantics)
        
        # From regional controls
        regional_semantics = [rc.semantic_label for rc in regional_controls if rc.confidence > 0.5]
        unique_regional = list(set(regional_semantics))
        text_components.extend(unique_regional)
        
        # Combine and clean up
        if text_components:
            generation_text = ", ".join(set(text_components))
        else:
            generation_text = "image"
        
        return generation_text
    
    def _calculate_overall_confidence(
        self,
        regional_controls: List[RegionalControl],
        diffusion_result: Dict[str, Any],
        semantic_features: List[SemanticFeature]
    ) -> float:
        """Calculate overall confidence in reverse analysis"""
        
        confidence_components = []
        
        # Regional control confidence
        if regional_controls:
            regional_conf = np.mean([rc.confidence for rc in regional_controls])
            confidence_components.append(regional_conf)
        
        # Diffusion reverse confidence
        if 'reverse_confidence' in diffusion_result:
            confidence_components.append(diffusion_result['reverse_confidence'])
        
        # Semantic feature confidence
        if semantic_features:
            semantic_conf = np.mean([sf.confidence for sf in semantic_features])
            confidence_components.append(semantic_conf)
        
        # Overall confidence
        if confidence_components:
            overall_confidence = np.mean(confidence_components)
        else:
            overall_confidence = 0.0
        
        return overall_confidence
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default analysis configuration"""
        
        return {
            'num_regions': 8,
            'diffusion_steps': 20,
            'feature_resolution': 16,
            'enable_diffusion_analysis': True,
            'enable_regional_extraction': True,
            'enable_semantic_extraction': True,
            'enable_token_generation': True
        }
