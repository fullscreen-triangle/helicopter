"""
Autonomous Reconstruction Core

Implements the revolutionary insight from Pakati: "Best way to analyze an image is if AI can draw the image perfectly."
Adapted for Helicopter's autonomous reconstruction approach using HuggingFace API for actual reconstruction tasks.

Core Insight: If an AI can perfectly reconstruct regions of an image from partial information,
it demonstrates true understanding of those regions.
"""

import numpy as np
import cv2
import requests
import base64
import io
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ReconstructionStrategy(Enum):
    """Different reconstruction strategies inspired by Pakati's regional control."""
    INPAINTING = "inpainting"
    IMAGE_TO_IMAGE = "image_to_image"
    CONDITIONAL_GENERATION = "conditional_generation"
    DEPTH_GUIDED = "depth_guided"
    EDGE_GUIDED = "edge_guided"


@dataclass
class ReconstructionTask:
    """Defines a reconstruction task similar to Pakati's regional tasks."""
    
    task_id: str
    original_image: np.ndarray
    masked_image: np.ndarray
    mask: np.ndarray
    strategy: ReconstructionStrategy
    prompt: str = ""
    difficulty_level: float = 0.5
    target_quality: float = 0.85
    
    # HuggingFace model configuration
    model_id: str = "runwayml/stable-diffusion-inpainting"
    api_params: Dict[str, Any] = field(default_factory=dict)
    
    # Results
    reconstructed_image: Optional[np.ndarray] = None
    quality_score: float = 0.0
    confidence_score: float = 0.0
    execution_time: float = 0.0
    success: bool = False


class HuggingFaceReconstructionAPI:
    """
    HuggingFace API interface for reconstruction tasks.
    Inspired by Pakati's API integration approach.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HuggingFace API key required. Set HUGGINGFACE_API_KEY environment variable.")
        
        self.base_url = "https://api-inference.huggingface.co/models"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Model configurations for different tasks
        self.model_configs = {
            ReconstructionStrategy.INPAINTING: {
                "default_model": "runwayml/stable-diffusion-inpainting",
                "alternatives": [
                    "stabilityai/stable-diffusion-2-inpainting",
                    "kandinsky-community/kandinsky-2-2-decoder-inpaint"
                ]
            },
            ReconstructionStrategy.IMAGE_TO_IMAGE: {
                "default_model": "runwayml/stable-diffusion-v1-5",
                "alternatives": [
                    "stabilityai/stable-diffusion-2-1",
                    "kandinsky-community/kandinsky-2-1"
                ]
            },
            ReconstructionStrategy.CONDITIONAL_GENERATION: {
                "default_model": "runwayml/stable-diffusion-v1-5",
                "alternatives": ["stabilityai/stable-diffusion-2-1"]
            },
            ReconstructionStrategy.DEPTH_GUIDED: {
                "default_model": "lllyasviel/sd-controlnet-depth",
                "alternatives": ["lllyasviel/control-sd15-depth-v1-1"]
            },
            ReconstructionStrategy.EDGE_GUIDED: {
                "default_model": "lllyasviel/sd-controlnet-canny",
                "alternatives": ["lllyasviel/control-sd15-canny-v1-1"]
            }
        }
    
    def reconstruct_region(self, task: ReconstructionTask) -> ReconstructionTask:
        """
        Reconstruct a region using HuggingFace API.
        
        Args:
            task: ReconstructionTask with all necessary information
            
        Returns:
            Updated ReconstructionTask with results
        """
        
        start_time = time.time()
        
        try:
            if task.strategy == ReconstructionStrategy.INPAINTING:
                result = self._inpaint_region(task)
            elif task.strategy == ReconstructionStrategy.IMAGE_TO_IMAGE:
                result = self._image_to_image_reconstruction(task)
            elif task.strategy == ReconstructionStrategy.CONDITIONAL_GENERATION:
                result = self._conditional_generation(task)
            elif task.strategy == ReconstructionStrategy.DEPTH_GUIDED:
                result = self._depth_guided_reconstruction(task)
            elif task.strategy == ReconstructionStrategy.EDGE_GUIDED:
                result = self._edge_guided_reconstruction(task)
            else:
                raise ValueError(f"Unsupported reconstruction strategy: {task.strategy}")
            
            task.reconstructed_image = result
            task.quality_score = self._calculate_quality_score(task)
            task.confidence_score = self._estimate_confidence(task)
            task.success = task.quality_score >= task.target_quality
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {str(e)}")
            task.reconstructed_image = task.masked_image.copy()  # Fallback
            task.quality_score = 0.0
            task.confidence_score = 0.0
            task.success = False
        
        task.execution_time = time.time() - start_time
        return task
    
    def _inpaint_region(self, task: ReconstructionTask) -> np.ndarray:
        """Perform inpainting using HuggingFace API."""
        
        model_id = task.model_id or self.model_configs[ReconstructionStrategy.INPAINTING]["default_model"]
        
        # Convert images to base64
        image_b64 = self._numpy_to_base64(task.masked_image)
        mask_b64 = self._numpy_to_base64(task.mask)
        
        # Prepare API request
        payload = {
            "inputs": {
                "image": image_b64,
                "mask_image": mask_b64,
                "prompt": task.prompt or "high quality, detailed reconstruction"
            },
            "parameters": {
                "num_inference_steps": task.api_params.get("steps", 50),
                "guidance_scale": task.api_params.get("guidance_scale", 7.5),
                "strength": task.api_params.get("strength", 1.0)
            }
        }
        
        # Make API request
        response = requests.post(
            f"{self.base_url}/{model_id}",
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            # Handle image response
            if response.headers.get('content-type', '').startswith('image/'):
                image = Image.open(io.BytesIO(response.content))
                return np.array(image)
            else:
                # Handle JSON response with base64 image
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    image_data = base64.b64decode(result[0])
                    image = Image.open(io.BytesIO(image_data))
                    return np.array(image)
        
        raise Exception(f"API request failed: {response.status_code} - {response.text}")
    
    def _image_to_image_reconstruction(self, task: ReconstructionTask) -> np.ndarray:
        """Perform image-to-image reconstruction."""
        
        model_id = task.model_id or self.model_configs[ReconstructionStrategy.IMAGE_TO_IMAGE]["default_model"]
        
        # Generate prompt based on visible regions
        if not task.prompt:
            task.prompt = self._generate_reconstruction_prompt(task)
        
        image_b64 = self._numpy_to_base64(task.masked_image)
        
        payload = {
            "inputs": {
                "image": image_b64,
                "prompt": task.prompt
            },
            "parameters": {
                "num_inference_steps": task.api_params.get("steps", 50),
                "guidance_scale": task.api_params.get("guidance_scale", 7.5),
                "strength": task.api_params.get("strength", 0.8)
            }
        }
        
        response = requests.post(
            f"{self.base_url}/{model_id}",
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            if response.headers.get('content-type', '').startswith('image/'):
                image = Image.open(io.BytesIO(response.content))
                return np.array(image)
        
        raise Exception(f"API request failed: {response.status_code} - {response.text}")
    
    def _conditional_generation(self, task: ReconstructionTask) -> np.ndarray:
        """Perform conditional generation based on visible context."""
        
        # This is a simplified version - in practice, you'd extract features from visible regions
        # and use them to guide generation
        
        model_id = task.model_id or self.model_configs[ReconstructionStrategy.CONDITIONAL_GENERATION]["default_model"]
        
        # Generate contextual prompt
        context_prompt = self._analyze_visible_context(task)
        full_prompt = f"{context_prompt}, {task.prompt}" if task.prompt else context_prompt
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "num_inference_steps": task.api_params.get("steps", 50),
                "guidance_scale": task.api_params.get("guidance_scale", 7.5),
                "width": task.original_image.shape[1],
                "height": task.original_image.shape[0]
            }
        }
        
        response = requests.post(
            f"{self.base_url}/{model_id}",
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            if response.headers.get('content-type', '').startswith('image/'):
                image = Image.open(io.BytesIO(response.content))
                generated = np.array(image)
                
                # Combine with original visible regions
                return self._combine_with_visible_regions(generated, task)
        
        raise Exception(f"API request failed: {response.status_code} - {response.text}")
    
    def _depth_guided_reconstruction(self, task: ReconstructionTask) -> np.ndarray:
        """Perform depth-guided reconstruction using ControlNet."""
        
        # Extract depth information from visible regions
        depth_map = self._estimate_depth_from_visible(task)
        
        model_id = task.model_id or self.model_configs[ReconstructionStrategy.DEPTH_GUIDED]["default_model"]
        
        depth_b64 = self._numpy_to_base64(depth_map)
        
        payload = {
            "inputs": {
                "image": depth_b64,
                "prompt": task.prompt or "detailed reconstruction maintaining depth structure"
            },
            "parameters": {
                "num_inference_steps": task.api_params.get("steps", 50),
                "guidance_scale": task.api_params.get("guidance_scale", 7.5),
                "controlnet_conditioning_scale": task.api_params.get("controlnet_scale", 1.0)
            }
        }
        
        response = requests.post(
            f"{self.base_url}/{model_id}",
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            if response.headers.get('content-type', '').startswith('image/'):
                image = Image.open(io.BytesIO(response.content))
                return np.array(image)
        
        raise Exception(f"API request failed: {response.status_code} - {response.text}")
    
    def _edge_guided_reconstruction(self, task: ReconstructionTask) -> np.ndarray:
        """Perform edge-guided reconstruction using ControlNet."""
        
        # Extract edge information from visible regions
        edge_map = self._extract_edges_from_visible(task)
        
        model_id = task.model_id or self.model_configs[ReconstructionStrategy.EDGE_GUIDED]["default_model"]
        
        edge_b64 = self._numpy_to_base64(edge_map)
        
        payload = {
            "inputs": {
                "image": edge_b64,
                "prompt": task.prompt or "detailed reconstruction following edge structure"
            },
            "parameters": {
                "num_inference_steps": task.api_params.get("steps", 50),
                "guidance_scale": task.api_params.get("guidance_scale", 7.5),
                "controlnet_conditioning_scale": task.api_params.get("controlnet_scale", 1.0)
            }
        }
        
        response = requests.post(
            f"{self.base_url}/{model_id}",
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            if response.headers.get('content-type', '').startswith('image/'):
                image = Image.open(io.BytesIO(response.content))
                return np.array(image)
        
        raise Exception(f"API request failed: {response.status_code} - {response.text}")
    
    def _numpy_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy array to base64 string."""
        
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        pil_image = Image.fromarray(image.astype(np.uint8))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _generate_reconstruction_prompt(self, task: ReconstructionTask) -> str:
        """Generate a prompt based on visible regions of the image."""
        
        # This is a simplified version - in practice, you'd use image captioning models
        # to analyze the visible regions and generate appropriate prompts
        
        visible_regions = task.masked_image[task.mask > 0]
        
        if len(visible_regions) == 0:
            return "high quality detailed image"
        
        # Analyze color distribution
        avg_color = np.mean(visible_regions, axis=0)
        
        # Simple color-based prompt generation
        if avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
            color_desc = "warm, reddish tones"
        elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
            color_desc = "natural, greenish tones"
        elif avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
            color_desc = "cool, bluish tones"
        else:
            color_desc = "balanced colors"
        
        return f"high quality detailed image with {color_desc}, photorealistic"
    
    def _analyze_visible_context(self, task: ReconstructionTask) -> str:
        """Analyze visible context to generate appropriate prompts."""
        
        # Extract features from visible regions
        visible_mask = task.mask > 0
        visible_image = task.masked_image.copy()
        visible_image[~visible_mask] = 0
        
        # Simple analysis - in practice, you'd use more sophisticated methods
        edges = cv2.Canny(cv2.cvtColor(visible_image, cv2.COLOR_RGB2GRAY), 50, 150)
        edge_density = np.sum(edges > 0) / np.sum(visible_mask)
        
        if edge_density > 0.1:
            return "detailed image with sharp edges and clear structures"
        else:
            return "smooth image with soft textures"
    
    def _combine_with_visible_regions(self, generated: np.ndarray, task: ReconstructionTask) -> np.ndarray:
        """Combine generated image with visible regions from original."""
        
        # Resize generated image to match original if needed
        if generated.shape[:2] != task.original_image.shape[:2]:
            generated = cv2.resize(generated, (task.original_image.shape[1], task.original_image.shape[0]))
        
        # Combine: use original where mask=1, generated where mask=0
        result = generated.copy()
        visible_mask = task.mask > 0
        result[visible_mask] = task.original_image[visible_mask]
        
        return result
    
    def _estimate_depth_from_visible(self, task: ReconstructionTask) -> np.ndarray:
        """Estimate depth map from visible regions."""
        
        # This is a simplified version - in practice, you'd use depth estimation models
        visible_image = task.masked_image.copy()
        gray = cv2.cvtColor(visible_image, cv2.COLOR_RGB2GRAY)
        
        # Simple depth estimation based on intensity gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        depth_estimate = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-255 range
        depth_estimate = (depth_estimate / depth_estimate.max() * 255).astype(np.uint8)
        
        # Convert to 3-channel for API compatibility
        return cv2.cvtColor(depth_estimate, cv2.COLOR_GRAY2RGB)
    
    def _extract_edges_from_visible(self, task: ReconstructionTask) -> np.ndarray:
        """Extract edge map from visible regions."""
        
        visible_image = task.masked_image.copy()
        gray = cv2.cvtColor(visible_image, cv2.COLOR_RGB2GRAY)
        
        # Extract edges using Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Convert to 3-channel for API compatibility
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    def _calculate_quality_score(self, task: ReconstructionTask) -> float:
        """Calculate reconstruction quality score."""
        
        if task.reconstructed_image is None:
            return 0.0
        
        # Focus on reconstructed regions (where mask == 0)
        reconstruction_mask = task.mask == 0
        
        if not np.any(reconstruction_mask):
            return 1.0  # Nothing to reconstruct
        
        # Calculate MSE in reconstructed regions
        original_region = task.original_image[reconstruction_mask]
        reconstructed_region = task.reconstructed_image[reconstruction_mask]
        
        mse = np.mean((original_region.astype(float) - reconstructed_region.astype(float)) ** 2)
        
        # Convert to quality score (0-1, higher is better)
        max_possible_error = 255 ** 2
        quality = 1.0 - (mse / max_possible_error)
        
        return max(0.0, min(1.0, quality))
    
    def _estimate_confidence(self, task: ReconstructionTask) -> float:
        """Estimate confidence in the reconstruction."""
        
        # This is a simplified confidence estimation
        # In practice, you'd use more sophisticated methods
        
        if task.reconstructed_image is None:
            return 0.0
        
        # Base confidence on quality score and difficulty
        base_confidence = task.quality_score
        
        # Adjust based on difficulty level
        difficulty_factor = 1.0 - (task.difficulty_level * 0.3)
        
        confidence = base_confidence * difficulty_factor
        
        return max(0.0, min(1.0, confidence))


class AutonomousReconstructionCore:
    """
    Core system for autonomous reconstruction testing.
    
    Implements Pakati's insight: "Best way to analyze an image is if AI can draw the image perfectly."
    Uses HuggingFace API for actual reconstruction while maintaining local control logic.
    """
    
    def __init__(self, api_key: str = None):
        self.api = HuggingFaceReconstructionAPI(api_key)
        self.reconstruction_history = []
        
        # Import mask generator from regional reconstruction engine
        from .regional_reconstruction_engine import RegionalMaskGenerator, MaskingStrategy
        self.mask_generator = RegionalMaskGenerator()
        self.MaskingStrategy = MaskingStrategy
    
    def test_understanding_through_reconstruction(self, image: np.ndarray,
                                                description: str = "",
                                                strategies: List[ReconstructionStrategy] = None,
                                                difficulty_levels: List[float] = None) -> Dict[str, Any]:
        """
        Test understanding of an image through reconstruction challenges.
        
        This implements the core insight: if AI can reconstruct regions perfectly,
        it demonstrates understanding of those regions.
        """
        
        if strategies is None:
            strategies = [ReconstructionStrategy.INPAINTING, ReconstructionStrategy.IMAGE_TO_IMAGE]
        
        if difficulty_levels is None:
            difficulty_levels = [0.2, 0.5, 0.8]
        
        test_results = {
            'image_shape': image.shape,
            'description': description,
            'strategies_tested': [s.value for s in strategies],
            'difficulty_levels': difficulty_levels,
            'strategy_results': {},
            'overall_understanding': {},
            'best_reconstruction': None,
            'insights': []
        }
        
        best_quality = 0.0
        best_task = None
        
        # Test each strategy at each difficulty level
        for strategy in strategies:
            strategy_results = {
                'strategy': strategy.value,
                'difficulty_results': [],
                'average_quality': 0.0,
                'success_rate': 0.0,
                'best_quality': 0.0
            }
            
            total_quality = 0.0
            successes = 0
            
            for difficulty in difficulty_levels:
                logger.info(f"Testing {strategy.value} at difficulty {difficulty}")
                
                # Create reconstruction task
                task = self._create_reconstruction_task(image, strategy, difficulty, description)
                
                # Perform reconstruction
                completed_task = self.api.reconstruct_region(task)
                
                # Record results
                difficulty_result = {
                    'difficulty': difficulty,
                    'quality_score': completed_task.quality_score,
                    'confidence_score': completed_task.confidence_score,
                    'execution_time': completed_task.execution_time,
                    'success': completed_task.success
                }
                
                strategy_results['difficulty_results'].append(difficulty_result)
                total_quality += completed_task.quality_score
                
                if completed_task.success:
                    successes += 1
                
                if completed_task.quality_score > best_quality:
                    best_quality = completed_task.quality_score
                    best_task = completed_task
                
                # Store in history
                self.reconstruction_history.append(completed_task)
            
            # Calculate strategy averages
            strategy_results['average_quality'] = total_quality / len(difficulty_levels)
            strategy_results['success_rate'] = successes / len(difficulty_levels)
            strategy_results['best_quality'] = max([r['quality_score'] for r in strategy_results['difficulty_results']])
            
            test_results['strategy_results'][strategy.value] = strategy_results
        
        # Generate overall assessment
        test_results['overall_understanding'] = self._generate_overall_understanding_assessment(test_results)
        test_results['best_reconstruction'] = self._serialize_task_for_results(best_task) if best_task else None
        test_results['insights'] = self._generate_reconstruction_insights(test_results)
        
        return test_results
    
    def _create_reconstruction_task(self, image: np.ndarray, 
                                  strategy: ReconstructionStrategy,
                                  difficulty: float,
                                  description: str) -> ReconstructionTask:
        """Create a reconstruction task with appropriate masking."""
        
        # Generate mask based on difficulty
        mask = self.mask_generator.generate_mask(
            self.MaskingStrategy.RANDOM_PATCHES,  # Default strategy
            image.shape[:2],
            difficulty
        )
        
        # Create masked image
        masked_image = image.copy()
        masked_image[mask == 0] = 0  # Zero out regions to reconstruct
        
        # Generate appropriate prompt based on strategy and description
        prompt = self._generate_strategy_prompt(strategy, description, difficulty)
        
        task = ReconstructionTask(
            task_id=f"task_{int(time.time())}_{strategy.value}_{difficulty}",
            original_image=image,
            masked_image=masked_image,
            mask=mask,
            strategy=strategy,
            prompt=prompt,
            difficulty_level=difficulty,
            api_params={
                "steps": max(20, int(50 * (1.0 - difficulty))),  # More steps for harder tasks
                "guidance_scale": 7.5 + (difficulty * 2.5),  # Higher guidance for harder tasks
                "strength": 0.8 + (difficulty * 0.2)
            }
        )
        
        return task
    
    def _generate_strategy_prompt(self, strategy: ReconstructionStrategy,
                                description: str, difficulty: float) -> str:
        """Generate appropriate prompt based on strategy and context."""
        
        base_prompt = description if description else "high quality detailed image"
        
        if strategy == ReconstructionStrategy.INPAINTING:
            return f"{base_prompt}, seamless inpainting, photorealistic"
        elif strategy == ReconstructionStrategy.IMAGE_TO_IMAGE:
            return f"{base_prompt}, detailed reconstruction, maintain consistency"
        elif strategy == ReconstructionStrategy.CONDITIONAL_GENERATION:
            return f"{base_prompt}, contextually appropriate, coherent"
        elif strategy == ReconstructionStrategy.DEPTH_GUIDED:
            return f"{base_prompt}, correct depth relationships, 3D structure"
        elif strategy == ReconstructionStrategy.EDGE_GUIDED:
            return f"{base_prompt}, sharp edges, clear boundaries"
        else:
            return base_prompt
    
    def _generate_overall_understanding_assessment(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall understanding assessment from test results."""
        
        strategy_qualities = []
        strategy_successes = []
        
        for strategy_result in test_results['strategy_results'].values():
            strategy_qualities.append(strategy_result['average_quality'])
            strategy_successes.append(strategy_result['success_rate'])
        
        overall_quality = np.mean(strategy_qualities) if strategy_qualities else 0.0
        overall_success_rate = np.mean(strategy_successes) if strategy_successes else 0.0
        
        # Classify understanding level
        if overall_quality >= 0.9:
            understanding_level = "excellent"
        elif overall_quality >= 0.8:
            understanding_level = "good"
        elif overall_quality >= 0.6:
            understanding_level = "moderate"
        elif overall_quality >= 0.4:
            understanding_level = "limited"
        else:
            understanding_level = "poor"
        
        return {
            'overall_quality': overall_quality,
            'overall_success_rate': overall_success_rate,
            'understanding_level': understanding_level,
            'strategies_mastered': sum(1 for sq in strategy_qualities if sq >= 0.8),
            'total_strategies_tested': len(strategy_qualities),
            'consistency_score': 1.0 - (np.std(strategy_qualities) / max(np.mean(strategy_qualities), 0.1)) if strategy_qualities else 0.0
        }
    
    def _serialize_task_for_results(self, task: ReconstructionTask) -> Dict[str, Any]:
        """Serialize task for inclusion in results (without large image arrays)."""
        
        return {
            'task_id': task.task_id,
            'strategy': task.strategy.value,
            'difficulty_level': task.difficulty_level,
            'prompt': task.prompt,
            'quality_score': task.quality_score,
            'confidence_score': task.confidence_score,
            'execution_time': task.execution_time,
            'success': task.success,
            'image_shape': task.original_image.shape,
            'mask_coverage': float(np.sum(task.mask) / task.mask.size)
        }
    
    def _generate_reconstruction_insights(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate insights from reconstruction test results."""
        
        insights = []
        overall = test_results['overall_understanding']
        
        # Overall performance insight
        insights.append(
            f"Achieved {overall['understanding_level']} understanding level "
            f"with {overall['overall_quality']:.3f} average quality"
        )
        
        # Strategy performance insights
        best_strategy = max(
            test_results['strategy_results'].items(),
            key=lambda x: x[1]['average_quality']
        )
        worst_strategy = min(
            test_results['strategy_results'].items(),
            key=lambda x: x[1]['average_quality']
        )
        
        insights.append(f"Best performing strategy: {best_strategy[0]} ({best_strategy[1]['average_quality']:.3f})")
        insights.append(f"Most challenging strategy: {worst_strategy[0]} ({worst_strategy[1]['average_quality']:.3f})")
        
        # Difficulty progression insights
        for strategy_name, strategy_data in test_results['strategy_results'].items():
            difficulties = [r['difficulty'] for r in strategy_data['difficulty_results']]
            qualities = [r['quality_score'] for r in strategy_data['difficulty_results']]
            
            if len(qualities) > 1:
                if qualities[-1] > qualities[0]:
                    insights.append(f"{strategy_name}: Quality improved with difficulty (unexpected)")
                elif qualities[0] > qualities[-1]:
                    insights.append(f"{strategy_name}: Quality degraded with difficulty (expected)")
        
        # Success rate insights
        if overall['overall_success_rate'] >= 0.8:
            insights.append("High success rate indicates robust understanding")
        elif overall['overall_success_rate'] < 0.3:
            insights.append("Low success rate suggests need for improvement")
        
        return insights
    
    def progressive_understanding_test(self, image: np.ndarray,
                                     description: str = "",
                                     max_difficulty: float = 0.9) -> Dict[str, Any]:
        """
        Perform progressive understanding test, increasing difficulty until failure.
        
        This implements the core Pakati insight of progressive mastery validation.
        """
        
        logger.info("Starting progressive understanding test")
        
        # Start with easy difficulty and progressively increase
        current_difficulty = 0.1
        difficulty_step = 0.1
        mastery_threshold = 0.85
        
        results = {
            'description': description,
            'image_shape': image.shape,
            'progression_results': [],
            'mastery_level': 0.0,
            'mastery_achieved': False,
            'understanding_pathway': [],
            'final_assessment': {}
        }
        
        while current_difficulty <= max_difficulty:
            logger.info(f"Testing difficulty level: {current_difficulty}")
            
            # Test at current difficulty with best strategy (inpainting)
            task = self._create_reconstruction_task(
                image, 
                ReconstructionStrategy.INPAINTING, 
                current_difficulty, 
                description
            )
            
            completed_task = self.api.reconstruct_region(task)
            
            level_result = {
                'difficulty': current_difficulty,
                'quality_score': completed_task.quality_score,
                'confidence_score': completed_task.confidence_score,
                'success': completed_task.success,
                'execution_time': completed_task.execution_time
            }
            
            results['progression_results'].append(level_result)
            
            if completed_task.success and completed_task.quality_score >= mastery_threshold:
                results['mastery_level'] = current_difficulty
                results['understanding_pathway'].append(
                    f"Mastered difficulty {current_difficulty} with quality {completed_task.quality_score:.3f}"
                )
            else:
                # Failed at this level
                results['understanding_pathway'].append(
                    f"Failed at difficulty {current_difficulty} with quality {completed_task.quality_score:.3f}"
                )
                break
            
            current_difficulty += difficulty_step
        
        # Determine if mastery achieved
        results['mastery_achieved'] = results['mastery_level'] >= 0.7
        
        # Generate final assessment
        results['final_assessment'] = {
            'mastery_level': results['mastery_level'],
            'mastery_achieved': results['mastery_achieved'],
            'peak_quality': max([r['quality_score'] for r in results['progression_results']] + [0.0]),
            'levels_completed': len([r for r in results['progression_results'] if r['success']]),
            'understanding_classification': self._classify_progressive_understanding(results['mastery_level'])
        }
        
        logger.info(f"Progressive test completed. Mastery level: {results['mastery_level']}")
        
        return results
    
    def _classify_progressive_understanding(self, mastery_level: float) -> str:
        """Classify understanding based on progressive mastery level."""
        
        if mastery_level >= 0.8:
            return "expert"
        elif mastery_level >= 0.6:
            return "proficient"
        elif mastery_level >= 0.4:
            return "developing"
        elif mastery_level >= 0.2:
            return "basic"
        else:
            return "novice"


# Example usage
if __name__ == "__main__":
    # Initialize core system
    core = AutonomousReconstructionCore()
    
    # Create test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Test understanding through reconstruction
    results = core.test_understanding_through_reconstruction(
        test_image,
        "test image for understanding validation"
    )
    
    print("Understanding Test Results:")
    print(f"Overall Understanding Level: {results['overall_understanding']['understanding_level']}")
    print(f"Overall Quality: {results['overall_understanding']['overall_quality']:.3f}")
    print(f"Success Rate: {results['overall_understanding']['overall_success_rate']:.3f}") 