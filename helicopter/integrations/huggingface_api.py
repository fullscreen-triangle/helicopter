"""
Hugging Face Model API Integration

This module provides intelligent integration with Hugging Face models for
computer vision tasks. It supports the core Helicopter principle by using
external models to validate and enhance autonomous reconstruction insights.

Key Features:
- Intelligent model selection based on task requirements
- Automatic model routing and orchestration
- Performance optimization and caching
- Cross-model validation and consensus building
"""

import os
import time
import json
import logging
import asyncio
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import cv2
from PIL import Image
import torch
import requests
from transformers import (
    AutoModel, AutoProcessor, AutoTokenizer,
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel,
    DetrImageProcessor, DetrForObjectDetection,
    SegformerImageProcessor, SegformerForSemanticSegmentation,
    pipeline
)
from huggingface_hub import HfApi, list_models
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Supported computer vision task types."""
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    SEMANTIC_SEGMENTATION = "semantic-segmentation"
    IMAGE_CAPTIONING = "image-to-text"
    VISUAL_QUESTION_ANSWERING = "visual-question-answering"
    IMAGE_SIMILARITY = "image-similarity"
    DEPTH_ESTIMATION = "depth-estimation"
    IMAGE_INPAINTING = "image-inpainting"
    IMAGE_GENERATION = "text-to-image"
    FEATURE_EXTRACTION = "feature-extraction"


@dataclass
class ModelCapabilities:
    """Defines capabilities and performance metrics for a model."""
    
    model_id: str
    task_type: TaskType
    accuracy_score: float = 0.0
    inference_speed: float = 0.0  # images per second
    memory_usage: float = 0.0     # GB
    supported_image_sizes: List[Tuple[int, int]] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ["en"])
    license: str = "unknown"
    downloads: int = 0
    last_updated: str = ""
    
    def __post_init__(self):
        if not self.supported_image_sizes:
            self.supported_image_sizes = [(224, 224), (384, 384), (512, 512)]


class ModelSelector:
    """Intelligently selects the best model for a given task and requirements."""
    
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token
        self.api = HfApi(token=hf_token)
        self.model_cache = {}
        self.performance_cache = {}
        
        # Curated list of high-performance models for each task
        self.recommended_models = {
            TaskType.IMAGE_CLASSIFICATION: [
                "google/vit-base-patch16-224",
                "microsoft/resnet-50",
                "facebook/convnext-base-224",
                "google/efficientnet-b0"
            ],
            TaskType.OBJECT_DETECTION: [
                "facebook/detr-resnet-50",
                "microsoft/table-transformer-detection",
                "hustvl/yolos-tiny"
            ],
            TaskType.SEMANTIC_SEGMENTATION: [
                "nvidia/segformer-b0-finetuned-ade-512-512",
                "facebook/mask2former-swin-base-coco-panoptic",
                "microsoft/beit-base-finetuned-ade-640-640"
            ],
            TaskType.IMAGE_CAPTIONING: [
                "Salesforce/blip-image-captioning-base",
                "microsoft/git-base-coco",
                "nlpconnect/vit-gpt2-image-captioning"
            ],
            TaskType.VISUAL_QUESTION_ANSWERING: [
                "Salesforce/blip-vqa-base",
                "microsoft/git-base-vqav2",
                "dandelin/vilt-b32-finetuned-vqa"
            ],
            TaskType.DEPTH_ESTIMATION: [
                "Intel/dpt-large",
                "facebook/dpt-dinov2-base-kitti",
                "vinvino02/glpn-kitti"
            ],
            TaskType.IMAGE_INPAINTING: [
                "runwayml/stable-diffusion-inpainting",
                "facebook/sam-vit-base",
                "microsoft/DialoGPT-medium"
            ],
            TaskType.FEATURE_EXTRACTION: [
                "openai/clip-vit-base-patch32",
                "facebook/dinov2-base",
                "google/vit-base-patch16-224"
            ]
        }
    
    def select_best_model(self, task_type: TaskType, 
                         requirements: Dict[str, Any] = None) -> str:
        """
        Select the best model for a given task based on requirements.
        
        Args:
            task_type: Type of computer vision task
            requirements: Dictionary of requirements (speed, accuracy, memory, etc.)
        
        Returns:
            Model ID of the selected model
        """
        if requirements is None:
            requirements = {}
        
        # Get candidate models
        candidates = self.recommended_models.get(task_type, [])
        
        if not candidates:
            logger.warning(f"No recommended models for task type: {task_type}")
            return self._fallback_model_selection(task_type)
        
        # Score models based on requirements
        scored_models = []
        for model_id in candidates:
            score = self._score_model(model_id, task_type, requirements)
            scored_models.append((model_id, score))
        
        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        selected_model = scored_models[0][0]
        logger.info(f"Selected model {selected_model} for task {task_type.value}")
        
        return selected_model
    
    def _score_model(self, model_id: str, task_type: TaskType, 
                    requirements: Dict[str, Any]) -> float:
        """Score a model based on requirements."""
        
        # Get model capabilities (cached or fetch)
        capabilities = self._get_model_capabilities(model_id, task_type)
        
        score = 0.0
        
        # Weight factors based on requirements
        accuracy_weight = requirements.get('accuracy_priority', 0.4)
        speed_weight = requirements.get('speed_priority', 0.3)
        memory_weight = requirements.get('memory_priority', 0.2)
        popularity_weight = requirements.get('popularity_priority', 0.1)
        
        # Accuracy score (normalized)
        if capabilities.accuracy_score > 0:
            score += accuracy_weight * (capabilities.accuracy_score / 100.0)
        
        # Speed score (higher is better)
        if capabilities.inference_speed > 0:
            speed_score = min(capabilities.inference_speed / 10.0, 1.0)  # Normalize to 0-1
            score += speed_weight * speed_score
        
        # Memory efficiency (lower is better)
        if capabilities.memory_usage > 0:
            memory_score = max(0, 1.0 - (capabilities.memory_usage / 16.0))  # 16GB as max
            score += memory_weight * memory_score
        
        # Popularity score (downloads)
        if capabilities.downloads > 0:
            popularity_score = min(np.log10(capabilities.downloads) / 6.0, 1.0)  # Log scale
            score += popularity_weight * popularity_score
        
        return score
    
    def _get_model_capabilities(self, model_id: str, task_type: TaskType) -> ModelCapabilities:
        """Get or fetch model capabilities."""
        
        if model_id in self.model_cache:
            return self.model_cache[model_id]
        
        try:
            # Fetch model info from Hugging Face
            model_info = self.api.model_info(model_id)
            
            capabilities = ModelCapabilities(
                model_id=model_id,
                task_type=task_type,
                downloads=getattr(model_info, 'downloads', 0),
                last_updated=str(getattr(model_info, 'lastModified', '')),
                license=getattr(model_info, 'license', 'unknown')
            )
            
            # Add performance estimates based on model type and size
            capabilities = self._estimate_performance(capabilities, model_info)
            
            self.model_cache[model_id] = capabilities
            return capabilities
            
        except Exception as e:
            logger.warning(f"Could not fetch info for model {model_id}: {e}")
            return ModelCapabilities(model_id=model_id, task_type=task_type)
    
    def _estimate_performance(self, capabilities: ModelCapabilities, model_info) -> ModelCapabilities:
        """Estimate performance metrics based on model characteristics."""
        
        # Simple heuristics based on model name and type
        model_name = capabilities.model_id.lower()
        
        # Accuracy estimates
        if 'large' in model_name:
            capabilities.accuracy_score = 85.0
            capabilities.inference_speed = 2.0
            capabilities.memory_usage = 8.0
        elif 'base' in model_name:
            capabilities.accuracy_score = 80.0
            capabilities.inference_speed = 5.0
            capabilities.memory_usage = 4.0
        elif 'small' in model_name or 'tiny' in model_name:
            capabilities.accuracy_score = 75.0
            capabilities.inference_speed = 10.0
            capabilities.memory_usage = 2.0
        else:
            capabilities.accuracy_score = 78.0
            capabilities.inference_speed = 4.0
            capabilities.memory_usage = 3.0
        
        return capabilities
    
    def _fallback_model_selection(self, task_type: TaskType) -> str:
        """Fallback model selection when no recommended models available."""
        
        fallback_models = {
            TaskType.IMAGE_CLASSIFICATION: "google/vit-base-patch16-224",
            TaskType.OBJECT_DETECTION: "facebook/detr-resnet-50",
            TaskType.SEMANTIC_SEGMENTATION: "nvidia/segformer-b0-finetuned-ade-512-512",
            TaskType.IMAGE_CAPTIONING: "Salesforce/blip-image-captioning-base",
            TaskType.VISUAL_QUESTION_ANSWERING: "Salesforce/blip-vqa-base",
            TaskType.FEATURE_EXTRACTION: "openai/clip-vit-base-patch32"
        }
        
        return fallback_models.get(task_type, "google/vit-base-patch16-224")


class TaskRouter:
    """Routes tasks to appropriate models and handles execution."""
    
    def __init__(self, model_selector: ModelSelector, device: str = "auto"):
        self.model_selector = model_selector
        self.device = self._setup_device(device)
        self.loaded_models = {}
        self.processors = {}
        
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def route_task(self, task_type: TaskType, image: np.ndarray, 
                  requirements: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Route a task to the appropriate model and execute it.
        
        Args:
            task_type: Type of computer vision task
            image: Input image as numpy array
            requirements: Task requirements
            **kwargs: Additional task-specific parameters
        
        Returns:
            Dictionary containing task results
        """
        
        # Select best model for the task
        model_id = self.model_selector.select_best_model(task_type, requirements)
        
        # Load model if not already loaded
        model, processor = self._load_model(model_id, task_type)
        
        # Execute task
        start_time = time.time()
        
        try:
            if task_type == TaskType.IMAGE_CLASSIFICATION:
                result = self._classify_image(model, processor, image, **kwargs)
            elif task_type == TaskType.OBJECT_DETECTION:
                result = self._detect_objects(model, processor, image, **kwargs)
            elif task_type == TaskType.SEMANTIC_SEGMENTATION:
                result = self._segment_image(model, processor, image, **kwargs)
            elif task_type == TaskType.IMAGE_CAPTIONING:
                result = self._caption_image(model, processor, image, **kwargs)
            elif task_type == TaskType.VISUAL_QUESTION_ANSWERING:
                result = self._answer_visual_question(model, processor, image, **kwargs)
            elif task_type == TaskType.FEATURE_EXTRACTION:
                result = self._extract_features(model, processor, image, **kwargs)
            elif task_type == TaskType.DEPTH_ESTIMATION:
                result = self._estimate_depth(model, processor, image, **kwargs)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            execution_time = time.time() - start_time
            
            return {
                'task_type': task_type.value,
                'model_id': model_id,
                'result': result,
                'execution_time': execution_time,
                'success': True,
                'device': self.device
            }
            
        except Exception as e:
            logger.error(f"Task execution failed for {task_type.value}: {e}")
            return {
                'task_type': task_type.value,
                'model_id': model_id,
                'result': None,
                'execution_time': time.time() - start_time,
                'success': False,
                'error': str(e),
                'device': self.device
            }
    
    def _load_model(self, model_id: str, task_type: TaskType) -> Tuple[Any, Any]:
        """Load model and processor."""
        
        if model_id in self.loaded_models:
            return self.loaded_models[model_id], self.processors[model_id]
        
        try:
            logger.info(f"Loading model: {model_id}")
            
            if task_type == TaskType.IMAGE_CLASSIFICATION:
                processor = AutoProcessor.from_pretrained(model_id)
                model = AutoModel.from_pretrained(model_id).to(self.device)
                
            elif task_type == TaskType.OBJECT_DETECTION:
                processor = DetrImageProcessor.from_pretrained(model_id)
                model = DetrForObjectDetection.from_pretrained(model_id).to(self.device)
                
            elif task_type == TaskType.SEMANTIC_SEGMENTATION:
                processor = SegformerImageProcessor.from_pretrained(model_id)
                model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(self.device)
                
            elif task_type == TaskType.IMAGE_CAPTIONING:
                processor = BlipProcessor.from_pretrained(model_id)
                model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
                
            elif task_type == TaskType.VISUAL_QUESTION_ANSWERING:
                processor = BlipProcessor.from_pretrained(model_id)
                model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
                
            elif task_type == TaskType.FEATURE_EXTRACTION:
                processor = CLIPProcessor.from_pretrained(model_id)
                model = CLIPModel.from_pretrained(model_id).to(self.device)
                
            else:
                # Fallback to pipeline
                pipe = pipeline(task_type.value, model=model_id, device=0 if self.device == "cuda" else -1)
                model = pipe
                processor = None
            
            self.loaded_models[model_id] = model
            self.processors[model_id] = processor
            
            logger.info(f"Successfully loaded model: {model_id}")
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def _classify_image(self, model, processor, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Classify image using the loaded model."""
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
        
        # Process image
        inputs = processor(images=image_pil, return_tensors="pt").to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get top predictions
        top_k = kwargs.get('top_k', 5)
        top_predictions = torch.topk(predictions, top_k)
        
        results = []
        for i in range(top_k):
            score = top_predictions.values[0][i].item()
            class_id = top_predictions.indices[0][i].item()
            
            results.append({
                'class_id': class_id,
                'score': score,
                'confidence': score
            })
        
        return {
            'predictions': results,
            'top_prediction': results[0] if results else None
        }
    
    def _detect_objects(self, model, processor, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Detect objects in image."""
        
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
        
        # Process image
        inputs = processor(images=image_pil, return_tensors="pt").to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([image_pil.size[::-1]]).to(self.device)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        
        detections = []
        confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > confidence_threshold:
                detections.append({
                    'label': label.item(),
                    'confidence': score.item(),
                    'bbox': box.tolist()  # [x1, y1, x2, y2]
                })
        
        return {
            'detections': detections,
            'num_detections': len(detections)
        }
    
    def _segment_image(self, model, processor, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Perform semantic segmentation."""
        
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
        
        # Process image
        inputs = processor(images=image_pil, return_tensors="pt").to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image_pil.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        
        predicted_segmentation = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
        
        return {
            'segmentation_map': predicted_segmentation,
            'num_classes': logits.shape[1],
            'shape': predicted_segmentation.shape
        }
    
    def _caption_image(self, model, processor, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generate image caption."""
        
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
        
        # Process image
        inputs = processor(image_pil, return_tensors="pt").to(self.device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=kwargs.get('max_length', 50))
        
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return {
            'caption': caption,
            'confidence': 1.0  # BLIP doesn't provide confidence scores directly
        }
    
    def _answer_visual_question(self, model, processor, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Answer visual question about image."""
        
        question = kwargs.get('question', 'What is in this image?')
        
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
        
        # Process image and question
        inputs = processor(image_pil, question, return_tensors="pt").to(self.device)
        
        # Generate answer
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=kwargs.get('max_length', 20))
        
        answer = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return {
            'question': question,
            'answer': answer,
            'confidence': 1.0
        }
    
    def _extract_features(self, model, processor, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Extract visual features from image."""
        
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
        
        # Process image
        inputs = processor(images=image_pil, return_tensors="pt").to(self.device)
        
        # Extract features
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return {
            'features': image_features.cpu().numpy(),
            'feature_dim': image_features.shape[-1],
            'normalized': True
        }
    
    def _estimate_depth(self, model, processor, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Estimate depth from image."""
        
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
        
        # Process image
        inputs = processor(images=image_pil, return_tensors="pt").to(self.device)
        
        # Get depth prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_pil.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        depth_map = prediction.squeeze().cpu().numpy()
        
        return {
            'depth_map': depth_map,
            'min_depth': float(depth_map.min()),
            'max_depth': float(depth_map.max()),
            'shape': depth_map.shape
        }


class HuggingFaceModelAPI:
    """Main API class for Hugging Face model integration."""
    
    def __init__(self, hf_token: Optional[str] = None, device: str = "auto"):
        """
        Initialize Hugging Face Model API.
        
        Args:
            hf_token: Hugging Face API token for private models
            device: Computation device ("cuda", "cpu", "auto")
        """
        self.hf_token = hf_token or os.getenv('HUGGINGFACE_TOKEN')
        self.model_selector = ModelSelector(self.hf_token)
        self.task_router = TaskRouter(self.model_selector, device)
        
        logger.info(f"Initialized Hugging Face API with device: {self.task_router.device}")
    
    def classify_image(self, image: np.ndarray, top_k: int = 5, 
                      requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Classify an image using the best available model.
        
        Args:
            image: Input image as numpy array
            top_k: Number of top predictions to return
            requirements: Model selection requirements
        
        Returns:
            Classification results
        """
        return self.task_router.route_task(
            TaskType.IMAGE_CLASSIFICATION, 
            image, 
            requirements, 
            top_k=top_k
        )
    
    def detect_objects(self, image: np.ndarray, confidence_threshold: float = 0.5,
                      requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detect objects in an image.
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence for detections
            requirements: Model selection requirements
        
        Returns:
            Object detection results
        """
        return self.task_router.route_task(
            TaskType.OBJECT_DETECTION,
            image,
            requirements,
            confidence_threshold=confidence_threshold
        )
    
    def segment_image(self, image: np.ndarray, 
                     requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform semantic segmentation on an image.
        
        Args:
            image: Input image as numpy array
            requirements: Model selection requirements
        
        Returns:
            Segmentation results
        """
        return self.task_router.route_task(
            TaskType.SEMANTIC_SEGMENTATION,
            image,
            requirements
        )
    
    def caption_image(self, image: np.ndarray, max_length: int = 50,
                     requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a caption for an image.
        
        Args:
            image: Input image as numpy array
            max_length: Maximum caption length
            requirements: Model selection requirements
        
        Returns:
            Image captioning results
        """
        return self.task_router.route_task(
            TaskType.IMAGE_CAPTIONING,
            image,
            requirements,
            max_length=max_length
        )
    
    def answer_visual_question(self, image: np.ndarray, question: str,
                              max_length: int = 20,
                              requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Answer a question about an image.
        
        Args:
            image: Input image as numpy array
            question: Question to answer
            max_length: Maximum answer length
            requirements: Model selection requirements
        
        Returns:
            Visual question answering results
        """
        return self.task_router.route_task(
            TaskType.VISUAL_QUESTION_ANSWERING,
            image,
            requirements,
            question=question,
            max_length=max_length
        )
    
    def extract_features(self, image: np.ndarray,
                        requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract visual features from an image.
        
        Args:
            image: Input image as numpy array
            requirements: Model selection requirements
        
        Returns:
            Feature extraction results
        """
        return self.task_router.route_task(
            TaskType.FEATURE_EXTRACTION,
            image,
            requirements
        )
    
    def estimate_depth(self, image: np.ndarray,
                      requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Estimate depth from an image.
        
        Args:
            image: Input image as numpy array
            requirements: Model selection requirements
        
        Returns:
            Depth estimation results
        """
        return self.task_router.route_task(
            TaskType.DEPTH_ESTIMATION,
            image,
            requirements
        )


class IntelligentModelOrchestrator:
    """
    Orchestrates multiple models for comprehensive image analysis.
    
    This class embodies the Helicopter principle by using multiple models
    to validate and enhance autonomous reconstruction insights.
    """
    
    def __init__(self, hf_api: HuggingFaceModelAPI):
        self.hf_api = hf_api
        self.analysis_cache = {}
    
    def comprehensive_analysis(self, image: np.ndarray, 
                             tasks: List[TaskType] = None,
                             requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis using multiple models.
        
        Args:
            image: Input image as numpy array
            tasks: List of tasks to perform (if None, performs all relevant tasks)
            requirements: Model selection requirements
        
        Returns:
            Comprehensive analysis results
        """
        
        if tasks is None:
            tasks = [
                TaskType.IMAGE_CLASSIFICATION,
                TaskType.OBJECT_DETECTION,
                TaskType.SEMANTIC_SEGMENTATION,
                TaskType.IMAGE_CAPTIONING,
                TaskType.FEATURE_EXTRACTION
            ]
        
        results = {
            'image_shape': image.shape,
            'tasks_performed': [],
            'task_results': {},
            'analysis_summary': {},
            'cross_validation': {}
        }
        
        # Perform each task
        for task in tasks:
            try:
                logger.info(f"Performing task: {task.value}")
                
                task_result = self.hf_api.task_router.route_task(
                    task, image, requirements
                )
                
                results['task_results'][task.value] = task_result
                results['tasks_performed'].append(task.value)
                
            except Exception as e:
                logger.error(f"Failed to perform task {task.value}: {e}")
                results['task_results'][task.value] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Generate analysis summary
        results['analysis_summary'] = self._generate_analysis_summary(results['task_results'])
        
        # Perform cross-validation between tasks
        results['cross_validation'] = self._cross_validate_results(results['task_results'])
        
        return results
    
    def _generate_analysis_summary(self, task_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of analysis results."""
        
        summary = {
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_execution_time': 0.0,
            'confidence_scores': {},
            'key_insights': []
        }
        
        for task_name, result in task_results.items():
            if result.get('success', False):
                summary['successful_tasks'] += 1
                summary['total_execution_time'] += result.get('execution_time', 0.0)
                
                # Extract confidence scores
                if 'result' in result and result['result']:
                    task_result = result['result']
                    
                    if task_name == 'image-classification' and 'top_prediction' in task_result:
                        summary['confidence_scores'][task_name] = task_result['top_prediction']['confidence']
                    elif task_name == 'image-to-text' and 'confidence' in task_result:
                        summary['confidence_scores'][task_name] = task_result['confidence']
                    elif task_name == 'object-detection' and 'detections' in task_result:
                        if task_result['detections']:
                            avg_conf = np.mean([d['confidence'] for d in task_result['detections']])
                            summary['confidence_scores'][task_name] = avg_conf
            else:
                summary['failed_tasks'] += 1
        
        # Generate key insights
        if summary['successful_tasks'] > 0:
            summary['key_insights'].append(f"Successfully analyzed image using {summary['successful_tasks']} different models")
            
            if 'image-to-text' in task_results and task_results['image-to-text'].get('success'):
                caption = task_results['image-to-text']['result']['caption']
                summary['key_insights'].append(f"Image description: {caption}")
            
            if 'object-detection' in task_results and task_results['object-detection'].get('success'):
                num_objects = task_results['object-detection']['result']['num_detections']
                summary['key_insights'].append(f"Detected {num_objects} objects in the image")
        
        return summary
    
    def _cross_validate_results(self, task_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate results between different tasks."""
        
        validation = {
            'consistency_score': 0.0,
            'conflicting_evidence': [],
            'supporting_evidence': [],
            'validation_insights': []
        }
        
        # Check consistency between classification and object detection
        if ('image-classification' in task_results and 
            'object-detection' in task_results and
            both_successful(task_results, ['image-classification', 'object-detection'])):
            
            # This is a simplified validation - in practice, you'd have more sophisticated logic
            validation['supporting_evidence'].append(
                "Both classification and object detection completed successfully"
            )
            validation['consistency_score'] += 0.3
        
        # Check consistency between captioning and other tasks
        if ('image-to-text' in task_results and 
            task_results['image-to-text'].get('success')):
            
            caption = task_results['image-to-text']['result']['caption'].lower()
            
            # Check if caption mentions detected objects
            if ('object-detection' in task_results and 
                task_results['object-detection'].get('success')):
                
                num_objects = task_results['object-detection']['result']['num_detections']
                if num_objects > 0 and any(word in caption for word in ['person', 'car', 'dog', 'cat', 'object']):
                    validation['supporting_evidence'].append(
                        "Caption content aligns with object detection results"
                    )
                    validation['consistency_score'] += 0.4
        
        # Normalize consistency score
        validation['consistency_score'] = min(validation['consistency_score'], 1.0)
        
        # Generate validation insights
        if validation['consistency_score'] > 0.7:
            validation['validation_insights'].append("High consistency across different analysis methods")
        elif validation['consistency_score'] > 0.4:
            validation['validation_insights'].append("Moderate consistency with some conflicting evidence")
        else:
            validation['validation_insights'].append("Low consistency - results may be unreliable")
        
        return validation


def both_successful(task_results: Dict[str, Any], task_names: List[str]) -> bool:
    """Check if both tasks completed successfully."""
    return all(
        task_name in task_results and task_results[task_name].get('success', False)
        for task_name in task_names
    )


# Example usage and testing
if __name__ == "__main__":
    # Initialize API
    hf_api = HuggingFaceModelAPI()
    
    # Create orchestrator
    orchestrator = IntelligentModelOrchestrator(hf_api)
    
    # Example image (you would load your actual image)
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Perform comprehensive analysis
    results = orchestrator.comprehensive_analysis(test_image)
    
    print("Comprehensive Analysis Results:")
    print(json.dumps(results['analysis_summary'], indent=2)) 