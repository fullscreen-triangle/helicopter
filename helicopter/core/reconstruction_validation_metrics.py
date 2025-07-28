"""
Reconstruction Validation Metrics

Implements the novel reconstruction-based evaluation metrics introduced in the paper:
1. Reconstruction Fidelity Score (RFS) - Overall reconstruction accuracy
2. Semantic Consistency Index (SCI) - Semantic preservation measure  
3. Partial Information Reconstruction Accuracy (PIRA) - Reconstruction from partial inputs

These metrics validate visual understanding through reconstruction capability
rather than traditional classification approaches.
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionMetrics:
    """Container for all reconstruction validation metrics"""
    rfs: float                          # Reconstruction Fidelity Score
    sci: float                          # Semantic Consistency Index  
    pira: float                         # Partial Information Reconstruction Accuracy
    
    # Component scores
    pixel_similarity: float             # Pixel-level similarity
    structural_similarity: float        # SSIM score
    perceptual_similarity: float        # LPIPS-style perceptual similarity
    semantic_embedding_similarity: float # Semantic embedding cosine similarity
    
    # Partial reconstruction scores
    partial_25_accuracy: float          # 25% information reconstruction
    partial_50_accuracy: float          # 50% information reconstruction
    partial_75_accuracy: float          # 75% information reconstruction
    
    # Additional metrics
    understanding_confidence: float     # Overall understanding confidence
    reconstruction_time: float          # Time taken for reconstruction


@dataclass
class ValidationConfig:
    """Configuration for reconstruction validation"""
    ssim_weight: float = 0.4            # Weight for SSIM in RFS
    lpips_weight: float = 0.4           # Weight for perceptual similarity in RFS
    semantic_weight: float = 0.2        # Weight for semantic consistency in RFS
    
    partial_levels: List[float] = None  # Information levels for PIRA
    semantic_model: str = "resnet50"    # Model for semantic embeddings
    perceptual_model: str = "vgg16"     # Model for perceptual similarity
    
    def __post_init__(self):
        if self.partial_levels is None:
            self.partial_levels = [0.25, 0.5, 0.75]


class SemanticEmbeddingExtractor:
    """Extract semantic embeddings for semantic consistency measurement"""
    
    def __init__(self, model_name: str = "resnet50"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained model
        if model_name == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove classifier
        elif model_name == "vit":
            # Could add Vision Transformer here
            raise NotImplementedError("ViT not implemented yet")
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Initialized semantic embedding extractor with {model_name}")
    
    def extract_features(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract semantic features from image"""
        if isinstance(image, np.ndarray):
            # Convert numpy to PIL for transforms
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray((image * 255).astype(np.uint8))
            image = self.transform(image).unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            features = self.model(image)
            features = features.view(features.size(0), -1)  # Flatten
        
        return features


class PerceptualSimilarityMeasure:
    """Measure perceptual similarity using pre-trained features"""
    
    def __init__(self, model_name: str = "vgg16"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load VGG features
        if model_name == "vgg16":
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
            self.features = vgg.features
        else:
            raise ValueError(f"Unknown perceptual model: {model_name}")
        
        self.features.to(self.device)
        self.features.eval()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Initialized perceptual similarity with {model_name}")
    
    def compute_similarity(
        self, 
        image1: Union[np.ndarray, torch.Tensor],
        image2: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """Compute perceptual similarity between two images"""
        
        # Preprocess images
        if isinstance(image1, np.ndarray):
            image1 = Image.fromarray((image1 * 255).astype(np.uint8) if image1.dtype != np.uint8 else image1)
            image1 = self.transform(image1).unsqueeze(0)
        
        if isinstance(image2, np.ndarray):
            image2 = Image.fromarray((image2 * 255).astype(np.uint8) if image2.dtype != np.uint8 else image2)
            image2 = self.transform(image2).unsqueeze(0)
        
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        
        # Extract features at multiple layers
        with torch.no_grad():
            features1 = []
            features2 = []
            
            x1, x2 = image1, image2
            
            for i, layer in enumerate(self.features):
                x1 = layer(x1)
                x2 = layer(x2)
                
                # Extract features at specific layers (conv layers)
                if isinstance(layer, nn.Conv2d) and i in [2, 5, 10, 17, 24]:
                    features1.append(x1)
                    features2.append(x2)
        
        # Compute similarity across layers
        similarities = []
        for f1, f2 in zip(features1, features2):
            # Normalize features
            f1_norm = f1 / (torch.norm(f1, dim=1, keepdim=True) + 1e-8)
            f2_norm = f2 / (torch.norm(f2, dim=1, keepdim=True) + 1e-8)
            
            # Compute cosine similarity
            similarity = torch.mean(f1_norm * f2_norm)
            similarities.append(similarity.item())
        
        # Average similarity across layers
        return np.mean(similarities)


class ReconstructionValidationMetrics:
    """
    Main class for computing reconstruction validation metrics.
    
    Implements RFS, SCI, and PIRA as described in the paper.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        
        # Initialize sub-components
        self.semantic_extractor = SemanticEmbeddingExtractor(self.config.semantic_model)
        self.perceptual_measure = PerceptualSimilarityMeasure(self.config.perceptual_model)
        
        logger.info("Initialized Reconstruction Validation Metrics")
    
    def compute_all_metrics(
        self,
        original_image: np.ndarray,
        reconstructed_image: np.ndarray,
        partial_reconstructions: Optional[Dict[float, np.ndarray]] = None,
        semantic_annotations: Optional[Dict] = None
    ) -> ReconstructionMetrics:
        """
        Compute all reconstruction validation metrics.
        
        Args:
            original_image: Ground truth image
            reconstructed_image: Reconstructed image from full information
            partial_reconstructions: Dict mapping information levels to reconstructions
            semantic_annotations: Optional semantic labels/annotations
            
        Returns:
            Complete reconstruction metrics
        """
        start_time = time.time()
        
        # Compute component similarities
        pixel_sim = self._compute_pixel_similarity(original_image, reconstructed_image)
        structural_sim = self._compute_structural_similarity(original_image, reconstructed_image)
        perceptual_sim = self._compute_perceptual_similarity(original_image, reconstructed_image)
        semantic_sim = self._compute_semantic_similarity(original_image, reconstructed_image)
        
        # Compute RFS (Reconstruction Fidelity Score)
        rfs = self._compute_rfs(structural_sim, perceptual_sim, semantic_sim)
        
        # Compute SCI (Semantic Consistency Index)
        sci = self._compute_sci(original_image, reconstructed_image, semantic_annotations)
        
        # Compute PIRA (Partial Information Reconstruction Accuracy)
        pira_scores = {}
        if partial_reconstructions:
            for level, partial_recon in partial_reconstructions.items():
                pira_scores[level] = self._compute_partial_reconstruction_accuracy(
                    original_image, partial_recon
                )
        else:
            # Generate partial reconstructions if not provided
            pira_scores = self._compute_pira_with_generated_partials(
                original_image, reconstructed_image
            )
        
        pira = np.mean(list(pira_scores.values()))
        
        # Overall understanding confidence
        understanding_confidence = (rfs + sci + pira) / 3.0
        
        reconstruction_time = time.time() - start_time
        
        return ReconstructionMetrics(
            rfs=rfs,
            sci=sci,
            pira=pira,
            pixel_similarity=pixel_sim,
            structural_similarity=structural_sim,
            perceptual_similarity=perceptual_sim,
            semantic_embedding_similarity=semantic_sim,
            partial_25_accuracy=pira_scores.get(0.25, 0.0),
            partial_50_accuracy=pira_scores.get(0.5, 0.0),
            partial_75_accuracy=pira_scores.get(0.75, 0.0),
            understanding_confidence=understanding_confidence,
            reconstruction_time=reconstruction_time
        )
    
    def _compute_pixel_similarity(
        self, 
        original: np.ndarray, 
        reconstructed: np.ndarray
    ) -> float:
        """Compute pixel-level similarity (MSE-based)"""
        if original.shape != reconstructed.shape:
            reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]))
        
        # Normalize to [0, 1]
        original_norm = original.astype(np.float32) / 255.0
        reconstructed_norm = reconstructed.astype(np.float32) / 255.0
        
        # Compute MSE and convert to similarity
        mse = np.mean((original_norm - reconstructed_norm) ** 2)
        similarity = 1.0 / (1.0 + mse)
        
        return similarity
    
    def _compute_structural_similarity(
        self, 
        original: np.ndarray, 
        reconstructed: np.ndarray
    ) -> float:
        """Compute SSIM (Structural Similarity Index)"""
        if original.shape != reconstructed.shape:
            reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]))
        
        # Convert to grayscale if needed
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            reconstructed_gray = cv2.cvtColor(reconstructed, cv2.COLOR_RGB2GRAY)
        else:
            original_gray = original
            reconstructed_gray = reconstructed
        
        # Compute SSIM
        ssim_score = ssim(original_gray, reconstructed_gray, data_range=255)
        
        return ssim_score
    
    def _compute_perceptual_similarity(
        self, 
        original: np.ndarray, 
        reconstructed: np.ndarray
    ) -> float:
        """Compute perceptual similarity using deep features"""
        similarity = self.perceptual_measure.compute_similarity(original, reconstructed)
        return similarity
    
    def _compute_semantic_similarity(
        self, 
        original: np.ndarray, 
        reconstructed: np.ndarray
    ) -> float:
        """Compute semantic similarity using embeddings"""
        # Extract semantic embeddings
        original_features = self.semantic_extractor.extract_features(original)
        reconstructed_features = self.semantic_extractor.extract_features(reconstructed)
        
        # Compute cosine similarity
        original_np = original_features.cpu().numpy()
        reconstructed_np = reconstructed_features.cpu().numpy()
        
        similarity = cosine_similarity(original_np, reconstructed_np)[0, 0]
        
        return similarity
    
    def _compute_rfs(
        self, 
        structural_sim: float, 
        perceptual_sim: float, 
        semantic_sim: float
    ) -> float:
        """
        Compute Reconstruction Fidelity Score (RFS).
        
        RFS = α·SSIM + β·LPIPS + γ·S_semantic
        """
        rfs = (
            self.config.ssim_weight * structural_sim +
            self.config.lpips_weight * perceptual_sim +
            self.config.semantic_weight * semantic_sim
        )
        
        return rfs
    
    def _compute_sci(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        semantic_annotations: Optional[Dict] = None
    ) -> float:
        """
        Compute Semantic Consistency Index (SCI).
        
        Measures how well semantic content is preserved in reconstruction.
        """
        # Base semantic similarity from embeddings
        base_sci = self._compute_semantic_similarity(original, reconstructed)
        
        # If semantic annotations are provided, use them for enhanced SCI
        if semantic_annotations:
            # This would involve comparing semantic segmentations, object detections, etc.
            # For now, we use the base semantic similarity
            # In practice, this would include:
            # - Object detection consistency
            # - Semantic segmentation overlap
            # - Scene classification agreement
            annotation_consistency = 1.0  # Placeholder
            sci = 0.7 * base_sci + 0.3 * annotation_consistency
        else:
            sci = base_sci
        
        return sci
    
    def _compute_partial_reconstruction_accuracy(
        self,
        original: np.ndarray,
        partial_reconstruction: np.ndarray
    ) -> float:
        """Compute accuracy for partial information reconstruction"""
        # This combines multiple similarity measures for partial reconstruction
        pixel_sim = self._compute_pixel_similarity(original, partial_reconstruction)
        structural_sim = self._compute_structural_similarity(original, partial_reconstruction)
        
        # Weight more heavily on structural similarity for partial reconstructions
        accuracy = 0.3 * pixel_sim + 0.7 * structural_sim
        
        return accuracy
    
    def _compute_pira_with_generated_partials(
        self,
        original: np.ndarray,
        full_reconstruction: np.ndarray
    ) -> Dict[float, float]:
        """
        Compute PIRA by generating partial reconstructions from full reconstruction.
        
        This is a simplified version - in practice, you would test reconstruction
        from actual partial information.
        """
        pira_scores = {}
        
        for level in self.config.partial_levels:
            # Simulate partial reconstruction by masking the full reconstruction
            partial_recon = self._simulate_partial_reconstruction(
                full_reconstruction, information_level=level
            )
            
            # Compute accuracy
            accuracy = self._compute_partial_reconstruction_accuracy(original, partial_recon)
            pira_scores[level] = accuracy
        
        return pira_scores
    
    def _simulate_partial_reconstruction(
        self,
        full_reconstruction: np.ndarray,
        information_level: float
    ) -> np.ndarray:
        """Simulate partial reconstruction by random masking"""
        height, width = full_reconstruction.shape[:2]
        
        # Create random mask based on information level
        mask = np.random.random((height, width)) < information_level
        
        # Apply mask to reconstruction
        partial_recon = full_reconstruction.copy()
        
        if len(partial_recon.shape) == 3:
            mask = mask[:, :, np.newaxis]
        
        # Set masked areas to gray (simulating missing information)
        partial_recon[~mask] = 128
        
        return partial_recon
    
    def get_validation_report(self, metrics: ReconstructionMetrics) -> str:
        """Generate human-readable validation report"""
        report = f"""
Reconstruction Validation Report:
===============================

Primary Metrics:
  • RFS (Reconstruction Fidelity Score): {metrics.rfs:.3f}
  • SCI (Semantic Consistency Index): {metrics.sci:.3f}  
  • PIRA (Partial Info Reconstruction Accuracy): {metrics.pira:.3f}

Component Scores:
  • Pixel Similarity: {metrics.pixel_similarity:.3f}
  • Structural Similarity (SSIM): {metrics.structural_similarity:.3f}
  • Perceptual Similarity: {metrics.perceptual_similarity:.3f}
  • Semantic Embedding Similarity: {metrics.semantic_embedding_similarity:.3f}

Partial Reconstruction Performance:
  • 25% Information: {metrics.partial_25_accuracy:.3f}
  • 50% Information: {metrics.partial_50_accuracy:.3f}
  • 75% Information: {metrics.partial_75_accuracy:.3f}

Overall Assessment:
  • Understanding Confidence: {metrics.understanding_confidence:.3f}
  • Reconstruction Time: {metrics.reconstruction_time:.2f}s

Interpretation:
{self._interpret_metrics(metrics)}
        """
        return report
    
    def _interpret_metrics(self, metrics: ReconstructionMetrics) -> str:
        """Provide interpretation of metrics"""
        interpretations = []
        
        # RFS interpretation
        if metrics.rfs >= 0.9:
            interpretations.append("Excellent reconstruction fidelity")
        elif metrics.rfs >= 0.8:
            interpretations.append("Good reconstruction fidelity")
        elif metrics.rfs >= 0.7:
            interpretations.append("Moderate reconstruction fidelity")
        else:
            interpretations.append("Poor reconstruction fidelity")
        
        # SCI interpretation
        if metrics.sci >= 0.9:
            interpretations.append("Excellent semantic consistency")
        elif metrics.sci >= 0.8:
            interpretations.append("Good semantic preservation")
        else:
            interpretations.append("Semantic inconsistencies detected")
        
        # PIRA interpretation
        if metrics.pira >= 0.8:
            interpretations.append("Strong partial reconstruction capability")
        elif metrics.pira >= 0.6:
            interpretations.append("Moderate partial reconstruction capability")
        else:
            interpretations.append("Weak partial reconstruction capability")
        
        # Overall assessment
        if metrics.understanding_confidence >= 0.85:
            interpretations.append("HIGH confidence in visual understanding")
        elif metrics.understanding_confidence >= 0.7:
            interpretations.append("MODERATE confidence in visual understanding")
        else:
            interpretations.append("LOW confidence in visual understanding")
        
        return " | ".join(interpretations)
    
    def compare_systems(
        self,
        system_metrics: Dict[str, ReconstructionMetrics]
    ) -> str:
        """Compare multiple systems using reconstruction metrics"""
        if not system_metrics:
            return "No systems to compare"
        
        report = "System Comparison Report:\n"
        report += "=" * 50 + "\n\n"
        
        # Create comparison table
        metrics_names = ["RFS", "SCI", "PIRA", "Understanding"]
        
        report += f"{'System':<15} {'RFS':<8} {'SCI':<8} {'PIRA':<8} {'Understanding':<12}\n"
        report += "-" * 60 + "\n"
        
        for system_name, metrics in system_metrics.items():
            report += f"{system_name:<15} "
            report += f"{metrics.rfs:<8.3f} "
            report += f"{metrics.sci:<8.3f} "
            report += f"{metrics.pira:<8.3f} "
            report += f"{metrics.understanding_confidence:<12.3f}\n"
        
        # Find best system
        best_overall = max(
            system_metrics.keys(),
            key=lambda k: system_metrics[k].understanding_confidence
        )
        
        report += f"\nBest Overall System: {best_overall} "
        report += f"(Understanding: {system_metrics[best_overall].understanding_confidence:.3f})\n"
        
        return report


# Import time for timing measurements
import time 