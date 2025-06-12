"""
Autonomous Reconstruction Engine - "Reverse Reverse Reverse Pakati"

The genius insight: The best way to know if an AI has truly analyzed an image is if it can 
perfectly reconstruct it. The path to reconstruction IS the analysis itself.

Core Principle:
- Give the system parts of an image sequentially
- Ask it to predict other parts
- The reconstruction process reveals true understanding
- No need to complicate with separate analysis methods
- Let the system work autonomously through iterative reconstruction

This is the ultimate test: Can you draw what you see?
If yes, you have truly seen it.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import random
import math
from collections import deque
import time

from .bayesian_objective_engine import BayesianObjectiveEngine
from .continuous_learning_engine import ContinuousLearningEngine

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionPatch:
    """A patch of the image for reconstruction"""
    x: int
    y: int
    width: int
    height: int
    pixels: np.ndarray
    is_known: bool = False
    confidence: float = 0.0
    reconstruction_attempts: int = 0


@dataclass
class ReconstructionState:
    """Current state of autonomous reconstruction"""
    iteration: int
    known_patches: List[ReconstructionPatch]
    unknown_patches: List[ReconstructionPatch]
    current_reconstruction: np.ndarray
    reconstruction_quality: float
    prediction_confidence: float
    next_target_patch: Optional[ReconstructionPatch]
    learning_progress: Dict[str, float]


class AutonomousReconstructionNetwork(nn.Module):
    """Neural network for autonomous image reconstruction"""
    
    def __init__(self, patch_size: int = 32, context_size: int = 96):
        super().__init__()
        
        self.patch_size = patch_size
        self.context_size = context_size
        
        # Context encoder - understands surrounding patches
        self.context_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Patch predictor - predicts missing patch from context
        self.patch_predictor = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, patch_size * patch_size * 3),
            nn.Sigmoid()
        )
        
        # Confidence estimator - how confident is the prediction
        self.confidence_estimator = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Quality assessor - how good is the overall reconstruction
        self.quality_assessor = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, context_region: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict patch from context
        
        Args:
            context_region: Context around missing patch [B, 3, H, W]
            
        Returns:
            predicted_patch: Predicted patch pixels [B, patch_size*patch_size*3]
            confidence: Confidence in prediction [B, 1]
        """
        
        # Encode context
        context_features = self.context_encoder(context_region)
        context_flat = context_features.view(context_features.size(0), -1)
        
        # Predict patch
        predicted_patch = self.patch_predictor(context_flat)
        
        # Estimate confidence
        confidence = self.confidence_estimator(context_flat)
        
        return predicted_patch, confidence
    
    def assess_quality(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Assess quality of current reconstruction"""
        return self.quality_assessor(reconstruction)


class AutonomousReconstructionEngine:
    """
    Main engine for autonomous image reconstruction
    
    The core insight: True image analysis is demonstrated by the ability to reconstruct.
    The system autonomously learns to predict image parts from other parts.
    """
    
    def __init__(self, 
                 patch_size: int = 32,
                 context_size: int = 96,
                 device: Optional[str] = None):
        
        self.patch_size = patch_size
        self.context_size = context_size
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Core reconstruction network
        self.reconstruction_network = AutonomousReconstructionNetwork(patch_size, context_size).to(self.device)
        self.optimizer = optim.AdamW(self.reconstruction_network.parameters(), lr=1e-4)
        
        # Autonomous learning components
        self.learning_engine = ContinuousLearningEngine("reconstruction", device)
        self.objective_engine = BayesianObjectiveEngine("reconstruction")
        
        # Reconstruction state
        self.current_state = None
        self.reconstruction_history = []
        
        # Autonomous strategy
        self.exploration_strategies = [
            'random_patch',
            'edge_guided',
            'content_aware',
            'uncertainty_guided',
            'progressive_refinement'
        ]
        
        logger.info(f"Initialized Autonomous Reconstruction Engine")
        logger.info(f"Patch size: {patch_size}, Context size: {context_size}")
    
    def autonomous_analyze(self, 
                          image: np.ndarray, 
                          max_iterations: int = 100,
                          target_quality: float = 0.95) -> Dict[str, Any]:
        """
        Autonomously analyze image through reconstruction
        
        The system will:
        1. Start with partial image information
        2. Iteratively predict missing parts
        3. Learn from reconstruction success/failure
        4. Continue until perfect reconstruction or convergence
        
        Args:
            image: Original image to analyze/reconstruct
            max_iterations: Maximum reconstruction iterations
            target_quality: Target reconstruction quality (0-1)
            
        Returns:
            Complete analysis through reconstruction process
        """
        
        logger.info("Starting autonomous reconstruction analysis")
        logger.info(f"Image shape: {image.shape}, Target quality: {target_quality}")
        
        # Initialize reconstruction state
        self.current_state = self._initialize_reconstruction_state(image)
        
        # Autonomous reconstruction loop
        for iteration in range(max_iterations):
            logger.debug(f"\n=== AUTONOMOUS ITERATION {iteration + 1} ===")
            
            # Select next patch to predict autonomously
            target_patch = self._autonomous_patch_selection(self.current_state)
            
            if target_patch is None:
                logger.info("No more patches to predict - reconstruction complete")
                break
            
            # Extract context for prediction
            context_region = self._extract_context_region(self.current_state, target_patch)
            
            # Predict patch using current network
            predicted_patch, confidence = self._predict_patch(context_region)
            
            # Update reconstruction
            self._update_reconstruction(self.current_state, target_patch, predicted_patch, confidence)
            
            # Assess current reconstruction quality
            quality = self._assess_reconstruction_quality(self.current_state, image)
            
            # Learn from this prediction
            learning_feedback = self._generate_learning_feedback(
                target_patch, predicted_patch, confidence, quality, image
            )
            
            # Update learning systems
            self._update_autonomous_learning(learning_feedback)
            
            # Update state
            self.current_state.iteration = iteration + 1
            self.current_state.reconstruction_quality = quality
            self.current_state.prediction_confidence = float(confidence.item())
            
            # Store history
            self.reconstruction_history.append({
                'iteration': iteration + 1,
                'quality': quality,
                'confidence': float(confidence.item()),
                'patch_location': (target_patch.x, target_patch.y),
                'learning_feedback': learning_feedback
            })
            
            logger.debug(f"Iteration {iteration + 1}: Quality = {quality:.3f}, "
                        f"Confidence = {confidence.item():.3f}")
            
            # Check if target quality reached
            if quality >= target_quality:
                logger.info(f"Target quality {target_quality} achieved in {iteration + 1} iterations")
                break
            
            # Autonomous adaptation - change strategy if not improving
            if iteration > 10 and self._should_adapt_strategy():
                self._adapt_reconstruction_strategy()
        
        # Generate final analysis results
        final_results = self._generate_autonomous_analysis_results(image)
        
        return final_results
    
    def _initialize_reconstruction_state(self, image: np.ndarray) -> ReconstructionState:
        """Initialize the reconstruction state with partial image information"""
        
        h, w = image.shape[:2]
        
        # Create patch grid
        patches = []
        for y in range(0, h - self.patch_size + 1, self.patch_size):
            for x in range(0, w - self.patch_size + 1, self.patch_size):
                patch = ReconstructionPatch(
                    x=x, y=y, 
                    width=self.patch_size, 
                    height=self.patch_size,
                    pixels=image[y:y+self.patch_size, x:x+self.patch_size].copy()
                )
                patches.append(patch)
        
        # Randomly select initial known patches (start with ~20% of image)
        num_initial_known = max(1, len(patches) // 5)
        known_indices = random.sample(range(len(patches)), num_initial_known)
        
        known_patches = []
        unknown_patches = []
        
        for i, patch in enumerate(patches):
            if i in known_indices:
                patch.is_known = True
                patch.confidence = 1.0
                known_patches.append(patch)
            else:
                patch.is_known = False
                unknown_patches.append(patch)
        
        # Initialize reconstruction with known patches
        reconstruction = np.zeros_like(image)
        for patch in known_patches:
            reconstruction[patch.y:patch.y+patch.height, patch.x:patch.x+patch.width] = patch.pixels
        
        return ReconstructionState(
            iteration=0,
            known_patches=known_patches,
            unknown_patches=unknown_patches,
            current_reconstruction=reconstruction,
            reconstruction_quality=0.0,
            prediction_confidence=0.0,
            next_target_patch=None,
            learning_progress={}
        )
    
    def _autonomous_patch_selection(self, state: ReconstructionState) -> Optional[ReconstructionPatch]:
        """Autonomously select next patch to predict"""
        
        if not state.unknown_patches:
            return None
        
        # Choose strategy based on current state and learning
        strategy = self._choose_reconstruction_strategy(state)
        
        if strategy == 'random_patch':
            return random.choice(state.unknown_patches)
        
        elif strategy == 'edge_guided':
            # Prefer patches adjacent to known patches
            edge_patches = []
            for unknown_patch in state.unknown_patches:
                if self._is_adjacent_to_known(unknown_patch, state.known_patches):
                    edge_patches.append(unknown_patch)
            
            return random.choice(edge_patches) if edge_patches else random.choice(state.unknown_patches)
        
        elif strategy == 'content_aware':
            # Prefer patches in high-detail areas
            return self._select_high_detail_patch(state)
        
        elif strategy == 'uncertainty_guided':
            # Prefer patches where we're most uncertain
            return self._select_uncertain_patch(state)
        
        elif strategy == 'progressive_refinement':
            # Systematically fill in patches
            return state.unknown_patches[0]  # Take first unknown
        
        else:
            return random.choice(state.unknown_patches)
    
    def _choose_reconstruction_strategy(self, state: ReconstructionState) -> str:
        """Choose reconstruction strategy based on current state"""
        
        # Early iterations - use edge-guided
        if state.iteration < 10:
            return 'edge_guided'
        
        # If quality is low - try content-aware
        elif state.reconstruction_quality < 0.5:
            return 'content_aware'
        
        # If confidence is low - use uncertainty-guided
        elif state.prediction_confidence < 0.6:
            return 'uncertainty_guided'
        
        # Otherwise progressive refinement
        else:
            return 'progressive_refinement'
    
    def _is_adjacent_to_known(self, patch: ReconstructionPatch, known_patches: List[ReconstructionPatch]) -> bool:
        """Check if patch is adjacent to any known patch"""
        
        for known_patch in known_patches:
            # Check if patches are adjacent (touching edges)
            x_adjacent = (abs(patch.x - known_patch.x) == self.patch_size and 
                         abs(patch.y - known_patch.y) <= self.patch_size)
            y_adjacent = (abs(patch.y - known_patch.y) == self.patch_size and 
                         abs(patch.x - known_patch.x) <= self.patch_size)
            
            if x_adjacent or y_adjacent:
                return True
        
        return False
    
    def _select_high_detail_patch(self, state: ReconstructionState) -> ReconstructionPatch:
        """Select patch in high-detail area based on surrounding context"""
        
        detail_scores = []
        
        for patch in state.unknown_patches:
            # Calculate detail score based on surrounding known patches
            detail_score = 0.0
            surrounding_count = 0
            
            for known_patch in state.known_patches:
                distance = math.sqrt((patch.x - known_patch.x)**2 + (patch.y - known_patch.y)**2)
                if distance < self.context_size:
                    # Calculate edge density in known patch
                    gray = cv2.cvtColor(known_patch.pixels, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
                    
                    detail_score += edge_density
                    surrounding_count += 1
            
            if surrounding_count > 0:
                detail_score /= surrounding_count
            
            detail_scores.append(detail_score)
        
        # Select patch with highest detail score
        if detail_scores:
            max_idx = np.argmax(detail_scores)
            return state.unknown_patches[max_idx]
        else:
            return random.choice(state.unknown_patches)
    
    def _select_uncertain_patch(self, state: ReconstructionState) -> ReconstructionPatch:
        """Select patch where prediction would be most uncertain"""
        
        # For now, select patches with least surrounding context
        context_scores = []
        
        for patch in state.unknown_patches:
            context_score = 0
            for known_patch in state.known_patches:
                distance = math.sqrt((patch.x - known_patch.x)**2 + (patch.y - known_patch.y)**2)
                if distance < self.context_size:
                    context_score += 1
            
            context_scores.append(context_score)
        
        # Select patch with least context (most uncertain)
        if context_scores:
            min_idx = np.argmin(context_scores)
            return state.unknown_patches[min_idx]
        else:
            return random.choice(state.unknown_patches)
    
    def _extract_context_region(self, state: ReconstructionState, target_patch: ReconstructionPatch) -> torch.Tensor:
        """Extract context region around target patch for prediction"""
        
        # Calculate context region bounds
        context_x = max(0, target_patch.x - (self.context_size - self.patch_size) // 2)
        context_y = max(0, target_patch.y - (self.context_size - self.patch_size) // 2)
        
        h, w = state.current_reconstruction.shape[:2]
        context_x = min(context_x, w - self.context_size)
        context_y = min(context_y, h - self.context_size)
        
        # Extract context region from current reconstruction
        context_region = state.current_reconstruction[
            context_y:context_y+self.context_size,
            context_x:context_x+self.context_size
        ].copy()
        
        # Mask out the target patch area (set to zero/unknown)
        target_rel_x = target_patch.x - context_x
        target_rel_y = target_patch.y - context_y
        
        context_region[
            target_rel_y:target_rel_y+self.patch_size,
            target_rel_x:target_rel_x+self.patch_size
        ] = 0
        
        # Convert to tensor
        context_tensor = torch.from_numpy(context_region).float().permute(2, 0, 1).unsqueeze(0)
        context_tensor = context_tensor.to(self.device) / 255.0
        
        return context_tensor
    
    def _predict_patch(self, context_region: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict patch from context using reconstruction network"""
        
        self.reconstruction_network.eval()
        
        with torch.no_grad():
            predicted_patch, confidence = self.reconstruction_network(context_region)
        
        return predicted_patch, confidence
    
    def _update_reconstruction(self, 
                             state: ReconstructionState, 
                             target_patch: ReconstructionPatch, 
                             predicted_patch: torch.Tensor, 
                             confidence: torch.Tensor):
        """Update reconstruction with predicted patch"""
        
        # Convert prediction to numpy
        predicted_pixels = predicted_patch.cpu().numpy().reshape(self.patch_size, self.patch_size, 3)
        predicted_pixels = (predicted_pixels * 255).astype(np.uint8)
        
        # Update patch
        target_patch.pixels = predicted_pixels
        target_patch.is_known = True
        target_patch.confidence = float(confidence.item())
        target_patch.reconstruction_attempts += 1
        
        # Move from unknown to known
        state.unknown_patches.remove(target_patch)
        state.known_patches.append(target_patch)
        
        # Update reconstruction
        state.current_reconstruction[
            target_patch.y:target_patch.y+target_patch.height,
            target_patch.x:target_patch.x+target_patch.width
        ] = predicted_pixels
    
    def _assess_reconstruction_quality(self, state: ReconstructionState, original_image: np.ndarray) -> float:
        """Assess quality of current reconstruction against original"""
        
        # Calculate SSIM-like quality measure
        reconstruction = state.current_reconstruction.astype(np.float32) / 255.0
        original = original_image.astype(np.float32) / 255.0
        
        # Only compare known regions
        mask = np.zeros(original.shape[:2], dtype=bool)
        for patch in state.known_patches:
            mask[patch.y:patch.y+patch.height, patch.x:patch.x+patch.width] = True
        
        if np.sum(mask) == 0:
            return 0.0
        
        # Calculate MSE in known regions
        mse = np.mean((reconstruction[mask] - original[mask])**2)
        
        # Convert to quality score (higher is better)
        quality = 1.0 / (1.0 + mse * 10)
        
        return quality
    
    def _generate_learning_feedback(self, 
                                  target_patch: ReconstructionPatch,
                                  predicted_patch: torch.Tensor,
                                  confidence: torch.Tensor,
                                  quality: float,
                                  original_image: np.ndarray) -> Dict[str, Any]:
        """Generate learning feedback for autonomous learning"""
        
        # Get ground truth patch
        gt_patch = original_image[
            target_patch.y:target_patch.y+target_patch.height,
            target_patch.x:target_patch.x+target_patch.width
        ]
        
        # Calculate prediction error
        predicted_pixels = predicted_patch.cpu().numpy().reshape(self.patch_size, self.patch_size, 3)
        predicted_pixels = (predicted_pixels * 255).astype(np.uint8)
        
        patch_mse = np.mean((predicted_pixels.astype(float) - gt_patch.astype(float))**2)
        patch_quality = 1.0 / (1.0 + patch_mse / 1000)
        
        return {
            'patch_location': (target_patch.x, target_patch.y),
            'prediction_quality': patch_quality,
            'confidence': float(confidence.item()),
            'overall_quality': quality,
            'patch_mse': patch_mse,
            'confidence_accuracy': abs(patch_quality - float(confidence.item())),
            'learning_signal': patch_quality - float(confidence.item())  # How much to adjust
        }
    
    def _update_autonomous_learning(self, learning_feedback: Dict[str, Any]):
        """Update autonomous learning systems"""
        
        # Train reconstruction network if we have learning signal
        if abs(learning_feedback['learning_signal']) > 0.1:
            self._train_reconstruction_network(learning_feedback)
        
        # Update objective function
        objective_evidence = {
            'reconstruction_quality': learning_feedback['overall_quality'],
            'prediction_confidence': learning_feedback['confidence'],
            'learning_progress': learning_feedback['prediction_quality']
        }
        
        self.objective_engine.update_objective(objective_evidence)
    
    def _train_reconstruction_network(self, learning_feedback: Dict[str, Any]):
        """Train the reconstruction network based on feedback"""
        
        # This would implement actual training
        # For now, just log the learning signal
        logger.debug(f"Learning signal: {learning_feedback['learning_signal']:.3f}")
        
        # In full implementation, would:
        # 1. Create training batch from recent predictions
        # 2. Calculate loss based on ground truth
        # 3. Backpropagate and update network
        # 4. Update learning rate based on progress
    
    def _should_adapt_strategy(self) -> bool:
        """Check if reconstruction strategy should be adapted"""
        
        if len(self.reconstruction_history) < 5:
            return False
        
        # Check if quality improvement has stalled
        recent_qualities = [h['quality'] for h in self.reconstruction_history[-5:]]
        quality_improvement = recent_qualities[-1] - recent_qualities[0]
        
        return quality_improvement < 0.01  # Less than 1% improvement
    
    def _adapt_reconstruction_strategy(self):
        """Adapt reconstruction strategy based on current performance"""
        
        logger.info("Adapting reconstruction strategy due to stalled progress")
        
        # Could implement strategy adaptation here
        # For now, just log the adaptation
        pass
    
    def _generate_autonomous_analysis_results(self, original_image: np.ndarray) -> Dict[str, Any]:
        """Generate final analysis results from autonomous reconstruction"""
        
        final_quality = self._assess_reconstruction_quality(self.current_state, original_image)
        
        # Calculate reconstruction metrics
        total_patches = len(self.current_state.known_patches) + len(self.current_state.unknown_patches)
        reconstructed_patches = len(self.current_state.known_patches)
        
        # Analyze reconstruction patterns
        reconstruction_analysis = self._analyze_reconstruction_patterns()
        
        # Generate understanding insights
        understanding_insights = self._generate_understanding_insights(original_image)
        
        return {
            'autonomous_reconstruction': {
                'final_quality': final_quality,
                'reconstruction_complete': len(self.current_state.unknown_patches) == 0,
                'patches_reconstructed': reconstructed_patches,
                'total_patches': total_patches,
                'completion_percentage': (reconstructed_patches / total_patches) * 100,
                'average_confidence': np.mean([p.confidence for p in self.current_state.known_patches]),
                'reconstruction_iterations': self.current_state.iteration
            },
            'reconstruction_analysis': reconstruction_analysis,
            'understanding_insights': understanding_insights,
            'reconstruction_image': self.current_state.current_reconstruction,
            'original_image': original_image,
            'reconstruction_history': self.reconstruction_history,
            'objective_state': self.objective_engine.get_belief_summary(),
            'learning_progress': self.current_state.learning_progress
        }
    
    def _analyze_reconstruction_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in the reconstruction process"""
        
        if not self.reconstruction_history:
            return {'note': 'No reconstruction history'}
        
        qualities = [h['quality'] for h in self.reconstruction_history]
        confidences = [h['confidence'] for h in self.reconstruction_history]
        
        return {
            'quality_progression': {
                'initial': qualities[0] if qualities else 0.0,
                'final': qualities[-1] if qualities else 0.0,
                'improvement': qualities[-1] - qualities[0] if len(qualities) > 1 else 0.0,
                'progression_rate': np.mean(np.diff(qualities)) if len(qualities) > 1 else 0.0
            },
            'confidence_analysis': {
                'average_confidence': np.mean(confidences),
                'confidence_stability': 1.0 - np.std(confidences),
                'confidence_trend': np.mean(np.diff(confidences)) if len(confidences) > 1 else 0.0
            },
            'learning_efficiency': {
                'iterations_to_convergence': len(self.reconstruction_history),
                'quality_per_iteration': qualities[-1] / len(qualities) if qualities else 0.0
            }
        }
    
    def _generate_understanding_insights(self, original_image: np.ndarray) -> Dict[str, Any]:
        """Generate insights about what the system understood from reconstruction"""
        
        insights = {
            'reconstruction_demonstrates': [],
            'understanding_level': 'partial',
            'key_insights': []
        }
        
        final_quality = self.current_state.reconstruction_quality
        
        # Determine understanding level
        if final_quality > 0.95:
            insights['understanding_level'] = 'excellent'
            insights['reconstruction_demonstrates'].append('Perfect pixel-level understanding')
        elif final_quality > 0.8:
            insights['understanding_level'] = 'good'
            insights['reconstruction_demonstrates'].append('Strong structural understanding')
        elif final_quality > 0.6:
            insights['understanding_level'] = 'moderate'
            insights['reconstruction_demonstrates'].append('Basic pattern recognition')
        else:
            insights['understanding_level'] = 'limited'
            insights['reconstruction_demonstrates'].append('Minimal understanding demonstrated')
        
        # Key insights based on reconstruction process
        if self.current_state.iteration < 20:
            insights['key_insights'].append('Rapid convergence suggests clear visual patterns')
        
        avg_confidence = np.mean([p.confidence for p in self.current_state.known_patches])
        if avg_confidence > 0.8:
            insights['key_insights'].append('High confidence indicates strong predictive capability')
        
        if len(self.current_state.unknown_patches) == 0:
            insights['key_insights'].append('Complete reconstruction achieved - full image understanding')
        
        return insights
    
    def save_reconstruction_state(self, save_path: str):
        """Save the current reconstruction state"""
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save reconstruction network
        torch.save({
            'model_state_dict': self.reconstruction_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_dir / 'reconstruction_network.pt')
        
        # Save reconstruction state and history
        import pickle
        with open(save_dir / 'reconstruction_state.pkl', 'wb') as f:
            pickle.dump({
                'current_state': self.current_state,
                'reconstruction_history': self.reconstruction_history,
                'patch_size': self.patch_size,
                'context_size': self.context_size
            }, f)
        
        logger.info(f"Saved autonomous reconstruction state to {save_path}")
    
    def load_reconstruction_state(self, save_path: str):
        """Load reconstruction state"""
        
        save_dir = Path(save_path)
        
        # Load reconstruction network
        network_path = save_dir / 'reconstruction_network.pt'
        if network_path.exists():
            checkpoint = torch.load(network_path, map_location=self.device)
            self.reconstruction_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load reconstruction state
        state_path = save_dir / 'reconstruction_state.pkl'
        if state_path.exists():
            import pickle
            with open(state_path, 'rb') as f:
                state_data = pickle.load(f)
                self.current_state = state_data['current_state']
                self.reconstruction_history = state_data['reconstruction_history']
        
        logger.info(f"Loaded autonomous reconstruction state from {save_path}") 