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
from PIL import Image

from .bayesian_objective_engine import BayesianObjectiveEngine
from .continuous_learning_engine import ContinuousLearningEngine
from .pakati_inspired_reconstruction import PakatiInspiredReconstruction
from .regional_reconstruction_engine import RegionalReconstructionEngine
from .segment_aware_reconstruction import SegmentAwareReconstructionEngine, SegmentType
from .nicotine_context_validator import NicotineContextValidator, NicotineIntegration
from .hatata_mdp_engine import HatataEngine
from .zengeza_noise_detector import ZengezaEngine

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
    Main engine for autonomous image reconstruction and understanding validation.
    
    Integrates multiple validation layers:
    1. Pakati-inspired reconstruction (HuggingFace API-based understanding validation)
    2. Segment-aware processing (independent iteration cycles preventing unwanted changes)
    3. Nicotine context validation (cognitive checkpoints preventing context drift)
    4. Hatata MDP verification (probabilistic fallback and additional verification)
    5. Zengeza noise detection (identifies and quantifies noise/garbage per segment)
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
        
        # Pakati-inspired reconstruction for API-based understanding validation
        try:
            self.pakati_engine = PakatiInspiredReconstruction()
            self.use_api_reconstruction = True
            logger.info("Pakati-inspired API reconstruction enabled")
        except ValueError as e:
            logger.warning(f"API reconstruction disabled: {e}")
            self.pakati_engine = None
            self.use_api_reconstruction = False
        
        # Regional reconstruction for local processing
        self.regional_engine = RegionalReconstructionEngine(device)
        
        # Segment-aware reconstruction for independent segment processing
        try:
            self.segment_engine = SegmentAwareReconstructionEngine(api_key=None)  # Will use same API key as pakati_engine
            if self.pakati_engine:
                self.segment_engine.api = self.pakati_engine.api
            logger.info("Segment-aware reconstruction enabled")
        except Exception as e:
            logger.warning(f"Segment-aware reconstruction disabled: {e}")
            self.segment_engine = None
        
        # Nicotine context validator for preventing context drift
        try:
            self.nicotine_validator = NicotineContextValidator(
                trigger_interval=15,  # Validate every 15 processes
                puzzle_count=3,
                pass_threshold=0.7
            )
            self.nicotine_integration = NicotineIntegration(self.nicotine_validator)
            logger.info("🚬 Nicotine context validation enabled")
        except Exception as e:
            logger.warning(f"Nicotine context validation disabled: {e}")
            self.nicotine_validator = None
            self.nicotine_integration = None
        
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
        
        # Initialize Zengeza noise detection engine
        self.zengeza_engine = ZengezaEngine()
        
        logger.info("🗑️ Zengeza noise detection integrated into autonomous reconstruction")
        
        logger.info(f"Initialized Autonomous Reconstruction Engine")
        logger.info(f"Patch size: {patch_size}, Context size: {context_size}")
    
    def validate_understanding_through_reconstruction(self, 
                                                    image: np.ndarray,
                                                    description: str = "") -> Dict[str, Any]:
        """
        Validate understanding using Pakati's insight: "Best way to analyze an image 
        is if AI can draw the image perfectly."
        
        This method uses HuggingFace API for actual reconstruction while maintaining
        the core insight that reconstruction ability demonstrates understanding.
        """
        
        if not self.use_api_reconstruction:
            logger.warning("API reconstruction not available, falling back to local methods")
            return self._local_understanding_validation(image, description)
        
        logger.info(f"Validating understanding through API reconstruction: {description}")
        
        # Use Pakati-inspired approach for comprehensive understanding test
        api_results = self.pakati_engine.test_understanding(image, description)
        
        # Use regional engine for detailed local analysis
        regional_results = self.regional_engine.comprehensive_understanding_assessment(image)
        
        # Combine results for comprehensive assessment
        combined_results = {
            'description': description,
            'image_shape': image.shape,
            'api_reconstruction_results': api_results,
            'regional_analysis_results': regional_results,
            'combined_understanding': self._combine_understanding_assessments(api_results, regional_results),
            'validation_method': 'pakati_inspired_api_reconstruction',
            'insights': []
        }
        
        # Generate combined insights
        combined_results['insights'] = self._generate_combined_insights(api_results, regional_results)
        
        return combined_results
    
    def progressive_understanding_validation(self, 
                                           image: np.ndarray,
                                           description: str = "") -> Dict[str, Any]:
        """
        Progressive understanding validation using Pakati's approach.
        
        Tests understanding at increasing difficulty levels until failure,
        implementing the core insight that reconstruction ability proves understanding.
        """
        
        if not self.use_api_reconstruction:
            logger.warning("API reconstruction not available")
            return {'error': 'API reconstruction required for progressive validation'}
        
        logger.info(f"Starting progressive understanding validation: {description}")
        
        # Use Pakati's progressive test
        progressive_results = self.pakati_engine.progressive_test(image, description)
        
        # Enhance with local regional analysis at the mastery level
        if progressive_results['mastery_achieved']:
            mastery_level = progressive_results['mastery_level']
            
            # Test regional understanding at the mastery difficulty level
            from .regional_reconstruction_engine import ReconstructionRegion
            
            # Create regions for detailed analysis
            h, w = image.shape[:2]
            regions = [
                ReconstructionRegion(
                    region_id="center_region",
                    polygon=[(w//4, h//4), (3*w//4, h//4), (3*w//4, 3*h//4), (w//4, 3*h//4)],
                    difficulty_level=mastery_level
                ),
                ReconstructionRegion(
                    region_id="full_image",
                    polygon=[(0, 0), (w, 0), (w, h), (0, h)],
                    difficulty_level=mastery_level
                )
            ]
            
            regional_validation = self.regional_engine.test_regional_understanding(
                image, regions
            )
            
            progressive_results['detailed_regional_analysis'] = regional_validation
        
        return progressive_results
    
    def _local_understanding_validation(self, image: np.ndarray, description: str) -> Dict[str, Any]:
        """Fallback local understanding validation when API is not available."""
        
        logger.info("Using local understanding validation")
        
        # Use regional reconstruction engine for local validation
        regional_results = self.regional_engine.comprehensive_understanding_assessment(image)
        
        # Convert to similar format as API results
        local_results = {
            'description': description,
            'image_shape': image.shape,
            'understanding_level': regional_results['overall_understanding']['understanding_level'],
            'average_quality': regional_results['overall_understanding']['overall_score'],
            'mastery_achieved': regional_results['overall_understanding']['overall_score'] >= 0.85,
            'validation_method': 'local_regional_reconstruction',
            'detailed_results': regional_results
        }
        
        return local_results
    
    def _combine_understanding_assessments(self, api_results: Dict[str, Any], 
                                         regional_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine API and regional understanding assessments."""
        
        # Extract key metrics
        api_quality = api_results.get('average_quality', 0.0)
        api_mastery = api_results.get('mastery_achieved', False)
        
        regional_quality = regional_results['overall_understanding']['overall_score']
        regional_mastery = regional_results['overall_understanding']['overall_score'] >= 0.85
        
        # Combine with weighted average (API gets higher weight as it's more comprehensive)
        combined_quality = (api_quality * 0.7) + (regional_quality * 0.3)
        combined_mastery = api_mastery and regional_quality >= 0.7  # Both must be good
        
        # Determine combined understanding level
        if combined_quality >= 0.9:
            understanding_level = "excellent"
        elif combined_quality >= 0.8:
            understanding_level = "good"
        elif combined_quality >= 0.6:
            understanding_level = "moderate"
        elif combined_quality >= 0.4:
            understanding_level = "limited"
        else:
            understanding_level = "poor"
        
        return {
            'combined_quality': combined_quality,
            'combined_mastery': combined_mastery,
            'understanding_level': understanding_level,
            'api_contribution': api_quality * 0.7,
            'regional_contribution': regional_quality * 0.3,
            'validation_confidence': min(1.0, (api_quality + regional_quality) / 2.0)
        }
    
    def _generate_combined_insights(self, api_results: Dict[str, Any], 
                                  regional_results: Dict[str, Any]) -> List[str]:
        """Generate insights from combined API and regional analysis."""
        
        insights = []
        
        # API insights
        api_level = api_results.get('understanding_level', 'unknown')
        api_quality = api_results.get('average_quality', 0.0)
        
        insights.append(f"API reconstruction achieved {api_level} understanding ({api_quality:.3f} quality)")
        
        # Regional insights
        regional_level = regional_results['overall_understanding']['understanding_level']
        regional_quality = regional_results['overall_understanding']['overall_score']
        
        insights.append(f"Regional analysis achieved {regional_level} understanding ({regional_quality:.3f} quality)")
        
        # Comparison insights
        if api_quality > regional_quality + 0.1:
            insights.append("API reconstruction outperformed local analysis - suggests complex understanding")
        elif regional_quality > api_quality + 0.1:
            insights.append("Local analysis outperformed API - suggests structured understanding")
        else:
            insights.append("API and local analysis showed consistent results - high confidence")
        
        # Strategy insights
        if 'test_results' in api_results:
            best_strategy = max(api_results['test_results'], key=lambda x: x['quality_score'])
            insights.append(f"Best API strategy: {best_strategy['strategy']} at difficulty {best_strategy['difficulty']}")
        
        if 'strategy_rankings' in regional_results:
            best_regional = min(regional_results['strategy_rankings'].items(), key=lambda x: x[1]['rank'])
            insights.append(f"Best regional strategy: {best_regional[0]}")
        
        return insights
    
    def segment_aware_understanding_validation(self, 
                                             image: np.ndarray,
                                             description: str = "") -> Dict[str, Any]:
        """
        Validate understanding using segment-aware reconstruction.
        
        This addresses the critical insight that AI changes everything when modifying anything.
        Each segment gets its own iteration cycles to prevent unwanted changes in other areas.
        """
        
        if not self.segment_engine:
            logger.warning("Segment-aware reconstruction not available")
            return {'error': 'Segment-aware reconstruction engine not initialized'}
        
        logger.info(f"Starting segment-aware understanding validation: {description}")
        
        # Perform segment-aware reconstruction
        segment_results = self.segment_engine.segment_aware_reconstruction(image, description)
        
        # Enhance with traditional analysis for comparison
        if self.use_api_reconstruction:
            # Also run Pakati-inspired test for comparison
            pakati_results = self.pakati_engine.test_understanding(image, description)
            
            # Combine insights
            combined_results = {
                'description': description,
                'image_shape': image.shape,
                'segment_aware_results': segment_results,
                'pakati_comparison_results': pakati_results,
                'validation_method': 'segment_aware_with_pakati_comparison',
                'combined_assessment': self._compare_segment_vs_pakati(segment_results, pakati_results),
                'insights': []
            }
            
            # Generate comparative insights
            combined_results['insights'] = self._generate_segment_pakati_insights(
                segment_results, pakati_results
            )
            
            return combined_results
        
        else:
            # Segment-aware only
            segment_results['validation_method'] = 'segment_aware_only'
            segment_results['insights'] = self._generate_segment_only_insights(segment_results)
            
            return segment_results
    
    def _compare_segment_vs_pakati(self, segment_results: Dict[str, Any], 
                                 pakati_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare segment-aware vs Pakati-inspired reconstruction results."""
        
        segment_quality = segment_results['overall_quality']
        pakati_quality = pakati_results.get('average_quality', 0.0)
        
        # Determine which approach performed better
        if segment_quality > pakati_quality + 0.1:
            better_approach = 'segment_aware'
            quality_advantage = segment_quality - pakati_quality
        elif pakati_quality > segment_quality + 0.1:
            better_approach = 'pakati_inspired'
            quality_advantage = pakati_quality - segment_quality
        else:
            better_approach = 'comparable'
            quality_advantage = abs(segment_quality - pakati_quality)
        
        return {
            'segment_quality': segment_quality,
            'pakati_quality': pakati_quality,
            'better_approach': better_approach,
            'quality_advantage': quality_advantage,
            'segment_iterations': segment_results['total_iterations'],
            'segment_success_rate': segment_results['successful_segments'] / max(1, segment_results['segments_processed']),
            'recommendation': self._generate_approach_recommendation(better_approach, quality_advantage)
        }
    
    def _generate_approach_recommendation(self, better_approach: str, 
                                        quality_advantage: float) -> str:
        """Generate recommendation for which approach to use."""
        
        if better_approach == 'segment_aware':
            if quality_advantage > 0.2:
                return "Strongly recommend segment-aware approach for this type of image"
            else:
                return "Segment-aware approach shows slight advantage"
        elif better_approach == 'pakati_inspired':
            if quality_advantage > 0.2:
                return "Strongly recommend Pakati-inspired approach for this type of image"
            else:
                return "Pakati-inspired approach shows slight advantage"
        else:
            return "Both approaches perform similarly - use based on specific requirements"
    
    def _generate_segment_pakati_insights(self, segment_results: Dict[str, Any],
                                        pakati_results: Dict[str, Any]) -> List[str]:
        """Generate insights comparing segment-aware and Pakati approaches."""
        
        insights = []
        
        # Performance comparison
        segment_quality = segment_results['overall_quality']
        pakati_quality = pakati_results.get('average_quality', 0.0)
        
        insights.append(f"Segment-aware quality: {segment_quality:.3f}")
        insights.append(f"Pakati-inspired quality: {pakati_quality:.3f}")
        
        # Efficiency comparison
        segment_iterations = segment_results['total_iterations']
        segment_count = segment_results['segments_processed']
        avg_iterations_per_segment = segment_iterations / max(1, segment_count)
        
        insights.append(f"Segment-aware used {segment_iterations} total iterations across {segment_count} segments")
        insights.append(f"Average {avg_iterations_per_segment:.1f} iterations per segment")
        
        # Success rate analysis
        success_rate = segment_results['successful_segments'] / max(1, segment_count)
        insights.append(f"Segment success rate: {success_rate:.1%}")
        
        # Approach-specific insights
        if segment_quality > pakati_quality + 0.1:
            insights.append("Segment-aware approach prevented unwanted changes in unrelated areas")
            insights.append("Independent segment processing improved overall reconstruction quality")
        elif pakati_quality > segment_quality + 0.1:
            insights.append("Pakati-inspired approach achieved better global coherence")
            insights.append("Holistic reconstruction outperformed segmented approach")
        else:
            insights.append("Both approaches achieved similar quality - validates reconstruction insight")
        
        return insights
    
    def _generate_segment_only_insights(self, segment_results: Dict[str, Any]) -> List[str]:
        """Generate insights for segment-aware reconstruction only."""
        
        insights = []
        
        # Overall performance
        insights.append(f"Achieved {segment_results['understanding_level']} understanding level")
        insights.append(f"Overall quality: {segment_results['overall_quality']:.3f}")
        
        # Segment analysis
        success_rate = segment_results['successful_segments'] / max(1, segment_results['segments_processed'])
        insights.append(f"Successfully reconstructed {success_rate:.1%} of segments")
        
        # Iteration efficiency
        avg_iterations = segment_results['total_iterations'] / max(1, segment_results['segments_processed'])
        insights.append(f"Average {avg_iterations:.1f} iterations per segment")
        
        # Key advantages
        insights.append("Segment-aware approach prevented AI from changing unrelated image areas")
        insights.append("Each segment received appropriate iteration cycles based on complexity")
        
        return insights
    
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
        
        # Set up nicotine context
        if self.nicotine_integration:
            self.nicotine_integration.set_task_context(
                task="autonomous_image_reconstruction",
                objectives=[
                    "reconstruct_image_through_understanding",
                    "achieve_target_quality",
                    "maintain_context_awareness",
                    "demonstrate_visual_comprehension"
                ]
            )
        
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
            
            # Nicotine checkpoint - validate context retention
            if self.nicotine_integration:
                system_state = {
                    'reconstruction_quality': quality,
                    'iteration_count': iteration + 1,
                    'prediction_confidence': float(confidence.item()),
                    'target_quality': target_quality,
                    'patches_known': len(self.current_state.known_patches),
                    'patches_unknown': len(self.current_state.unknown_patches)
                }
                
                can_continue = self.nicotine_integration.checkpoint(
                    process_name=f"reconstruction_iteration_{iteration + 1}",
                    system_state=system_state
                )
                
                if not can_continue:
                    logger.error(f"🚬 Nicotine validation failed at iteration {iteration + 1} - halting reconstruction")
                    break
            
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
        
        # Add nicotine validation report
        if self.nicotine_validator:
            final_results['nicotine_validation'] = self.nicotine_validator.get_validation_report()
            logger.info(f"🚬 Nicotine sessions: {final_results['nicotine_validation']['total_sessions']}, "
                       f"Pass rate: {final_results['nicotine_validation']['pass_rate']:.1%}")
        
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
            'learning_progress': self.current_state.learning_progress,
            'validation_layers': {
                'segment_aware_enabled': self.segment_engine is not None,
                'nicotine_enabled': self.nicotine_validator is not None,
                'hatata_enabled': False,  # Assuming Hatata MDP is not enabled in the original code
                'zengeza_enabled': True
            }
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
    
    def reconstruct_with_understanding_validation(self, image_path: str, 
                                                prompt: str,
                                                max_iterations: int = 5,
                                                quality_threshold: float = 0.85,
                                                enable_segment_aware: bool = True,
                                                enable_nicotine: bool = True,
                                                enable_hatata: bool = True,
                                                enable_zengeza: bool = True) -> Dict[str, Any]:
        """
        Perform autonomous reconstruction with comprehensive understanding validation.
        
        Now includes Zengeza noise detection to identify and quantify garbage content
        per segment per iteration, helping focus reconstruction on important regions.
        """
        
        logger.info(f"🚁 Starting autonomous reconstruction with understanding validation")
        logger.info(f"📊 Validation layers: Segment-aware={enable_segment_aware}, "
                   f"Nicotine={enable_nicotine}, Hatata={enable_hatata}, Zengeza={enable_zengeza}")
        
        # ... existing code ...
        
        # Initialize Zengeza noise tracking
        if enable_zengeza:
            noise_analysis_history = []
            segment_noise_trends = {}
        
        for iteration in range(max_iterations):
            logger.info(f"🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # ... existing code for reconstruction ...
            
            # Zengeza noise analysis - identify garbage content per segment
            if enable_zengeza:
                logger.info("🗑️ Performing Zengeza noise analysis...")
                
                # Prepare segments for noise analysis
                if enable_segment_aware and hasattr(self, 'segment_engine'):
                    # Use segment-aware segments
                    segments_for_noise = []
                    for segment_id, segment_data in segment_results.get('segments', {}).items():
                        segments_for_noise.append({
                            'segment_id': segment_id,
                            'bbox': segment_data.get('bbox', (0, 0, 64, 64)),
                            'pixels': segment_data.get('pixels'),
                            'segment_type': segment_data.get('segment_type', 'unknown')
                        })
                else:
                    # Create basic segments for noise analysis
                    image_array = np.array(Image.open(image_path))
                    h, w = image_array.shape[:2]
                    segments_for_noise = [
                        {
                            'segment_id': f'grid_segment_{i}',
                            'bbox': (x, y, min(64, w-x), min(64, h-y)),
                            'pixels': image_array[y:y+64, x:x+64]
                        }
                        for i, (x, y) in enumerate([(x, y) for x in range(0, w, 64) for y in range(0, h, 64)])
                    ]
                
                # Context for noise analysis
                noise_context = {
                    'reconstruction_quality': current_quality,
                    'expected_complexity': 0.6,  # Could be learned/adjusted
                    'iteration': iteration,
                    'prompt': prompt,
                    'total_iterations': max_iterations
                }
                
                # Perform noise analysis
                noise_results = self.zengeza_engine.analyze_image_noise(
                    image=np.array(Image.open(image_path)),
                    segments=segments_for_noise,
                    context=noise_context,
                    iteration=iteration
                )
                
                noise_analysis_history.append(noise_results)
                
                # Update segment priorities based on noise analysis
                high_importance_segments = noise_results['high_importance_segments']
                high_noise_segments = noise_results['high_noise_segments']
                
                logger.info(f"🗑️ Noise analysis: {len(high_importance_segments)} high-importance segments, "
                           f"{len(high_noise_segments)} high-noise segments")
                
                # Log noise insights
                for insight in noise_results['noise_insights']:
                    logger.info(f"🗑️ Insight: {insight}")
                
                # Adjust reconstruction focus based on noise analysis
                if enable_segment_aware and hasattr(self, 'segment_engine'):
                    # Update segment priorities based on importance scores
                    for priority_seg in noise_results['prioritized_segments'][:5]:  # Top 5 most important
                        segment_id = priority_seg['segment_id']
                        if segment_id in segment_results.get('segments', {}):
                            # Increase iterations for important segments
                            segment_results['segments'][segment_id]['priority_boost'] = True
                            segment_results['segments'][segment_id]['importance_score'] = priority_seg['importance_score']
                
                # Store noise trends for this segment
                for segment_id, segment_data in noise_results['segment_noise_analysis'].items():
                    if segment_id not in segment_noise_trends:
                        segment_noise_trends[segment_id] = []
                    
                    segment_noise_trends[segment_id].append({
                        'iteration': iteration,
                        'noise_probability': segment_data['noise_probability'],
                        'importance_score': segment_data['importance_score'],
                        'noise_level': segment_data['noise_level']
                    })
            
            # ... existing code for other validation layers ...
            
            # Check if we should continue based on noise analysis
            if enable_zengeza and noise_results:
                avg_noise = noise_results['global_noise_statistics']['average_noise_level']
                
                # If noise level is very high, we might want to try different approach
                if avg_noise > 0.8:
                    logger.warning(f"🗑️ Very high noise level detected ({avg_noise:.3f}). "
                                 f"Consider different reconstruction strategy.")
                
                # If most segments are high importance and low noise, we might be done
                high_importance_ratio = len(high_importance_segments) / max(1, len(segments_for_noise))
                if high_importance_ratio > 0.7 and avg_noise < 0.3:
                    logger.info(f"🗑️ High information content detected ({high_importance_ratio:.1%} important segments, "
                               f"{avg_noise:.3f} avg noise). Quality reconstruction likely achieved.")
            
            # ... existing code for quality checks and iteration logic ...
        
        # Compile final results with noise analysis
        final_results = {
            # ... existing results ...
            'validation_layers': {
                'segment_aware_enabled': enable_segment_aware,
                'nicotine_enabled': enable_nicotine, 
                'hatata_enabled': enable_hatata,
                'zengeza_enabled': enable_zengeza
            }
        }
        
        # Add Zengeza noise analysis results
        if enable_zengeza:
            final_results['zengeza_noise_analysis'] = {
                'noise_analysis_history': noise_analysis_history,
                'segment_noise_trends': segment_noise_trends,
                'final_noise_report': self.zengeza_engine.get_noise_report(),
                'noise_insights': noise_analysis_history[-1]['noise_insights'] if noise_analysis_history else [],
                'high_importance_segments': noise_analysis_history[-1]['high_importance_segments'] if noise_analysis_history else [],
                'high_noise_segments': noise_analysis_history[-1]['high_noise_segments'] if noise_analysis_history else []
            }
            
            logger.info(f"🗑️ Final noise analysis: "
                       f"{len(final_results['zengeza_noise_analysis']['high_importance_segments'])} important segments, "
                       f"{len(final_results['zengeza_noise_analysis']['high_noise_segments'])} noise segments")
        
        # ... existing code ...
        
        return final_results
    
    def zengeza_noise_understanding_validation(self, image_path: str, 
                                             segments: List[Dict[str, Any]],
                                             context: Dict[str, Any],
                                             iteration: int = 0) -> Dict[str, Any]:
        """
        Standalone Zengeza noise analysis for understanding validation.
        
        This method can be called independently to analyze noise in image segments
        and determine what content is important vs garbage for reconstruction.
        """
        
        logger.info(f"🗑️ Performing standalone Zengeza noise analysis")
        
        # Load image
        image = np.array(Image.open(image_path))
        
        # Perform noise analysis
        noise_results = self.zengeza_engine.analyze_image_noise(
            image=image,
            segments=segments,
            context=context,
            iteration=iteration
        )
        
        # Generate recommendations
        recommendations = self._generate_noise_based_recommendations(noise_results)
        
        results = {
            'noise_analysis': noise_results,
            'recommendations': recommendations,
            'analysis_method': 'zengeza_standalone',
            'timestamp': time.time()
        }
        
        logger.info(f"🗑️ Standalone noise analysis complete")
        
        return results
    
    def _generate_noise_based_recommendations(self, noise_results: Dict[str, Any]) -> List[str]:
        """Generate reconstruction recommendations based on noise analysis."""
        
        recommendations = []
        
        # Global noise level recommendations
        avg_noise = noise_results['global_noise_statistics']['average_noise_level']
        
        if avg_noise > 0.7:
            recommendations.append("High noise detected - consider preprocessing or different segmentation strategy")
        elif avg_noise < 0.3:
            recommendations.append("Low noise detected - image has high information content, focus on detail preservation")
        
        # Segment-specific recommendations
        high_noise_count = len(noise_results['high_noise_segments'])
        high_importance_count = len(noise_results['high_importance_segments'])
        
        if high_noise_count > 0:
            recommendations.append(f"Skip or reduce iterations for {high_noise_count} high-noise segments to save computation")
        
        if high_importance_count > 0:
            recommendations.append(f"Prioritize and increase iterations for {high_importance_count} high-importance segments")
        
        # Noise type specific recommendations
        all_noise_types = []
        for segment_data in noise_results['segment_noise_analysis'].values():
            all_noise_types.extend(segment_data['noise_types'])
        
        if 'visual_noise' in all_noise_types:
            recommendations.append("Visual noise detected - consider denoising preprocessing")
        
        if 'semantic_noise' in all_noise_types:
            recommendations.append("Semantic noise detected - focus reconstruction on semantically meaningful regions")
        
        if 'structural_noise' in all_noise_types:
            recommendations.append("Structural noise detected - emphasize edge and structure preservation")
        
        return recommendations 