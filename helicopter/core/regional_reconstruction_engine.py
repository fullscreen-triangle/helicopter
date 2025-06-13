"""
Regional Reconstruction Engine

Inspired by Pakati's regional control approach, this engine implements
autonomous reconstruction with regional masking strategies to test
understanding depth. Instead of generating from prompts, we reconstruct
from partial information to prove visual comprehension.

Core Insight: If an AI can perfectly reconstruct specific regions of an image
from surrounding context, it demonstrates true understanding of those regions.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MaskingStrategy(Enum):
    """Different masking strategies to test understanding depth."""
    RANDOM_PATCHES = "random_patches"
    CENTER_OUT = "center_out"
    EDGE_IN = "edge_in"
    PROGRESSIVE_REVEAL = "progressive_reveal"
    QUADRANT_REVEAL = "quadrant_reveal"
    FREQUENCY_BANDS = "frequency_bands"
    SEMANTIC_REGIONS = "semantic_regions"
    SPIRAL_REVEAL = "spiral_reveal"
    CHECKERBOARD = "checkerboard"
    RADIAL_BANDS = "radial_bands"


@dataclass
class ReconstructionRegion:
    """Defines a region for reconstruction testing."""
    
    region_id: str
    polygon: List[Tuple[int, int]]  # Vertices defining the region
    mask: np.ndarray = None
    difficulty_level: float = 0.5  # 0.0 = easy, 1.0 = very hard
    masking_strategy: MaskingStrategy = MaskingStrategy.RANDOM_PATCHES
    context_radius: int = 32  # How much surrounding context to provide
    target_quality: float = 0.85
    
    def __post_init__(self):
        if self.mask is None and self.polygon:
            self.mask = self._create_mask_from_polygon()
    
    def _create_mask_from_polygon(self) -> np.ndarray:
        """Create binary mask from polygon vertices."""
        # This would be implemented based on the polygon
        # For now, return a placeholder
        return np.ones((224, 224), dtype=np.uint8)


@dataclass
class ReconstructionChallenge:
    """Defines a reconstruction challenge with multiple difficulty levels."""
    
    challenge_id: str
    image: np.ndarray
    regions: List[ReconstructionRegion]
    masking_strategies: List[MaskingStrategy]
    difficulty_progression: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 0.9])
    success_threshold: float = 0.85
    max_attempts_per_level: int = 3
    
    def get_next_difficulty(self, current_level: float) -> Optional[float]:
        """Get next difficulty level in progression."""
        current_idx = -1
        for i, level in enumerate(self.difficulty_progression):
            if abs(level - current_level) < 0.01:
                current_idx = i
                break
        
        if current_idx >= 0 and current_idx < len(self.difficulty_progression) - 1:
            return self.difficulty_progression[current_idx + 1]
        return None


class RegionalMaskGenerator:
    """Generates masks for different reconstruction strategies."""
    
    def __init__(self):
        self.mask_cache = {}
    
    def generate_mask(self, strategy: MaskingStrategy, image_shape: Tuple[int, int], 
                     difficulty: float = 0.5, region_mask: np.ndarray = None) -> np.ndarray:
        """
        Generate a mask based on the specified strategy.
        
        Args:
            strategy: Masking strategy to use
            image_shape: (height, width) of the image
            difficulty: 0.0 = easy (more context), 1.0 = hard (less context)
            region_mask: Optional region constraint mask
        
        Returns:
            Binary mask where 1 = known pixels, 0 = pixels to reconstruct
        """
        
        cache_key = f"{strategy.value}_{image_shape}_{difficulty}"
        if cache_key in self.mask_cache:
            base_mask = self.mask_cache[cache_key].copy()
        else:
            base_mask = self._generate_base_mask(strategy, image_shape, difficulty)
            self.mask_cache[cache_key] = base_mask.copy()
        
        # Apply region constraint if provided
        if region_mask is not None:
            # Only reconstruct within the specified region
            final_mask = base_mask.copy()
            final_mask[region_mask == 0] = 1  # Keep pixels outside region as known
            return final_mask
        
        return base_mask
    
    def _generate_base_mask(self, strategy: MaskingStrategy, 
                           image_shape: Tuple[int, int], difficulty: float) -> np.ndarray:
        """Generate base mask for the given strategy."""
        
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Calculate how much to reveal based on difficulty
        # Lower difficulty = more revealed context
        reveal_ratio = 1.0 - difficulty
        
        if strategy == MaskingStrategy.RANDOM_PATCHES:
            return self._random_patches_mask(h, w, reveal_ratio)
        
        elif strategy == MaskingStrategy.CENTER_OUT:
            return self._center_out_mask(h, w, reveal_ratio)
        
        elif strategy == MaskingStrategy.EDGE_IN:
            return self._edge_in_mask(h, w, reveal_ratio)
        
        elif strategy == MaskingStrategy.PROGRESSIVE_REVEAL:
            return self._progressive_reveal_mask(h, w, reveal_ratio)
        
        elif strategy == MaskingStrategy.QUADRANT_REVEAL:
            return self._quadrant_reveal_mask(h, w, reveal_ratio)
        
        elif strategy == MaskingStrategy.FREQUENCY_BANDS:
            return self._frequency_bands_mask(h, w, reveal_ratio)
        
        elif strategy == MaskingStrategy.SPIRAL_REVEAL:
            return self._spiral_reveal_mask(h, w, reveal_ratio)
        
        elif strategy == MaskingStrategy.CHECKERBOARD:
            return self._checkerboard_mask(h, w, reveal_ratio)
        
        elif strategy == MaskingStrategy.RADIAL_BANDS:
            return self._radial_bands_mask(h, w, reveal_ratio)
        
        else:
            # Default to random patches
            return self._random_patches_mask(h, w, reveal_ratio)
    
    def _random_patches_mask(self, h: int, w: int, reveal_ratio: float) -> np.ndarray:
        """Generate random patches mask."""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create random patches of known pixels
        patch_size = max(8, int(min(h, w) * 0.1))
        num_patches = int((h * w * reveal_ratio) / (patch_size * patch_size))
        
        for _ in range(num_patches):
            y = np.random.randint(0, h - patch_size)
            x = np.random.randint(0, w - patch_size)
            mask[y:y+patch_size, x:x+patch_size] = 1
        
        return mask
    
    def _center_out_mask(self, h: int, w: int, reveal_ratio: float) -> np.ndarray:
        """Generate center-out reveal mask."""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        center_y, center_x = h // 2, w // 2
        max_radius = min(center_y, center_x)
        reveal_radius = int(max_radius * reveal_ratio)
        
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        mask[distance <= reveal_radius] = 1
        
        return mask
    
    def _edge_in_mask(self, h: int, w: int, reveal_ratio: float) -> np.ndarray:
        """Generate edge-in reveal mask."""
        mask = np.ones((h, w), dtype=np.uint8)
        
        # Create a border of known pixels, hide center
        border_size = int(min(h, w) * reveal_ratio * 0.5)
        
        mask[border_size:h-border_size, border_size:w-border_size] = 0
        
        return mask
    
    def _progressive_reveal_mask(self, h: int, w: int, reveal_ratio: float) -> np.ndarray:
        """Generate progressive reveal mask (left to right)."""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        reveal_width = int(w * reveal_ratio)
        mask[:, :reveal_width] = 1
        
        return mask
    
    def _quadrant_reveal_mask(self, h: int, w: int, reveal_ratio: float) -> np.ndarray:
        """Generate quadrant-based reveal mask."""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        mid_h, mid_w = h // 2, w // 2
        
        # Reveal quadrants based on reveal_ratio
        if reveal_ratio >= 0.25:
            mask[:mid_h, :mid_w] = 1  # Top-left
        if reveal_ratio >= 0.5:
            mask[:mid_h, mid_w:] = 1  # Top-right
        if reveal_ratio >= 0.75:
            mask[mid_h:, :mid_w] = 1  # Bottom-left
        if reveal_ratio >= 1.0:
            mask[mid_h:, mid_w:] = 1  # Bottom-right
        
        return mask
    
    def _frequency_bands_mask(self, h: int, w: int, reveal_ratio: float) -> np.ndarray:
        """Generate frequency bands mask (structure vs details)."""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create a pattern that reveals low-frequency (structure) information first
        # This is a simplified version - in practice, you'd use FFT
        
        # Create a grid pattern that gets finer as reveal_ratio increases
        grid_size = max(2, int(32 * (1.0 - reveal_ratio)))
        
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                # Reveal a small patch at each grid point
                patch_size = max(1, grid_size // 4)
                mask[y:min(y+patch_size, h), x:min(x+patch_size, w)] = 1
        
        return mask
    
    def _spiral_reveal_mask(self, h: int, w: int, reveal_ratio: float) -> np.ndarray:
        """Generate spiral reveal mask."""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        center_y, center_x = h // 2, w // 2
        total_pixels = h * w
        reveal_pixels = int(total_pixels * reveal_ratio)
        
        # Generate spiral coordinates
        spiral_coords = self._generate_spiral_coordinates(h, w, center_y, center_x)
        
        # Reveal pixels along the spiral
        for i, (y, x) in enumerate(spiral_coords[:reveal_pixels]):
            if 0 <= y < h and 0 <= x < w:
                mask[y, x] = 1
        
        return mask
    
    def _generate_spiral_coordinates(self, h: int, w: int, 
                                   center_y: int, center_x: int) -> List[Tuple[int, int]]:
        """Generate coordinates in spiral order."""
        coords = []
        x, y = center_x, center_y
        dx, dy = 0, -1
        
        for _ in range(max(h, w) ** 2):
            if (-w//2 < x <= w//2) and (-h//2 < y <= h//2):
                coords.append((center_y + y, center_x + x))
            
            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
                dx, dy = -dy, dx
            
            x, y = x + dx, y + dy
        
        return coords
    
    def _checkerboard_mask(self, h: int, w: int, reveal_ratio: float) -> np.ndarray:
        """Generate checkerboard pattern mask."""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Checkerboard size based on reveal ratio
        check_size = max(1, int(16 * (1.0 - reveal_ratio)))
        
        for y in range(0, h, check_size):
            for x in range(0, w, check_size):
                if ((y // check_size) + (x // check_size)) % 2 == 0:
                    mask[y:min(y+check_size, h), x:min(x+check_size, w)] = 1
        
        return mask
    
    def _radial_bands_mask(self, h: int, w: int, reveal_ratio: float) -> np.ndarray:
        """Generate radial bands mask."""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        center_y, center_x = h // 2, w // 2
        max_radius = min(center_y, center_x)
        
        # Create concentric bands
        num_bands = max(3, int(10 * reveal_ratio))
        band_width = max_radius / num_bands
        
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        
        for i in range(0, num_bands, 2):  # Reveal every other band
            inner_radius = i * band_width
            outer_radius = (i + 1) * band_width
            band_mask = (distance >= inner_radius) & (distance < outer_radius)
            mask[band_mask] = 1
        
        return mask


class RegionalReconstructionNetwork(nn.Module):
    """Neural network for regional reconstruction."""
    
    def __init__(self, input_channels: int = 3, hidden_dim: int = 256):
        super().__init__()
        
        # Encoder for context
        self.context_encoder = nn.Sequential(
            nn.Conv2d(input_channels + 1, 64, 3, padding=1),  # +1 for mask channel
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16))
        )
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for regional reconstruction.
        
        Args:
            masked_image: Image with masked regions (B, C, H, W)
            mask: Binary mask (B, 1, H, W) where 1=known, 0=reconstruct
        
        Returns:
            Tuple of (reconstructed_image, confidence_score)
        """
        
        # Combine image and mask as input
        x = torch.cat([masked_image, mask], dim=1)
        
        # Encode context
        features = self.context_encoder(x)
        
        # Estimate confidence
        confidence = self.confidence_head(features)
        
        # Decode reconstruction
        reconstruction = self.decoder(features)
        
        # Combine known and reconstructed regions
        final_image = masked_image * mask + reconstruction * (1 - mask)
        
        return final_image, confidence


class RegionalReconstructionEngine:
    """
    Main engine for regional reconstruction testing.
    
    This engine implements the core insight: if an AI can perfectly reconstruct
    specific regions from surrounding context, it demonstrates understanding.
    """
    
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.mask_generator = RegionalMaskGenerator()
        self.network = RegionalReconstructionNetwork().to(self.device)
        self.reconstruction_history = []
        
        # Performance tracking
        self.strategy_performance = {strategy: [] for strategy in MaskingStrategy}
        
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def test_regional_understanding(self, image: np.ndarray, 
                                  regions: List[ReconstructionRegion],
                                  strategies: List[MaskingStrategy] = None) -> Dict[str, Any]:
        """
        Test understanding of specific regions using various masking strategies.
        
        Args:
            image: Input image to test understanding on
            regions: List of regions to test
            strategies: Masking strategies to use (if None, uses all)
        
        Returns:
            Comprehensive understanding assessment
        """
        
        if strategies is None:
            strategies = list(MaskingStrategy)
        
        results = {
            'image_shape': image.shape,
            'regions_tested': len(regions),
            'strategies_used': [s.value for s in strategies],
            'region_results': {},
            'overall_understanding': {},
            'strategy_performance': {},
            'reconstruction_quality': {}
        }
        
        total_quality = 0.0
        total_tests = 0
        
        # Test each region with each strategy
        for region in regions:
            region_results = {
                'region_id': region.region_id,
                'strategy_results': {},
                'best_strategy': None,
                'worst_strategy': None,
                'average_quality': 0.0
            }
            
            strategy_qualities = {}
            
            for strategy in strategies:
                logger.info(f"Testing region {region.region_id} with strategy {strategy.value}")
                
                strategy_result = self._test_region_with_strategy(
                    image, region, strategy
                )
                
                region_results['strategy_results'][strategy.value] = strategy_result
                strategy_qualities[strategy] = strategy_result['reconstruction_quality']
                
                # Update global tracking
                self.strategy_performance[strategy].append(strategy_result['reconstruction_quality'])
                total_quality += strategy_result['reconstruction_quality']
                total_tests += 1
            
            # Find best and worst strategies for this region
            if strategy_qualities:
                best_strategy = max(strategy_qualities.keys(), key=lambda k: strategy_qualities[k])
                worst_strategy = min(strategy_qualities.keys(), key=lambda k: strategy_qualities[k])
                
                region_results['best_strategy'] = {
                    'strategy': best_strategy.value,
                    'quality': strategy_qualities[best_strategy]
                }
                region_results['worst_strategy'] = {
                    'strategy': worst_strategy.value,
                    'quality': strategy_qualities[worst_strategy]
                }
                region_results['average_quality'] = np.mean(list(strategy_qualities.values()))
            
            results['region_results'][region.region_id] = region_results
        
        # Calculate overall metrics
        results['overall_understanding'] = {
            'average_quality': total_quality / max(1, total_tests),
            'understanding_level': self._classify_understanding_level(total_quality / max(1, total_tests)),
            'total_tests_performed': total_tests,
            'regions_mastered': sum(1 for r in results['region_results'].values() 
                                  if r['average_quality'] > 0.85)
        }
        
        # Strategy performance summary
        for strategy in strategies:
            if self.strategy_performance[strategy]:
                results['strategy_performance'][strategy.value] = {
                    'average_quality': np.mean(self.strategy_performance[strategy]),
                    'std_quality': np.std(self.strategy_performance[strategy]),
                    'success_rate': np.mean([q > 0.8 for q in self.strategy_performance[strategy]])
                }
        
        return results
    
    def _test_region_with_strategy(self, image: np.ndarray, 
                                 region: ReconstructionRegion,
                                 strategy: MaskingStrategy) -> Dict[str, Any]:
        """Test a specific region with a specific masking strategy."""
        
        start_time = time.time()
        
        # Generate mask for this strategy and region
        mask = self.mask_generator.generate_mask(
            strategy, 
            image.shape[:2], 
            region.difficulty_level,
            region.mask
        )
        
        # Prepare tensors
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Create masked image
        masked_image = image_tensor * mask_tensor
        
        # Perform reconstruction
        with torch.no_grad():
            reconstructed, confidence = self.network(masked_image, mask_tensor)
        
        # Calculate quality metrics
        reconstruction_quality = self._calculate_reconstruction_quality(
            image_tensor, reconstructed, mask_tensor, region.mask
        )
        
        execution_time = time.time() - start_time
        
        result = {
            'strategy': strategy.value,
            'region_id': region.region_id,
            'difficulty_level': region.difficulty_level,
            'reconstruction_quality': float(reconstruction_quality),
            'confidence_score': float(confidence.item()),
            'execution_time': execution_time,
            'mask_coverage': float(mask.sum() / mask.size),  # How much context was provided
            'success': reconstruction_quality > region.target_quality
        }
        
        # Store in history
        self.reconstruction_history.append(result)
        
        return result
    
    def _calculate_reconstruction_quality(self, original: torch.Tensor, 
                                        reconstructed: torch.Tensor,
                                        mask: torch.Tensor,
                                        region_mask: np.ndarray = None) -> float:
        """Calculate reconstruction quality score."""
        
        # Focus on the reconstructed regions (where mask == 0)
        reconstruction_mask = (1 - mask)
        
        if region_mask is not None:
            # Further constrain to the specific region of interest
            region_tensor = torch.from_numpy(region_mask).float().to(self.device)
            reconstruction_mask = reconstruction_mask * region_tensor
        
        if reconstruction_mask.sum() == 0:
            return 0.0
        
        # Calculate MSE in reconstructed regions
        mse = torch.mean((original - reconstructed) ** 2 * reconstruction_mask)
        
        # Convert to quality score (higher is better)
        quality = 1.0 / (1.0 + mse.item() * 10)
        
        return quality
    
    def _classify_understanding_level(self, quality_score: float) -> str:
        """Classify understanding level based on reconstruction quality."""
        
        if quality_score >= 0.95:
            return "excellent"
        elif quality_score >= 0.85:
            return "good"
        elif quality_score >= 0.70:
            return "moderate"
        elif quality_score >= 0.50:
            return "limited"
        else:
            return "poor"
    
    def progressive_difficulty_test(self, image: np.ndarray,
                                  region: ReconstructionRegion,
                                  strategy: MaskingStrategy) -> Dict[str, Any]:
        """
        Test understanding with progressively increasing difficulty.
        
        This implements the core insight: true understanding should be robust
        across difficulty levels.
        """
        
        challenge = ReconstructionChallenge(
            challenge_id=f"{region.region_id}_{strategy.value}",
            image=image,
            regions=[region],
            masking_strategies=[strategy]
        )
        
        results = {
            'challenge_id': challenge.challenge_id,
            'strategy': strategy.value,
            'region_id': region.region_id,
            'difficulty_progression': [],
            'mastery_achieved': False,
            'mastery_level': 0.0,
            'understanding_pathway': []
        }
        
        for difficulty in challenge.difficulty_progression:
            logger.info(f"Testing difficulty level {difficulty}")
            
            # Update region difficulty
            test_region = ReconstructionRegion(
                region_id=region.region_id,
                polygon=region.polygon,
                mask=region.mask,
                difficulty_level=difficulty,
                masking_strategy=strategy,
                target_quality=challenge.success_threshold
            )
            
            # Test at this difficulty level
            level_result = self._test_region_with_strategy(image, test_region, strategy)
            
            results['difficulty_progression'].append({
                'difficulty': difficulty,
                'quality': level_result['reconstruction_quality'],
                'confidence': level_result['confidence_score'],
                'success': level_result['success']
            })
            
            # Check if mastery achieved at this level
            if level_result['success']:
                results['mastery_level'] = difficulty
                
                # Record the understanding pathway
                results['understanding_pathway'].append({
                    'level': difficulty,
                    'strategy': strategy.value,
                    'quality_achieved': level_result['reconstruction_quality'],
                    'insights': f"Mastered {strategy.value} at difficulty {difficulty}"
                })
            else:
                # Failed at this level - stop progression
                break
        
        # Determine if overall mastery achieved
        results['mastery_achieved'] = results['mastery_level'] >= 0.8
        
        return results
    
    def comprehensive_understanding_assessment(self, image: np.ndarray,
                                             regions: List[ReconstructionRegion] = None) -> Dict[str, Any]:
        """
        Perform comprehensive understanding assessment using all strategies.
        
        This is the main method that implements the full testing protocol.
        """
        
        if regions is None:
            # Create default regions covering the whole image
            h, w = image.shape[:2]
            regions = [
                ReconstructionRegion(
                    region_id="full_image",
                    polygon=[(0, 0), (w, 0), (w, h), (0, h)],
                    difficulty_level=0.5
                )
            ]
        
        assessment = {
            'image_shape': image.shape,
            'timestamp': time.time(),
            'regions_assessed': len(regions),
            'strategies_tested': len(list(MaskingStrategy)),
            'regional_assessments': {},
            'strategy_rankings': {},
            'overall_understanding': {},
            'mastery_analysis': {},
            'recommendations': []
        }
        
        # Test each region comprehensively
        for region in regions:
            logger.info(f"Comprehensive assessment of region: {region.region_id}")
            
            region_assessment = {
                'region_id': region.region_id,
                'strategy_results': {},
                'progressive_tests': {},
                'mastery_strategies': [],
                'challenging_strategies': [],
                'understanding_score': 0.0
            }
            
            total_quality = 0.0
            strategy_count = 0
            
            # Test with all strategies
            for strategy in MaskingStrategy:
                # Basic test
                basic_result = self._test_region_with_strategy(image, region, strategy)
                region_assessment['strategy_results'][strategy.value] = basic_result
                
                # Progressive difficulty test
                progressive_result = self.progressive_difficulty_test(image, region, strategy)
                region_assessment['progressive_tests'][strategy.value] = progressive_result
                
                # Track mastery
                if progressive_result['mastery_achieved']:
                    region_assessment['mastery_strategies'].append(strategy.value)
                else:
                    region_assessment['challenging_strategies'].append(strategy.value)
                
                total_quality += basic_result['reconstruction_quality']
                strategy_count += 1
            
            # Calculate region understanding score
            region_assessment['understanding_score'] = total_quality / max(1, strategy_count)
            assessment['regional_assessments'][region.region_id] = region_assessment
        
        # Generate overall assessment
        assessment['overall_understanding'] = self._generate_overall_assessment(assessment)
        assessment['strategy_rankings'] = self._rank_strategies(assessment)
        assessment['mastery_analysis'] = self._analyze_mastery_patterns(assessment)
        assessment['recommendations'] = self._generate_recommendations(assessment)
        
        return assessment
    
    def _generate_overall_assessment(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall understanding assessment."""
        
        region_scores = [
            r['understanding_score'] 
            for r in assessment['regional_assessments'].values()
        ]
        
        if not region_scores:
            return {'understanding_level': 'unknown', 'confidence': 0.0}
        
        overall_score = np.mean(region_scores)
        understanding_level = self._classify_understanding_level(overall_score)
        
        return {
            'overall_score': overall_score,
            'understanding_level': understanding_level,
            'confidence': min(overall_score, 1.0),
            'regions_mastered': sum(1 for score in region_scores if score > 0.85),
            'total_regions': len(region_scores),
            'mastery_percentage': sum(1 for score in region_scores if score > 0.85) / len(region_scores) * 100
        }
    
    def _rank_strategies(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Rank masking strategies by effectiveness."""
        
        strategy_scores = {strategy.value: [] for strategy in MaskingStrategy}
        
        for region_data in assessment['regional_assessments'].values():
            for strategy, result in region_data['strategy_results'].items():
                strategy_scores[strategy].append(result['reconstruction_quality'])
        
        rankings = {}
        for strategy, scores in strategy_scores.items():
            if scores:
                rankings[strategy] = {
                    'average_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'success_rate': np.mean([s > 0.8 for s in scores]),
                    'rank': 0  # Will be filled in after sorting
                }
        
        # Sort by average score
        sorted_strategies = sorted(rankings.items(), key=lambda x: x[1]['average_score'], reverse=True)
        
        for i, (strategy, data) in enumerate(sorted_strategies):
            rankings[strategy]['rank'] = i + 1
        
        return rankings
    
    def _analyze_mastery_patterns(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in mastery across strategies and regions."""
        
        mastery_patterns = {
            'consistent_mastery': [],  # Strategies mastered across all regions
            'inconsistent_mastery': [],  # Strategies mastered in some regions
            'no_mastery': [],  # Strategies not mastered anywhere
            'region_difficulty_ranking': {},
            'insights': []
        }
        
        # Analyze strategy consistency
        strategy_mastery = {strategy.value: 0 for strategy in MaskingStrategy}
        total_regions = len(assessment['regional_assessments'])
        
        for region_data in assessment['regional_assessments'].values():
            for strategy in region_data['mastery_strategies']:
                strategy_mastery[strategy] += 1
        
        for strategy, mastery_count in strategy_mastery.items():
            if mastery_count == total_regions:
                mastery_patterns['consistent_mastery'].append(strategy)
            elif mastery_count > 0:
                mastery_patterns['inconsistent_mastery'].append(strategy)
            else:
                mastery_patterns['no_mastery'].append(strategy)
        
        # Generate insights
        if mastery_patterns['consistent_mastery']:
            mastery_patterns['insights'].append(
                f"Strong understanding demonstrated in {len(mastery_patterns['consistent_mastery'])} strategies"
            )
        
        if mastery_patterns['no_mastery']:
            mastery_patterns['insights'].append(
                f"Challenges identified in {len(mastery_patterns['no_mastery'])} strategies"
            )
        
        return mastery_patterns
    
    def _generate_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on assessment results."""
        
        recommendations = []
        overall = assessment['overall_understanding']
        
        if overall['understanding_level'] == 'excellent':
            recommendations.append("Excellent understanding demonstrated across all tested regions and strategies")
        elif overall['understanding_level'] == 'good':
            recommendations.append("Good understanding with room for improvement in challenging strategies")
        elif overall['understanding_level'] == 'moderate':
            recommendations.append("Moderate understanding - focus on improving reconstruction quality")
        else:
            recommendations.append("Limited understanding - significant training needed")
        
        # Strategy-specific recommendations
        rankings = assessment['strategy_rankings']
        worst_strategies = [s for s, data in rankings.items() if data['average_score'] < 0.6]
        
        if worst_strategies:
            recommendations.append(f"Focus training on challenging strategies: {', '.join(worst_strategies)}")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Initialize engine
    engine = RegionalReconstructionEngine()
    
    # Create test image
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Define test regions
    regions = [
        ReconstructionRegion(
            region_id="center_region",
            polygon=[(56, 56), (168, 56), (168, 168), (56, 168)],
            difficulty_level=0.5
        )
    ]
    
    # Perform comprehensive assessment
    results = engine.comprehensive_understanding_assessment(test_image, regions)
    
    print("Comprehensive Understanding Assessment:")
    print(f"Overall Understanding Level: {results['overall_understanding']['understanding_level']}")
    print(f"Overall Score: {results['overall_understanding']['overall_score']:.3f}")
    print(f"Mastery Percentage: {results['overall_understanding']['mastery_percentage']:.1f}%") 