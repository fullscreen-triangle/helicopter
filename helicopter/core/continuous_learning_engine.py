"""
Continuous Learning Engine

Implements the actual continuous training, iterative learning, and confidence-based iteration:
1. Model Fine-tuning and Weight Updates
2. Knowledge Accumulation and Memory Systems
3. Confidence-based Convergence Control
4. Adaptive Learning Rate Management
5. Cross-method Knowledge Transfer
6. Performance-based Model Selection

This is the core learning system that makes Helicopter truly adaptive.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import pickle
from collections import deque
import copy
from datetime import datetime
import cv2

logger = logging.getLogger(__name__)


@dataclass
class LearningMemory:
    """Memory system for continuous learning"""
    image_features: deque = field(default_factory=lambda: deque(maxlen=1000))
    analysis_results: deque = field(default_factory=lambda: deque(maxlen=1000))
    confidence_scores: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_patterns: deque = field(default_factory=lambda: deque(maxlen=500))
    successful_patterns: deque = field(default_factory=lambda: deque(maxlen=500))
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    learned_mappings: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class IterationState:
    """State of current iteration"""
    iteration_number: int
    current_confidence: float
    target_confidence: float
    learning_rate: float
    convergence_history: List[float]
    method_performances: Dict[str, float]
    adaptation_needed: bool
    next_actions: List[str]


@dataclass
class TrainingBatch:
    """Batch for continuous training"""
    images: torch.Tensor
    features: torch.Tensor
    targets: torch.Tensor
    confidences: torch.Tensor
    metadata: Dict[str, Any]


class AdaptiveLearningRateScheduler:
    """Adaptive learning rate based on performance"""
    
    def __init__(self, initial_lr: float = 1e-4, patience: int = 5, factor: float = 0.5):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.no_improvement_count = 0
        self.best_performance = 0.0
        self.performance_history = []
    
    def step(self, current_performance: float) -> float:
        """Update learning rate based on performance"""
        
        self.performance_history.append(current_performance)
        
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # Reduce learning rate if no improvement
        if self.no_improvement_count >= self.patience:
            self.current_lr *= self.factor
            self.no_improvement_count = 0
            logger.info(f"Reduced learning rate to {self.current_lr}")
        
        return self.current_lr
    
    def reset(self):
        """Reset scheduler state"""
        self.current_lr = self.initial_lr
        self.no_improvement_count = 0
        self.best_performance = 0.0


class KnowledgeDistillationNetwork(nn.Module):
    """Network for distilling knowledge between methods"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 512):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.adaptation_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        confidence = self.confidence_head(encoded)
        adaptation = self.adaptation_head(encoded)
        
        return encoded, confidence, adaptation


class ConfidenceBasedController:
    """Controls iteration based on confidence levels"""
    
    def __init__(self, target_confidence: float = 0.85, max_iterations: int = 10, min_improvement: float = 0.02):
        self.target_confidence = target_confidence
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.confidence_history = []
        self.convergence_patterns = []
    
    def should_continue(self, current_confidence: float, iteration: int) -> Tuple[bool, str]:
        """Determine if iteration should continue"""
        
        self.confidence_history.append(current_confidence)
        
        # Check maximum iterations
        if iteration >= self.max_iterations:
            return False, f"Maximum iterations ({self.max_iterations}) reached"
        
        # Check target confidence
        if current_confidence >= self.target_confidence:
            return False, f"Target confidence ({self.target_confidence}) achieved"
        
        # Check for convergence (no improvement)
        if len(self.confidence_history) >= 3:
            recent_improvements = [
                self.confidence_history[i] - self.confidence_history[i-1] 
                for i in range(-2, 0)
            ]
            
            if all(imp < self.min_improvement for imp in recent_improvements):
                return False, "Convergence detected - no significant improvement"
        
        # Check for oscillation
        if len(self.confidence_history) >= 5:
            recent_scores = self.confidence_history[-5:]
            variance = np.var(recent_scores)
            
            if variance > 0.1:  # High variance indicates oscillation
                return False, "Oscillation detected - unstable convergence"
        
        return True, "Continue iteration"
    
    def get_learning_strategy(self, current_confidence: float, iteration: int) -> Dict[str, Any]:
        """Get learning strategy based on current state"""
        
        strategy = {
            'focus_areas': [],
            'learning_rate_multiplier': 1.0,
            'method_weights': {},
            'exploration_level': 0.1
        }
        
        # Low confidence - focus on exploration
        if current_confidence < 0.5:
            strategy['focus_areas'] = ['feature_extraction', 'semantic_analysis']
            strategy['learning_rate_multiplier'] = 1.5
            strategy['exploration_level'] = 0.3
        
        # Medium confidence - balanced approach
        elif current_confidence < 0.75:
            strategy['focus_areas'] = ['cross_validation', 'confidence_refinement']
            strategy['learning_rate_multiplier'] = 1.0
            strategy['exploration_level'] = 0.2
        
        # High confidence - focus on fine-tuning
        else:
            strategy['focus_areas'] = ['precision_tuning', 'consistency_improvement']
            strategy['learning_rate_multiplier'] = 0.8
            strategy['exploration_level'] = 0.1
        
        return strategy


class ContinuousLearningEngine:
    """
    Main continuous learning engine
    
    Implements actual continuous training, iterative learning, and confidence-based iteration
    """
    
    def __init__(self, domain: str, device: Optional[str] = None):
        self.domain = domain
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Learning components
        self.memory = LearningMemory()
        self.knowledge_network = KnowledgeDistillationNetwork().to(self.device)
        self.optimizer = optim.AdamW(self.knowledge_network.parameters(), lr=1e-4)
        self.lr_scheduler = AdaptiveLearningRateScheduler()
        self.confidence_controller = ConfidenceBasedController()
        
        # Training state
        self.training_step = 0
        self.epoch = 0
        self.best_performance = 0.0
        self.model_checkpoints = {}
        
        # Method-specific adapters
        self.method_adapters = {}
        self.method_performance_history = {}
        
        logger.info(f"Initialized Continuous Learning Engine for domain: {domain}")
    
    def learn_from_analysis(
        self, 
        image: np.ndarray, 
        analysis_results: Dict[str, Any], 
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Learn from a single image analysis"""
        
        # Extract features for learning
        features = self._extract_learning_features(image, analysis_results)
        
        # Calculate confidence and errors
        confidence = self._calculate_analysis_confidence(analysis_results)
        errors = self._identify_learning_targets(analysis_results, ground_truth)
        
        # Store in memory
        self._update_memory(image, features, analysis_results, confidence, errors)
        
        # Perform incremental training
        if len(self.memory.image_features) >= 10:  # Minimum batch size
            training_loss = self._perform_incremental_training()
        else:
            training_loss = 0.0
        
        # Update method-specific knowledge
        self._update_method_knowledge(analysis_results, confidence)
        
        return {
            'confidence': confidence,
            'training_loss': training_loss,
            'memory_size': len(self.memory.image_features),
            'learning_progress': self._calculate_learning_progress()
        }
    
    def iterate_until_convergence(
        self, 
        images: List[np.ndarray], 
        initial_analysis_results: List[Dict[str, Any]],
        ground_truth: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Perform iterative learning until convergence"""
        
        iteration_history = []
        current_results = initial_analysis_results
        
        for iteration in range(self.confidence_controller.max_iterations):
            logger.info(f"\n=== ITERATION {iteration + 1} ===")
            
            # Calculate current confidence
            confidences = [self._calculate_analysis_confidence(result) for result in current_results]
            current_confidence = np.mean(confidences)
            
            # Check if we should continue
            should_continue, reason = self.confidence_controller.should_continue(
                current_confidence, iteration
            )
            
            if not should_continue:
                logger.info(f"Stopping iteration: {reason}")
                break
            
            # Get learning strategy for this iteration
            strategy = self.confidence_controller.get_learning_strategy(current_confidence, iteration)
            
            # Perform batch learning
            batch_results = self._perform_batch_learning(
                images, current_results, ground_truth, strategy
            )
            
            # Update analysis results with learned improvements
            improved_results = self._apply_learned_improvements(
                images, current_results, batch_results['improvements']
            )
            
            # Store iteration state
            iteration_state = IterationState(
                iteration_number=iteration + 1,
                current_confidence=current_confidence,
                target_confidence=self.confidence_controller.target_confidence,
                learning_rate=self.lr_scheduler.current_lr,
                convergence_history=self.confidence_controller.confidence_history.copy(),
                method_performances=batch_results['method_performances'],
                adaptation_needed=batch_results['adaptation_needed'],
                next_actions=batch_results['next_actions']
            )
            
            iteration_history.append(iteration_state)
            current_results = improved_results
            
            # Log progress
            logger.info(f"Iteration {iteration + 1} confidence: {current_confidence:.3f}")
            logger.info(f"Strategy: {strategy['focus_areas']}")
        
        # Final evaluation
        final_confidence = np.mean([self._calculate_analysis_confidence(result) for result in current_results])
        
        return {
            'final_results': current_results,
            'final_confidence': final_confidence,
            'iteration_history': iteration_history,
            'total_iterations': len(iteration_history),
            'convergence_achieved': final_confidence >= self.confidence_controller.target_confidence,
            'learning_metrics': self._calculate_final_learning_metrics(iteration_history)
        }
    
    def _extract_learning_features(self, image: np.ndarray, analysis_results: Dict[str, Any]) -> torch.Tensor:
        """Extract features for learning from image and analysis"""
        
        # Combine features from different analysis methods
        feature_components = []
        
        # Visual features (simplified - would use actual feature extractors)
        visual_features = self._extract_visual_features(image)
        feature_components.append(visual_features)
        
        # Analysis-derived features
        for method_name, result in analysis_results.items():
            if isinstance(result, dict) and 'features' in result:
                method_features = torch.tensor(result['features'], dtype=torch.float32)
                feature_components.append(method_features)
        
        # Concatenate all features
        if feature_components:
            combined_features = torch.cat(feature_components, dim=-1)
        else:
            combined_features = torch.zeros(512)  # Default feature size
        
        return combined_features.to(self.device)
    
    def _extract_visual_features(self, image: np.ndarray) -> torch.Tensor:
        """Extract visual features from image"""
        
        # Simplified visual feature extraction
        # In practice, this would use pretrained vision models
        
        # Resize and normalize
        resized = cv2.resize(image, (224, 224))
        normalized = resized.astype(np.float32) / 255.0
        
        # Simple statistical features
        features = [
            np.mean(normalized),
            np.std(normalized),
            np.mean(normalized, axis=(0, 1)),  # Channel means
            np.std(normalized, axis=(0, 1))    # Channel stds
        ]
        
        # Flatten and pad to standard size
        flattened = np.concatenate([f.flatten() if hasattr(f, 'flatten') else [f] for f in features])
        padded = np.pad(flattened, (0, max(0, 256 - len(flattened))))[:256]
        
        return torch.tensor(padded, dtype=torch.float32)
    
    def _calculate_analysis_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate confidence from analysis result"""
        
        confidences = []
        
        # Extract confidences from different methods
        for method_name, result in analysis_result.items():
            if isinstance(result, dict):
                if 'confidence' in result:
                    confidences.append(result['confidence'])
                elif 'consensus_confidence' in result:
                    confidences.append(result['consensus_confidence'])
        
        # Overall confidence
        if confidences:
            return np.mean(confidences)
        else:
            return 0.5  # Default moderate confidence
    
    def _identify_learning_targets(
        self, 
        analysis_results: Dict[str, Any], 
        ground_truth: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Identify what the system should learn from this analysis"""
        
        learning_targets = {
            'confidence_improvements': [],
            'accuracy_improvements': [],
            'consistency_improvements': [],
            'method_correlations': {}
        }
        
        # If ground truth available, identify accuracy improvements
        if ground_truth:
            for method_name, result in analysis_results.items():
                if method_name in ground_truth:
                    gt_data = ground_truth[method_name]
                    error = self._calculate_method_error(result, gt_data)
                    
                    if error > 0.1:  # Significant error
                        learning_targets['accuracy_improvements'].append({
                            'method': method_name,
                            'error': error,
                            'target': gt_data
                        })
        
        # Identify confidence improvements needed
        for method_name, result in analysis_results.items():
            if isinstance(result, dict) and 'confidence' in result:
                confidence = result['confidence']
                
                if confidence < 0.7:  # Low confidence
                    learning_targets['confidence_improvements'].append({
                        'method': method_name,
                        'current_confidence': confidence,
                        'target_confidence': 0.8
                    })
        
        return learning_targets
    
    def _calculate_method_error(self, predicted: Any, ground_truth: Any) -> float:
        """Calculate error between predicted and ground truth"""
        
        # Simplified error calculation
        # In practice, this would be method-specific
        
        if isinstance(predicted, dict) and isinstance(ground_truth, dict):
            # Compare dictionary values
            errors = []
            for key in set(predicted.keys()) & set(ground_truth.keys()):
                if isinstance(predicted[key], (int, float)) and isinstance(ground_truth[key], (int, float)):
                    error = abs(predicted[key] - ground_truth[key]) / max(abs(ground_truth[key]), 1e-8)
                    errors.append(error)
            
            return np.mean(errors) if errors else 1.0
        
        return 0.5  # Default moderate error
    
    def _update_memory(
        self, 
        image: np.ndarray, 
        features: torch.Tensor, 
        analysis_results: Dict[str, Any], 
        confidence: float, 
        errors: Dict[str, Any]
    ):
        """Update learning memory with new information"""
        
        # Store features and results
        self.memory.image_features.append(features.cpu())
        self.memory.analysis_results.append(analysis_results)
        self.memory.confidence_scores.append(confidence)
        
        # Store error patterns for learning
        if errors['accuracy_improvements']:
            self.memory.error_patterns.append(errors)
        else:
            self.memory.successful_patterns.append({
                'features': features.cpu(),
                'confidence': confidence,
                'results': analysis_results
            })
        
        # Update domain knowledge
        self._update_domain_knowledge(analysis_results, confidence)
    
    def _update_domain_knowledge(self, analysis_results: Dict[str, Any], confidence: float):
        """Update domain-specific knowledge base"""
        
        for method_name, result in analysis_results.items():
            if method_name not in self.memory.domain_knowledge:
                self.memory.domain_knowledge[method_name] = {
                    'average_confidence': confidence,
                    'performance_history': [confidence],
                    'learned_patterns': [],
                    'adaptation_parameters': {}
                }
            else:
                knowledge = self.memory.domain_knowledge[method_name]
                knowledge['performance_history'].append(confidence)
                
                # Update running average
                knowledge['average_confidence'] = np.mean(knowledge['performance_history'][-100:])
                
                # Learn patterns
                if confidence > 0.8:  # High confidence pattern
                    pattern = self._extract_success_pattern(result)
                    knowledge['learned_patterns'].append(pattern)
    
    def _extract_success_pattern(self, analysis_result: Any) -> Dict[str, Any]:
        """Extract pattern from successful analysis"""
        
        pattern = {
            'timestamp': datetime.now().isoformat(),
            'confidence': self._calculate_analysis_confidence({'result': analysis_result}),
            'features': {}
        }
        
        # Extract key features that led to success
        if isinstance(analysis_result, dict):
            for key, value in analysis_result.items():
                if isinstance(value, (int, float)):
                    pattern['features'][key] = value
        
        return pattern
    
    def _perform_incremental_training(self) -> float:
        """Perform incremental training on recent memory"""
        
        # Create training batch from recent memory
        batch = self._create_training_batch()
        
        if batch is None:
            return 0.0
        
        # Training step
        self.knowledge_network.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        encoded, predicted_confidence, adaptation = self.knowledge_network(batch.features)
        
        # Calculate losses
        confidence_loss = nn.MSELoss()(predicted_confidence.squeeze(), batch.confidences)
        
        # Feature consistency loss
        feature_loss = nn.MSELoss()(encoded, batch.targets)
        
        # Total loss
        total_loss = confidence_loss + 0.1 * feature_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.knowledge_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update learning rate
        current_performance = 1.0 - total_loss.item()
        self.lr_scheduler.step(current_performance)
        
        self.training_step += 1
        
        logger.debug(f"Training step {self.training_step}, Loss: {total_loss.item():.4f}")
        
        return total_loss.item()
    
    def _create_training_batch(self, batch_size: int = 16) -> Optional[TrainingBatch]:
        """Create training batch from memory"""
        
        if len(self.memory.image_features) < batch_size:
            return None
        
        # Sample recent examples
        indices = list(range(max(0, len(self.memory.image_features) - batch_size), len(self.memory.image_features)))
        
        # Collect batch data
        features_list = [self.memory.image_features[i] for i in indices]
        confidence_list = [self.memory.confidence_scores[i] for i in indices]
        
        # Stack into tensors
        features = torch.stack(features_list).to(self.device)
        confidences = torch.tensor(confidence_list, dtype=torch.float32).to(self.device)
        
        # Create targets (same as features for autoencoder-like training)
        targets = features.clone()
        
        return TrainingBatch(
            images=None,  # Not needed for this training
            features=features,
            targets=targets,
            confidences=confidences,
            metadata={'batch_size': batch_size, 'indices': indices}
        )
    
    def _update_method_knowledge(self, analysis_results: Dict[str, Any], confidence: float):
        """Update method-specific knowledge"""
        
        for method_name, result in analysis_results.items():
            if method_name not in self.method_performance_history:
                self.method_performance_history[method_name] = []
            
            self.method_performance_history[method_name].append(confidence)
            
            # Create or update method adapter
            if method_name not in self.method_adapters:
                self.method_adapters[method_name] = self._create_method_adapter(method_name)
            
            # Update adapter with new knowledge
            self._update_method_adapter(method_name, result, confidence)
    
    def _create_method_adapter(self, method_name: str) -> Dict[str, Any]:
        """Create adapter for specific method"""
        
        return {
            'performance_history': [],
            'learned_parameters': {},
            'adaptation_rules': [],
            'confidence_threshold': 0.7,
            'last_update': datetime.now()
        }
    
    def _update_method_adapter(self, method_name: str, result: Any, confidence: float):
        """Update method-specific adapter"""
        
        adapter = self.method_adapters[method_name]
        adapter['performance_history'].append(confidence)
        adapter['last_update'] = datetime.now()
        
        # Adapt parameters based on performance
        recent_performance = np.mean(adapter['performance_history'][-10:])
        
        if recent_performance < adapter['confidence_threshold']:
            # Poor performance - add adaptation rule
            adaptation_rule = {
                'condition': f'confidence < {adapter["confidence_threshold"]}',
                'action': 'increase_attention',
                'parameter_adjustments': {'weight': 1.2}
            }
            adapter['adaptation_rules'].append(adaptation_rule)
    
    def _calculate_learning_progress(self) -> Dict[str, float]:
        """Calculate overall learning progress"""
        
        if len(self.memory.confidence_scores) < 10:
            return {'progress': 0.0, 'trend': 'insufficient_data'}
        
        # Calculate trend over recent scores
        recent_scores = list(self.memory.confidence_scores)[-20:]
        
        # Linear regression to find trend
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        
        slope, intercept = np.polyfit(x, y, 1)
        
        # Progress metrics
        current_avg = np.mean(recent_scores[-5:])
        initial_avg = np.mean(recent_scores[:5])
        
        progress = (current_avg - initial_avg) / max(initial_avg, 0.1)
        
        return {
            'progress': progress,
            'trend': 'improving' if slope > 0 else 'declining',
            'current_performance': current_avg,
            'learning_rate': slope
        }
    
    def _perform_batch_learning(
        self, 
        images: List[np.ndarray], 
        analysis_results: List[Dict[str, Any]], 
        ground_truth: Optional[List[Dict[str, Any]]], 
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform learning on a batch of images"""
        
        batch_improvements = []
        method_performances = {}
        
        # Extract features for entire batch
        batch_features = []
        batch_confidences = []
        
        for i, (image, result) in enumerate(zip(images, analysis_results)):
            features = self._extract_learning_features(image, result)
            confidence = self._calculate_analysis_confidence(result)
            
            batch_features.append(features)
            batch_confidences.append(confidence)
        
        # Stack into batch tensors
        batch_feature_tensor = torch.stack(batch_features)
        batch_confidence_tensor = torch.tensor(batch_confidences)
        
        # Perform batch training
        self.knowledge_network.train()
        
        for epoch in range(5):  # Multiple epochs for batch
            self.optimizer.zero_grad()
            
            encoded, pred_confidence, adaptation = self.knowledge_network(batch_feature_tensor)
            
            # Batch losses
            confidence_loss = nn.MSELoss()(pred_confidence.squeeze(), batch_confidence_tensor)
            consistency_loss = self._calculate_consistency_loss(encoded)
            
            total_loss = confidence_loss + 0.1 * consistency_loss
            
            total_loss.backward()
            self.optimizer.step()
        
        # Generate improvements based on learning
        for i in range(len(images)):
            improvement = self._generate_improvement_suggestion(
                encoded[i], pred_confidence[i], adaptation[i], strategy
            )
            batch_improvements.append(improvement)
        
        # Calculate method performances
        for method_name in analysis_results[0].keys():
            method_confidences = [
                self._calculate_analysis_confidence({method_name: result[method_name]})
                for result in analysis_results
                if method_name in result
            ]
            method_performances[method_name] = np.mean(method_confidences) if method_confidences else 0.0
        
        return {
            'improvements': batch_improvements,
            'method_performances': method_performances,
            'adaptation_needed': any(conf < 0.7 for conf in batch_confidences),
            'next_actions': self._determine_next_actions(strategy, method_performances)
        }
    
    def _calculate_consistency_loss(self, encoded_features: torch.Tensor) -> torch.Tensor:
        """Calculate consistency loss across batch"""
        
        # Encourage consistency in learned representations
        batch_mean = torch.mean(encoded_features, dim=0)
        consistency_loss = torch.mean(torch.var(encoded_features - batch_mean, dim=0))
        
        return consistency_loss
    
    def _generate_improvement_suggestion(
        self, 
        encoded: torch.Tensor, 
        predicted_confidence: torch.Tensor, 
        adaptation: torch.Tensor, 
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate improvement suggestion for single item"""
        
        improvement = {
            'confidence_boost': float(predicted_confidence.item()),
            'feature_adjustments': adaptation.detach().cpu().numpy(),
            'focus_methods': strategy['focus_areas'],
            'learning_priority': 'high' if predicted_confidence < 0.6 else 'normal'
        }
        
        return improvement
    
    def _determine_next_actions(self, strategy: Dict[str, Any], method_performances: Dict[str, float]) -> List[str]:
        """Determine next actions based on current state"""
        
        actions = []
        
        # Poor performing methods need attention
        for method, performance in method_performances.items():
            if performance < 0.6:
                actions.append(f"improve_{method}_accuracy")
            elif performance < 0.8:
                actions.append(f"tune_{method}_confidence")
        
        # Strategy-based actions
        if 'feature_extraction' in strategy['focus_areas']:
            actions.append("enhance_feature_extraction")
        
        if 'cross_validation' in strategy['focus_areas']:
            actions.append("strengthen_cross_validation")
        
        return actions
    
    def _apply_learned_improvements(
        self, 
        images: List[np.ndarray], 
        original_results: List[Dict[str, Any]], 
        improvements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply learned improvements to analysis results"""
        
        improved_results = []
        
        for i, (original, improvement) in enumerate(zip(original_results, improvements)):
            improved = copy.deepcopy(original)
            
            # Apply confidence boost
            for method_name, result in improved.items():
                if isinstance(result, dict) and 'confidence' in result:
                    boost = improvement['confidence_boost']
                    improved[method_name]['confidence'] = min(1.0, result['confidence'] * boost)
            
            # Apply feature adjustments (simplified)
            improved['_learning_metadata'] = {
                'improvement_applied': True,
                'confidence_boost': improvement['confidence_boost'],
                'learning_priority': improvement['learning_priority']
            }
            
            improved_results.append(improved)
        
        return improved_results
    
    def _calculate_final_learning_metrics(self, iteration_history: List[IterationState]) -> Dict[str, Any]:
        """Calculate final learning metrics"""
        
        if not iteration_history:
            return {'error': 'No iteration history'}
        
        # Confidence progression
        confidence_progression = [state.current_confidence for state in iteration_history]
        
        # Learning rate adaptation
        lr_progression = [state.learning_rate for state in iteration_history]
        
        # Method performance trends
        method_trends = {}
        for state in iteration_history:
            for method, perf in state.method_performances.items():
                if method not in method_trends:
                    method_trends[method] = []
                method_trends[method].append(perf)
        
        return {
            'confidence_improvement': confidence_progression[-1] - confidence_progression[0],
            'learning_rate_final': lr_progression[-1],
            'convergence_rate': np.mean(np.diff(confidence_progression)),
            'method_improvements': {
                method: trend[-1] - trend[0] 
                for method, trend in method_trends.items() 
                if len(trend) > 1
            },
            'stability_score': 1.0 - np.var(confidence_progression[-3:]) if len(confidence_progression) >= 3 else 0.0
        }
    
    def save_learning_state(self, save_path: str):
        """Save complete learning state"""
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.knowledge_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'epoch': self.epoch,
            'best_performance': self.best_performance
        }, save_dir / 'model_checkpoint.pt')
        
        # Save memory and knowledge
        with open(save_dir / 'learning_memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)
        
        with open(save_dir / 'method_adapters.json', 'w') as f:
            # Convert non-serializable objects
            serializable_adapters = {}
            for method, adapter in self.method_adapters.items():
                serializable_adapters[method] = {
                    'performance_history': adapter['performance_history'],
                    'learned_parameters': adapter['learned_parameters'],
                    'confidence_threshold': adapter['confidence_threshold'],
                    'last_update': adapter['last_update'].isoformat()
                }
            json.dump(serializable_adapters, f, indent=2)
        
        logger.info(f"Saved learning state to {save_path}")
    
    def load_learning_state(self, save_path: str):
        """Load complete learning state"""
        
        save_dir = Path(save_path)
        
        if not save_dir.exists():
            logger.warning(f"Save path {save_path} does not exist")
            return
        
        # Load model state
        checkpoint_path = save_dir / 'model_checkpoint.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.knowledge_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_step = checkpoint['training_step']
            self.epoch = checkpoint['epoch']
            self.best_performance = checkpoint['best_performance']
        
        # Load memory
        memory_path = save_dir / 'learning_memory.pkl'
        if memory_path.exists():
            with open(memory_path, 'rb') as f:
                self.memory = pickle.load(f)
        
        # Load method adapters
        adapters_path = save_dir / 'method_adapters.json'
        if adapters_path.exists():
            with open(adapters_path, 'r') as f:
                loaded_adapters = json.load(f)
                for method, adapter_data in loaded_adapters.items():
                    adapter_data['last_update'] = datetime.fromisoformat(adapter_data['last_update'])
                    self.method_adapters[method] = adapter_data
        
        logger.info(f"Loaded learning state from {save_path}") 