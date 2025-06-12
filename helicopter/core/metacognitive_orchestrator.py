"""
Metacognitive Orchestrator

The high-level orchestrator that uses the Bayesian objective function to optimize
the entire analysis process. This addresses the fundamental insight that images
are probabilistic collections of pixels, not deterministic binary data.

Key Components:
1. Objective Function Optimization - Uses Bayesian belief network as target
2. Fuzzy Logic Decision Making - Handles uncertainty and continuous values
3. Method Selection and Weighting - Dynamically adjusts analysis methods
4. Convergence Detection - Recognizes when optimization has reached optimum
5. Meta-Learning - Learns about the learning process itself

This orchestrator treats image analysis as a continuous optimization problem
rather than a discrete classification task.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
from collections import deque
import time
from pathlib import Path
import json

from .bayesian_objective_engine import BayesianObjectiveEngine, FuzzyEvidence
from .continuous_learning_engine import ContinuousLearningEngine

logger = logging.getLogger(__name__)


@dataclass
class OptimizationState:
    """Current state of the optimization process"""
    iteration: int
    objective_value: float
    gradient: Dict[str, float]
    uncertainty_map: Dict[str, float]
    fuzzy_state: Dict[str, Any]
    method_weights: Dict[str, float]
    convergence_score: float
    next_actions: List[str]


@dataclass
class MetaLearningMemory:
    """Memory for meta-learning about the optimization process"""
    successful_strategies: deque = field(default_factory=lambda: deque(maxlen=100))
    failed_strategies: deque = field(default_factory=lambda: deque(maxlen=50))
    optimization_patterns: deque = field(default_factory=lambda: deque(maxlen=200))
    method_effectiveness: Dict[str, List[float]] = field(default_factory=dict)
    domain_adaptations: Dict[str, Any] = field(default_factory=dict)


class FuzzyOptimizer:
    """Fuzzy logic-based optimizer for continuous optimization"""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.momentum = 0.9
        self.velocity = {}
        
        # Fuzzy rules for optimization
        self.fuzzy_rules = [
            # High gradient, low uncertainty -> aggressive step
            {'gradient': 'high', 'uncertainty': 'low', 'action': 'aggressive_step'},
            # High gradient, high uncertainty -> cautious step
            {'gradient': 'high', 'uncertainty': 'high', 'action': 'cautious_step'},
            # Low gradient, low uncertainty -> fine_tune
            {'gradient': 'low', 'uncertainty': 'low', 'action': 'fine_tune'},
            # Low gradient, high uncertainty -> explore
            {'gradient': 'low', 'uncertainty': 'high', 'action': 'explore'},
        ]
    
    def optimize_step(self, 
                     current_weights: Dict[str, float], 
                     gradient: Dict[str, float], 
                     uncertainty: Dict[str, float]) -> Dict[str, float]:
        """Perform one optimization step using fuzzy logic"""
        
        new_weights = {}
        
        for param_name, current_weight in current_weights.items():
            grad = gradient.get(param_name, 0.0)
            uncert = uncertainty.get(param_name, 0.5)
            
            # Determine fuzzy action
            action = self._get_fuzzy_action(grad, uncert)
            
            # Calculate step size based on fuzzy action
            step_size = self._calculate_fuzzy_step_size(action, grad, uncert)
            
            # Apply momentum
            if param_name not in self.velocity:
                self.velocity[param_name] = 0.0
            
            self.velocity[param_name] = (self.momentum * self.velocity[param_name] + 
                                       (1 - self.momentum) * grad)
            
            # Update weight
            new_weight = current_weight + step_size * self.velocity[param_name]
            new_weights[param_name] = np.clip(new_weight, 0.0, 1.0)
        
        return new_weights
    
    def _get_fuzzy_action(self, gradient: float, uncertainty: float) -> str:
        """Determine fuzzy action based on gradient and uncertainty"""
        
        # Fuzzify inputs
        grad_high = self._sigmoid(abs(gradient) - 0.1, steepness=10)
        grad_low = 1.0 - grad_high
        
        uncert_high = self._sigmoid(uncertainty - 0.5, steepness=5)
        uncert_low = 1.0 - uncert_high
        
        # Evaluate fuzzy rules
        rule_activations = {
            'aggressive_step': min(grad_high, uncert_low),
            'cautious_step': min(grad_high, uncert_high),
            'fine_tune': min(grad_low, uncert_low),
            'explore': min(grad_low, uncert_high)
        }
        
        # Return action with highest activation
        return max(rule_activations.items(), key=lambda x: x[1])[0]
    
    def _calculate_fuzzy_step_size(self, action: str, gradient: float, uncertainty: float) -> float:
        """Calculate step size based on fuzzy action"""
        
        base_step = self.learning_rate
        
        if action == 'aggressive_step':
            return base_step * 2.0
        elif action == 'cautious_step':
            return base_step * 0.5
        elif action == 'fine_tune':
            return base_step * 0.1
        elif action == 'explore':
            return base_step * 1.5 * np.random.uniform(0.5, 1.5)  # Add randomness for exploration
        
        return base_step
    
    def _sigmoid(self, x: float, steepness: float = 1.0) -> float:
        """Sigmoid function for fuzzy membership"""
        return 1.0 / (1.0 + np.exp(-steepness * x))


class MethodSelector:
    """Selects and weights analysis methods based on current optimization state"""
    
    def __init__(self, available_methods: List[str]):
        self.available_methods = available_methods
        self.method_performance_history = {method: deque(maxlen=50) for method in available_methods}
        self.method_weights = {method: 1.0 for method in available_methods}
        
        # Fuzzy sets for method selection
        self.performance_fuzzy_sets = {
            'poor': (0.0, 0.3),
            'average': (0.3, 0.7),
            'excellent': (0.7, 1.0)
        }
    
    def select_methods(self, 
                      optimization_state: OptimizationState, 
                      resource_budget: float = 1.0) -> Dict[str, float]:
        """Select methods and their weights based on current state"""
        
        # Calculate method relevance based on current objective state
        method_relevance = self._calculate_method_relevance(optimization_state)
        
        # Apply fuzzy logic for method selection
        fuzzy_weights = self._apply_fuzzy_method_selection(method_relevance, optimization_state)
        
        # Normalize weights to fit resource budget
        total_weight = sum(fuzzy_weights.values())
        if total_weight > 0:
            normalized_weights = {
                method: (weight / total_weight) * resource_budget 
                for method, weight in fuzzy_weights.items()
            }
        else:
            # Fallback to equal weights
            normalized_weights = {
                method: resource_budget / len(self.available_methods) 
                for method in self.available_methods
            }
        
        # Update method weights
        self.method_weights.update(normalized_weights)
        
        return normalized_weights
    
    def _calculate_method_relevance(self, optimization_state: OptimizationState) -> Dict[str, float]:
        """Calculate how relevant each method is to current optimization state"""
        
        relevance = {}
        
        for method in self.available_methods:
            base_relevance = 0.5  # Default relevance
            
            # Adjust based on gradient information
            if method in optimization_state.gradient:
                gradient_magnitude = abs(optimization_state.gradient[method])
                base_relevance += gradient_magnitude * 0.3
            
            # Adjust based on uncertainty
            if method in optimization_state.uncertainty_map:
                uncertainty = optimization_state.uncertainty_map[method]
                # Higher uncertainty means method might be more important to improve
                base_relevance += uncertainty * 0.2
            
            # Adjust based on historical performance
            if method in self.method_performance_history:
                recent_performance = list(self.method_performance_history[method])
                if recent_performance:
                    avg_performance = np.mean(recent_performance)
                    # Poor performing methods get more attention
                    if avg_performance < 0.5:
                        base_relevance += 0.3
                    elif avg_performance > 0.8:
                        base_relevance += 0.1  # Still important but less urgent
            
            relevance[method] = np.clip(base_relevance, 0.0, 1.0)
        
        return relevance
    
    def _apply_fuzzy_method_selection(self, 
                                    method_relevance: Dict[str, float], 
                                    optimization_state: OptimizationState) -> Dict[str, float]:
        """Apply fuzzy logic to determine method weights"""
        
        fuzzy_weights = {}
        
        for method, relevance in method_relevance.items():
            # Fuzzify relevance
            relevance_membership = self._calculate_fuzzy_membership(relevance, self.performance_fuzzy_sets)
            
            # Apply fuzzy rules
            if relevance_membership['excellent'] > 0.7:
                weight = 1.0  # High weight for excellent relevance
            elif relevance_membership['poor'] > 0.7:
                weight = 0.8  # High weight for poor performance (needs improvement)
            elif relevance_membership['average'] > 0.5:
                weight = 0.6  # Medium weight for average relevance
            else:
                # Weighted combination based on memberships
                weight = (relevance_membership['excellent'] * 1.0 + 
                         relevance_membership['average'] * 0.6 + 
                         relevance_membership['poor'] * 0.8)
            
            fuzzy_weights[method] = weight
        
        return fuzzy_weights
    
    def _calculate_fuzzy_membership(self, value: float, fuzzy_sets: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Calculate membership degrees for fuzzy sets"""
        
        memberships = {}
        
        for set_name, (low, high) in fuzzy_sets.items():
            if value <= low:
                membership = 1.0 if set_name == 'poor' else 0.0
            elif value >= high:
                membership = 1.0 if set_name == 'excellent' else 0.0
            else:
                # Triangular membership function
                if set_name == 'poor':
                    membership = max(0.0, (low + 0.2 - value) / 0.2)
                elif set_name == 'excellent':
                    membership = max(0.0, (value - high + 0.2) / 0.2)
                else:  # average
                    center = (low + high) / 2
                    width = (high - low) / 2
                    membership = max(0.0, 1.0 - abs(value - center) / width)
            
            memberships[set_name] = membership
        
        return memberships
    
    def update_method_performance(self, method: str, performance: float):
        """Update performance history for a method"""
        
        if method in self.method_performance_history:
            self.method_performance_history[method].append(performance)


class ConvergenceDetector:
    """Detects when optimization has converged using fuzzy logic"""
    
    def __init__(self, patience: int = 5, min_improvement: float = 0.01):
        self.patience = patience
        self.min_improvement = min_improvement
        self.objective_history = deque(maxlen=20)
        self.gradient_history = deque(maxlen=10)
        self.no_improvement_count = 0
    
    def check_convergence(self, optimization_state: OptimizationState) -> Tuple[bool, str, float]:
        """Check if optimization has converged"""
        
        self.objective_history.append(optimization_state.objective_value)
        self.gradient_history.append(np.mean(list(optimization_state.gradient.values())))
        
        # Calculate convergence indicators
        improvement_score = self._calculate_improvement_score()
        stability_score = self._calculate_stability_score()
        gradient_score = self._calculate_gradient_score()
        
        # Fuzzy convergence decision
        convergence_score = self._fuzzy_convergence_decision(
            improvement_score, stability_score, gradient_score
        )
        
        # Determine convergence
        converged = convergence_score > 0.8
        
        if converged:
            reason = f"Fuzzy convergence achieved (score: {convergence_score:.3f})"
        elif len(self.objective_history) >= self.patience:
            recent_improvement = self.objective_history[-1] - self.objective_history[-self.patience]
            if recent_improvement < self.min_improvement:
                self.no_improvement_count += 1
                if self.no_improvement_count >= 3:
                    converged = True
                    reason = f"No improvement for {self.no_improvement_count} checks"
            else:
                self.no_improvement_count = 0
                reason = "Continuing optimization"
        else:
            reason = "Insufficient history for convergence check"
        
        return converged, reason, convergence_score
    
    def _calculate_improvement_score(self) -> float:
        """Calculate improvement score based on objective history"""
        
        if len(self.objective_history) < 3:
            return 0.0
        
        recent_values = list(self.objective_history)[-5:]
        if len(recent_values) < 2:
            return 0.0
        
        # Calculate trend
        improvements = [recent_values[i] - recent_values[i-1] for i in range(1, len(recent_values))]
        avg_improvement = np.mean(improvements)
        
        # Normalize to [0, 1] where 1 means good improvement
        if avg_improvement > 0:
            return min(1.0, avg_improvement * 10)  # Scale up small improvements
        else:
            return 0.0
    
    def _calculate_stability_score(self) -> float:
        """Calculate stability score based on objective variance"""
        
        if len(self.objective_history) < 3:
            return 0.0
        
        recent_values = list(self.objective_history)[-5:]
        variance = np.var(recent_values)
        
        # Lower variance = higher stability
        stability = 1.0 / (1.0 + variance * 100)  # Scale variance
        return stability
    
    def _calculate_gradient_score(self) -> float:
        """Calculate gradient score based on gradient magnitude"""
        
        if len(self.gradient_history) < 2:
            return 0.0
        
        recent_gradients = list(self.gradient_history)[-3:]
        avg_gradient_magnitude = np.mean([abs(g) for g in recent_gradients])
        
        # Lower gradient magnitude = closer to optimum
        gradient_score = 1.0 / (1.0 + avg_gradient_magnitude * 10)
        return gradient_score
    
    def _fuzzy_convergence_decision(self, 
                                  improvement_score: float, 
                                  stability_score: float, 
                                  gradient_score: float) -> float:
        """Make fuzzy convergence decision"""
        
        # Fuzzy rules for convergence
        # Rule 1: High stability AND low gradient -> converged
        rule1 = min(stability_score, gradient_score)
        
        # Rule 2: Low improvement AND high stability -> converged
        rule2 = min(1.0 - improvement_score, stability_score)
        
        # Rule 3: All scores high -> converged
        rule3 = min(improvement_score, stability_score, gradient_score)
        
        # Combine rules using fuzzy OR (maximum)
        convergence_score = max(rule1, rule2, rule3 * 0.8)  # Weight rule3 less
        
        return convergence_score


class MetacognitiveOrchestrator:
    """
    Main metacognitive orchestrator that optimizes the analysis process
    
    Uses Bayesian objective function with fuzzy logic to handle the probabilistic
    nature of image analysis. Treats the entire process as continuous optimization
    rather than discrete decision making.
    """
    
    def __init__(self, 
                 domain: str, 
                 available_methods: List[str],
                 max_iterations: int = 20,
                 resource_budget: float = 1.0):
        
        self.domain = domain
        self.available_methods = available_methods
        self.max_iterations = max_iterations
        self.resource_budget = resource_budget
        
        # Core components
        self.objective_engine = BayesianObjectiveEngine(domain)
        self.fuzzy_optimizer = FuzzyOptimizer()
        self.method_selector = MethodSelector(available_methods)
        self.convergence_detector = ConvergenceDetector()
        
        # Meta-learning
        self.meta_memory = MetaLearningMemory()
        
        # State tracking
        self.current_state = None
        self.optimization_history = []
        
        logger.info(f"Initialized Metacognitive Orchestrator for domain: {domain}")
        logger.info(f"Available methods: {available_methods}")
    
    def orchestrate_analysis(self, 
                           image_data: np.ndarray, 
                           initial_analysis_results: Dict[str, Any],
                           analysis_function: Callable[[np.ndarray, Dict[str, float]], Dict[str, Any]],
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Orchestrate the complete analysis process using metacognitive optimization
        
        Args:
            image_data: The image to analyze
            initial_analysis_results: Initial analysis results from all methods
            analysis_function: Function that performs analysis given image and method weights
            metadata: Optional metadata about the image
            
        Returns:
            Optimized analysis results with metacognitive insights
        """
        
        logger.info("Starting metacognitive orchestration")
        
        # Initialize objective function with initial results
        initial_objective = self.objective_engine.update_objective(initial_analysis_results, image_data)
        
        # Initialize optimization state
        current_weights = {method: 1.0 for method in self.available_methods}
        current_results = initial_analysis_results
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            logger.debug(f"\n=== METACOGNITIVE ITERATION {iteration + 1} ===")
            
            # Get current optimization state
            optimization_state = self._get_optimization_state(iteration, current_results, current_weights)
            self.current_state = optimization_state
            
            # Check convergence
            converged, reason, convergence_score = self.convergence_detector.check_convergence(optimization_state)
            
            if converged:
                logger.info(f"Optimization converged: {reason}")
                break
            
            # Select methods and weights for next iteration
            new_weights = self.method_selector.select_methods(optimization_state, self.resource_budget)
            
            # Apply fuzzy optimization step
            optimized_weights = self.fuzzy_optimizer.optimize_step(
                current_weights, 
                optimization_state.gradient, 
                optimization_state.uncertainty_map
            )
            
            # Combine method selection with optimization
            final_weights = self._combine_weights(new_weights, optimized_weights)
            
            # Perform analysis with new weights
            new_results = analysis_function(image_data, final_weights)
            
            # Update objective function
            new_objective = self.objective_engine.update_objective(new_results, image_data)
            
            # Update method performance
            self._update_method_performance(new_results, final_weights)
            
            # Meta-learning update
            self._update_meta_learning(optimization_state, final_weights, new_objective)
            
            # Store history
            self.optimization_history.append({
                'iteration': iteration + 1,
                'objective_value': new_objective,
                'weights': final_weights.copy(),
                'convergence_score': convergence_score,
                'state': optimization_state
            })
            
            # Update for next iteration
            current_weights = final_weights
            current_results = new_results
            
            logger.debug(f"Iteration {iteration + 1}: objective = {new_objective:.3f}, "
                        f"convergence = {convergence_score:.3f}")
        
        # Generate final results with metacognitive insights
        final_results = self._generate_final_results(current_results, image_data, metadata)
        
        return final_results
    
    def _get_optimization_state(self, 
                              iteration: int, 
                              current_results: Dict[str, Any], 
                              current_weights: Dict[str, float]) -> OptimizationState:
        """Get current optimization state"""
        
        objective_value = self.objective_engine.get_optimization_target()
        gradient = self.objective_engine.get_optimization_gradient()
        uncertainty_map = self.objective_engine.get_uncertainty_map()
        fuzzy_state = self.objective_engine.get_fuzzy_state()
        
        # Calculate convergence score
        if hasattr(self.convergence_detector, 'objective_history') and self.convergence_detector.objective_history:
            convergence_score = self.convergence_detector._fuzzy_convergence_decision(
                self.convergence_detector._calculate_improvement_score(),
                self.convergence_detector._calculate_stability_score(),
                self.convergence_detector._calculate_gradient_score()
            )
        else:
            convergence_score = 0.0
        
        # Determine next actions based on current state
        next_actions = self._determine_next_actions(gradient, uncertainty_map, fuzzy_state)
        
        return OptimizationState(
            iteration=iteration,
            objective_value=objective_value,
            gradient=gradient,
            uncertainty_map=uncertainty_map,
            fuzzy_state=fuzzy_state,
            method_weights=current_weights.copy(),
            convergence_score=convergence_score,
            next_actions=next_actions
        )
    
    def _determine_next_actions(self, 
                              gradient: Dict[str, float], 
                              uncertainty_map: Dict[str, float], 
                              fuzzy_state: Dict[str, Any]) -> List[str]:
        """Determine next actions based on current optimization state"""
        
        actions = []
        
        # High gradient methods need attention
        for method, grad in gradient.items():
            if abs(grad) > 0.1:
                actions.append(f"optimize_{method}")
        
        # High uncertainty methods need exploration
        for method, uncertainty in uncertainty_map.items():
            if uncertainty > 0.7:
                actions.append(f"explore_{method}")
        
        # Fuzzy state-based actions
        linguistic_labels = fuzzy_state.get('linguistic_labels', {})
        for belief, label in linguistic_labels.items():
            if label == 'low':
                actions.append(f"improve_{belief}")
        
        return actions
    
    def _combine_weights(self, 
                        method_weights: Dict[str, float], 
                        optimization_weights: Dict[str, float]) -> Dict[str, float]:
        """Combine method selection weights with optimization weights"""
        
        combined = {}
        
        for method in self.available_methods:
            method_weight = method_weights.get(method, 0.5)
            opt_weight = optimization_weights.get(method, 0.5)
            
            # Fuzzy combination using weighted average
            combined[method] = 0.6 * method_weight + 0.4 * opt_weight
        
        return combined
    
    def _update_method_performance(self, results: Dict[str, Any], weights: Dict[str, float]):
        """Update method performance based on results"""
        
        for method_name, result in results.items():
            if isinstance(result, dict) and 'confidence' in result:
                confidence = result['confidence']
                self.method_selector.update_method_performance(method_name, confidence)
    
    def _update_meta_learning(self, 
                            optimization_state: OptimizationState, 
                            weights: Dict[str, float], 
                            new_objective: float):
        """Update meta-learning memory"""
        
        # Store optimization pattern
        pattern = {
            'state': optimization_state,
            'weights': weights.copy(),
            'objective_improvement': new_objective - optimization_state.objective_value,
            'timestamp': time.time()
        }
        
        self.meta_memory.optimization_patterns.append(pattern)
        
        # Update method effectiveness
        for method, weight in weights.items():
            if method not in self.meta_memory.method_effectiveness:
                self.meta_memory.method_effectiveness[method] = []
            
            # Effectiveness = weight * objective improvement
            effectiveness = weight * max(0, new_objective - optimization_state.objective_value)
            self.meta_memory.method_effectiveness[method].append(effectiveness)
        
        # Store successful/failed strategies
        if new_objective > optimization_state.objective_value:
            strategy = {
                'weights': weights.copy(),
                'context': optimization_state.fuzzy_state,
                'improvement': new_objective - optimization_state.objective_value
            }
            self.meta_memory.successful_strategies.append(strategy)
        else:
            strategy = {
                'weights': weights.copy(),
                'context': optimization_state.fuzzy_state,
                'decline': optimization_state.objective_value - new_objective
            }
            self.meta_memory.failed_strategies.append(strategy)
    
    def _generate_final_results(self, 
                              final_results: Dict[str, Any], 
                              image_data: np.ndarray, 
                              metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate final results with metacognitive insights"""
        
        # Get final belief state
        belief_summary = self.objective_engine.belief_network.get_belief_summary()
        
        # Calculate optimization metrics
        optimization_metrics = self._calculate_optimization_metrics()
        
        # Meta-learning insights
        meta_insights = self._generate_meta_insights()
        
        # Combine everything
        comprehensive_results = {
            'analysis_results': final_results,
            'metacognitive_insights': {
                'final_objective_value': belief_summary['objective_value'],
                'belief_state': belief_summary,
                'optimization_metrics': optimization_metrics,
                'meta_learning_insights': meta_insights,
                'fuzzy_state': self.objective_engine.get_fuzzy_state(),
                'uncertainty_map': self.objective_engine.get_uncertainty_map(),
                'optimization_history': self.optimization_history
            },
            'image_metadata': {
                'shape': image_data.shape,
                'dtype': str(image_data.dtype),
                'analysis_metadata': metadata
            }
        }
        
        return comprehensive_results
    
    def _calculate_optimization_metrics(self) -> Dict[str, Any]:
        """Calculate metrics about the optimization process"""
        
        if not self.optimization_history:
            return {'note': 'No optimization history'}
        
        objectives = [h['objective_value'] for h in self.optimization_history]
        convergence_scores = [h['convergence_score'] for h in self.optimization_history]
        
        return {
            'total_iterations': len(self.optimization_history),
            'initial_objective': objectives[0] if objectives else 0.0,
            'final_objective': objectives[-1] if objectives else 0.0,
            'objective_improvement': objectives[-1] - objectives[0] if len(objectives) > 1 else 0.0,
            'convergence_rate': np.mean(np.diff(convergence_scores)) if len(convergence_scores) > 1 else 0.0,
            'optimization_efficiency': (objectives[-1] - objectives[0]) / len(objectives) if objectives else 0.0,
            'final_convergence_score': convergence_scores[-1] if convergence_scores else 0.0
        }
    
    def _generate_meta_insights(self) -> Dict[str, Any]:
        """Generate insights from meta-learning"""
        
        insights = {
            'most_effective_methods': {},
            'optimization_patterns': {},
            'domain_adaptations': self.meta_memory.domain_adaptations.copy()
        }
        
        # Most effective methods
        for method, effectiveness_history in self.meta_memory.method_effectiveness.items():
            if effectiveness_history:
                insights['most_effective_methods'][method] = {
                    'average_effectiveness': np.mean(effectiveness_history),
                    'consistency': 1.0 - np.std(effectiveness_history),
                    'total_uses': len(effectiveness_history)
                }
        
        # Optimization patterns
        if self.meta_memory.optimization_patterns:
            recent_patterns = list(self.meta_memory.optimization_patterns)[-10:]
            
            # Average improvement per pattern
            improvements = [p['objective_improvement'] for p in recent_patterns]
            insights['optimization_patterns'] = {
                'average_improvement': np.mean(improvements),
                'improvement_consistency': 1.0 - np.std(improvements),
                'successful_pattern_count': len([i for i in improvements if i > 0])
            }
        
        return insights
    
    def save_state(self, save_path: str):
        """Save the complete orchestrator state"""
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save objective engine state
        self.objective_engine.save_state(str(save_dir))
        
        # Save orchestrator-specific state
        orchestrator_state = {
            'domain': self.domain,
            'available_methods': self.available_methods,
            'max_iterations': self.max_iterations,
            'resource_budget': self.resource_budget,
            'meta_memory': self.meta_memory,
            'optimization_history': self.optimization_history,
            'method_weights': self.method_selector.method_weights
        }
        
        with open(save_dir / 'orchestrator_state.json', 'w') as f:
            # Convert non-serializable objects
            serializable_state = self._make_serializable(orchestrator_state)
            json.dump(serializable_state, f, indent=2)
        
        logger.info(f"Saved metacognitive orchestrator state to {save_path}")
    
    def load_state(self, save_path: str):
        """Load the complete orchestrator state"""
        
        save_dir = Path(save_path)
        
        # Load objective engine state
        self.objective_engine.load_state(str(save_dir))
        
        # Load orchestrator-specific state
        state_file = save_dir / 'orchestrator_state.json'
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.domain = state['domain']
            self.available_methods = state['available_methods']
            self.max_iterations = state['max_iterations']
            self.resource_budget = state['resource_budget']
            self.optimization_history = state['optimization_history']
            
            # Restore method weights
            if 'method_weights' in state:
                self.method_selector.method_weights = state['method_weights']
            
            logger.info(f"Loaded metacognitive orchestrator state from {save_path}")
        else:
            logger.warning(f"No orchestrator state found at {save_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable"""
        
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, deque):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj 