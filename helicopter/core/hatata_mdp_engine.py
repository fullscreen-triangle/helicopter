"""
Hatata Markov Decision Process Engine

A probabilistic state transition engine for image understanding verification.

Core Innovation:
- Image understanding is not a straight probabilistic process
- Uses MDP to model transitions between understanding states
- Provides probabilistic fallback when deterministic methods fail
- Acts as additional verification layer for reconstruction quality
- Models uncertainty and decision-making in visual analysis

The name "Hatata" reflects the iterative, probabilistic nature of understanding -
like saying "aha!" repeatedly as understanding gradually emerges through state transitions.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class UnderstandingState(Enum):
    """States of image understanding in the MDP."""
    INITIAL = "initial"                    # Starting state, no understanding
    PARTIAL_RECOGNITION = "partial_recognition"  # Some features recognized
    FEATURE_EXTRACTION = "feature_extraction"   # Extracting visual features
    PATTERN_DETECTION = "pattern_detection"     # Detecting patterns/structures
    SEMANTIC_MAPPING = "semantic_mapping"       # Mapping to semantic concepts
    CONTEXTUAL_ANALYSIS = "contextual_analysis" # Understanding context/relationships
    RECONSTRUCTION_READY = "reconstruction_ready" # Ready for reconstruction
    VALIDATION_PHASE = "validation_phase"       # Validating understanding
    UNDERSTANDING_ACHIEVED = "understanding_achieved" # Complete understanding
    UNDERSTANDING_FAILED = "understanding_failed"     # Failed to understand
    UNCERTAINTY_STATE = "uncertainty_state"          # High uncertainty, need more info


class HatataAction(Enum):
    """Actions available in the MDP for progressing understanding."""
    ANALYZE_FEATURES = "analyze_features"       # Extract more features
    DETECT_PATTERNS = "detect_patterns"         # Look for patterns
    MAP_SEMANTICS = "map_semantics"            # Map to semantic concepts
    ANALYZE_CONTEXT = "analyze_context"         # Analyze contextual relationships
    ATTEMPT_RECONSTRUCTION = "attempt_reconstruction" # Try reconstruction
    VALIDATE_UNDERSTANDING = "validate_understanding" # Validate current understanding
    GATHER_MORE_INFO = "gather_more_info"      # Collect additional information
    BACKTRACK = "backtrack"                    # Return to previous state
    TERMINATE_SUCCESS = "terminate_success"     # Declare understanding achieved
    TERMINATE_FAILURE = "terminate_failure"     # Declare understanding failed


@dataclass
class HatataTransition:
    """Represents a state transition in the MDP."""
    from_state: UnderstandingState
    action: HatataAction
    to_state: UnderstandingState
    probability: float
    reward: float
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HatataObservation:
    """Observation from the environment (image analysis results)."""
    timestamp: float
    reconstruction_quality: float
    feature_confidence: float
    pattern_strength: float
    semantic_clarity: float
    context_coherence: float
    uncertainty_level: float
    additional_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class HatataEpisode:
    """A complete episode of understanding through MDP."""
    episode_id: str
    start_time: float
    initial_state: UnderstandingState
    state_sequence: List[UnderstandingState] = field(default_factory=list)
    action_sequence: List[HatataAction] = field(default_factory=list)
    reward_sequence: List[float] = field(default_factory=list)
    observation_sequence: List[HatataObservation] = field(default_factory=list)
    
    # Episode results
    final_state: Optional[UnderstandingState] = None
    total_reward: float = 0.0
    episode_length: int = 0
    success: bool = False
    end_time: float = 0.0


class HatataMDPModel:
    """The core MDP model for image understanding state transitions."""
    
    def __init__(self):
        self.states = list(UnderstandingState)
        self.actions = list(HatataAction)
        
        # Transition probabilities: P(s'|s,a)
        self.transition_probs: Dict[Tuple[UnderstandingState, HatataAction, UnderstandingState], float] = {}
        
        # Reward function: R(s,a,s')
        self.rewards: Dict[Tuple[UnderstandingState, HatataAction, UnderstandingState], float] = {}
        
        # State values (learned through experience)
        self.state_values: Dict[UnderstandingState, float] = {}
        
        # Action values: Q(s,a)
        self.q_values: Dict[Tuple[UnderstandingState, HatataAction], float] = {}
        
        # Initialize the MDP
        self._initialize_mdp()
    
    def _initialize_mdp(self):
        """Initialize transition probabilities and rewards."""
        
        # Initialize state values
        for state in self.states:
            self.state_values[state] = 0.0
        
        # Initialize Q-values
        for state in self.states:
            for action in self.actions:
                self.q_values[(state, action)] = 0.0
        
        # Define transition probabilities and rewards
        self._define_transitions()
        
        logger.info("Hatata MDP model initialized with transition probabilities and rewards")
    
    def _define_transitions(self):
        """Define the transition probabilities and rewards for the MDP."""
        
        # Define key transitions with probabilities and rewards
        transitions = [
            # From INITIAL state
            (UnderstandingState.INITIAL, HatataAction.ANALYZE_FEATURES, 
             UnderstandingState.FEATURE_EXTRACTION, 0.8, 1.0),
            (UnderstandingState.INITIAL, HatataAction.ANALYZE_FEATURES, 
             UnderstandingState.UNCERTAINTY_STATE, 0.2, -0.5),
            
            # From FEATURE_EXTRACTION
            (UnderstandingState.FEATURE_EXTRACTION, HatataAction.DETECT_PATTERNS,
             UnderstandingState.PATTERN_DETECTION, 0.7, 2.0),
            (UnderstandingState.FEATURE_EXTRACTION, HatataAction.DETECT_PATTERNS,
             UnderstandingState.PARTIAL_RECOGNITION, 0.3, 1.0),
            
            # From PATTERN_DETECTION
            (UnderstandingState.PATTERN_DETECTION, HatataAction.MAP_SEMANTICS,
             UnderstandingState.SEMANTIC_MAPPING, 0.6, 3.0),
            (UnderstandingState.PATTERN_DETECTION, HatataAction.MAP_SEMANTICS,
             UnderstandingState.UNCERTAINTY_STATE, 0.4, -1.0),
            
            # From SEMANTIC_MAPPING
            (UnderstandingState.SEMANTIC_MAPPING, HatataAction.ANALYZE_CONTEXT,
             UnderstandingState.CONTEXTUAL_ANALYSIS, 0.8, 4.0),
            (UnderstandingState.SEMANTIC_MAPPING, HatataAction.ATTEMPT_RECONSTRUCTION,
             UnderstandingState.RECONSTRUCTION_READY, 0.5, 3.0),
            
            # From CONTEXTUAL_ANALYSIS
            (UnderstandingState.CONTEXTUAL_ANALYSIS, HatataAction.ATTEMPT_RECONSTRUCTION,
             UnderstandingState.RECONSTRUCTION_READY, 0.9, 5.0),
            (UnderstandingState.CONTEXTUAL_ANALYSIS, HatataAction.VALIDATE_UNDERSTANDING,
             UnderstandingState.VALIDATION_PHASE, 0.7, 4.0),
            
            # From RECONSTRUCTION_READY
            (UnderstandingState.RECONSTRUCTION_READY, HatataAction.VALIDATE_UNDERSTANDING,
             UnderstandingState.VALIDATION_PHASE, 0.8, 5.0),
            (UnderstandingState.RECONSTRUCTION_READY, HatataAction.TERMINATE_SUCCESS,
             UnderstandingState.UNDERSTANDING_ACHIEVED, 0.6, 10.0),
            
            # From VALIDATION_PHASE
            (UnderstandingState.VALIDATION_PHASE, HatataAction.TERMINATE_SUCCESS,
             UnderstandingState.UNDERSTANDING_ACHIEVED, 0.8, 15.0),
            (UnderstandingState.VALIDATION_PHASE, HatataAction.BACKTRACK,
             UnderstandingState.SEMANTIC_MAPPING, 0.2, -2.0),
            
            # From UNCERTAINTY_STATE
            (UnderstandingState.UNCERTAINTY_STATE, HatataAction.GATHER_MORE_INFO,
             UnderstandingState.FEATURE_EXTRACTION, 0.6, 0.5),
            (UnderstandingState.UNCERTAINTY_STATE, HatataAction.BACKTRACK,
             UnderstandingState.INITIAL, 0.3, -1.0),
            (UnderstandingState.UNCERTAINTY_STATE, HatataAction.TERMINATE_FAILURE,
             UnderstandingState.UNDERSTANDING_FAILED, 0.1, -10.0),
            
            # Failure transitions from various states
            (UnderstandingState.PARTIAL_RECOGNITION, HatataAction.TERMINATE_FAILURE,
             UnderstandingState.UNDERSTANDING_FAILED, 0.1, -5.0),
        ]
        
        # Store transitions
        for from_state, action, to_state, prob, reward in transitions:
            self.transition_probs[(from_state, action, to_state)] = prob
            self.rewards[(from_state, action, to_state)] = reward
        
        # Normalize probabilities for each (state, action) pair
        self._normalize_transition_probabilities()
    
    def _normalize_transition_probabilities(self):
        """Ensure transition probabilities sum to 1 for each (state, action) pair."""
        
        # Group transitions by (state, action)
        state_action_transitions = defaultdict(list)
        
        for (from_state, action, to_state), prob in self.transition_probs.items():
            state_action_transitions[(from_state, action)].append((to_state, prob))
        
        # Normalize each group
        for (from_state, action), transitions in state_action_transitions.items():
            total_prob = sum(prob for _, prob in transitions)
            
            if total_prob > 0:
                for to_state, prob in transitions:
                    normalized_prob = prob / total_prob
                    self.transition_probs[(from_state, action, to_state)] = normalized_prob
    
    def get_transition_probability(self, from_state: UnderstandingState, 
                                 action: HatataAction, 
                                 to_state: UnderstandingState) -> float:
        """Get transition probability P(s'|s,a)."""
        return self.transition_probs.get((from_state, action, to_state), 0.0)
    
    def get_reward(self, from_state: UnderstandingState, 
                   action: HatataAction, 
                   to_state: UnderstandingState) -> float:
        """Get reward R(s,a,s')."""
        return self.rewards.get((from_state, action, to_state), 0.0)
    
    def get_possible_actions(self, state: UnderstandingState) -> List[HatataAction]:
        """Get possible actions from a given state."""
        possible_actions = []
        
        for (from_state, action, _), prob in self.transition_probs.items():
            if from_state == state and prob > 0:
                if action not in possible_actions:
                    possible_actions.append(action)
        
        return possible_actions
    
    def sample_next_state(self, current_state: UnderstandingState, 
                         action: HatataAction) -> Tuple[UnderstandingState, float]:
        """Sample next state given current state and action."""
        
        # Get all possible transitions
        possible_transitions = []
        for (from_state, act, to_state), prob in self.transition_probs.items():
            if from_state == current_state and act == action:
                possible_transitions.append((to_state, prob))
        
        if not possible_transitions:
            # No valid transitions, stay in current state
            return current_state, -1.0
        
        # Sample based on probabilities
        states, probs = zip(*possible_transitions)
        next_state = np.random.choice(states, p=probs)
        
        # Get reward
        reward = self.get_reward(current_state, action, next_state)
        
        return next_state, reward


class HatataDecisionEngine:
    """Decision engine that selects actions based on MDP policy."""
    
    def __init__(self, mdp_model: HatataMDPModel):
        self.mdp_model = mdp_model
        self.epsilon = 0.1  # Exploration rate
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        
        # Policy: maps state to action probabilities
        self.policy: Dict[UnderstandingState, Dict[HatataAction, float]] = {}
        self._initialize_policy()
    
    def _initialize_policy(self):
        """Initialize a uniform random policy."""
        
        for state in self.mdp_model.states:
            possible_actions = self.mdp_model.get_possible_actions(state)
            
            if possible_actions:
                action_prob = 1.0 / len(possible_actions)
                self.policy[state] = {action: action_prob for action in possible_actions}
            else:
                self.policy[state] = {}
    
    def select_action(self, state: UnderstandingState, 
                     observation: HatataObservation) -> HatataAction:
        """Select action based on current policy and observation."""
        
        possible_actions = self.mdp_model.get_possible_actions(state)
        
        if not possible_actions:
            return HatataAction.TERMINATE_FAILURE
        
        # Epsilon-greedy action selection with observation-based adjustments
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(possible_actions)
        else:
            # Exploitation: best action based on Q-values and observation
            return self._select_best_action(state, observation, possible_actions)
    
    def _select_best_action(self, state: UnderstandingState, 
                           observation: HatataObservation,
                           possible_actions: List[HatataAction]) -> HatataAction:
        """Select best action based on Q-values and current observation."""
        
        action_scores = {}
        
        for action in possible_actions:
            # Base Q-value
            q_value = self.mdp_model.q_values.get((state, action), 0.0)
            
            # Adjust based on observation
            observation_bonus = self._calculate_observation_bonus(action, observation)
            
            action_scores[action] = q_value + observation_bonus
        
        # Select action with highest score
        best_action = max(action_scores.items(), key=lambda x: x[1])[0]
        return best_action
    
    def _calculate_observation_bonus(self, action: HatataAction, 
                                   observation: HatataObservation) -> float:
        """Calculate bonus/penalty based on current observation."""
        
        bonus = 0.0
        
        # Action-specific bonuses based on observation
        if action == HatataAction.ANALYZE_FEATURES:
            bonus += (1.0 - observation.feature_confidence) * 2.0
        
        elif action == HatataAction.DETECT_PATTERNS:
            bonus += observation.pattern_strength * 1.5
        
        elif action == HatataAction.MAP_SEMANTICS:
            bonus += observation.semantic_clarity * 2.0
        
        elif action == HatataAction.ANALYZE_CONTEXT:
            bonus += observation.context_coherence * 1.8
        
        elif action == HatataAction.ATTEMPT_RECONSTRUCTION:
            bonus += observation.reconstruction_quality * 3.0
        
        elif action == HatataAction.VALIDATE_UNDERSTANDING:
            if observation.reconstruction_quality > 0.8:
                bonus += 2.0
            else:
                bonus -= 1.0
        
        elif action == HatataAction.GATHER_MORE_INFO:
            bonus += observation.uncertainty_level * 1.5
        
        elif action == HatataAction.TERMINATE_SUCCESS:
            if observation.reconstruction_quality > 0.9:
                bonus += 5.0
            else:
                bonus -= 3.0
        
        elif action == HatataAction.TERMINATE_FAILURE:
            if observation.uncertainty_level > 0.8:
                bonus += 1.0
            else:
                bonus -= 2.0
        
        return bonus
    
    def update_q_values(self, state: UnderstandingState, action: HatataAction,
                       reward: float, next_state: UnderstandingState):
        """Update Q-values using Q-learning."""
        
        current_q = self.mdp_model.q_values.get((state, action), 0.0)
        
        # Get maximum Q-value for next state
        next_actions = self.mdp_model.get_possible_actions(next_state)
        if next_actions:
            max_next_q = max(self.mdp_model.q_values.get((next_state, a), 0.0) 
                           for a in next_actions)
        else:
            max_next_q = 0.0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.mdp_model.q_values[(state, action)] = new_q


class HatataEngine:
    """
    Main Hatata MDP engine for probabilistic image understanding verification.
    
    Provides a fallback option when deterministic methods fail and acts as
    an additional verification layer using Markov Decision Process.
    """
    
    def __init__(self):
        self.mdp_model = HatataMDPModel()
        self.decision_engine = HatataDecisionEngine(self.mdp_model)
        
        # Episode tracking
        self.episodes: List[HatataEpisode] = []
        self.current_episode: Optional[HatataEpisode] = None
        
        # Performance metrics
        self.success_rate = 0.0
        self.average_episode_length = 0.0
        self.average_reward = 0.0
        
        logger.info("🎯 Hatata MDP engine initialized")
    
    def start_understanding_episode(self, image_data: Dict[str, Any]) -> str:
        """Start a new understanding episode."""
        
        episode_id = f"hatata_{int(time.time())}_{random.randint(1000, 9999)}"
        
        self.current_episode = HatataEpisode(
            episode_id=episode_id,
            start_time=time.time(),
            initial_state=UnderstandingState.INITIAL
        )
        
        self.current_episode.state_sequence.append(UnderstandingState.INITIAL)
        
        logger.info(f"🎯 Started Hatata episode {episode_id}")
        return episode_id
    
    def step(self, observation: HatataObservation) -> Tuple[HatataAction, UnderstandingState, float, bool]:
        """Take one step in the MDP."""
        
        if not self.current_episode:
            raise ValueError("No active episode. Call start_understanding_episode first.")
        
        # Get current state
        current_state = self.current_episode.state_sequence[-1]
        
        # Check for terminal states
        if current_state in [UnderstandingState.UNDERSTANDING_ACHIEVED, 
                           UnderstandingState.UNDERSTANDING_FAILED]:
            return HatataAction.TERMINATE_SUCCESS, current_state, 0.0, True
        
        # Select action
        action = self.decision_engine.select_action(current_state, observation)
        
        # Execute action (sample next state)
        next_state, reward = self.mdp_model.sample_next_state(current_state, action)
        
        # Update episode
        self.current_episode.action_sequence.append(action)
        self.current_episode.state_sequence.append(next_state)
        self.current_episode.reward_sequence.append(reward)
        self.current_episode.observation_sequence.append(observation)
        self.current_episode.total_reward += reward
        self.current_episode.episode_length += 1
        
        # Update Q-values
        self.decision_engine.update_q_values(current_state, action, reward, next_state)
        
        # Check if episode is done
        done = next_state in [UnderstandingState.UNDERSTANDING_ACHIEVED, 
                            UnderstandingState.UNDERSTANDING_FAILED]
        
        if done:
            self._finish_episode(next_state)
        
        logger.debug(f"Hatata step: {current_state.value} --{action.value}--> {next_state.value} (reward: {reward:.2f})")
        
        return action, next_state, reward, done
    
    def _finish_episode(self, final_state: UnderstandingState):
        """Finish the current episode."""
        
        if not self.current_episode:
            return
        
        self.current_episode.final_state = final_state
        self.current_episode.end_time = time.time()
        self.current_episode.success = (final_state == UnderstandingState.UNDERSTANDING_ACHIEVED)
        
        self.episodes.append(self.current_episode)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        logger.info(f"🎯 Finished Hatata episode {self.current_episode.episode_id}: "
                   f"{'SUCCESS' if self.current_episode.success else 'FAILURE'} "
                   f"(reward: {self.current_episode.total_reward:.2f}, "
                   f"length: {self.current_episode.episode_length})")
        
        self.current_episode = None
    
    def _update_performance_metrics(self):
        """Update overall performance metrics."""
        
        if not self.episodes:
            return
        
        # Success rate
        successes = sum(1 for ep in self.episodes if ep.success)
        self.success_rate = successes / len(self.episodes)
        
        # Average episode length
        self.average_episode_length = np.mean([ep.episode_length for ep in self.episodes])
        
        # Average reward
        self.average_reward = np.mean([ep.total_reward for ep in self.episodes])
    
    def probabilistic_understanding_verification(self, 
                                               reconstruction_results: Dict[str, Any],
                                               max_steps: int = 20) -> Dict[str, Any]:
        """
        Perform probabilistic understanding verification using MDP.
        
        This serves as a fallback when deterministic methods are uncertain
        and as an additional verification layer.
        """
        
        logger.info("🎯 Starting Hatata probabilistic understanding verification")
        
        # Start episode
        episode_id = self.start_understanding_episode(reconstruction_results)
        
        # Create initial observation from reconstruction results
        observation = self._create_observation_from_results(reconstruction_results)
        
        # Run MDP episode
        step_count = 0
        trajectory = []
        
        while step_count < max_steps:
            action, next_state, reward, done = self.step(observation)
            
            trajectory.append({
                'step': step_count,
                'action': action.value,
                'state': next_state.value,
                'reward': reward,
                'observation': {
                    'reconstruction_quality': observation.reconstruction_quality,
                    'feature_confidence': observation.feature_confidence,
                    'uncertainty_level': observation.uncertainty_level
                }
            })
            
            if done:
                break
            
            # Update observation based on action taken (simulate environment response)
            observation = self._update_observation(observation, action, next_state)
            step_count += 1
        
        # Get final episode
        final_episode = self.episodes[-1] if self.episodes else None
        
        # Generate verification results
        results = {
            'episode_id': episode_id,
            'verification_method': 'hatata_mdp',
            'success': final_episode.success if final_episode else False,
            'final_state': final_episode.final_state.value if final_episode else 'unknown',
            'total_reward': final_episode.total_reward if final_episode else 0.0,
            'episode_length': final_episode.episode_length if final_episode else 0,
            'trajectory': trajectory,
            'probabilistic_confidence': self._calculate_probabilistic_confidence(final_episode),
            'understanding_probability': self._estimate_understanding_probability(),
            'verification_insights': self._generate_verification_insights(final_episode),
            'mdp_performance': {
                'success_rate': self.success_rate,
                'average_episode_length': self.average_episode_length,
                'average_reward': self.average_reward
            }
        }
        
        logger.info(f"🎯 Hatata verification completed: "
                   f"{'SUCCESS' if results['success'] else 'FAILURE'} "
                   f"(confidence: {results['probabilistic_confidence']:.3f})")
        
        return results
    
    def _create_observation_from_results(self, results: Dict[str, Any]) -> HatataObservation:
        """Create observation from reconstruction results."""
        
        return HatataObservation(
            timestamp=time.time(),
            reconstruction_quality=results.get('reconstruction_quality', 0.5),
            feature_confidence=results.get('feature_confidence', 0.5),
            pattern_strength=results.get('pattern_strength', 0.5),
            semantic_clarity=results.get('semantic_clarity', 0.5),
            context_coherence=results.get('context_coherence', 0.5),
            uncertainty_level=results.get('uncertainty_level', 0.5),
            additional_metrics=results.get('additional_metrics', {})
        )
    
    def _update_observation(self, current_obs: HatataObservation, 
                          action: HatataAction, 
                          new_state: UnderstandingState) -> HatataObservation:
        """Update observation based on action taken and new state."""
        
        # Simulate how observations change based on actions
        new_obs = HatataObservation(
            timestamp=time.time(),
            reconstruction_quality=current_obs.reconstruction_quality,
            feature_confidence=current_obs.feature_confidence,
            pattern_strength=current_obs.pattern_strength,
            semantic_clarity=current_obs.semantic_clarity,
            context_coherence=current_obs.context_coherence,
            uncertainty_level=current_obs.uncertainty_level
        )
        
        # Action-specific observation updates
        if action == HatataAction.ANALYZE_FEATURES:
            new_obs.feature_confidence = min(1.0, current_obs.feature_confidence + 0.1)
            new_obs.uncertainty_level = max(0.0, current_obs.uncertainty_level - 0.05)
        
        elif action == HatataAction.DETECT_PATTERNS:
            new_obs.pattern_strength = min(1.0, current_obs.pattern_strength + 0.15)
        
        elif action == HatataAction.MAP_SEMANTICS:
            new_obs.semantic_clarity = min(1.0, current_obs.semantic_clarity + 0.2)
        
        elif action == HatataAction.ANALYZE_CONTEXT:
            new_obs.context_coherence = min(1.0, current_obs.context_coherence + 0.15)
        
        elif action == HatataAction.ATTEMPT_RECONSTRUCTION:
            new_obs.reconstruction_quality = min(1.0, current_obs.reconstruction_quality + 0.1)
        
        elif action == HatataAction.GATHER_MORE_INFO:
            new_obs.uncertainty_level = max(0.0, current_obs.uncertainty_level - 0.1)
        
        return new_obs
    
    def _calculate_probabilistic_confidence(self, episode: Optional[HatataEpisode]) -> float:
        """Calculate confidence in the probabilistic verification."""
        
        if not episode:
            return 0.0
        
        # Base confidence from success/failure
        base_confidence = 0.8 if episode.success else 0.2
        
        # Adjust based on episode characteristics
        reward_factor = min(1.0, max(0.0, episode.total_reward / 20.0))
        length_factor = max(0.5, min(1.0, 10.0 / episode.episode_length))
        
        confidence = base_confidence * 0.6 + reward_factor * 0.3 + length_factor * 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _estimate_understanding_probability(self) -> float:
        """Estimate probability of true understanding based on MDP history."""
        
        if not self.episodes:
            return 0.5
        
        # Use recent episodes to estimate probability
        recent_episodes = self.episodes[-10:] if len(self.episodes) >= 10 else self.episodes
        
        success_count = sum(1 for ep in recent_episodes if ep.success)
        probability = success_count / len(recent_episodes)
        
        return probability
    
    def _generate_verification_insights(self, episode: Optional[HatataEpisode]) -> List[str]:
        """Generate insights from the MDP verification process."""
        
        insights = []
        
        if not episode:
            insights.append("No episode data available for analysis")
            return insights
        
        # Episode-specific insights
        if episode.success:
            insights.append(f"MDP successfully navigated to understanding state")
            insights.append(f"Achieved understanding through {episode.episode_length} decision steps")
        else:
            insights.append(f"MDP failed to reach understanding state")
            insights.append(f"Process terminated after {episode.episode_length} steps")
        
        # Reward analysis
        if episode.total_reward > 10:
            insights.append("High reward indicates strong understanding progression")
        elif episode.total_reward < 0:
            insights.append("Negative reward suggests understanding difficulties")
        
        # State transition analysis
        if len(episode.state_sequence) > 1:
            unique_states = len(set(episode.state_sequence))
            insights.append(f"Explored {unique_states} different understanding states")
        
        # Performance context
        insights.append(f"Overall MDP success rate: {self.success_rate:.1%}")
        
        return insights
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        
        return {
            'total_episodes': len(self.episodes),
            'success_rate': self.success_rate,
            'average_episode_length': self.average_episode_length,
            'average_reward': self.average_reward,
            'recent_performance': {
                'last_10_episodes': len([ep for ep in self.episodes[-10:] if ep.success]) / min(10, len(self.episodes)) if self.episodes else 0.0
            },
            'state_statistics': self._get_state_statistics(),
            'action_statistics': self._get_action_statistics()
        }
    
    def _get_state_statistics(self) -> Dict[str, Any]:
        """Get statistics about state visits."""
        
        state_counts = defaultdict(int)
        
        for episode in self.episodes:
            for state in episode.state_sequence:
                state_counts[state.value] += 1
        
        total_visits = sum(state_counts.values())
        
        return {
            'state_visit_counts': dict(state_counts),
            'state_visit_frequencies': {
                state: count / total_visits for state, count in state_counts.items()
            } if total_visits > 0 else {}
        }
    
    def _get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics about action usage."""
        
        action_counts = defaultdict(int)
        
        for episode in self.episodes:
            for action in episode.action_sequence:
                action_counts[action.value] += 1
        
        total_actions = sum(action_counts.values())
        
        return {
            'action_usage_counts': dict(action_counts),
            'action_usage_frequencies': {
                action: count / total_actions for action, count in action_counts.items()
            } if total_actions > 0 else {}
        }


# Example usage and integration
if __name__ == "__main__":
    # Initialize Hatata engine
    hatata = HatataEngine()
    
    # Simulate reconstruction results for verification
    reconstruction_results = {
        'reconstruction_quality': 0.75,
        'feature_confidence': 0.68,
        'pattern_strength': 0.72,
        'semantic_clarity': 0.65,
        'context_coherence': 0.70,
        'uncertainty_level': 0.35
    }
    
    # Perform probabilistic verification
    verification_results = hatata.probabilistic_understanding_verification(
        reconstruction_results, 
        max_steps=15
    )
    
    print(f"🎯 Hatata MDP Verification Results:")
    print(f"Success: {verification_results['success']}")
    print(f"Final State: {verification_results['final_state']}")
    print(f"Probabilistic Confidence: {verification_results['probabilistic_confidence']:.3f}")
    print(f"Understanding Probability: {verification_results['understanding_probability']:.3f}")
    print(f"Episode Length: {verification_results['episode_length']}")
    print(f"Total Reward: {verification_results['total_reward']:.2f}")
    
    print(f"\nInsights:")
    for insight in verification_results['verification_insights']:
        print(f"  • {insight}")
    
    # Get performance report
    performance = hatata.get_performance_report()
    print(f"\nMDP Performance:")
    print(f"  Success Rate: {performance['success_rate']:.1%}")
    print(f"  Average Episode Length: {performance['average_episode_length']:.1f}")
    print(f"  Average Reward: {performance['average_reward']:.2f}") 