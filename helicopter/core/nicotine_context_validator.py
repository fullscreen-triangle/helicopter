"""
Nicotine Context Validator

A "cigarette break" for AI systems to prevent context drift and task amnesia.

Core Innovation:
- AI systems lose context over time and forget what they're supposed to be doing
- Nicotine kicks in after N processes to present machine-readable puzzles
- If the system solves the puzzle, it proves understanding and context retention
- System must summarize current context to continue processing
- Prevents cognitive drift and maintains task focus

The name "nicotine" reflects the idea of a brief pause (like a cigarette break) 
that actually helps maintain focus and cognitive performance.
"""

import json
import time
import hashlib
import random
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class PuzzleType(Enum):
    """Types of machine-readable puzzles for context validation."""
    SEQUENCE_COMPLETION = "sequence_completion"      # Complete a logical sequence
    PATTERN_RECOGNITION = "pattern_recognition"      # Identify patterns in data
    CONTEXT_SUMMARIZATION = "context_summarization"  # Summarize current context
    TASK_OBJECTIVE_RECALL = "task_objective_recall"  # Recall primary objectives
    STATE_CONSISTENCY_CHECK = "state_consistency"    # Verify state consistency
    DEPENDENCY_MAPPING = "dependency_mapping"        # Map process dependencies
    PROGRESS_ASSESSMENT = "progress_assessment"      # Assess current progress
    ERROR_DETECTION = "error_detection"              # Detect logical errors


class ContextDriftSeverity(Enum):
    """Severity levels of context drift."""
    NONE = "none"           # No drift detected
    MINIMAL = "minimal"     # Slight drift, continue with caution
    MODERATE = "moderate"   # Noticeable drift, require validation
    SEVERE = "severe"       # Significant drift, halt and reorient
    CRITICAL = "critical"   # Complete context loss, restart required


@dataclass
class ContextSnapshot:
    """Snapshot of system context at a specific point in time."""
    
    timestamp: float
    process_count: int
    current_task: str
    primary_objectives: List[str]
    active_processes: List[str]
    system_state: Dict[str, Any]
    recent_decisions: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    context_hash: str = ""
    
    def __post_init__(self):
        """Generate context hash for integrity checking."""
        context_data = {
            'task': self.current_task,
            'objectives': sorted(self.primary_objectives),
            'processes': sorted(self.active_processes),
            'decisions': self.recent_decisions[-5:] if self.recent_decisions else []
        }
        self.context_hash = hashlib.md5(
            json.dumps(context_data, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class NicotinePuzzle:
    """A machine-readable puzzle for context validation."""
    
    puzzle_id: str
    puzzle_type: PuzzleType
    difficulty: float  # 0.0 to 1.0
    question: Dict[str, Any]
    expected_answer: Dict[str, Any]
    context_requirements: List[str]
    time_limit: float  # seconds
    
    # Validation state
    presented_at: float = 0.0
    answered_at: float = 0.0
    provided_answer: Optional[Dict[str, Any]] = None
    is_correct: bool = False
    confidence_score: float = 0.0


@dataclass
class NicotineSession:
    """A context validation session."""
    
    session_id: str
    triggered_at: float
    trigger_reason: str
    context_before: ContextSnapshot
    puzzles: List[NicotinePuzzle]
    
    # Session results
    completed_at: float = 0.0
    puzzles_solved: int = 0
    total_puzzles: int = 0
    average_confidence: float = 0.0
    context_drift_detected: ContextDriftSeverity = ContextDriftSeverity.NONE
    context_after: Optional[ContextSnapshot] = None
    session_passed: bool = False
    continuation_approved: bool = False


class NicotinePuzzleGenerator:
    """Generates machine-readable puzzles for context validation."""
    
    def __init__(self):
        self.puzzle_templates = self._initialize_puzzle_templates()
    
    def generate_puzzle(self, 
                       puzzle_type: PuzzleType,
                       context: ContextSnapshot,
                       difficulty: float = 0.5) -> NicotinePuzzle:
        """Generate a context-specific puzzle."""
        
        puzzle_id = f"{puzzle_type.value}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        if puzzle_type == PuzzleType.SEQUENCE_COMPLETION:
            return self._generate_sequence_puzzle(puzzle_id, context, difficulty)
        elif puzzle_type == PuzzleType.PATTERN_RECOGNITION:
            return self._generate_pattern_puzzle(puzzle_id, context, difficulty)
        elif puzzle_type == PuzzleType.CONTEXT_SUMMARIZATION:
            return self._generate_summarization_puzzle(puzzle_id, context, difficulty)
        elif puzzle_type == PuzzleType.TASK_OBJECTIVE_RECALL:
            return self._generate_objective_recall_puzzle(puzzle_id, context, difficulty)
        elif puzzle_type == PuzzleType.STATE_CONSISTENCY_CHECK:
            return self._generate_consistency_puzzle(puzzle_id, context, difficulty)
        elif puzzle_type == PuzzleType.DEPENDENCY_MAPPING:
            return self._generate_dependency_puzzle(puzzle_id, context, difficulty)
        elif puzzle_type == PuzzleType.PROGRESS_ASSESSMENT:
            return self._generate_progress_puzzle(puzzle_id, context, difficulty)
        elif puzzle_type == PuzzleType.ERROR_DETECTION:
            return self._generate_error_detection_puzzle(puzzle_id, context, difficulty)
        else:
            raise ValueError(f"Unknown puzzle type: {puzzle_type}")
    
    def _generate_sequence_puzzle(self, puzzle_id: str, 
                                context: ContextSnapshot, 
                                difficulty: float) -> NicotinePuzzle:
        """Generate a sequence completion puzzle."""
        
        # Create a sequence based on process counts or metrics
        base_sequence = [1, 2, 4, 8, 16]  # Powers of 2
        if context.performance_metrics:
            # Use actual metrics to create sequence
            metrics_values = list(context.performance_metrics.values())
            if len(metrics_values) >= 3:
                base_sequence = metrics_values[:4]
        
        # Remove last element as the answer
        question_sequence = base_sequence[:-1]
        expected_answer = base_sequence[-1]
        
        question = {
            "type": "sequence_completion",
            "sequence": question_sequence,
            "instruction": "Complete the sequence based on the pattern",
            "context_hint": f"Consider the progression in {context.current_task}"
        }
        
        expected = {
            "next_value": expected_answer,
            "pattern_explanation": "Powers of 2 progression"
        }
        
        return NicotinePuzzle(
            puzzle_id=puzzle_id,
            puzzle_type=PuzzleType.SEQUENCE_COMPLETION,
            difficulty=difficulty,
            question=question,
            expected_answer=expected,
            context_requirements=["mathematical_reasoning", "pattern_recognition"],
            time_limit=30.0
        )
    
    def _generate_context_summarization_puzzle(self, puzzle_id: str,
                                             context: ContextSnapshot,
                                             difficulty: float) -> NicotinePuzzle:
        """Generate a context summarization puzzle."""
        
        question = {
            "type": "context_summarization",
            "instruction": "Summarize the current system context",
            "required_elements": [
                "primary_task",
                "active_processes", 
                "recent_decisions",
                "current_objectives"
            ],
            "format": "structured_summary"
        }
        
        expected = {
            "primary_task": context.current_task,
            "active_processes": context.active_processes,
            "objectives_count": len(context.primary_objectives),
            "recent_decisions_count": len(context.recent_decisions),
            "context_integrity": context.context_hash[:8]  # First 8 chars
        }
        
        return NicotinePuzzle(
            puzzle_id=puzzle_id,
            puzzle_type=PuzzleType.CONTEXT_SUMMARIZATION,
            difficulty=difficulty,
            question=question,
            expected_answer=expected,
            context_requirements=["context_awareness", "summarization"],
            time_limit=60.0
        )
    
    def _generate_objective_recall_puzzle(self, puzzle_id: str,
                                        context: ContextSnapshot,
                                        difficulty: float) -> NicotinePuzzle:
        """Generate a task objective recall puzzle."""
        
        # Scramble objectives and ask to identify primary ones
        all_objectives = context.primary_objectives.copy()
        
        # Add some decoy objectives
        decoy_objectives = [
            "optimize_memory_usage",
            "minimize_processing_time", 
            "maximize_throughput",
            "reduce_error_rate"
        ]
        
        # Mix real and decoy objectives
        mixed_objectives = all_objectives + random.sample(decoy_objectives, 2)
        random.shuffle(mixed_objectives)
        
        question = {
            "type": "objective_recall",
            "instruction": "Identify the primary objectives for the current task",
            "objective_list": mixed_objectives,
            "select_count": len(all_objectives),
            "context": f"Current task: {context.current_task}"
        }
        
        expected = {
            "primary_objectives": sorted(all_objectives),
            "objective_count": len(all_objectives)
        }
        
        return NicotinePuzzle(
            puzzle_id=puzzle_id,
            puzzle_type=PuzzleType.TASK_OBJECTIVE_RECALL,
            difficulty=difficulty,
            question=question,
            expected_answer=expected,
            context_requirements=["objective_awareness", "task_memory"],
            time_limit=45.0
        )
    
    def _generate_consistency_puzzle(self, puzzle_id: str,
                                   context: ContextSnapshot,
                                   difficulty: float) -> NicotinePuzzle:
        """Generate a state consistency check puzzle."""
        
        # Create a consistency check based on system state
        state_items = list(context.system_state.items())[:5]  # First 5 items
        
        # Introduce one inconsistency
        inconsistent_items = state_items.copy()
        if inconsistent_items:
            # Modify one item to be inconsistent
            idx = random.randint(0, len(inconsistent_items) - 1)
            key, value = inconsistent_items[idx]
            if isinstance(value, (int, float)):
                inconsistent_items[idx] = (key, value * -1)  # Make negative
            elif isinstance(value, str):
                inconsistent_items[idx] = (key, value + "_CORRUPTED")
            elif isinstance(value, bool):
                inconsistent_items[idx] = (key, not value)
        
        question = {
            "type": "consistency_check",
            "instruction": "Identify inconsistencies in the system state",
            "state_snapshot": dict(inconsistent_items),
            "expected_state_properties": {
                "all_values_positive": True,
                "no_corrupted_strings": True,
                "boolean_consistency": True
            }
        }
        
        expected = {
            "inconsistencies_found": 1,
            "inconsistent_key": inconsistent_items[idx][0] if inconsistent_items else None,
            "consistency_status": "failed"
        }
        
        return NicotinePuzzle(
            puzzle_id=puzzle_id,
            puzzle_type=PuzzleType.STATE_CONSISTENCY_CHECK,
            difficulty=difficulty,
            question=question,
            expected_answer=expected,
            context_requirements=["state_analysis", "error_detection"],
            time_limit=40.0
        )
    
    def _initialize_puzzle_templates(self) -> Dict[PuzzleType, Dict[str, Any]]:
        """Initialize puzzle templates for different types."""
        
        return {
            PuzzleType.SEQUENCE_COMPLETION: {
                "base_difficulty": 0.3,
                "time_multiplier": 1.0,
                "context_weight": 0.7
            },
            PuzzleType.PATTERN_RECOGNITION: {
                "base_difficulty": 0.4,
                "time_multiplier": 1.2,
                "context_weight": 0.6
            },
            PuzzleType.CONTEXT_SUMMARIZATION: {
                "base_difficulty": 0.6,
                "time_multiplier": 2.0,
                "context_weight": 1.0
            },
            PuzzleType.TASK_OBJECTIVE_RECALL: {
                "base_difficulty": 0.5,
                "time_multiplier": 1.5,
                "context_weight": 0.9
            },
            PuzzleType.STATE_CONSISTENCY_CHECK: {
                "base_difficulty": 0.7,
                "time_multiplier": 1.3,
                "context_weight": 0.8
            }
        }


class NicotineContextValidator:
    """
    Main context validator that acts as a "cigarette break" for AI systems.
    
    Prevents context drift by periodically validating that the system still
    understands what it's supposed to be doing through machine-readable puzzles.
    """
    
    def __init__(self, 
                 trigger_interval: int = 10,  # Trigger after N processes
                 puzzle_count: int = 3,       # Number of puzzles per session
                 pass_threshold: float = 0.7, # Minimum score to pass
                 max_retries: int = 2):       # Max retries before halt
        
        self.trigger_interval = trigger_interval
        self.puzzle_count = puzzle_count
        self.pass_threshold = pass_threshold
        self.max_retries = max_retries
        
        self.puzzle_generator = NicotinePuzzleGenerator()
        
        # State tracking
        self.process_count = 0
        self.last_validation = 0.0
        self.context_history: List[ContextSnapshot] = []
        self.validation_sessions: List[NicotineSession] = []
        
        # Context drift detection
        self.baseline_context: Optional[ContextSnapshot] = None
        self.drift_threshold = 0.3  # Threshold for context drift detection
        
        logger.info(f"Nicotine context validator initialized: "
                   f"interval={trigger_interval}, puzzles={puzzle_count}")
    
    def register_process(self, 
                        process_name: str,
                        current_task: str,
                        objectives: List[str],
                        system_state: Dict[str, Any],
                        recent_decisions: List[Dict[str, Any]] = None) -> bool:
        """
        Register a process and check if validation is needed.
        
        Returns True if system can continue, False if validation failed.
        """
        
        self.process_count += 1
        
        # Create context snapshot
        context = ContextSnapshot(
            timestamp=time.time(),
            process_count=self.process_count,
            current_task=current_task,
            primary_objectives=objectives,
            active_processes=[process_name],
            system_state=system_state,
            recent_decisions=recent_decisions or [],
            performance_metrics=self._extract_performance_metrics(system_state)
        )
        
        self.context_history.append(context)
        
        # Set baseline if first process
        if self.baseline_context is None:
            self.baseline_context = context
            logger.info("Baseline context established")
            return True
        
        # Check if validation is needed
        if self._should_trigger_validation(context):
            logger.info(f"🚬 Nicotine break triggered after {self.process_count} processes")
            return self._perform_validation(context)
        
        return True
    
    def _should_trigger_validation(self, current_context: ContextSnapshot) -> bool:
        """Determine if validation should be triggered."""
        
        # Trigger based on process count
        if self.process_count % self.trigger_interval == 0:
            return True
        
        # Trigger based on context drift
        if self._detect_context_drift(current_context) >= ContextDriftSeverity.MODERATE:
            logger.warning("Context drift detected - triggering validation")
            return True
        
        # Trigger based on time since last validation
        time_since_last = time.time() - self.last_validation
        if time_since_last > 300:  # 5 minutes
            return True
        
        return False
    
    def _detect_context_drift(self, current_context: ContextSnapshot) -> ContextDriftSeverity:
        """Detect context drift by comparing with baseline and recent contexts."""
        
        if not self.baseline_context:
            return ContextDriftSeverity.NONE
        
        drift_indicators = []
        
        # Check task consistency
        if current_context.current_task != self.baseline_context.current_task:
            drift_indicators.append("task_change")
        
        # Check objective consistency
        baseline_objectives = set(self.baseline_context.primary_objectives)
        current_objectives = set(current_context.primary_objectives)
        
        objective_overlap = len(baseline_objectives & current_objectives)
        objective_drift = 1.0 - (objective_overlap / max(len(baseline_objectives), 1))
        
        if objective_drift > 0.5:
            drift_indicators.append("objective_drift")
        
        # Check context hash similarity with recent contexts
        if len(self.context_history) > 3:
            recent_hashes = [ctx.context_hash for ctx in self.context_history[-3:]]
            if len(set(recent_hashes)) == len(recent_hashes):  # All different
                drift_indicators.append("context_instability")
        
        # Determine severity
        drift_count = len(drift_indicators)
        
        if drift_count == 0:
            return ContextDriftSeverity.NONE
        elif drift_count == 1:
            return ContextDriftSeverity.MINIMAL
        elif drift_count == 2:
            return ContextDriftSeverity.MODERATE
        elif drift_count == 3:
            return ContextDriftSeverity.SEVERE
        else:
            return ContextDriftSeverity.CRITICAL
    
    def _perform_validation(self, context: ContextSnapshot) -> bool:
        """Perform context validation through puzzles."""
        
        session_id = f"nicotine_{int(time.time())}_{random.randint(100, 999)}"
        
        # Determine trigger reason
        drift_level = self._detect_context_drift(context)
        if drift_level >= ContextDriftSeverity.MODERATE:
            trigger_reason = f"context_drift_{drift_level.value}"
        else:
            trigger_reason = f"interval_trigger_{self.process_count}"
        
        # Create validation session
        session = NicotineSession(
            session_id=session_id,
            triggered_at=time.time(),
            trigger_reason=trigger_reason,
            context_before=context,
            puzzles=[],
            total_puzzles=self.puzzle_count
        )
        
        logger.info(f"Starting nicotine session {session_id}: {trigger_reason}")
        
        # Generate puzzles
        puzzle_types = [
            PuzzleType.CONTEXT_SUMMARIZATION,
            PuzzleType.TASK_OBJECTIVE_RECALL,
            PuzzleType.STATE_CONSISTENCY_CHECK
        ]
        
        for i in range(self.puzzle_count):
            puzzle_type = puzzle_types[i % len(puzzle_types)]
            difficulty = 0.5 + (i * 0.1)  # Increasing difficulty
            
            puzzle = self.puzzle_generator.generate_puzzle(
                puzzle_type, context, difficulty
            )
            session.puzzles.append(puzzle)
        
        # Present puzzles (in real implementation, this would interface with the AI system)
        session_passed = self._present_puzzles(session)
        
        # Update session results
        session.completed_at = time.time()
        session.session_passed = session_passed
        session.continuation_approved = session_passed
        session.context_after = context  # In real implementation, capture post-validation context
        
        self.validation_sessions.append(session)
        self.last_validation = time.time()
        
        if session_passed:
            logger.info(f"✅ Nicotine session {session_id} PASSED - system can continue")
            return True
        else:
            logger.warning(f"❌ Nicotine session {session_id} FAILED - context validation failed")
            return False
    
    def _present_puzzles(self, session: NicotineSession) -> bool:
        """Present puzzles to the system for solving."""
        
        # This is a simplified implementation
        # In practice, this would interface with the actual AI system
        
        solved_count = 0
        total_confidence = 0.0
        
        for puzzle in session.puzzles:
            puzzle.presented_at = time.time()
            
            # Simulate puzzle solving (in real implementation, this would call the AI system)
            is_correct, confidence = self._simulate_puzzle_solving(puzzle)
            
            puzzle.answered_at = time.time()
            puzzle.is_correct = is_correct
            puzzle.confidence_score = confidence
            
            if is_correct:
                solved_count += 1
            
            total_confidence += confidence
        
        session.puzzles_solved = solved_count
        session.average_confidence = total_confidence / len(session.puzzles)
        
        # Determine if session passed
        success_rate = solved_count / len(session.puzzles)
        passed = (success_rate >= self.pass_threshold and 
                 session.average_confidence >= 0.6)
        
        logger.info(f"Puzzle results: {solved_count}/{len(session.puzzles)} solved, "
                   f"avg confidence: {session.average_confidence:.3f}")
        
        return passed
    
    def _simulate_puzzle_solving(self, puzzle: NicotinePuzzle) -> Tuple[bool, float]:
        """Simulate puzzle solving (placeholder for actual AI system interface)."""
        
        # This is a placeholder - in real implementation, this would:
        # 1. Present the puzzle to the AI system
        # 2. Collect the system's response
        # 3. Validate the response against expected answer
        # 4. Calculate confidence score
        
        # For now, simulate with random success based on difficulty
        success_probability = max(0.3, 1.0 - puzzle.difficulty)
        is_correct = random.random() < success_probability
        confidence = random.uniform(0.4, 0.9) if is_correct else random.uniform(0.1, 0.5)
        
        return is_correct, confidence
    
    def _extract_performance_metrics(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics from system state."""
        
        metrics = {}
        
        for key, value in system_state.items():
            if isinstance(value, (int, float)):
                if 'quality' in key.lower() or 'score' in key.lower():
                    metrics[key] = float(value)
                elif 'time' in key.lower() or 'duration' in key.lower():
                    metrics[key] = float(value)
                elif 'count' in key.lower() or 'iteration' in key.lower():
                    metrics[key] = float(value)
        
        return metrics
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report."""
        
        if not self.validation_sessions:
            return {
                'total_sessions': 0,
                'status': 'no_validations_performed',
                'process_count': self.process_count
            }
        
        recent_session = self.validation_sessions[-1]
        
        # Calculate overall statistics
        total_sessions = len(self.validation_sessions)
        passed_sessions = sum(1 for s in self.validation_sessions if s.session_passed)
        pass_rate = passed_sessions / total_sessions
        
        # Calculate average performance
        avg_puzzles_solved = np.mean([s.puzzles_solved for s in self.validation_sessions])
        avg_confidence = np.mean([s.average_confidence for s in self.validation_sessions])
        
        return {
            'total_sessions': total_sessions,
            'passed_sessions': passed_sessions,
            'pass_rate': pass_rate,
            'average_puzzles_solved': avg_puzzles_solved,
            'average_confidence': avg_confidence,
            'recent_session': {
                'session_id': recent_session.session_id,
                'trigger_reason': recent_session.trigger_reason,
                'passed': recent_session.session_passed,
                'puzzles_solved': recent_session.puzzles_solved,
                'total_puzzles': recent_session.total_puzzles,
                'confidence': recent_session.average_confidence
            },
            'context_drift_history': [
                self._detect_context_drift(ctx).value for ctx in self.context_history[-5:]
            ],
            'process_count': self.process_count,
            'last_validation': self.last_validation,
            'context_drift_detected': len([
                ctx for ctx in self.context_history[-5:] 
                if self._detect_context_drift(ctx) >= ContextDriftSeverity.MODERATE
            ]) > 0
        }


# Example usage and integration
if __name__ == "__main__":
    # Initialize nicotine validator
    validator = NicotineContextValidator(
        trigger_interval=5,  # Validate every 5 processes
        puzzle_count=3,
        pass_threshold=0.7
    )
    
    # Simulate system processes
    for i in range(12):
        system_state = {
            'reconstruction_quality': 0.8 + random.uniform(-0.1, 0.1),
            'iteration_count': i + 1,
            'confidence_score': 0.7 + random.uniform(-0.2, 0.2),
            'processing_time': random.uniform(1.0, 3.0)
        }
        
        objectives = [
            "validate_image_understanding",
            "improve_reconstruction_quality", 
            "maintain_context_awareness"
        ]
        
        can_continue = validator.register_process(
            process_name=f"reconstruction_process_{i}",
            current_task="autonomous_image_reconstruction",
            objectives=objectives,
            system_state=system_state
        )
        
        if not can_continue:
            print(f"🛑 Process {i} halted due to validation failure")
            break
        
        print(f"✅ Process {i} completed successfully")
    
    # Get validation report
    report = validator.get_validation_report()
    print(f"\n📊 Validation Report:")
    print(f"Total sessions: {report['total_sessions']}")
    print(f"Pass rate: {report['pass_rate']:.1%}")
    print(f"Average confidence: {report['average_confidence']:.3f}")
    print(f"Recent session passed: {report['recent_session']['passed']}")
    print(f"Context drift detected: {report['context_drift_detected']}") 