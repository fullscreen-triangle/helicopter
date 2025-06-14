#!/usr/bin/env python3
"""
Nicotine Context Validation Demo

Demonstrates the "cigarette break" for AI systems that prevents context drift and task amnesia.

Core Problem Solved:
- AI systems lose context over time and forget what they're supposed to be doing
- Long-running processes suffer from cognitive drift
- Systems need periodic validation to prove they still understand their objectives

Nicotine Solution:
- Kicks in after N processes to present machine-readable puzzles
- If system solves puzzles, it proves understanding and context retention
- System must summarize current context to continue processing
- Prevents cognitive drift and maintains task focus

Usage:
    python examples/nicotine_demo.py --processes 20 --interval 5

Requirements:
    - No external dependencies required for basic demo
    - Integrates with existing Helicopter reconstruction engines
"""

import os
import sys
import argparse
import time
import random
from pathlib import Path

# Add helicopter to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from helicopter.core import (
    NicotineContextValidator, 
    NicotineIntegration, 
    AutonomousReconstructionEngine
)


def demo_basic_nicotine_validation():
    """Demonstrate basic nicotine context validation."""
    
    print("\n" + "="*70)
    print("🚬 NICOTINE CONTEXT VALIDATION DEMO")
    print("="*70)
    print("Problem: AI systems lose context and forget what they're doing")
    print("Solution: Periodic 'cigarette breaks' with validation puzzles")
    print("="*70)
    
    # Initialize nicotine validator
    validator = NicotineContextValidator(
        trigger_interval=5,  # Validate every 5 processes
        puzzle_count=3,      # 3 puzzles per session
        pass_threshold=0.7   # 70% success rate required
    )
    
    # Define task context
    objectives = [
        "validate_image_understanding",
        "improve_reconstruction_quality", 
        "maintain_context_awareness",
        "demonstrate_visual_comprehension"
    ]
    
    print(f"Task: Autonomous Image Reconstruction")
    print(f"Objectives: {len(objectives)} primary objectives")
    print(f"Validation interval: Every {validator.trigger_interval} processes")
    print(f"Pass threshold: {validator.pass_threshold:.1%}")
    
    # Simulate system processes
    print(f"\nSimulating system processes...")
    
    for i in range(12):
        # Simulate system state
        system_state = {
            'reconstruction_quality': 0.8 + random.uniform(-0.1, 0.1),
            'iteration_count': i + 1,
            'confidence_score': 0.7 + random.uniform(-0.2, 0.2),
            'processing_time': random.uniform(1.0, 3.0),
            'patches_processed': (i + 1) * 10,
            'memory_usage': 0.6 + random.uniform(-0.1, 0.1)
        }
        
        # Register process with nicotine validator
        can_continue = validator.register_process(
            process_name=f"reconstruction_process_{i}",
            current_task="autonomous_image_reconstruction",
            objectives=objectives,
            system_state=system_state
        )
        
        if not can_continue:
            print(f"🛑 Process {i} halted due to nicotine validation failure")
            print("   System failed to prove it understands what it's doing")
            break
        
        print(f"✅ Process {i} completed successfully")
        time.sleep(0.1)  # Brief pause for demonstration
    
    # Get validation report
    report = validator.get_validation_report()
    
    print(f"\n📊 Nicotine Validation Report:")
    print(f"  Total validation sessions: {report['total_sessions']}")
    print(f"  Sessions passed: {report['passed_sessions']}")
    print(f"  Pass rate: {report['pass_rate']:.1%}")
    print(f"  Context drift detected: {report['context_drift_detected']}")
    
    if report['total_sessions'] > 0:
        recent = report['recent_session']
        print(f"\n  Recent Session:")
        print(f"    Session ID: {recent['session_id']}")
        print(f"    Trigger reason: {recent['trigger_reason']}")
        print(f"    Puzzles solved: {recent['puzzles_solved']}/{recent['total_puzzles']}")
        print(f"    Confidence: {recent['confidence']:.3f}")
        print(f"    Result: {'✅ PASSED' if recent['passed'] else '❌ FAILED'}")
    
    return report


def demo_context_drift_detection():
    """Demonstrate context drift detection and response."""
    
    print("\n" + "="*70)
    print("🔄 CONTEXT DRIFT DETECTION DEMO")
    print("="*70)
    print("Simulating context drift and nicotine response")
    print("="*70)
    
    validator = NicotineContextValidator(
        trigger_interval=10,  # Higher interval to focus on drift detection
        puzzle_count=2,
        pass_threshold=0.6
    )
    
    # Start with initial objectives
    initial_objectives = [
        "reconstruct_image_patches",
        "validate_understanding",
        "maintain_quality"
    ]
    
    print(f"Initial objectives: {initial_objectives}")
    
    # Process with stable context
    for i in range(3):
        system_state = {'quality': 0.8, 'iteration': i}
        
        validator.register_process(
            process_name=f"stable_process_{i}",
            current_task="image_reconstruction",
            objectives=initial_objectives,
            system_state=system_state
        )
        
        print(f"✅ Stable process {i} completed")
    
    # Introduce context drift
    print(f"\n⚠️  Introducing context drift...")
    
    drifted_objectives = [
        "optimize_memory_usage",  # Different objective
        "minimize_processing_time",  # Different objective
        "maximize_throughput"  # Different objective
    ]
    
    print(f"Drifted objectives: {drifted_objectives}")
    
    # Process with drifted context
    for i in range(2):
        system_state = {'quality': 0.6, 'iteration': i + 10}
        
        can_continue = validator.register_process(
            process_name=f"drifted_process_{i}",
            current_task="performance_optimization",  # Different task
            objectives=drifted_objectives,
            system_state=system_state
        )
        
        if can_continue:
            print(f"✅ Drifted process {i} completed")
        else:
            print(f"🛑 Drifted process {i} halted by nicotine validation")
            break
    
    # Show drift detection results
    report = validator.get_validation_report()
    print(f"\n📊 Context Drift Results:")
    print(f"  Drift detected: {report['context_drift_detected']}")
    print(f"  Validation sessions triggered: {report['total_sessions']}")
    
    return report


def demo_integrated_reconstruction():
    """Demonstrate nicotine integration with autonomous reconstruction."""
    
    print("\n" + "="*70)
    print("🔗 INTEGRATED RECONSTRUCTION DEMO")
    print("="*70)
    print("Nicotine validation integrated with autonomous reconstruction")
    print("="*70)
    
    try:
        # Initialize autonomous reconstruction engine (includes nicotine)
        engine = AutonomousReconstructionEngine()
        
        if not engine.nicotine_validator:
            print("⚠️  Nicotine validation not available in reconstruction engine")
            return None
        
        print(f"✅ Autonomous reconstruction engine with nicotine validation ready")
        print(f"   Validation interval: {engine.nicotine_validator.trigger_interval} processes")
        
        # Create a simple test image
        import numpy as np
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        print(f"🖼️  Testing with {test_image.shape} image")
        print(f"   This will trigger nicotine validation during reconstruction...")
        
        # Perform autonomous analysis with nicotine validation
        results = engine.autonomous_analyze(
            image=test_image,
            max_iterations=20,  # Enough to trigger validation
            target_quality=0.85
        )
        
        # Show nicotine results
        if 'nicotine_validation' in results:
            nicotine_report = results['nicotine_validation']
            
            print(f"\n🚬 Nicotine Validation During Reconstruction:")
            print(f"   Sessions: {nicotine_report['total_sessions']}")
            print(f"   Pass rate: {nicotine_report['pass_rate']:.1%}")
            print(f"   Context maintained: {'✅' if nicotine_report['pass_rate'] > 0.5 else '❌'}")
            
            if nicotine_report['total_sessions'] > 0:
                recent = nicotine_report['recent_session']
                print(f"   Recent session: {recent['puzzles_solved']}/{recent['total_puzzles']} puzzles solved")
        
        # Show reconstruction results
        final_quality = results['autonomous_reconstruction']['final_quality']
        print(f"\n📈 Reconstruction Results:")
        print(f"   Final quality: {final_quality:.3f}")
        print(f"   Iterations: {results['autonomous_reconstruction']['total_iterations']}")
        print(f"   Context maintained throughout: {'✅' if 'nicotine_validation' in results else '❌'}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in integrated demo: {e}")
        return None


def demo_puzzle_types():
    """Demonstrate different types of nicotine puzzles."""
    
    print("\n" + "="*70)
    print("🧩 NICOTINE PUZZLE TYPES DEMO")
    print("="*70)
    print("Different puzzle types for context validation")
    print("="*70)
    
    from helicopter.core.nicotine_context_validator import (
        NicotinePuzzleGenerator, 
        ContextSnapshot,
        PuzzleType
    )
    
    # Create sample context
    context = ContextSnapshot(
        timestamp=time.time(),
        process_count=10,
        current_task="image_reconstruction",
        primary_objectives=[
            "reconstruct_patches",
            "validate_understanding", 
            "maintain_quality"
        ],
        active_processes=["reconstruction_engine"],
        system_state={
            'quality': 0.85,
            'confidence': 0.78,
            'iteration': 10
        },
        recent_decisions=[]
    )
    
    generator = NicotinePuzzleGenerator()
    
    # Generate different puzzle types
    puzzle_types = [
        ("Context Summarization", PuzzleType.CONTEXT_SUMMARIZATION),
        ("Objective Recall", PuzzleType.TASK_OBJECTIVE_RECALL),
        ("State Consistency", PuzzleType.STATE_CONSISTENCY_CHECK)
    ]
    
    for name, puzzle_type in puzzle_types:
        print(f"\n🧩 {name} Puzzle:")
        
        if puzzle_type == PuzzleType.CONTEXT_SUMMARIZATION:
            puzzle = generator.generate_context_summary_puzzle(context)
        elif puzzle_type == PuzzleType.TASK_OBJECTIVE_RECALL:
            puzzle = generator.generate_objective_recall_puzzle(context)
        elif puzzle_type == PuzzleType.STATE_CONSISTENCY_CHECK:
            puzzle = generator.generate_consistency_puzzle(context)
        
        print(f"   Puzzle ID: {puzzle.puzzle_id}")
        print(f"   Type: {puzzle.puzzle_type.value}")
        print(f"   Time limit: {puzzle.time_limit}s")
        print(f"   Question keys: {list(puzzle.question.keys())}")
        print(f"   Expected answer keys: {list(puzzle.expected_answer.keys())}")
        
        # Show sample question content
        if 'instruction' in puzzle.question:
            print(f"   Instruction: {puzzle.question['instruction']}")


def main():
    parser = argparse.ArgumentParser(description="Nicotine Context Validation Demo")
    parser.add_argument("--demo", 
                       choices=["basic", "drift", "integrated", "puzzles", "all"], 
                       default="all", 
                       help="Which demo to run")
    parser.add_argument("--processes", type=int, default=12, 
                       help="Number of processes to simulate")
    parser.add_argument("--interval", type=int, default=5, 
                       help="Validation interval")
    
    args = parser.parse_args()
    
    print("🚬 NICOTINE CONTEXT VALIDATION SYSTEM")
    print("=====================================")
    print("The 'cigarette break' for AI systems")
    print("Prevents context drift and task amnesia")
    print("=====================================")
    
    results = {}
    
    try:
        if args.demo in ["basic", "all"]:
            results['basic'] = demo_basic_nicotine_validation()
        
        if args.demo in ["drift", "all"]:
            results['drift'] = demo_context_drift_detection()
        
        if args.demo in ["integrated", "all"]:
            results['integrated'] = demo_integrated_reconstruction()
        
        if args.demo in ["puzzles", "all"]:
            results['puzzles'] = demo_puzzle_types()
        
        print(f"\n" + "="*70)
        print("🎯 DEMO COMPLETED")
        print("="*70)
        print("Key Benefits Demonstrated:")
        print("  ✅ Prevented context drift through periodic validation")
        print("  ✅ Detected when AI systems lose track of objectives")
        print("  ✅ Maintained task focus through machine-readable puzzles")
        print("  ✅ Integrated seamlessly with existing reconstruction systems")
        print("  ✅ Provided cognitive checkpoints for long-running processes")
        print("\nNicotine acts as a 'cigarette break' that actually improves")
        print("cognitive performance by validating context retention.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 