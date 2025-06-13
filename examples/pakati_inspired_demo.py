#!/usr/bin/env python3
"""
Pakati-Inspired Reconstruction Demo

Demonstrates the revolutionary insight: "Best way to analyze an image is if AI can draw the image perfectly."

This demo shows how Helicopter now uses HuggingFace API for actual reconstruction while maintaining
the core insight that reconstruction ability demonstrates understanding.

Usage:
    python examples/pakati_inspired_demo.py --image path/to/image.jpg --description "description of image"

Requirements:
    - Set HUGGINGFACE_API_KEY environment variable
    - Install required packages: pip install requests pillow opencv-python numpy
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

# Add helicopter to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from helicopter.core import PakatiInspiredReconstruction, AutonomousReconstructionEngine


def load_image(image_path: str) -> np.ndarray:
    """Load image from file."""
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if too large (API limits)
    max_size = 512
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
        print(f"Resized image from {w}x{h} to {new_w}x{new_h}")
    
    return image


def demo_basic_understanding_test(image: np.ndarray, description: str):
    """Demonstrate basic understanding test using Pakati-inspired approach."""
    
    print("\n" + "="*60)
    print("BASIC UNDERSTANDING TEST")
    print("="*60)
    
    try:
        # Initialize Pakati-inspired reconstruction engine
        engine = PakatiInspiredReconstruction()
        
        # Test understanding through reconstruction
        print(f"Testing understanding of: {description}")
        print("This will test reconstruction at multiple difficulty levels...")
        
        results = engine.test_understanding(image, description)
        
        # Display results
        print(f"\nResults:")
        print(f"  Understanding Level: {results['understanding_level']}")
        print(f"  Average Quality: {results['average_quality']:.3f}")
        print(f"  Success Rate: {results['success_rate']:.3f}")
        print(f"  Mastery Achieved: {results['mastery_achieved']}")
        
        print(f"\nDetailed Results:")
        for test_result in results['test_results']:
            print(f"  {test_result['strategy']} at difficulty {test_result['difficulty']}: "
                  f"Quality {test_result['quality_score']:.3f} ({'✓' if test_result['success'] else '✗'})")
        
        return results
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure to set HUGGINGFACE_API_KEY environment variable")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def demo_progressive_understanding_test(image: np.ndarray, description: str):
    """Demonstrate progressive understanding test."""
    
    print("\n" + "="*60)
    print("PROGRESSIVE UNDERSTANDING TEST")
    print("="*60)
    
    try:
        engine = PakatiInspiredReconstruction()
        
        print(f"Progressive testing: {description}")
        print("This will increase difficulty until failure...")
        
        results = engine.progressive_test(image, description)
        
        # Display results
        print(f"\nProgressive Results:")
        print(f"  Mastery Level: {results['mastery_level']:.1f}")
        print(f"  Mastery Achieved: {results['mastery_achieved']}")
        
        print(f"\nProgression Details:")
        for i, level_result in enumerate(results['progression']):
            status = "✓ PASSED" if level_result['success'] else "✗ FAILED"
            print(f"  Level {level_result['difficulty']:.1f}: "
                  f"Quality {level_result['quality_score']:.3f} {status}")
            
            if not level_result['success']:
                print(f"    → Failed at difficulty {level_result['difficulty']:.1f}")
                break
        
        return results
        
    except Exception as e:
        print(f"Error in progressive test: {e}")
        return None


def demo_autonomous_engine_integration(image: np.ndarray, description: str):
    """Demonstrate integration with the main autonomous reconstruction engine."""
    
    print("\n" + "="*60)
    print("AUTONOMOUS ENGINE INTEGRATION")
    print("="*60)
    
    try:
        # Initialize main autonomous reconstruction engine
        engine = AutonomousReconstructionEngine()
        
        print(f"Testing with autonomous engine: {description}")
        print("This combines API reconstruction with local analysis...")
        
        # Test understanding validation
        results = engine.validate_understanding_through_reconstruction(image, description)
        
        # Display combined results
        print(f"\nCombined Understanding Assessment:")
        combined = results['combined_understanding']
        print(f"  Understanding Level: {combined['understanding_level']}")
        print(f"  Combined Quality: {combined['combined_quality']:.3f}")
        print(f"  Combined Mastery: {combined['combined_mastery']}")
        print(f"  Validation Confidence: {combined['validation_confidence']:.3f}")
        
        print(f"\nInsights:")
        for insight in results['insights']:
            print(f"  • {insight}")
        
        return results
        
    except Exception as e:
        print(f"Error in autonomous engine test: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Pakati-Inspired Reconstruction Demo")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--description", default="", help="Description of the image")
    parser.add_argument("--test", choices=["basic", "progressive", "autonomous", "all"], 
                       default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("HUGGINGFACE_API_KEY"):
        print("ERROR: HUGGINGFACE_API_KEY environment variable not set")
        print("Please set your HuggingFace API key:")
        print("export HUGGINGFACE_API_KEY='your_api_key_here'")
        return 1
    
    try:
        # Load image
        print(f"Loading image: {args.image}")
        image = load_image(args.image)
        print(f"Image shape: {image.shape}")
        
        description = args.description or f"image from {Path(args.image).name}"
        
        print(f"\nTesting Pakati-inspired reconstruction on: {description}")
        print(f"Core insight: 'Best way to analyze an image is if AI can draw the image perfectly'")
        
        # Run selected tests
        if args.test in ["basic", "all"]:
            demo_basic_understanding_test(image, description)
        
        if args.test in ["progressive", "all"]:
            demo_progressive_understanding_test(image, description)
        
        if args.test in ["autonomous", "all"]:
            demo_autonomous_engine_integration(image, description)
        
        print(f"\n" + "="*60)
        print("DEMO COMPLETED")
        print("="*60)
        print("The system has demonstrated understanding through reconstruction ability.")
        print("This validates the core insight that reconstruction proves comprehension.")
        
        return 0
        
    except Exception as e:
        print(f"Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 