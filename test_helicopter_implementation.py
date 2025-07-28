#!/usr/bin/env python3
"""
Test script for Helicopter Framework Implementation

This script tests the core components implemented from the paper to ensure
they're working correctly.
"""

import numpy as np
import torch
import cv2
import sys
from pathlib import Path

# Add helicopter to path
sys.path.append(str(Path(__file__).parent))

try:
    from helicopter.core import (
        ThermodynamicPixelEngine,
        HierarchicalBayesianProcessor,
        ReconstructionValidationMetrics,
        HelicopterProcessingEngine,
        create_helicopter_engine
    )
    print("✅ Successfully imported all Helicopter components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def create_test_image():
    """Create a simple test image"""
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    
    # Add some patterns
    cv2.circle(image, (32, 32), 20, (255, 0, 0), -1)
    cv2.rectangle(image, (60, 60), (100, 100), (0, 255, 0), -1)
    
    # Add noise
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


def test_thermodynamic_processing():
    """Test thermodynamic pixel processing"""
    print("\n🔥 Testing Thermodynamic Pixel Processing...")
    
    try:
        engine = ThermodynamicPixelEngine(
            base_temperature=1.0,
            max_temperature=5.0,
            max_iterations=10  # Reduced for testing
        )
        
        test_image = create_test_image()
        processed_image, metrics = engine.process_image_thermodynamically(
            test_image, return_metrics=True
        )
        
        print(f"   ✅ Processed image shape: {processed_image.shape}")
        print(f"   ✅ Average temperature: {metrics.average_temperature:.2f}")
        print(f"   ✅ Processing efficiency: {metrics.processing_efficiency:.2%}")
        print(f"   ✅ Equilibrium percentage: {metrics.equilibrium_percentage:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Thermodynamic processing failed: {e}")
        return False


def test_bayesian_processing():
    """Test hierarchical Bayesian processing"""
    print("\n🧠 Testing Hierarchical Bayesian Processing...")
    
    try:
        processor = HierarchicalBayesianProcessor(
            input_dim=256,
            molecular_dim=32,  # Reduced for testing
            neural_dim=64,
            cognitive_dim=128
        )
        
        # Create test features
        test_features = torch.randn(4, 256)  # Batch of 4 samples
        
        result = processor.process_hierarchically(test_features)
        
        print(f"   ✅ Molecular uncertainty: {result.molecular_state.uncertainty:.4f}")
        print(f"   ✅ Neural uncertainty: {result.neural_state.uncertainty:.4f}")
        print(f"   ✅ Cognitive uncertainty: {result.cognitive_state.uncertainty:.4f}")
        print(f"   ✅ Total uncertainty: {result.total_uncertainty:.4f}")
        print(f"   ✅ Calibration score: {result.calibration_score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Bayesian processing failed: {e}")
        return False


def test_reconstruction_validation():
    """Test reconstruction validation metrics"""
    print("\n📊 Testing Reconstruction Validation Metrics...")
    
    try:
        validator = ReconstructionValidationMetrics()
        
        original_image = create_test_image()
        
        # Create a slightly modified "reconstruction" 
        reconstructed_image = cv2.GaussianBlur(original_image, (3, 3), 1.0)
        
        metrics = validator.compute_all_metrics(
            original_image=original_image,
            reconstructed_image=reconstructed_image
        )
        
        print(f"   ✅ RFS (Reconstruction Fidelity Score): {metrics.rfs:.3f}")
        print(f"   ✅ SCI (Semantic Consistency Index): {metrics.sci:.3f}")
        print(f"   ✅ PIRA (Partial Info Reconstruction Accuracy): {metrics.pira:.3f}")
        print(f"   ✅ Understanding confidence: {metrics.understanding_confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Validation metrics failed: {e}")
        return False


def test_integrated_processing():
    """Test the integrated processing engine"""
    print("\n🚁 Testing Integrated Helicopter Processing Engine...")
    
    try:
        # Create engine with reduced settings for testing
        config_overrides = {
            'max_thermodynamic_iterations': 5,
            'molecular_dim': 32,
            'neural_dim': 64,
            'cognitive_dim': 128
        }
        
        engine = create_helicopter_engine(config_overrides)
        
        test_image = create_test_image()
        
        # Process through complete pipeline
        results = engine.process_image(test_image)
        
        print(f"   ✅ Processing completed in {results.processing_time:.2f}s")
        print(f"   ✅ Understanding confidence: {results.understanding_confidence:.3f}")
        print(f"   ✅ Computational speedup: {results.computational_speedup:.1f}×")
        print(f"   ✅ Resource efficiency: {results.resource_efficiency:.1%}")
        print(f"   ✅ Reconstruction quality: {results.reconstruction_quality:.3f}")
        
        # Check that all components produced results
        assert results.thermodynamic_metrics is not None
        assert results.hierarchical_result is not None
        assert results.validation_metrics is not None
        assert results.reconstructed_image is not None
        
        print("   ✅ All pipeline components executed successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integrated processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🧪 HELICOPTER FRAMEWORK IMPLEMENTATION TEST")
    print("=" * 50)
    print("Testing core components from the paper:")
    print("- Thermodynamic Pixel Processing")
    print("- Hierarchical Bayesian Processing")
    print("- Reconstruction Validation Metrics")
    print("- Integrated Processing Engine")
    print("=" * 50)
    
    tests = [
        test_thermodynamic_processing,
        test_bayesian_processing,
        test_reconstruction_validation,
        test_integrated_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📋 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The Helicopter framework is working correctly.")
        print("\nYou can now:")
        print("1. Run the demo: python examples/complete_helicopter_demo.py --demo")
        print("2. Process your own images: python examples/complete_helicopter_demo.py --image path/to/image.jpg")
        print("3. Import components: from helicopter.core import HelicopterProcessingEngine")
    else:
        print("❌ Some tests failed. Check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 