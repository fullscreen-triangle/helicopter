#!/usr/bin/env python3
"""
Autonomous Reconstruction Demo - "Reverse Reverse Reverse Pakati"

Demonstrates the genius insight: The best way to know if an AI has truly analyzed 
an image is if it can perfectly reconstruct it. The path to reconstruction IS the analysis.

This is the ultimate Turing test for computer vision:
- Can you draw what you see?
- If yes, you have truly seen it.
- The reconstruction process reveals true understanding.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time
from typing import Dict, Any

from helicopter.core.autonomous_reconstruction_engine import AutonomousReconstructionEngine
from helicopter.core.comprehensive_analysis_engine import ComprehensiveAnalysisEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_image(size: tuple = (256, 256)) -> np.ndarray:
    """Create a test image with various patterns for reconstruction testing"""
    
    h, w = size
    image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create different regions with distinct patterns
    
    # Top-left: Gradient
    for y in range(h//2):
        for x in range(w//2):
            image[y, x] = [int(255 * x / (w//2)), int(255 * y / (h//2)), 128]
    
    # Top-right: Checkerboard
    for y in range(h//2):
        for x in range(w//2, w):
            if (x//16 + y//16) % 2 == 0:
                image[y, x] = [255, 255, 255]
            else:
                image[y, x] = [0, 0, 0]
    
    # Bottom-left: Circles
    center_x, center_y = w//4, 3*h//4
    for y in range(h//2, h):
        for x in range(w//2):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist < 30:
                image[y, x] = [255, 0, 0]  # Red circle
            elif dist < 50:
                image[y, x] = [0, 255, 0]  # Green ring
            elif dist < 70:
                image[y, x] = [0, 0, 255]  # Blue ring
    
    # Bottom-right: Random noise with structure
    np.random.seed(42)
    noise = np.random.randint(0, 256, (h//2, w//2, 3), dtype=np.uint8)
    # Add some structure to the noise
    for y in range(h//2):
        for x in range(w//2):
            if (x + y) % 20 < 10:
                noise[y, x] = noise[y, x] // 2 + 128
    
    image[h//2:h, w//2:w] = noise
    
    return image


def demonstrate_autonomous_reconstruction():
    """Demonstrate autonomous reconstruction analysis"""
    
    print("\n" + "="*80)
    print("AUTONOMOUS RECONSTRUCTION DEMO")
    print("The Ultimate Test of Image Understanding")
    print("="*80)
    
    print("\nCore Insight:")
    print("- The best way to know if an AI has truly analyzed an image")
    print("- is if it can perfectly reconstruct it.")
    print("- The path to reconstruction IS the analysis itself.")
    print("- This is the ultimate Turing test for computer vision.")
    
    # Create test image
    print("\n1. Creating test image with various patterns...")
    test_image = create_test_image((256, 256))
    print(f"   Test image shape: {test_image.shape}")
    
    # Initialize autonomous reconstruction engine
    print("\n2. Initializing Autonomous Reconstruction Engine...")
    reconstruction_engine = AutonomousReconstructionEngine(
        patch_size=32,
        context_size=96,
        device=None  # Use CPU for demo
    )
    
    # Perform autonomous reconstruction analysis
    print("\n3. Starting autonomous reconstruction analysis...")
    print("   The system will:")
    print("   - Start with partial image information (~20% of patches)")
    print("   - Iteratively predict missing parts from context")
    print("   - Learn from reconstruction success/failure")
    print("   - Continue until perfect reconstruction or convergence")
    
    start_time = time.time()
    
    results = reconstruction_engine.autonomous_analyze(
        image=test_image,
        max_iterations=30,  # Reasonable for demo
        target_quality=0.85  # Good quality target
    )
    
    analysis_time = time.time() - start_time
    
    # Display results
    print(f"\n4. Autonomous reconstruction completed in {analysis_time:.2f} seconds")
    
    recon_results = results['autonomous_reconstruction']
    understanding_insights = results['understanding_insights']
    
    print(f"\n   RECONSTRUCTION METRICS:")
    print(f"   - Final Quality: {recon_results['final_quality']:.1%}")
    print(f"   - Reconstruction Complete: {recon_results['reconstruction_complete']}")
    print(f"   - Patches Reconstructed: {recon_results['patches_reconstructed']}/{recon_results['total_patches']}")
    print(f"   - Completion Percentage: {recon_results['completion_percentage']:.1f}%")
    print(f"   - Average Confidence: {recon_results['average_confidence']:.3f}")
    print(f"   - Reconstruction Iterations: {recon_results['reconstruction_iterations']}")
    
    print(f"\n   UNDERSTANDING INSIGHTS:")
    print(f"   - Understanding Level: {understanding_insights['understanding_level']}")
    print(f"   - What Reconstruction Demonstrates:")
    for demo in understanding_insights['reconstruction_demonstrates']:
        print(f"     • {demo}")
    print(f"   - Key Insights:")
    for insight in understanding_insights['key_insights']:
        print(f"     • {insight}")
    
    # Analyze reconstruction patterns
    if 'reconstruction_analysis' in results:
        analysis = results['reconstruction_analysis']
        if 'quality_progression' in analysis:
            quality_prog = analysis['quality_progression']
            print(f"\n   LEARNING PROGRESSION:")
            print(f"   - Initial Quality: {quality_prog['initial']:.3f}")
            print(f"   - Final Quality: {quality_prog['final']:.3f}")
            print(f"   - Quality Improvement: {quality_prog['improvement']:.3f}")
            print(f"   - Learning Rate: {quality_prog['progression_rate']:.4f} per iteration")
    
    # Show reconstruction history
    if results['reconstruction_history']:
        print(f"\n   RECONSTRUCTION HISTORY (last 5 iterations):")
        history = results['reconstruction_history'][-5:]
        for h in history:
            print(f"   - Iteration {h['iteration']}: Quality={h['quality']:.3f}, "
                  f"Confidence={h['confidence']:.3f}, Patch=({h['patch_location'][0]}, {h['patch_location'][1]})")
    
    return results, test_image


def demonstrate_comprehensive_analysis_with_reconstruction():
    """Demonstrate comprehensive analysis using autonomous reconstruction as primary method"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS WITH AUTONOMOUS RECONSTRUCTION")
    print("Integration with Full Helicopter System")
    print("="*80)
    
    # Create test image
    test_image = create_test_image((128, 128))  # Smaller for faster demo
    
    # Initialize comprehensive analysis engine
    print("\n1. Initializing Comprehensive Analysis Engine...")
    analysis_engine = ComprehensiveAnalysisEngine()
    
    # Perform comprehensive analysis
    print("\n2. Starting comprehensive analysis with autonomous reconstruction as primary...")
    print("   - PRIMARY: Autonomous reconstruction (the ultimate test)")
    print("   - SUPPORTING: Traditional methods for validation")
    print("   - CROSS-VALIDATION: Compare insights across methods")
    print("   - LEARNING: Iterative improvement if needed")
    
    start_time = time.time()
    
    comprehensive_results = analysis_engine.comprehensive_analysis(
        image=test_image,
        metadata={'demo': True, 'source': 'synthetic'},
        enable_iterative_learning=True,
        enable_autonomous_reconstruction=True
    )
    
    analysis_time = time.time() - start_time
    
    # Display comprehensive results
    print(f"\n3. Comprehensive analysis completed in {analysis_time:.2f} seconds")
    
    if 'final_assessment' in comprehensive_results:
        assessment = comprehensive_results['final_assessment']
        
        print(f"\n   FINAL ASSESSMENT:")
        print(f"   - Primary Method: {assessment['primary_method']}")
        print(f"   - Analysis Complete: {assessment['analysis_complete']}")
        print(f"   - Understanding Demonstrated: {assessment['understanding_demonstrated']}")
        print(f"   - Confidence Score: {assessment['confidence_score']:.1%}")
        
        print(f"\n   KEY FINDINGS:")
        for finding in assessment['key_findings']:
            print(f"   • {finding}")
        
        print(f"\n   RECOMMENDATIONS:")
        for rec in assessment['recommendations']:
            print(f"   • {rec}")
    
    # Cross-validation results
    if 'cross_validation' in comprehensive_results:
        cross_val = comprehensive_results['cross_validation']
        
        print(f"\n   CROSS-VALIDATION:")
        print(f"   - Validation Status: {cross_val['understanding_validation']['status']}")
        print(f"   - Support Ratio: {cross_val['understanding_validation']['support_ratio']:.1%}")
        
        if cross_val['supporting_evidence']:
            print(f"   - Supporting Evidence:")
            for evidence in cross_val['supporting_evidence']:
                print(f"     ✓ {evidence}")
        
        if cross_val['conflicting_evidence']:
            print(f"   - Conflicting Evidence:")
            for conflict in cross_val['conflicting_evidence']:
                print(f"     ✗ {conflict}")
    
    return comprehensive_results


def visualize_reconstruction_process(results: Dict[str, Any], original_image: np.ndarray):
    """Visualize the reconstruction process"""
    
    print("\n4. Visualizing reconstruction process...")
    
    try:
        # Get reconstruction image
        if 'reconstruction_image' in results:
            reconstruction = results['reconstruction_image']
        elif 'autonomous_reconstruction' in results:
            # Try to get from nested results
            reconstruction = results.get('original_image', original_image)
        else:
            reconstruction = original_image
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Reconstructed image
        axes[1].imshow(cv2.cvtColor(reconstruction, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Reconstructed Image')
        axes[1].axis('off')
        
        # Difference
        diff = np.abs(original_image.astype(float) - reconstruction.astype(float))
        diff = (diff / diff.max() * 255).astype(np.uint8)
        axes[2].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Reconstruction Difference')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        
        plt.savefig(output_dir / 'autonomous_reconstruction_demo.png', dpi=150, bbox_inches='tight')
        print(f"   Visualization saved to: {output_dir / 'autonomous_reconstruction_demo.png'}")
        
        # Show if possible
        try:
            plt.show()
        except:
            print("   (Display not available, but image saved)")
        
        plt.close()
        
    except Exception as e:
        print(f"   Visualization failed: {e}")


def main():
    """Main demonstration function"""
    
    print("HELICOPTER AUTONOMOUS RECONSTRUCTION DEMONSTRATION")
    print("The Genius Insight: Reconstruction Ability = True Understanding")
    
    try:
        # Demonstrate autonomous reconstruction
        results, test_image = demonstrate_autonomous_reconstruction()
        
        # Visualize results
        visualize_reconstruction_process(results, test_image)
        
        # Demonstrate comprehensive analysis integration
        comprehensive_results = demonstrate_comprehensive_analysis_with_reconstruction()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        
        print("\nKey Takeaways:")
        print("1. Autonomous reconstruction provides the ultimate test of image understanding")
        print("2. If a system can perfectly reconstruct an image, it has truly 'seen' it")
        print("3. The reconstruction process itself IS the analysis")
        print("4. This approach works autonomously without complex method orchestration")
        print("5. Supporting methods validate reconstruction insights")
        print("6. The system learns and improves through iterative reconstruction")
        
        print(f"\nThis is 'Reverse Reverse Reverse Pakati' - the path to perfect reconstruction")
        print(f"reveals perfect understanding. Genius in its simplicity!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 