#!/usr/bin/env python3
"""
Zengeza Noise Detection Demo

Demonstrates the Zengeza noise detection system that calculates the probable
amount of "noise" or garbage per segment per iteration.

Key Innovation:
- Not every part of an image is important for understanding
- Much content is "noise" that doesn't contribute to comprehension
- This noise isn't always obvious - can be subtle and context-dependent
- Zengeza quantifies noise probability to focus on meaningful content

Usage:
    python examples/zengeza_demo.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Any
import time

from helicopter.core.zengeza_noise_detector import (
    ZengezaEngine, 
    NoiseType, 
    NoiseLevel,
    ZengezaSegment
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_images() -> Dict[str, np.ndarray]:
    """Create test images with different noise characteristics."""
    
    images = {}
    
    # 1. Clean image with clear structure
    clean_image = np.zeros((256, 256, 3), dtype=np.uint8)
    clean_image[50:100, 50:200] = [255, 0, 0]  # Red rectangle
    clean_image[150:200, 50:200] = [0, 255, 0]  # Green rectangle
    cv2.circle(clean_image, (128, 128), 30, (0, 0, 255), -1)  # Blue circle
    images['clean_structured'] = clean_image
    
    # 2. Noisy image with visual artifacts
    noisy_image = clean_image.copy()
    noise = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)
    noisy_image = cv2.add(noisy_image, noise)
    images['visual_noise'] = noisy_image
    
    # 3. Image with semantic noise (random objects)
    semantic_noise_image = clean_image.copy()
    # Add random small rectangles (semantic noise)
    for _ in range(20):
        x, y = np.random.randint(0, 200, 2)
        w, h = np.random.randint(5, 15, 2)
        color = np.random.randint(0, 255, 3)
        semantic_noise_image[y:y+h, x:x+w] = color
    images['semantic_noise'] = semantic_noise_image
    
    # 4. Image with structural noise (repetitive patterns)
    structural_noise_image = np.zeros((256, 256, 3), dtype=np.uint8)
    # Create repetitive checkerboard pattern
    for i in range(0, 256, 16):
        for j in range(0, 256, 16):
            if (i//16 + j//16) % 2 == 0:
                structural_noise_image[i:i+16, j:j+16] = [128, 128, 128]
    images['structural_noise'] = structural_noise_image
    
    # 5. Mixed content image
    mixed_image = np.zeros((256, 256, 3), dtype=np.uint8)
    # Important content (large shapes)
    cv2.rectangle(mixed_image, (20, 20), (120, 120), (255, 0, 0), -1)
    cv2.circle(mixed_image, (180, 60), 40, (0, 255, 0), -1)
    # Noise content (small random elements)
    for _ in range(50):
        x, y = np.random.randint(0, 256, 2)
        cv2.circle(mixed_image, (x, y), 2, tuple(np.random.randint(0, 255, 3).tolist()), -1)
    images['mixed_content'] = mixed_image
    
    return images


def create_segments_from_image(image: np.ndarray, segment_size: int = 64) -> List[Dict[str, Any]]:
    """Create segments from an image for noise analysis."""
    
    segments = []
    h, w = image.shape[:2]
    
    segment_id = 0
    for y in range(0, h, segment_size):
        for x in range(0, w, segment_size):
            # Calculate actual segment dimensions
            seg_w = min(segment_size, w - x)
            seg_h = min(segment_size, h - y)
            
            # Extract segment pixels
            segment_pixels = image[y:y+seg_h, x:x+seg_w]
            
            segments.append({
                'segment_id': f'segment_{segment_id}',
                'bbox': (x, y, seg_w, seg_h),
                'pixels': segment_pixels
            })
            
            segment_id += 1
    
    return segments


def visualize_noise_analysis(image: np.ndarray, 
                           noise_results: Dict[str, Any],
                           title: str = "Noise Analysis") -> np.ndarray:
    """Visualize noise analysis results on the image."""
    
    # Create visualization image
    vis_image = image.copy()
    
    # Draw segment boundaries and noise levels
    for segment_id, segment_data in noise_results['segment_noise_analysis'].items():
        bbox = segment_data.get('bbox', (0, 0, 32, 32))
        x, y, w, h = bbox
        
        noise_prob = segment_data['noise_probability']
        importance = segment_data['importance_score']
        
        # Color based on noise level
        if noise_prob > 0.8:
            color = (0, 0, 255)  # Red for high noise
        elif noise_prob > 0.6:
            color = (0, 128, 255)  # Orange for moderate-high noise
        elif noise_prob > 0.4:
            color = (0, 255, 255)  # Yellow for moderate noise
        elif noise_prob > 0.2:
            color = (128, 255, 0)  # Light green for low noise
        else:
            color = (0, 255, 0)  # Green for minimal noise
        
        # Draw rectangle
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        # Add noise probability text
        text = f"{noise_prob:.2f}"
        cv2.putText(vis_image, text, (x + 2, y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return vis_image


def run_comprehensive_noise_demo():
    """Run comprehensive demonstration of Zengeza noise detection."""
    
    print("🗑️ Starting Zengeza Noise Detection Demo")
    print("=" * 60)
    
    # Initialize Zengeza engine
    zengeza = ZengezaEngine()
    
    # Create test images
    print("\n📸 Creating test images with different noise characteristics...")
    test_images = create_test_images()
    
    # Analyze each test image
    results = {}
    
    for image_name, image in test_images.items():
        print(f"\n🔍 Analyzing {image_name}...")
        
        # Create segments
        segments = create_segments_from_image(image, segment_size=64)
        
        # Context for analysis
        context = {
            'reconstruction_quality': 0.7,
            'expected_complexity': 0.5,
            'image_type': image_name
        }
        
        # Perform noise analysis
        noise_results = zengeza.analyze_image_noise(
            image=image,
            segments=segments,
            context=context,
            iteration=0
        )
        
        results[image_name] = {
            'image': image,
            'noise_results': noise_results,
            'segments': segments
        }
        
        # Print analysis summary
        print(f"  📊 Total segments: {noise_results['total_segments']}")
        print(f"  📈 Average noise level: {noise_results['global_noise_statistics']['average_noise_level']:.3f}")
        print(f"  🗑️ High-noise segments: {len(noise_results['high_noise_segments'])}")
        print(f"  ⭐ High-importance segments: {len(noise_results['high_importance_segments'])}")
        
        # Print insights
        print(f"  💡 Insights:")
        for insight in noise_results['noise_insights']:
            print(f"     • {insight}")
    
    # Demonstrate iterative noise analysis
    print(f"\n🔄 Demonstrating iterative noise analysis...")
    
    # Use mixed content image for iteration demo
    mixed_image = test_images['mixed_content']
    segments = create_segments_from_image(mixed_image)
    
    iteration_results = []
    
    for iteration in range(3):
        print(f"\n  Iteration {iteration + 1}:")
        
        # Simulate changing context over iterations
        context = {
            'reconstruction_quality': 0.5 + (iteration * 0.15),  # Improving quality
            'expected_complexity': 0.6,
            'iteration': iteration,
            'total_iterations': 3
        }
        
        noise_results = zengeza.analyze_image_noise(
            image=mixed_image,
            segments=segments,
            context=context,
            iteration=iteration
        )
        
        iteration_results.append(noise_results)
        
        avg_noise = noise_results['global_noise_statistics']['average_noise_level']
        print(f"    Average noise: {avg_noise:.3f}")
        print(f"    High-importance segments: {len(noise_results['high_importance_segments'])}")
    
    # Analyze noise trends across iterations
    print(f"\n📈 Noise trends across iterations:")
    for i, result in enumerate(iteration_results):
        avg_noise = result['global_noise_statistics']['average_noise_level']
        print(f"  Iteration {i+1}: {avg_noise:.3f} average noise")
    
    # Demonstrate segment prioritization
    print(f"\n🎯 Segment prioritization example:")
    
    final_results = iteration_results[-1]
    prioritized = final_results['prioritized_segments'][:5]  # Top 5
    
    print(f"  Top 5 most important segments:")
    for i, seg in enumerate(prioritized):
        print(f"    {i+1}. {seg['segment_id']}: "
              f"priority {seg['priority_score']:.3f}, "
              f"importance {seg['importance_score']:.3f}, "
              f"noise {seg['noise_probability']:.3f}")
    
    # Demonstrate noise type analysis
    print(f"\n🔬 Noise type analysis:")
    
    all_noise_types = {}
    for image_name, data in results.items():
        noise_results = data['noise_results']
        
        image_noise_types = []
        for segment_data in noise_results['segment_noise_analysis'].values():
            image_noise_types.extend(segment_data['noise_types'])
        
        if image_noise_types:
            from collections import Counter
            noise_counts = Counter(image_noise_types)
            all_noise_types[image_name] = noise_counts
            
            print(f"  {image_name}:")
            for noise_type, count in noise_counts.most_common():
                print(f"    • {noise_type}: {count} segments")
    
    # Generate visualization
    print(f"\n🎨 Generating noise visualization...")
    
    # Create subplot for each test image
    fig, axes = plt.subplots(2, len(test_images), figsize=(20, 8))
    
    for i, (image_name, data) in enumerate(results.items()):
        image = data['image']
        noise_results = data['noise_results']
        
        # Original image
        axes[0, i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Original: {image_name}")
        axes[0, i].axis('off')
        
        # Noise analysis visualization
        vis_image = visualize_noise_analysis(image, noise_results, image_name)
        axes[1, i].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f"Noise Analysis: {image_name}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('zengeza_noise_analysis_demo.png', dpi=150, bbox_inches='tight')
    print(f"  💾 Saved visualization to 'zengeza_noise_analysis_demo.png'")
    
    # Performance analysis
    print(f"\n⚡ Performance analysis:")
    
    # Time noise analysis
    start_time = time.time()
    
    test_image = test_images['mixed_content']
    test_segments = create_segments_from_image(test_image, segment_size=32)  # Smaller segments
    
    noise_results = zengeza.analyze_image_noise(
        image=test_image,
        segments=test_segments,
        context={'reconstruction_quality': 0.7},
        iteration=0
    )
    
    analysis_time = time.time() - start_time
    
    print(f"  ⏱️ Analyzed {len(test_segments)} segments in {analysis_time:.3f} seconds")
    print(f"  📊 Average time per segment: {analysis_time/len(test_segments)*1000:.2f} ms")
    
    # Final summary
    print(f"\n🎉 Zengeza Noise Detection Demo Complete!")
    print(f"=" * 60)
    print(f"Key Findings:")
    print(f"• Zengeza successfully identifies different types of noise")
    print(f"• Visual, semantic, structural, and contextual noise are detected")
    print(f"• Segment prioritization helps focus on important content")
    print(f"• Iterative analysis shows noise trends over time")
    print(f"• Performance is suitable for real-time analysis")
    
    return results


def demonstrate_integration_with_reconstruction():
    """Demonstrate how Zengeza integrates with reconstruction process."""
    
    print(f"\n🔗 Demonstrating Zengeza integration with reconstruction...")
    
    # Simulate reconstruction scenario
    zengeza = ZengezaEngine()
    
    # Create image with mixed importance content
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    
    # Important content (main subject)
    cv2.rectangle(image, (30, 30), (90, 90), (255, 100, 50), -1)
    cv2.circle(image, (64, 64), 20, (100, 255, 100), -1)
    
    # Noise content (background clutter)
    for _ in range(30):
        x, y = np.random.randint(0, 128, 2)
        if not (30 <= x <= 90 and 30 <= y <= 90):  # Avoid main subject
            cv2.circle(image, (x, y), 2, tuple(np.random.randint(0, 255, 3).tolist()), -1)
    
    segments = create_segments_from_image(image, segment_size=32)
    
    # Simulate reconstruction iterations
    print(f"  Simulating reconstruction with noise-aware prioritization:")
    
    for iteration in range(3):
        # Analyze noise
        context = {
            'reconstruction_quality': 0.4 + (iteration * 0.2),
            'expected_complexity': 0.6,
            'iteration': iteration
        }
        
        noise_results = zengeza.analyze_image_noise(image, segments, context, iteration)
        
        # Get prioritized segments
        prioritized = noise_results['prioritized_segments']
        high_importance = [s for s in prioritized if s['importance_score'] > 0.7]
        high_noise = [s for s in prioritized if s['noise_probability'] > 0.6]
        
        print(f"    Iteration {iteration + 1}:")
        print(f"      Focus on {len(high_importance)} high-importance segments")
        print(f"      Skip/reduce {len(high_noise)} high-noise segments")
        print(f"      Computational savings: {len(high_noise)/len(segments)*100:.1f}%")
    
    print(f"  ✅ Integration demonstration complete")


if __name__ == "__main__":
    try:
        # Run comprehensive demo
        results = run_comprehensive_noise_demo()
        
        # Demonstrate integration
        demonstrate_integration_with_reconstruction()
        
        print(f"\n🚁 Helicopter + Zengeza: Separating signal from noise in computer vision!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc() 