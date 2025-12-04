#!/usr/bin/env python
"""
Run HCCC algorithm benchmarks.

Usage:
    python -m scripts.run_benchmark [options]
"""

import argparse
import sys
from pathlib import Path
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vision.bmd import HardwareBMDStream
from algorithm import HCCCAlgorithm
from validation import BenchmarkSuite


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description='Run HCCC benchmarks')
    parser.add_argument('--n-images', type=int, default=5,
                       help='Number of test images')
    parser.add_argument('--image-type', type=str, default='geometric',
                       choices=['geometric', 'gradient', 'texture', 'random'],
                       help='Type of synthetic images')
    parser.add_argument('--segments', type=int, default=50,
                       help='Number of segments')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')

    args = parser.parse_args()

    print("=" * 80)
    print("HCCC Algorithm Benchmark")
    print("=" * 80)

    # Generate test images
    print(f"\nGenerating {args.n_images} test images ({args.image_type})...")
    benchmark = BenchmarkSuite()

    test_images = []
    for i in range(args.n_images):
        image = benchmark.generate_synthetic_image(
            image_type=args.image_type,
            size=(256, 256)
        )
        test_images.append(image)

    # Initialize algorithm
    print("\nInitializing algorithm...")
    hardware_stream = HardwareBMDStream()
    hccc = HCCCAlgorithm(
        hardware_stream=hardware_stream,
        lambda_stream=0.5,
        coherence_threshold=1.0
    )

    # Run benchmark
    print(f"\nRunning benchmark...")
    start_time = time.time()

    results = benchmark.benchmark_algorithm(
        hccc,
        test_images,
        segmentation_method='slic',
        segmentation_params={'n_segments': args.segments}
    )

    total_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 80)
    print("Benchmark Results")
    print("=" * 80)

    if 'aggregate' in results:
        agg = results['aggregate']
        print(f"\nAggregate Metrics:")
        print(f"  Success rate: {agg['success_rate']:.2%}")
        print(f"  Mean time: {agg['mean_time']:.2f}s per image")
        print(f"  Mean iterations: {agg['mean_iterations']:.1f}")
        print(f"  Convergence rate: {agg['convergence_rate']:.2%}")
        print(f"  Total time: {total_time:.2f}s")

    print(f"\nPer-Image Results:")
    for i, img_result in enumerate(results['per_image']):
        if img_result['success']:
            print(f"  Image {i+1}: ✓ {img_result['time']:.2f}s, "
                  f"{img_result['iterations']} iterations")
        else:
            print(f"  Image {i+1}: ✗ {img_result.get('error', 'Unknown error')}")

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
