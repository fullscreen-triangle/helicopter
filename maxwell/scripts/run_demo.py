#!/usr/bin/env python
"""
Run HCCC algorithm demo.

Usage:
    python -m scripts.run_demo [options]

Options:
    --image PATH        Path to image file (default: synthetic)
    --segments N        Number of segments (default: 50)
    --lambda FLOAT      Lambda stream parameter (default: 0.5)
    --threshold FLOAT   Coherence threshold (default: 1.0)
    --output DIR        Output directory for results (default: results/)
    --visualize         Generate visualizations
    --validate          Run validation suite
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vision.bmd import HardwareBMDStream
from categorical import AmbiguityCalculator, CategoricalCompletion
from algorithm import HCCCAlgorithm
from validation import (
    ValidationMetrics,
    ResultVisualizer,
    BenchmarkSuite,
    BiologicalValidator,
    PhysicalValidator
)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Run HCCC algorithm demo')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--segments', type=int, default=50, help='Number of segments')
    parser.add_argument('--lambda', dest='lambda_stream', type=float, default=0.5,
                       help='Lambda stream parameter')
    parser.add_argument('--threshold', type=float, default=1.0,
                       help='Coherence threshold')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation suite')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("HCCC Algorithm Demo")
    print("=" * 80)

    # 1. Load or generate image
    if args.image:
        print(f"\nLoading image: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            return 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        print("\nGenerating synthetic image...")
        benchmark = BenchmarkSuite()
        image = benchmark.generate_synthetic_image('geometric', size=(256, 256))

    print(f"Image shape: {image.shape}")

    # 2. Initialize hardware stream
    print("\nInitializing hardware BMD stream...")
    hardware_stream = HardwareBMDStream()
    initial_stream = hardware_stream.measure_stream()
    print(f"Stream richness: {initial_stream.categorical_richness():.2e}")

    # 3. Create algorithm
    print("\nInitializing HCCC algorithm...")
    hccc = HCCCAlgorithm(
        hardware_stream=hardware_stream,
        lambda_stream=args.lambda_stream,
        coherence_threshold=args.threshold,
        max_iterations=100
    )

    # 4. Process image
    print("\nProcessing image...")
    results = hccc.process_image(
        image,
        segmentation_method='slic',
        segmentation_params={'n_segments': args.segments}
    )

    print(f"\nProcessing complete!")
    print(f"  Iterations: {results['convergence_step']}")
    print(f"  Regions processed: {results['regions_processed']}/{results['regions_total']}")
    print(f"  Final ambiguity: {results['metrics']['final_ambiguity']:.3f}")
    print(f"  Network richness: {results['interpretation']['network_richness']:.2e}")

    # 5. Visualize if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        visualizer = ResultVisualizer()

        from regions import ImageSegmenter
        segmenter = ImageSegmenter()
        regions = segmenter.segment(image, 'slic', n_segments=args.segments)

        visualizer.visualize_complete_results(
            image,
            regions,
            results,
            save_dir=str(output_dir)
        )
        print(f"  Saved to: {output_dir}/")

    # 6. Validate if requested
    if args.validate:
        print("\nRunning validation...")

        bio_validator = BiologicalValidator()
        bio_results = bio_validator.comprehensive_biological_validation(
            network_bmd=results['network_bmd_final'],
            hardware_stream=hardware_stream.get_stream_state(),
            divergence_history=results['stream_divergence_history'],
            richness_history=results['network_richness_history']
        )

        print(f"  Biological: {'✓ PASS' if bio_results['overall']['validated'] else '✗ FAIL'}")

        phys_validator = PhysicalValidator()
        phys_results = phys_validator.comprehensive_physical_validation(
            initial_bmd=initial_stream,
            final_network_bmd=results['network_bmd_final'],
            hardware_stream=hardware_stream
        )

        print(f"  Physical: {'✓ PASS' if phys_results['overall']['validated'] else '✗ FAIL'}")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
