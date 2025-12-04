#!/usr/bin/env python
"""
Process a single image with HCCC algorithm.

Usage:
    python -m scripts.process_image input.jpg [options]
"""

import argparse
import sys
from pathlib import Path
import cv2
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vision.bmd import HardwareBMDStream
from algorithm import HCCCAlgorithm
from validation import ResultVisualizer
from regions import ImageSegmenter


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description='Process image with HCCC')
    parser.add_argument('input', type=str, help='Input image path')
    parser.add_argument('--output', type=str, help='Output directory (default: same as input)')
    parser.add_argument('--segments', type=int, default=50, help='Number of segments')
    parser.add_argument('--method', type=str, default='slic',
                       choices=['slic', 'felzenszwalb', 'watershed', 'hierarchical'],
                       help='Segmentation method')
    parser.add_argument('--lambda', dest='lambda_stream', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=1.0)
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--save-results', action='store_true', help='Save results JSON')

    args = parser.parse_args()

    # Load image
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Image not found: {input_path}")
        return 1

    print(f"Loading image: {input_path}")
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Error: Could not load image")
        return 1

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {image.shape}")

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_path.parent / f"{input_path.stem}_results"

    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize and run algorithm
    print("\nInitializing algorithm...")
    hardware_stream = HardwareBMDStream()

    hccc = HCCCAlgorithm(
        hardware_stream=hardware_stream,
        lambda_stream=args.lambda_stream,
        coherence_threshold=args.threshold
    )

    print("Processing image...")
    results = hccc.process_image(
        image,
        segmentation_method=args.method,
        segmentation_params={'n_segments': args.segments}
    )

    # Print results
    print("\nResults:")
    print(f"  Iterations: {results['convergence_step']}")
    print(f"  Regions processed: {results['regions_processed']}/{results['regions_total']}")
    print(f"  Final ambiguity: {results['metrics']['final_ambiguity']:.3f}")
    print(f"  Final divergence: {results['metrics']['final_divergence']:.3f}")
    print(f"  Network richness: {results['interpretation']['network_richness']:.2e}")
    print(f"  Compounds formed: {results['interpretation']['n_compounds']}")

    # Save results JSON
    if args.save_results:
        results_file = output_dir / "results.json"

        # Convert results to JSON-serializable format
        results_json = {
            'convergence_step': results['convergence_step'],
            'regions_processed': results['regions_processed'],
            'regions_total': results['regions_total'],
            'metrics': results['metrics'],
            'interpretation': {
                k: v for k, v in results['interpretation'].items()
                if not k.endswith('_bmd')  # Skip BMD objects
            },
            'processing_sequence': results['processing_sequence'],
            'ambiguity_history': results['ambiguity_history'],
            'stream_divergence_history': results['stream_divergence_history'],
            'network_richness_history': results['network_richness_history']
        }

        with open(results_file, 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    # Visualize
    if args.visualize:
        print("\nGenerating visualizations...")

        segmenter = ImageSegmenter()
        regions = segmenter.segment(
            image,
            method=args.method,
            n_segments=args.segments
        )

        visualizer = ResultVisualizer()
        visualizer.visualize_complete_results(
            image,
            regions,
            results,
            save_dir=str(output_dir)
        )

        print(f"Visualizations saved to: {output_dir}/")

    print("\nProcessing complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
