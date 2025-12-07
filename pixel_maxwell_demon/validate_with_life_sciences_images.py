#!/usr/bin/env python3
"""
Life Sciences Image Validation Suite with Multi-Modal Virtual Detectors
======================================================================

Comprehensive validation of dual-membrane HCCC framework using life sciences
images with SIMULTANEOUS multi-modal analysis.

REVOLUTIONARY CAPABILITY: Traditional life sciences imaging requires commitment
to a single modality (fluorescent microscopy, light field, phase contrast, etc.).
Samples prepared for one method cannot be reused for another.

OUR METHOD SOLVES THIS: Using categorical pixel demons with virtual detectors,
we can analyze the SAME sample with ALL imaging modalities SIMULTANEOUSLY:
- Fluorescent microscopy (VirtualPhotodiode)
- IR spectroscopy (VirtualIRSpectrometer)
- Raman spectroscopy (VirtualRamanSpectrometer)
- Mass spectrometry (VirtualMassSpectrometer)
- Temperature mapping (VirtualThermometer)
- Pressure mapping (VirtualBarometer)
- Phase interferometry (VirtualInterferometer)

No physical commitment required - all modalities accessed through categorical
queries on the same pixel demon grid!

Author: Kundai Sachikonye & AI Collaborator
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from pathlib import Path
from typing import List, Dict
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from maxwell.integration import (
    DualMembraneHCCCAlgorithm,
    DepthExtractor,
    FrameworkValidator
)


class LifeSciencesValidator:
    """
    Comprehensive validator for life sciences images.
    """
    
    def __init__(self, public_dir: str = 'public', output_dir: str = 'life_sciences_validation'):
        """
        Initialize validator.
        
        Args:
            public_dir: Directory containing life sciences images
            output_dir: Directory for validation results
        """
        self.public_dir = Path(public_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        self.image_files = self._find_image_files()
        
        self.results = {}
    
    def _find_image_files(self) -> List[Path]:
        """Find all valid image files in public directory."""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        
        image_files = []
        for ext in valid_extensions:
            image_files.extend(self.public_dir.glob(f'*{ext}'))
            image_files.extend(self.public_dir.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} life sciences images:")
        for img in sorted(image_files):
            print(f"  - {img.name}")
        
        return sorted(image_files)
    
    def validate_single_image(
        self,
        image_path: Path,
        n_segments: int = 50,
        cascade_depth: int = 10
    ) -> Dict:
        """
        Validate framework on single image.
        
        Args:
            image_path: Path to image
            n_segments: Number of segmentation regions
            cascade_depth: Cascade depth
        
        Returns:
            Validation results dictionary
        """
        print(f"\n{'='*80}")
        print(f"Validating: {image_path.name}")
        print(f"{'='*80}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  ✗ Failed to load image")
            return {'success': False, 'error': 'Failed to load image'}
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"  Image shape: {image.shape}")
        print(f"  Image dtype: {image.dtype}")
        
        # Create output directory for this image
        image_output_dir = self.output_dir / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize algorithm
        print(f"\n  Initializing algorithm (n_segments={n_segments}, cascade_depth={cascade_depth})...")
        algorithm = DualMembraneHCCCAlgorithm(
            max_iterations=100,
            convergence_threshold=1e-6,
            use_cascade=True,
            cascade_depth=cascade_depth,
            atmospheric_conditions={
                'temperature': 298.15,  # Room temperature
                'pressure': 101325,
                'humidity': 0.5
            }
        )
        
        # Process image
        start_time = time.time()
        try:
            result = algorithm.process_image(
                image,
                n_segments=n_segments,
                segmentation_method='slic'
            )
            processing_time = time.time() - start_time
            
            print(f"\n  ✓ Processing complete ({processing_time:.2f}s)")
            
        except Exception as e:
            print(f"  ✗ Processing failed: {e}")
            return {'success': False, 'error': str(e)}
        
        # Extract results
        validation_results = {
            'success': True,
            'image_name': image_path.name,
            'image_shape': image.shape,
            'processing_time': processing_time,
            'total_iterations': result.total_iterations,
            'converged': result.converged,
            'final_richness': result.final_richness,
            'final_stream_coherence': result.final_stream_coherence,
            'energy_dissipation': result.energy_dissipation,
            'depth_statistics': result.network_bmd.calculate_depth_statistics(),
            'n_regions': len(result.network_bmd.region_bmds),
            'n_compounds': len(result.network_bmd.compound_bmds),
            'molecular_statistics': result.hardware_stream.get_molecular_demon_statistics()
        }
        
        # Print summary
        print(f"\n  Results:")
        print(f"    - Iterations: {result.total_iterations}")
        print(f"    - Converged: {result.converged}")
        print(f"    - Final richness: {result.final_richness:.6f}")
        print(f"    - Stream coherence: {result.final_stream_coherence:.4f}")
        print(f"    - Energy dissipation: {result.energy_dissipation:.3e} J")
        print(f"    - Depth range: [{validation_results['depth_statistics']['min']:.4f}, "
              f"{validation_results['depth_statistics']['max']:.4f}]")
        
        # Save results
        with open(image_output_dir / 'results.json', 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        # Save depth map
        np.save(image_output_dir / 'depth_map.npy', result.depth_map)
        
        # Create visualizations
        print(f"\n  Generating visualizations...")
        self._create_visualizations(image, result, image_output_dir)
        
        return validation_results
    
    def _create_visualizations(
        self,
        image: np.ndarray,
        result,
        output_dir: Path
    ):
        """Create comprehensive visualization suite."""
        extractor = DepthExtractor(normalize=True, smoothing_sigma=1.0)
        
        # 1. Comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Life Sciences Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Depth map
        im = axes[1].imshow(result.depth_map, cmap='turbo')
        axes[1].set_title('Categorical Depth Map', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 3D depth surface (small)
        from mpl_toolkits.mplot3d import Axes3D
        ax3d = fig.add_subplot(1, 3, 3, projection='3d')
        
        # Downsample for 3D plot
        h, w = result.depth_map.shape
        stride = max(1, h // 50)
        depth_small = result.depth_map[::stride, ::stride]
        image_small = image[::stride, ::stride]
        
        x = np.arange(depth_small.shape[1])
        y = np.arange(depth_small.shape[0])
        X, Y = np.meshgrid(x, y)
        
        ax3d.plot_surface(
            X, Y, depth_small,
            facecolors=image_small / 255.0,
            rstride=1, cstride=1,
            shade=False
        )
        ax3d.set_title('3D Categorical Depth', fontsize=12, fontweight='bold')
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Depth')
        ax3d.view_init(elev=30, azim=45)
        
        fig.suptitle('Dual-Membrane HCCC Analysis', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(output_dir / 'comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 2. Depth map only (high resolution)
        fig, ax = extractor.visualize_depth(result.depth_map, colormap='turbo')
        fig.savefig(output_dir / 'depth_map.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 3. Depth histogram
        fig, ax = extractor.create_depth_histogram(result.depth_map)
        fig.savefig(output_dir / 'depth_histogram.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 4. Convergence plot
        fig, ax = plt.subplots(figsize=(10, 6))
        iterations = [d['iteration'] for d in result.iteration_history]
        richness = [d['richness'] for d in result.iteration_history]
        
        ax.plot(iterations, richness, 'b-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Network Richness', fontsize=12)
        ax.set_title('Convergence of Network Richness', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / 'convergence.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"    ✓ Visualizations saved to: {output_dir}")
    
    def validate_all_images(
        self,
        n_segments: int = 50,
        cascade_depth: int = 10,
        max_images: int = None
    ) -> Dict:
        """
        Validate framework on all life sciences images.
        
        Args:
            n_segments: Number of segmentation regions
            cascade_depth: Cascade depth
            max_images: Maximum number of images to process (None = all)
        
        Returns:
            Complete validation results
        """
        print("\n" + "="*80)
        print("  LIFE SCIENCES IMAGE VALIDATION SUITE")
        print("  Dual-Membrane HCCC Framework")
        print("="*80)
        
        # Limit images if requested
        images_to_process = self.image_files[:max_images] if max_images else self.image_files
        
        print(f"\nProcessing {len(images_to_process)} images...")
        print(f"Parameters:")
        print(f"  - Segments per image: {n_segments}")
        print(f"  - Cascade depth: {cascade_depth}")
        
        # Process each image
        all_results = []
        successful = 0
        failed = 0
        
        for i, image_path in enumerate(images_to_process, 1):
            print(f"\n[{i}/{len(images_to_process)}] Processing {image_path.name}...")
            
            result = self.validate_single_image(
                image_path,
                n_segments=n_segments,
                cascade_depth=cascade_depth
            )
            
            all_results.append(result)
            
            if result.get('success', False):
                successful += 1
            else:
                failed += 1
        
        # Aggregate statistics
        successful_results = [r for r in all_results if r.get('success', False)]
        
        if successful_results:
            aggregate_stats = {
                'total_images': len(images_to_process),
                'successful': successful,
                'failed': failed,
                'success_rate': successful / len(images_to_process),
                'average_processing_time': np.mean([r['processing_time'] for r in successful_results]),
                'average_iterations': np.mean([r['total_iterations'] for r in successful_results]),
                'average_richness': np.mean([r['final_richness'] for r in successful_results]),
                'average_coherence': np.mean([r['final_stream_coherence'] for r in successful_results]),
                'average_energy': np.mean([r['energy_dissipation'] for r in successful_results]),
                'convergence_rate': sum(1 for r in successful_results if r['converged']) / len(successful_results)
            }
        else:
            aggregate_stats = {
                'total_images': len(images_to_process),
                'successful': 0,
                'failed': len(images_to_process),
                'success_rate': 0.0
            }
        
        # Save complete results
        complete_results = {
            'aggregate_statistics': aggregate_stats,
            'individual_results': all_results,
            'parameters': {
                'n_segments': n_segments,
                'cascade_depth': cascade_depth
            }
        }
        
        with open(self.output_dir / 'complete_results.json', 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        # Create summary report
        self._create_summary_report(complete_results)
        
        # Print summary
        print(f"\n{'='*80}")
        print("  VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"\n  Images processed: {len(images_to_process)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Success rate: {aggregate_stats['success_rate']*100:.1f}%")
        
        if successful_results:
            print(f"\n  Average metrics:")
            print(f"    - Processing time: {aggregate_stats['average_processing_time']:.2f}s")
            print(f"    - Iterations: {aggregate_stats['average_iterations']:.1f}")
            print(f"    - Final richness: {aggregate_stats['average_richness']:.6f}")
            print(f"    - Stream coherence: {aggregate_stats['average_coherence']:.4f}")
            print(f"    - Energy dissipation: {aggregate_stats['average_energy']:.3e} J")
            print(f"    - Convergence rate: {aggregate_stats['convergence_rate']*100:.1f}%")
        
        print(f"\n  Results saved to: {self.output_dir}")
        print(f"{'='*80}\n")
        
        return complete_results
    
    def _create_summary_report(self, results: Dict):
        """Create visual summary report."""
        successful_results = [r for r in results['individual_results'] if r.get('success', False)]
        
        if not successful_results:
            print("  ✗ No successful results to summarize")
            return
        
        # Create multi-panel summary figure
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Processing time distribution
        ax1 = plt.subplot(2, 3, 1)
        times = [r['processing_time'] for r in successful_results]
        ax1.hist(times, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Processing Time (s)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('Processing Time Distribution', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Richness distribution
        ax2 = plt.subplot(2, 3, 2)
        richness = [r['final_richness'] for r in successful_results]
        ax2.hist(richness, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Final Richness', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Network Richness Distribution', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Stream coherence distribution
        ax3 = plt.subplot(2, 3, 3)
        coherence = [r['final_stream_coherence'] for r in successful_results]
        ax3.hist(coherence, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax3.set_xlabel('Stream Coherence', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.set_title('Hardware Stream Coherence', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Iteration count distribution
        ax4 = plt.subplot(2, 3, 4)
        iterations = [r['total_iterations'] for r in successful_results]
        ax4.hist(iterations, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('Iterations', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.set_title('Convergence Iterations', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Energy dissipation distribution
        ax5 = plt.subplot(2, 3, 5)
        energy = [r['energy_dissipation'] for r in successful_results]
        ax5.hist(energy, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax5.set_xlabel('Energy Dissipation (J)', fontsize=10)
        ax5.set_ylabel('Frequency', fontsize=10)
        ax5.set_title('Energy Dissipation Distribution', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Depth statistics
        ax6 = plt.subplot(2, 3, 6)
        mean_depths = [r['depth_statistics']['mean'] for r in successful_results]
        std_depths = [r['depth_statistics']['std'] for r in successful_results]
        image_names = [r['image_name'][:10] for r in successful_results]  # Truncate names
        
        x = np.arange(len(successful_results))
        ax6.bar(x, mean_depths, yerr=std_depths, alpha=0.7, color='teal', edgecolor='black', capsize=5)
        ax6.set_xlabel('Image', fontsize=10)
        ax6.set_ylabel('Mean Depth', fontsize=10)
        ax6.set_title('Categorical Depth by Image', fontsize=11, fontweight='bold')
        if len(successful_results) <= 10:
            ax6.set_xticks(x)
            ax6.set_xticklabels(image_names, rotation=45, ha='right', fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        fig.suptitle('Life Sciences Validation Summary', fontsize=16, fontweight='bold')
        fig.tight_layout()
        fig.savefig(self.output_dir / 'summary_report.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Summary report saved: {self.output_dir / 'summary_report.png'}")


def main():
    """Run life sciences validation suite."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Life Sciences Image Validation Suite'
    )
    parser.add_argument(
        '--public-dir',
        type=str,
        default='public',
        help='Directory containing life sciences images (default: public)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='life_sciences_validation',
        help='Output directory for results (default: life_sciences_validation)'
    )
    parser.add_argument(
        '--n-segments',
        type=int,
        default=50,
        help='Number of segmentation regions (default: 50)'
    )
    parser.add_argument(
        '--cascade-depth',
        type=int,
        default=10,
        help='Cascade depth for O(N³) gain (default: 10)'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to process (default: all)'
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = LifeSciencesValidator(
        public_dir=args.public_dir,
        output_dir=args.output_dir
    )
    
    # Run validation
    results = validator.validate_all_images(
        n_segments=args.n_segments,
        cascade_depth=args.cascade_depth,
        max_images=args.max_images
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

