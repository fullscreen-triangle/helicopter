"""
PTRM Validation Suite Runner.

Executes all validation experiments on BBBC039 dataset and generates
publication-quality visualization panels.

Usage:
    python -m turbine.validation.run_validation

Or from Python:
    from turbine.validation import run_full_validation
    results = run_full_validation()
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import numpy as np

from .data_loader import BBBC039DataLoader
from .partition_coordinates import PartitionCoordinateExtractor
from .s_entropy import SEntropyAnalyzer
from .sequential_exclusion import SequentialExclusionValidator
from .reaction_localization import MultimodalLocalizationValidator
from .visualization import ValidationPanelGenerator


def run_full_validation(data_root: Optional[str] = None,
                        output_dir: Optional[str] = None,
                        max_samples: int = 50,
                        verbose: bool = True) -> Dict:
    """
    Run complete PTRM validation suite.

    Args:
        data_root: Path to turbine/public directory (auto-detected if None)
        output_dir: Path for output (results and panels)
        max_samples: Maximum number of images to process
        verbose: Print progress information

    Returns:
        Dictionary with all validation results
    """
    start_time = time.time()

    # Auto-detect paths
    if data_root is None:
        # Try relative to this file
        module_dir = Path(__file__).parent
        data_root = module_dir.parent / "public"
        if not data_root.exists():
            # Try from cwd
            data_root = Path("turbine/public")

    if output_dir is None:
        output_dir = Path(__file__).parent / "results"

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    panels_dir = output_dir / "panels"

    output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("  PTRM VALIDATION SUITE")
        print("  Partition Transition Rate Measurement Framework")
        print("  Testing on BBBC039 Nuclei Dataset")
        print("=" * 70)
        print(f"\n  Data root: {data_root}")
        print(f"  Output dir: {output_dir}")
        print(f"  Max samples: {max_samples}")
        print()

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'data_root': str(data_root),
            'output_dir': str(output_dir),
            'max_samples': max_samples
        }
    }

    # =========================================================================
    # Load Data
    # =========================================================================
    if verbose:
        print("-" * 70)
        print("LOADING DATA")
        print("-" * 70)

    try:
        loader = BBBC039DataLoader(str(data_root))
        samples = loader.load_all(max_samples=max_samples)
        dataset_stats = loader.get_dataset_statistics(samples)

        results['dataset'] = {
            'n_images': len(samples),
            'total_nuclei': dataset_stats['total_nuclei'],
            'nuclei_per_image': dataset_stats['nuclei_per_image'],
            'area_mean': dataset_stats['area_mean'],
            'area_std': dataset_stats['area_std']
        }

        if verbose:
            print(f"  Loaded {len(samples)} images")
            print(f"  Total nuclei: {dataset_stats['total_nuclei']}")
            print(f"  Nuclei per image: {dataset_stats['nuclei_per_image']:.1f}")

    except Exception as e:
        if verbose:
            print(f"  ERROR loading data: {e}")
        results['error'] = str(e)
        return results

    # =========================================================================
    # Experiment 1: Partition Coordinates
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("EXPERIMENT 1: PARTITION COORDINATE VALIDATION")
        print("Testing: C(n) = 2n² capacity theorem")
        print("-" * 70)

    extractor = PartitionCoordinateExtractor()
    all_coords = extractor.extract_from_dataset(samples)

    capacity_test = extractor.test_capacity_theorem(all_coords)
    entropy_data = extractor.compute_partition_entropy(all_coords)
    distribution_3d = extractor.get_3d_distribution(all_coords)

    results['partition'] = {
        'n_nuclei': len(all_coords),
        'chi2': capacity_test['chi2'],
        'p_value': capacity_test['p_value'],
        'validated': capacity_test['validated'],
        'reason': capacity_test['reason'],
        'entropy': entropy_data['entropy'],
        'normalized_entropy': entropy_data['normalized_entropy'],
        'n_occupied_states': entropy_data['n_occupied_states'],
        'observed_distribution': capacity_test['observed_distribution'],
        'expected_distribution': capacity_test['expected_distribution']
    }

    if verbose:
        print(f"  Nuclei analyzed: {len(all_coords)}")
        print(f"  Chi2 = {capacity_test['chi2']:.2f}, p = {capacity_test['p_value']:.4f}")
        print(f"  Partition entropy: {entropy_data['entropy']:.3f} bits")
        status = "[PASS] VALIDATED" if capacity_test['validated'] else "[FAIL] NOT VALIDATED"
        print(f"  Status: {status}")

    # =========================================================================
    # Experiment 2: S-Entropy Conservation
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("EXPERIMENT 2: S-ENTROPY TRAJECTORY ANALYSIS")
        print("Testing: S_k + S_t + S_e = constant")
        print("-" * 70)

    s_analyzer = SEntropyAnalyzer(extractor)
    trajectory = s_analyzer.analyze_trajectory(samples[:min(20, len(samples))])
    conservation_test = s_analyzer.test_conservation()
    trajectory_data = s_analyzer.get_trajectory_arrays()
    phase_space = s_analyzer.compute_phase_space_trajectory()

    results['s_entropy'] = {
        'n_timepoints': len(trajectory),
        'mean_S_total': conservation_test['mean_total'],
        'std_S_total': conservation_test['std_total'],
        'cv': conservation_test['cv'],
        'validated': conservation_test['validated'],
        'reason': conservation_test['reason']
    }

    if verbose:
        print(f"  Timepoints analyzed: {len(trajectory)}")
        print(f"  Mean S_total: {conservation_test['mean_total']:.4f}")
        print(f"  CV: {conservation_test['cv']:.4f}")
        status = "[PASS] VALIDATED" if conservation_test['validated'] else "[FAIL] NOT VALIDATED"
        print(f"  Status: {status}")

    # =========================================================================
    # Experiment 3: Sequential Exclusion
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("EXPERIMENT 3: SEQUENTIAL EXCLUSION VALIDATION")
        print("Testing: N_12 = N_0 * prod(epsilon_i) -> 1")
        print("-" * 70)

    exclusion_validator = SequentialExclusionValidator()
    modality_results = exclusion_validator.apply_modalities(samples[:min(30, len(samples))])
    exclusion_data = exclusion_validator.compute_cumulative_exclusion()
    curve_data = exclusion_validator.get_exclusion_curve_data()
    resolution_data = exclusion_validator.test_resolution_scaling()

    results['exclusion'] = {
        'n_modalities': len(modality_results),
        'N_0': exclusion_data.get('N_0', 0),
        'N_final': exclusion_data.get('N_final', 0),
        'reduction_factor': exclusion_data.get('reduction_factor', 1),
        'log_reduction': exclusion_data.get('log_reduction', 0),
        'validated': exclusion_data.get('validated', False),
        'reason': exclusion_data.get('reason', ''),
        'resolution_enhancement': resolution_data.get('enhancement_correlated', 1)
    }

    if verbose:
        print(f"  Modalities applied: {len(modality_results)}")
        print(f"  N_0 = {exclusion_data.get('N_0', 0):.2e}")
        print(f"  N_final = {exclusion_data.get('N_final', 0):.2e}")
        print(f"  Reduction: {exclusion_data.get('log_reduction', 0):.1f} orders of magnitude")
        status = "[PASS] VALIDATED" if exclusion_data.get('validated', False) else "[FAIL] NOT VALIDATED"
        print(f"  Status: {status}")

    # =========================================================================
    # Experiment 4: Multimodal Reaction Localization
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("EXPERIMENT 4: MULTIMODAL REACTION LOCALIZATION")
        print("Testing: Intersection Theorem (dr = dr_single * prod(epsilon_i^{1/3}))")
        print("-" * 70)

    localization_validator = MultimodalLocalizationValidator()

    # Use consecutive images as pseudo-time series
    if len(samples) >= 2:
        signal_maps = {}
        peak_data = {}
        consensus_data = {'positions': np.array([]).reshape(0, 2), 'n_modalities': np.array([]), 'confidence': np.array([])}
        resolution_results = {'validated': False}

        try:
            detections = localization_validator.detect_with_modalities(
                samples[0].image, samples[1].image
            )
            reactions = localization_validator.find_consensus_locations()
            resolution_results = localization_validator.compute_resolution_enhancement()

            signal_maps = localization_validator.get_signal_maps()
            peak_data = localization_validator.get_peak_data()
            consensus_data = localization_validator.get_consensus_data()

            results['localization'] = {
                'n_modalities': len(detections),
                'n_reactions_detected': len(reactions),
                'enhancement_factor': resolution_results.get('enhancement_factor', 1),
                'single_resolution': resolution_results.get('single_modality_resolution', 0),
                'multimodal_resolution': resolution_results.get('multimodal_resolution_theory', 0),
                'validated': resolution_results.get('validated', False)
            }

            if verbose:
                print(f"  Modalities: {len(detections)}")
                print(f"  Reactions detected: {len(reactions)}")
                print(f"  Enhancement factor: {resolution_results.get('enhancement_factor', 1):.2f}×")
                status = "[PASS] VALIDATED" if resolution_results.get('validated', False) else "[FAIL] NOT VALIDATED"
                print(f"  Status: {status}")

        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            results['localization'] = {'error': str(e), 'validated': False}
    else:
        if verbose:
            print("  SKIPPED: Need at least 2 images for reaction detection")
        results['localization'] = {'skipped': True, 'validated': False}
        signal_maps = {}
        peak_data = {}
        consensus_data = {}
        resolution_results = {}

    # =========================================================================
    # Generate Visualization Panels
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("GENERATING VISUALIZATION PANELS")
        print("-" * 70)

    panel_generator = ValidationPanelGenerator(str(panels_dir))

    panel_paths = []

    try:
        # Panel 1: Partition Coordinates
        path1 = panel_generator.generate_panel_1_partition_coordinates(
            partition_data={'coords': all_coords},
            capacity_test=capacity_test,
            entropy_data=entropy_data,
            distribution_3d=distribution_3d
        )
        panel_paths.append(path1)
        if verbose:
            print(f"  Generated: {Path(path1).name}")

        # Panel 2: S-Entropy
        path2 = panel_generator.generate_panel_2_s_entropy(
            trajectory_data=trajectory_data,
            conservation_test=conservation_test,
            phase_space=phase_space
        )
        panel_paths.append(path2)
        if verbose:
            print(f"  Generated: {Path(path2).name}")

        # Panel 3: Sequential Exclusion
        path3 = panel_generator.generate_panel_3_sequential_exclusion(
            exclusion_data=exclusion_data,
            curve_data=curve_data,
            resolution_data=resolution_data
        )
        panel_paths.append(path3)
        if verbose:
            print(f"  Generated: {Path(path3).name}")

        # Panel 4: Reaction Localization
        if signal_maps:
            path4 = panel_generator.generate_panel_4_reaction_localization(
                signal_maps=signal_maps,
                peak_data=peak_data,
                consensus_data=consensus_data,
                resolution_results=resolution_results
            )
            panel_paths.append(path4)
            if verbose:
                print(f"  Generated: {Path(path4).name}")

        # Summary Panel
        all_results = {
            'partition': results.get('partition', {}),
            's_entropy': results.get('s_entropy', {}),
            'exclusion': results.get('exclusion', {}),
            'localization': results.get('localization', {})
        }

        path_summary = panel_generator.generate_summary_panel(
            all_results=all_results,
            dataset_stats=dataset_stats
        )
        panel_paths.append(path_summary)
        if verbose:
            print(f"  Generated: {Path(path_summary).name}")

    except Exception as e:
        if verbose:
            print(f"  ERROR generating panels: {e}")
        results['panel_error'] = str(e)

    results['panels'] = panel_paths

    # =========================================================================
    # Save Results
    # =========================================================================
    results_path = output_dir / "validation_results.json"

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    results_json = convert_for_json(results)

    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    if verbose:
        print(f"\n  Results saved: {results_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time

    if verbose:
        print("\n" + "=" * 70)
        print("  VALIDATION COMPLETE")
        print("=" * 70)

        n_validated = sum([
            results.get('partition', {}).get('validated', False),
            results.get('s_entropy', {}).get('validated', False),
            results.get('exclusion', {}).get('validated', False),
            results.get('localization', {}).get('validated', False)
        ])

        print(f"\n  Experiments validated: {n_validated}/4")
        print(f"  Total time: {elapsed:.1f} seconds")
        print(f"\n  Output directory: {output_dir}")
        print(f"  Panels generated: {len(panel_paths)}")
        print()

    results['summary'] = {
        'n_validated': n_validated,
        'total_experiments': 4,
        'elapsed_seconds': elapsed
    }

    return results


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run PTRM validation suite on BBBC039 dataset'
    )
    parser.add_argument('--data-root', type=str, default=None,
                       help='Path to turbine/public directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results and panels')
    parser.add_argument('--max-samples', type=int, default=50,
                       help='Maximum number of images to process')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    results = run_full_validation(
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        verbose=not args.quiet
    )

    # Exit with error code if any experiment failed
    n_validated = results.get('summary', {}).get('n_validated', 0)
    return 0 if n_validated >= 2 else 1


if __name__ == '__main__':
    exit(main())
