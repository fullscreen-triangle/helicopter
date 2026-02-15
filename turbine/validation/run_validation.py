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
from .quintupartite_validation import QuintupartiteValidator
from .dual_membrane_validation import DualMembraneValidator
from .oxygen_dynamics_validation import OxygenDynamicsValidator
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
    # Experiment 5: Quintupartite Virtual Microscopy
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("EXPERIMENT 5: QUINTUPARTITE VIRTUAL MICROSCOPY")
        print("Testing: Multi-Modal Uniqueness Theorem (N_5 = N_0 x prod(eps) -> 1)")
        print("-" * 70)

    quintupartite_validator = QuintupartiteValidator(n_modalities=5)

    try:
        # Apply all 5 modalities
        modality_results = quintupartite_validator.apply_all_modalities(samples[:min(10, len(samples))])

        # Test multi-modal uniqueness
        uniqueness_test = quintupartite_validator.test_multimodal_uniqueness()

        # Test metabolic GPS
        gps_test = quintupartite_validator.test_metabolic_gps(samples[0].mask)

        # Test temporal-causal consistency
        causal_test = quintupartite_validator.test_temporal_causal_consistency(samples[:min(5, len(samples))])

        # Get visualization data
        quintu_signal_maps = quintupartite_validator.get_signal_maps()
        exclusion_curve = quintupartite_validator.get_exclusion_curve_data()

        results['quintupartite'] = {
            'n_modalities': len(modality_results),
            'unique_determination': uniqueness_test.get('unique_determination', False),
            'N_0': uniqueness_test.get('N_0', 0),
            'N_final': uniqueness_test.get('N_final', 0),
            'log_reduction': uniqueness_test.get('log_reduction', 0),
            'gps_validated': gps_test.get('validated', False),
            'gps_error': gps_test.get('mean_localization_error', 0),
            'causal_consistent': causal_test.get('causal_consistent', False),
            'validated': uniqueness_test.get('validated', False) or gps_test.get('validated', False),
            'reason': uniqueness_test.get('reason', '')
        }

        if verbose:
            print(f"  Modalities applied: {len(modality_results)}")
            print(f"  N_0 = {uniqueness_test.get('N_0', 0):.2e}")
            print(f"  N_final = {uniqueness_test.get('N_final', 0):.2e}")
            print(f"  Log reduction: {uniqueness_test.get('log_reduction', 0):.1f} orders")
            print(f"  GPS triangulation: {'SUCCESS' if gps_test.get('validated', False) else 'FAILED'}")
            status = "[PASS] VALIDATED" if results['quintupartite']['validated'] else "[FAIL] NOT VALIDATED"
            print(f"  Status: {status}")

    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        results['quintupartite'] = {'error': str(e), 'validated': False}
        quintu_signal_maps = {}
        uniqueness_test = {}
        gps_test = {}
        causal_test = {}

    # =========================================================================
    # Experiment 6: Dual-Membrane Pixel Maxwell Demon
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("EXPERIMENT 6: DUAL-MEMBRANE PIXEL MAXWELL DEMON")
        print("Testing: Conjugate State Theorem (S_k^back = -S_k^front, r = -1.000)")
        print("-" * 70)

    dual_membrane_validator = DualMembraneValidator()

    try:
        # Test conjugate states
        conjugate_test = dual_membrane_validator.test_conjugate_states(samples[:min(10, len(samples))])

        # Test platform independence
        platform_test = dual_membrane_validator.test_platform_independence(
            samples[0].image, n_runs=3, delay_ms=50
        )

        # Test quadratic scaling
        cascade_test = dual_membrane_validator.compute_reflectance_cascade(
            samples[0].image, n_levels=10
        )

        # Test zero-backaction
        backaction_test = dual_membrane_validator.test_zero_backaction(samples[:min(5, len(samples))])

        # Test membrane thickness
        thickness_test = dual_membrane_validator.test_membrane_thickness(samples[:min(10, len(samples))])

        # Get visualization data
        membrane_vis = dual_membrane_validator.get_membrane_visualization_data(samples[0].image)
        cascade_data = dual_membrane_validator.get_cascade_data()

        results['dual_membrane'] = {
            'mean_anti_correlation': conjugate_test.get('mean_anti_correlation', 0),
            'expected_correlation': -1.0,
            'mean_conjugate_sum': conjugate_test.get('mean_conjugate_sum', 0),
            'platform_independent': platform_test.get('validated', False),
            'max_platform_difference': platform_test.get('max_difference', 0),
            'cascade_enhancement': cascade_test.get('enhancement_factor', 1),
            'zero_backaction': backaction_test.get('zero_backaction', False),
            'membrane_thickness_consistent': thickness_test.get('categorical_depth_consistent', False),
            'validated': conjugate_test.get('validated', False),
            'reason': conjugate_test.get('reason', '')
        }

        if verbose:
            print(f"  Anti-correlation: r = {conjugate_test.get('mean_anti_correlation', 0):.6f}")
            print(f"  Conjugate sum: {conjugate_test.get('mean_conjugate_sum', 0):.2e}")
            print(f"  Platform independent: {'YES' if platform_test.get('validated', False) else 'NO'}")
            print(f"  Cascade enhancement: {cascade_test.get('enhancement_factor', 1):.1f}x")
            print(f"  Zero-backaction: {'YES' if backaction_test.get('zero_backaction', False) else 'NO'}")
            status = "[PASS] VALIDATED" if results['dual_membrane']['validated'] else "[FAIL] NOT VALIDATED"
            print(f"  Status: {status}")

    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        results['dual_membrane'] = {'error': str(e), 'validated': False}
        conjugate_test = {}
        platform_test = {}
        cascade_test = {}
        membrane_vis = {}
        cascade_data = {}

    # =========================================================================
    # Experiment 7: Oxygen-Mediated Categorical Microscopy
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("EXPERIMENT 7: OXYGEN-MEDIATED CATEGORICAL MICROSCOPY")
        print("Testing: Ternary State Dynamics (Absorption/Ground/Emission)")
        print("-" * 70)

    oxygen_validator = OxygenDynamicsValidator()

    try:
        # Test ternary dynamics
        ternary_test = oxygen_validator.test_ternary_dynamics(samples[:min(5, len(samples))])

        # Test capacitor architecture
        capacitor_test = oxygen_validator.test_capacitor_architecture()

        # Test virtual light source
        virtual_light_test = oxygen_validator.test_virtual_light_source()

        # Get visualization data
        oxygen_vis_data = oxygen_validator.get_visualization_data()
        state_history = oxygen_validator.get_state_history_data()

        results['oxygen_dynamics'] = {
            'distribution_valid': ternary_test.get('distribution_valid', False),
            'resolution_valid': ternary_test.get('resolution_valid', False),
            'spatial_resolution_nm': ternary_test.get('spatial_resolution_nm', 0),
            'temporal_resolution_fs': ternary_test.get('temporal_resolution_fs', 0),
            'snr': ternary_test.get('snr', 0),
            'capacitance_pF': capacitor_test.get('capacitance_pF', 0),
            'electric_field_Vm': capacitor_test.get('electric_field_Vm', 0),
            'wavelength_um': virtual_light_test.get('wavelength_um', 0),
            'validated': ternary_test.get('validated', False) or capacitor_test.get('validated', False),
            'reason': ternary_test.get('reason', '')
        }

        if verbose:
            final_dist = ternary_test.get('final_distribution', {})
            print(f"  State distribution: Abs={final_dist.get('absorption', 0):.2%}, "
                  f"Gnd={final_dist.get('ground', 0):.2%}, Emit={final_dist.get('emission', 0):.2%}")
            print(f"  Spatial resolution: {ternary_test.get('spatial_resolution_nm', 0):.1f} nm")
            print(f"  Capacitance: {capacitor_test.get('capacitance_pF', 0):.2f} pF")
            print(f"  Virtual light: {virtual_light_test.get('wavelength_um', 0):.2f} um (Mid-IR)")
            status = "[PASS] VALIDATED" if results['oxygen_dynamics']['validated'] else "[FAIL] NOT VALIDATED"
            print(f"  Status: {status}")

    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        results['oxygen_dynamics'] = {'error': str(e), 'validated': False}
        ternary_test = {}
        capacitor_test = {}
        virtual_light_test = {}
        oxygen_vis_data = {}
        state_history = {}

    # =========================================================================
    # Experiment 8: Electrostatic Chambers & Atomic Spectrometry
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("EXPERIMENT 8: ELECTROSTATIC CHAMBERS & ATOMIC SPECTROMETRY")
        print("Testing: Transient Bioreactors and Protein Atom Arrays")
        print("-" * 70)

    try:
        # Test electrostatic chamber formation
        chamber_test = oxygen_validator.simulate_electrostatic_chambers(
            samples[0].image, num_steps=100
        )

        # Test atomic ternary spectrometry
        spectrometry_test = oxygen_validator.test_atomic_ternary_spectrometry(
            samples[0].image
        )

        # Get virtual image
        virtual_image = oxygen_vis_data.get('virtual_image', None)

        results['electrostatic_chambers'] = {
            'num_chamber_events': chamber_test.get('num_chamber_events', 0),
            'mean_chamber_size_nm': chamber_test.get('mean_chamber_size_nm', 0),
            'rate_enhancement': chamber_test.get('rate_enhancement', 0),
            'spectrometry_validated': spectrometry_test.get('validated', False),
            'absorption_intensity': spectrometry_test.get('absorption_intensity', 0),
            'emission_intensity': spectrometry_test.get('emission_intensity', 0),
            'validated': chamber_test.get('validated', False) or spectrometry_test.get('validated', False),
            'reason': chamber_test.get('reason', '') + '; ' + spectrometry_test.get('reason', '')
        }

        if verbose:
            print(f"  Chamber events: {chamber_test.get('num_chamber_events', 0)}")
            print(f"  Mean chamber size: {chamber_test.get('mean_chamber_size_nm', 0):.1f} nm")
            print(f"  Rate enhancement: {chamber_test.get('rate_enhancement', 0):.0f}x")
            spec_dist = spectrometry_test.get('state_distribution', {})
            print(f"  Atomic states: G={spec_dist.get('ground', 0):.2%}, "
                  f"N={spec_dist.get('natural', 0):.2%}, E={spec_dist.get('excited', 0):.2%}")
            status = "[PASS] VALIDATED" if results['electrostatic_chambers']['validated'] else "[FAIL] NOT VALIDATED"
            print(f"  Status: {status}")

    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        results['electrostatic_chambers'] = {'error': str(e), 'validated': False}
        chamber_test = {}
        spectrometry_test = {}
        virtual_image = None

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

        # Panel 5: Quintupartite
        if 'quintupartite' in results and 'error' not in results['quintupartite']:
            try:
                path5 = panel_generator.generate_panel_5_quintupartite(
                    uniqueness_data=uniqueness_test if 'uniqueness_test' in dir() else {},
                    gps_data=gps_test if 'gps_test' in dir() else {},
                    causal_data=causal_test if 'causal_test' in dir() else {},
                    signal_maps=quintu_signal_maps if 'quintu_signal_maps' in dir() else {}
                )
                panel_paths.append(path5)
                if verbose:
                    print(f"  Generated: {Path(path5).name}")
            except Exception as e:
                if verbose:
                    print(f"  Error generating Panel 5: {e}")

        # Panel 6: Dual-Membrane
        if 'dual_membrane' in results and 'error' not in results['dual_membrane']:
            try:
                path6 = panel_generator.generate_panel_6_dual_membrane(
                    conjugate_data=conjugate_test if 'conjugate_test' in dir() else {},
                    platform_data=platform_test if 'platform_test' in dir() else {},
                    cascade_data=cascade_data if 'cascade_data' in dir() else {},
                    membrane_vis=membrane_vis if 'membrane_vis' in dir() else {}
                )
                panel_paths.append(path6)
                if verbose:
                    print(f"  Generated: {Path(path6).name}")
            except Exception as e:
                if verbose:
                    print(f"  Error generating Panel 6: {e}")

        # Panel 7: Oxygen Dynamics
        if 'oxygen_dynamics' in results and 'error' not in results['oxygen_dynamics']:
            try:
                path7 = panel_generator.generate_panel_7_oxygen_dynamics(
                    ternary_data=ternary_test if 'ternary_test' in dir() else {},
                    capacitor_data=capacitor_test if 'capacitor_test' in dir() else {},
                    virtual_light_data=virtual_light_test if 'virtual_light_test' in dir() else {},
                    state_history=state_history if 'state_history' in dir() else {}
                )
                panel_paths.append(path7)
                if verbose:
                    print(f"  Generated: {Path(path7).name}")
            except Exception as e:
                if verbose:
                    print(f"  Error generating Panel 7: {e}")

        # Panel 8: Electrostatic Chambers
        if 'electrostatic_chambers' in results and 'error' not in results['electrostatic_chambers']:
            try:
                path8 = panel_generator.generate_panel_8_electrostatic_chambers(
                    chamber_data=chamber_test if 'chamber_test' in dir() else {},
                    spectrometry_data=spectrometry_test if 'spectrometry_test' in dir() else {},
                    virtual_image=virtual_image if 'virtual_image' in dir() else None
                )
                panel_paths.append(path8)
                if verbose:
                    print(f"  Generated: {Path(path8).name}")
            except Exception as e:
                if verbose:
                    print(f"  Error generating Panel 8: {e}")

        # Extended Summary Panel (all 8 experiments)
        all_results = {
            'partition': results.get('partition', {}),
            's_entropy': results.get('s_entropy', {}),
            'exclusion': results.get('exclusion', {}),
            'localization': results.get('localization', {}),
            'quintupartite': results.get('quintupartite', {}),
            'dual_membrane': results.get('dual_membrane', {}),
            'oxygen_dynamics': results.get('oxygen_dynamics', {}),
            'electrostatic_chambers': results.get('electrostatic_chambers', {})
        }

        path_summary = panel_generator.generate_extended_summary_panel(
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
            results.get('localization', {}).get('validated', False),
            results.get('quintupartite', {}).get('validated', False),
            results.get('dual_membrane', {}).get('validated', False),
            results.get('oxygen_dynamics', {}).get('validated', False),
            results.get('electrostatic_chambers', {}).get('validated', False)
        ])

        print(f"\n  Experiments validated: {n_validated}/8")
        print(f"  Total time: {elapsed:.1f} seconds")
        print(f"\n  Output directory: {output_dir}")
        print(f"  Panels generated: {len(panel_paths)}")
        print()

    results['summary'] = {
        'n_validated': n_validated,
        'total_experiments': 8,
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

    # Exit with error code if less than half the experiments passed
    n_validated = results.get('summary', {}).get('n_validated', 0)
    return 0 if n_validated >= 4 else 1


if __name__ == '__main__':
    exit(main())
