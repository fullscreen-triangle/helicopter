#!/usr/bin/env python3
"""
Visualize Multi-Modal Detector Results from JSON

Creates comprehensive panel charts with:
- Radar charts for each detector performance
- Electromagnetic spectrum representations
- Temporal analysis
- Comparative metrics
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import json
from pathlib import Path
from typing import Dict, List
import matplotlib.cm as cm


def load_multi_modal_results(json_path: Path) -> Dict:
    """Load the complete multi-modal results JSON"""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_detector_radar_chart(ax: plt.Axes, detector_name: str, 
                                detector_data: List[Dict], title: str):
    """
    Create radar chart for single detector showing performance across images
    """
    # Extract statistics across all images
    means = [img['detector_statistics'][detector_name]['mean'] 
             for img in detector_data if detector_name in img['detector_statistics']]
    stds = [img['detector_statistics'][detector_name]['std'] 
            for img in detector_data if detector_name in img['detector_statistics']]
    times = [img['detector_statistics'][detector_name]['measurement_time_s'] 
             for img in detector_data if detector_name in img['detector_statistics']]
    
    if len(means) == 0:
        ax.text(0.5, 0.5, f'No data for\n{detector_name}', 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    # Normalize metrics to 0-1 scale for radar chart
    metrics = {
        'Consistency\n(low std)': 1.0 / (1.0 + np.mean(stds) / (np.mean(means) + 1e-10)),
        'Speed\n(fast)': 1.0 / (1.0 + np.mean(times) / 10.0),
        'Signal\n(high)': np.mean(means) / (max(means) + 1e-10),
        'Precision\n(low var)': 1.0 - (np.std(means) / (np.mean(means) + 1e-10)),
        'Reliability': 1.0 if len(means) == len(detector_data) else 0.5
    }
    
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # Number of variables
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]
    
    # Plot
    ax.plot(angles, values, 'o-', linewidth=3, color='#FF6B6B', markersize=8)
    ax.fill(angles, values, alpha=0.25, color='#FF6B6B')
    
    # Fix axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=20)
    
    # Add center dot
    ax.plot(0, 0, 'ko', markersize=5)


def create_em_spectrum_radar(ax: plt.Axes, detector_name: str, title: str):
    """
    Create EM spectrum representation as radar chart
    
    Shows which part of electromagnetic spectrum each detector is sensitive to
    """
    # Define EM spectrum regions (wavelength in nm)
    em_regions = {
        'Gamma\nRays': (0.001, 0.01),
        'X-Rays': (0.01, 10),
        'UV': (10, 400),
        'Visible': (400, 700),
        'Near-IR': (700, 2500),
        'Mid-IR': (2500, 25000),
        'Far-IR': (25000, 1e6),
        'Microwave': (1e6, 1e9)
    }
    
    # Define detector sensitivities
    detector_sensitivities = {
        'Photodiode': {'Visible': 1.0, 'Near-IR': 0.7, 'UV': 0.5},
        'IR_Spectrometer': {'Near-IR': 1.0, 'Mid-IR': 1.0, 'Visible': 0.3},
        'Raman_Spectrometer': {'Visible': 0.9, 'Near-IR': 0.8},
        'Mass_Spectrometer': {},  # Not EM-based
        'Thermometer': {'Far-IR': 1.0, 'Mid-IR': 0.8},  # Thermal radiation
        'Barometer': {},  # Not EM-based
        'Hygrometer': {},  # Not EM-based
        'Interferometer': {'Visible': 1.0, 'Near-IR': 0.9, 'UV': 0.7}
    }
    
    # Get sensitivities for this detector
    sensitivities = detector_sensitivities.get(detector_name, {})
    
    if not sensitivities:
        # For non-EM detectors, show as text
        ax.text(0.5, 0.5, f'{detector_name}\nNot EM-based\n(Mechanical/Chemical)', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title(title, fontsize=11, fontweight='bold')
        return
    
    # Prepare data for radar chart
    categories = list(em_regions.keys())
    values = [sensitivities.get(cat, 0.0) for cat in categories]
    
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    # Plot with EM spectrum colors
    colors_map = {
        'Gamma\nRays': '#9400D3',
        'X-Rays': '#4B0082',
        'UV': '#8B00FF',
        'Visible': '#FFFFFF',
        'Near-IR': '#FF0000',
        'Mid-IR': '#8B0000',
        'Far-IR': '#FF6347',
        'Microwave': '#FF8C00'
    }
    
    # Fill with gradient
    for i in range(len(angles) - 1):
        if values[i] > 0:
            color = colors_map.get(categories[i % len(categories)], '#888888')
            ax.fill([0, angles[i], angles[i+1]], [0, values[i], values[i+1]], 
                   alpha=0.6, color=color)
    
    # Plot outline
    ax.plot(angles, values, 'ko-', linewidth=2, markersize=6)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(['0.5', '1.0'], fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=20)


def create_detector_comparison_bars(ax: plt.Axes, results: Dict, title: str):
    """Create grouped bar chart comparing detector metrics"""
    detector_types = results['detector_types']
    individual_results = results['individual_results']
    
    # Compute average metrics for each detector
    avg_means = []
    avg_stds = []
    avg_times = []
    
    for detector in detector_types:
        means = []
        stds = []
        times = []
        
        for img in individual_results:
            if detector in img['detector_statistics']:
                stats = img['detector_statistics'][detector]
                means.append(stats['mean'])
                stds.append(stats['std'])
                times.append(stats['measurement_time_s'])
        
        avg_means.append(np.mean(means) if means else 0)
        avg_stds.append(np.mean(stds) if stds else 0)
        avg_times.append(np.mean(times) if times else 0)
    
    # Normalize for comparison
    norm_means = [m / (max(avg_means) + 1e-10) for m in avg_means]
    norm_times = [t / (max(avg_times) + 1e-10) for t in avg_times]
    norm_stds = [s / (max(avg_stds) + 1e-10) for s in avg_stds]
    
    x = np.arange(len(detector_types))
    width = 0.25
    
    bars1 = ax.bar(x - width, norm_means, width, label='Signal (norm)', 
                  color='#4ECDC4', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, norm_times, width, label='Time (norm)', 
                  color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, norm_stds, width, label='Noise (norm)', 
                  color='#FFA07A', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Detector', fontsize=10, fontweight='bold')
    ax.set_ylabel('Normalized Value', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('_', '\n') for d in detector_types], 
                       fontsize=7, rotation=0)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.2)


def create_time_analysis(ax: plt.Axes, results: Dict, title: str):
    """Analyze measurement times across detectors"""
    detector_types = results['detector_types']
    individual_results = results['individual_results']
    
    # Collect times for each detector
    times_per_detector = {det: [] for det in detector_types}
    
    for img in individual_results:
        for detector in detector_types:
            if detector in img['detector_statistics']:
                times_per_detector[detector].append(
                    img['detector_statistics'][detector]['measurement_time_s']
                )
    
    # Box plot
    data_for_plot = [times_per_detector[det] for det in detector_types]
    labels = [d.replace('_', '\n') for d in detector_types]
    
    bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True,
                   showfliers=True)
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(detector_types)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Measurement Time (s)', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=7)


def create_revolutionary_advantage_visual(ax: plt.Axes, results: Dict, title: str):
    """Visualize the revolutionary advantage"""
    agg = results['aggregate_statistics']
    
    traditional = agg['traditional_samples_required']
    ours = agg['our_samples_required']
    savings = agg['sample_savings']
    
    # Create comparison bars
    categories = ['Traditional\nApproach', 'Our\nApproach']
    values = [traditional, ours]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val}\nsamples', ha='center', va='bottom', 
               fontsize=12, fontweight='bold')
    
    # Add savings annotation
    ax.annotate('', xy=(1, traditional), xytext=(1, ours),
               arrowprops=dict(arrowstyle='<->', color='green', lw=3))
    ax.text(1.3, (traditional + ours) / 2, 
           f'Savings:\n{savings}\nsamples\n({savings/traditional*100:.0f}%)',
           fontsize=11, fontweight='bold', color='green',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.set_ylabel('Samples Required', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, traditional * 1.3)
    ax.grid(axis='y', alpha=0.3)


def create_consistency_heatmap(ax: plt.Axes, results: Dict, title: str):
    """Create heatmap showing consistency across images"""
    detector_types = results['detector_types']
    individual_results = results['individual_results']
    
    # Build matrix: detectors × images
    n_detectors = len(detector_types)
    n_images = len(individual_results)
    
    consistency_matrix = np.zeros((n_detectors, n_images))
    
    for i, detector in enumerate(detector_types):
        means_for_detector = []
        for j, img in enumerate(individual_results):
            if detector in img['detector_statistics']:
                means_for_detector.append(img['detector_statistics'][detector]['mean'])
        
        if means_for_detector:
            # Normalized standard deviation as consistency metric
            consistency = 1.0 - (np.std(means_for_detector) / (np.mean(means_for_detector) + 1e-10))
            consistency_matrix[i, :] = consistency
    
    # Plot
    im = ax.imshow(consistency_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_yticks(np.arange(n_detectors))
    ax.set_yticklabels([d.replace('_', ' ') for d in detector_types], fontsize=8)
    ax.set_xticks(np.arange(n_images))
    ax.set_xticklabels([f'Img {i+1}' for i in range(n_images)], fontsize=8)
    ax.set_xlabel('Image', fontsize=10, fontweight='bold')
    ax.set_ylabel('Detector', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Consistency', fontsize=9)
    
    # Add values
    for i in range(n_detectors):
        for j in range(n_images):
            text = ax.text(j, i, f'{consistency_matrix[i, j]:.2f}',
                         ha="center", va="center", 
                         color="white" if consistency_matrix[i, j] < 0.5 else "black",
                         fontsize=7)


def create_comprehensive_panel(results: Dict, output_path: Path):
    """Create comprehensive 4×4 panel chart"""
    print("  Creating comprehensive multi-modal detector panel...")
    
    detector_types = results['detector_types']
    individual_results = results['individual_results']
    
    fig = plt.figure(figsize=(22, 22))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)
    
    # Row 1: Detector radar charts (first 4 detectors)
    for idx in range(min(4, len(detector_types))):
        ax = fig.add_subplot(gs[0, idx], projection='polar')
        create_detector_radar_chart(ax, detector_types[idx], individual_results,
                                    f'{detector_types[idx].replace("_", " ")} Performance')
    
    # Row 2: Detector radar charts (next 4 detectors) 
    for idx in range(4, min(8, len(detector_types))):
        ax = fig.add_subplot(gs[1, idx-4], projection='polar')
        create_detector_radar_chart(ax, detector_types[idx], individual_results,
                                    f'{detector_types[idx].replace("_", " ")} Performance')
    
    # Row 3: EM spectrum representations (first 4)
    for idx in range(min(4, len(detector_types))):
        ax = fig.add_subplot(gs[2, idx], projection='polar')
        create_em_spectrum_radar(ax, detector_types[idx],
                                f'{detector_types[idx].replace("_", " ")}\nEM Spectrum')
    
    # Row 4: Comparative analysis
    ax_comp = fig.add_subplot(gs[3, 0])
    create_detector_comparison_bars(ax_comp, results, 'Detector Comparison')
    
    ax_time = fig.add_subplot(gs[3, 1])
    create_time_analysis(ax_time, results, 'Measurement Times')
    
    ax_adv = fig.add_subplot(gs[3, 2])
    create_revolutionary_advantage_visual(ax_adv, results, 'Revolutionary Advantage')
    
    ax_cons = fig.add_subplot(gs[3, 3])
    create_consistency_heatmap(ax_cons, results, 'Cross-Image Consistency')
    
    # Main title
    fig.suptitle('Multi-Modal Detector Analysis with EM Spectrum Mapping',
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path}")


def main():
    """Main processing function"""
    print("=" * 80)
    print("MULTI-MODAL DETECTOR VISUALIZATION")
    print("=" * 80)
    
    # Load JSON data
    json_path = Path("../maxwell/multi_modal_validation/complete_multi_modal_results.json")
    if not json_path.exists():
        json_path = Path("maxwell/multi_modal_validation/complete_multi_modal_results.json")
    
    if not json_path.exists():
        print(f"✗ File not found: {json_path}")
        return
    
    print(f"\nLoading: {json_path}")
    results = load_multi_modal_results(json_path)
    
    print(f"  Total images: {results['aggregate_statistics']['total_images']}")
    print(f"  Detectors: {len(results['detector_types'])}")
    for detector in results['detector_types']:
        print(f"    • {detector}")
    
    # Create output directory
    output_dir = Path("multi_modal_detector_panels")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comprehensive panel
    print("\nGenerating panel charts...")
    create_comprehensive_panel(results, 
                               output_dir / 'multi_modal_detector_analysis.png')
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput: {output_dir}/multi_modal_detector_analysis.png")
    print("\nPanel includes:")
    print("  • Row 1-2: Radar charts for each detector (performance metrics)")
    print("  • Row 3: EM spectrum sensitivity (shows which wavelengths each detects)")
    print("  • Row 4: Comparative analysis (bars, times, advantage, consistency)")
    print("=" * 80)


if __name__ == '__main__':
    main()

