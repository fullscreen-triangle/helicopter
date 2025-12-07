#!/usr/bin/env python3
"""
Visualize Categorical Depth Results

Creates comprehensive visualizations of categorical depth from dual-membrane structures:
- 3D depth maps
- Cross-sections
- Statistical distributions  
- EM spectrum correlation
- Depth-based segmentation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy import stats
import matplotlib.patches as mpatches


def load_depth_data(npy_path: Path) -> np.ndarray:
    """Load categorical depth NPY file"""
    return np.load(npy_path)


def create_3d_surface_plot(ax: Axes3D, depth_data: np.ndarray, title: str):
    """Create 3D surface plot of depth"""
    h, w = depth_data.shape
    
    # Subsample for performance
    step = max(1, h // 100)
    depth_sub = depth_data[::step, ::step]
    h_sub, w_sub = depth_sub.shape
    
    # Create meshgrid
    x = np.linspace(0, w, w_sub)
    y = np.linspace(0, h, h_sub)
    X, Y = np.meshgrid(x, y)
    
    # Plot surface
    surf = ax.plot_surface(X, Y, depth_sub, cmap='viridis',
                          linewidth=0, antialiased=True, alpha=0.9)
    
    ax.set_xlabel('X (pixels)', fontsize=9)
    ax.set_ylabel('Y (pixels)', fontsize=9)
    ax.set_zlabel('Categorical Depth', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    # Colorbar
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)


def create_depth_heatmap(ax: plt.Axes, depth_data: np.ndarray, title: str):
    """Create 2D heatmap of depth"""
    im = ax.imshow(depth_data, cmap='plasma', aspect='auto')
    ax.set_xlabel('X (pixels)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y (pixels)', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Depth', fontsize=9)


def create_depth_histogram(ax: plt.Axes, depth_data: np.ndarray, title: str):
    """Create histogram of depth values"""
    flat_depth = depth_data.flatten()
    
    # Histogram
    n, bins, patches = ax.hist(flat_depth, bins=50, color='steelblue', 
                               alpha=0.7, edgecolor='black')
    
    # Color bars by value
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm.plasma(c))
    
    # Add statistics
    mean_depth = np.mean(flat_depth)
    median_depth = np.median(flat_depth)
    std_depth = np.std(flat_depth)
    
    ax.axvline(mean_depth, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_depth:.3f}')
    ax.axvline(median_depth, color='green', linestyle='--', linewidth=2, label=f'Median: {median_depth:.3f}')
    
    ax.set_xlabel('Categorical Depth', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)


def create_cross_sections(ax: plt.Axes, depth_data: np.ndarray, title: str):
    """Create horizontal and vertical cross-sections"""
    h, w = depth_data.shape
    
    # Horizontal cross-section (middle row)
    h_section = depth_data[h//2, :]
    # Vertical cross-section (middle column)
    v_section = depth_data[:, w//2]
    
    x_h = np.arange(w)
    x_v = np.arange(h)
    
    ax.plot(x_h, h_section, 'b-', linewidth=2, label='Horizontal (middle row)', alpha=0.7)
    ax.plot(x_v, v_section, 'r-', linewidth=2, label='Vertical (middle col)', alpha=0.7)
    
    ax.set_xlabel('Position (pixels)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Categorical Depth', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def create_depth_gradient_map(ax: plt.Axes, depth_data: np.ndarray, title: str):
    """Create depth gradient magnitude map"""
    # Compute gradients
    grad_y, grad_x = np.gradient(depth_data)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    im = ax.imshow(gradient_magnitude, cmap='hot', aspect='auto')
    ax.set_xlabel('X (pixels)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y (pixels)', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Gradient Magnitude', fontsize=9)


def create_depth_segmentation(ax: plt.Axes, depth_data: np.ndarray, title: str):
    """Segment depth into layers"""
    flat_depth = depth_data.flatten()
    
    # Quantize into layers
    n_layers = 5
    percentiles = np.linspace(0, 100, n_layers + 1)
    thresholds = np.percentile(flat_depth, percentiles)
    
    # Create segmented image
    segmented = np.zeros_like(depth_data, dtype=int)
    for i in range(n_layers):
        mask = (depth_data >= thresholds[i]) & (depth_data < thresholds[i+1])
        segmented[mask] = i
    
    # Plot with discrete colors
    cmap = plt.cm.get_cmap('Set3', n_layers)
    im = ax.imshow(segmented, cmap=cmap, aspect='auto', vmin=0, vmax=n_layers-1)
    
    ax.set_xlabel('X (pixels)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y (pixels)', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Create legend
    legend_elements = [mpatches.Patch(facecolor=cmap(i), label=f'Layer {i+1}')
                      for i in range(n_layers)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
             bbox_to_anchor=(1.15, 1))


def create_depth_contour(ax: plt.Axes, depth_data: np.ndarray, title: str):
    """Create contour plot of depth"""
    h, w = depth_data.shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    # Smooth for better contours
    depth_smooth = gaussian_filter(depth_data, sigma=2.0)
    
    # Contour plot
    levels = 15
    contour = ax.contourf(X, Y, depth_smooth, levels=levels, cmap='twilight', alpha=0.8)
    contour_lines = ax.contour(X, Y, depth_smooth, levels=levels, colors='black',
                               linewidths=0.5, alpha=0.5)
    
    ax.set_xlabel('X (pixels)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y (pixels)', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Depth', fontsize=9)


def create_em_spectrum_depth_correlation(ax: plt.Axes, depth_data: np.ndarray, title: str):
    """
    Correlate depth with EM spectrum (conceptual)
    
    Deeper structures might correlate with longer wavelengths (penetration)
    """
    # EM spectrum regions (wavelength in nm)
    em_regions = ['UV\n(400)', 'Blue\n(450)', 'Green\n(550)', 'Red\n(650)', 
                  'Near-IR\n(800)', 'Mid-IR\n(2500)']
    
    # Wavelengths in nm
    wavelengths = [400, 450, 550, 650, 800, 2500]
    
    # Conceptual correlation: deeper = longer wavelength penetration
    flat_depth = depth_data.flatten()
    depth_normalized = (flat_depth - flat_depth.min()) / (flat_depth.max() - flat_depth.min() + 1e-10)
    
    # Simulate penetration for each wavelength
    penetrations = []
    for wl in wavelengths:
        # Longer wavelengths penetrate deeper
        penetration_factor = np.log(wl) / np.log(max(wavelengths))
        # Fraction of depth values that would be visible at this wavelength
        visible_fraction = np.mean(depth_normalized <= penetration_factor)
        penetrations.append(visible_fraction * 100)
    
    # Bar chart
    colors = ['#8B00FF', '#0000FF', '#00FF00', '#FF0000', '#8B0000', '#FF6347']
    bars = ax.bar(range(len(em_regions)), penetrations, color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_xticks(range(len(em_regions)))
    ax.set_xticklabels(em_regions, fontsize=9)
    ax.set_ylabel('Depth Penetration (%)', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, penetrations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=8)


def create_depth_statistics_table(ax: plt.Axes, depth_data: np.ndarray, title: str):
    """Create table with depth statistics"""
    ax.axis('off')
    
    flat_depth = depth_data.flatten()
    
    # Compute statistics
    stats_dict = {
        'Mean': np.mean(flat_depth),
        'Median': np.median(flat_depth),
        'Std Dev': np.std(flat_depth),
        'Min': np.min(flat_depth),
        'Max': np.max(flat_depth),
        'Range': np.max(flat_depth) - np.min(flat_depth),
        'Q25': np.percentile(flat_depth, 25),
        'Q75': np.percentile(flat_depth, 75),
        'Skewness': stats.skew(flat_depth),
        'Kurtosis': stats.kurtosis(flat_depth)
    }
    
    # Create table
    table_data = [[metric, f'{value:.6f}'] for metric, value in stats_dict.items()]
    
    table = ax.table(cellText=table_data,
                    colLabels=['Statistic', 'Value'],
                    cellLoc='left',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats_dict) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=20)


def create_cumulative_depth_distribution(ax: plt.Axes, depth_data: np.ndarray, title: str):
    """Create cumulative distribution of depth"""
    flat_depth = depth_data.flatten()
    sorted_depth = np.sort(flat_depth)
    cdf = np.arange(1, len(sorted_depth) + 1) / len(sorted_depth)
    
    ax.plot(sorted_depth, cdf, 'b-', linewidth=2)
    ax.fill_between(sorted_depth, cdf, alpha=0.3)
    
    ax.set_xlabel('Categorical Depth', fontsize=10, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add quartile lines
    for q, label in [(0.25, 'Q1'), (0.5, 'Median'), (0.75, 'Q3')]:
        val = np.percentile(flat_depth, q * 100)
        ax.axvline(val, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(val, q, label, fontsize=8, va='center', ha='right')


def create_depth_radial_profile(ax: plt.Axes, depth_data: np.ndarray, title: str):
    """Create radial profile from center"""
    h, w = depth_data.shape
    center_y, center_x = h // 2, w // 2
    
    # Create distance map from center
    y, x = np.indices((h, w))
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r = r.astype(int)
    
    # Compute radial average
    max_r = min(center_x, center_y)
    radial_bins = np.arange(0, max_r)
    radial_profile = np.zeros(len(radial_bins))
    
    for i, radius in enumerate(radial_bins):
        mask = (r >= radius) & (r < radius + 1)
        if np.sum(mask) > 0:
            radial_profile[i] = np.mean(depth_data[mask])
    
    ax.plot(radial_bins, radial_profile, 'purple', linewidth=2)
    ax.fill_between(radial_bins, radial_profile, alpha=0.3, color='purple')
    
    ax.set_xlabel('Radial Distance from Center (pixels)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Average Depth', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)


def create_comprehensive_panel(depth_data: np.ndarray, output_path: Path):
    """Create comprehensive 4×3 panel with 3D plot"""
    print("  Creating comprehensive categorical depth panel...")
    
    fig = plt.figure(figsize=(24, 18))
    
    # Create grid: 4 rows × 3 cols, with 3D plot spanning 2 cols
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: 3D surface (spanning 2 cols) + depth heatmap
    ax_3d = fig.add_subplot(gs[0, :2], projection='3d')
    create_3d_surface_plot(ax_3d, depth_data, '3D Categorical Depth Surface')
    
    ax_heatmap = fig.add_subplot(gs[0, 2])
    create_depth_heatmap(ax_heatmap, depth_data, 'Depth Heatmap')
    
    # Row 2: Histogram, Cross-sections, Gradient
    ax_hist = fig.add_subplot(gs[1, 0])
    create_depth_histogram(ax_hist, depth_data, 'Depth Distribution')
    
    ax_cross = fig.add_subplot(gs[1, 1])
    create_cross_sections(ax_cross, depth_data, 'Cross-Sections')
    
    ax_grad = fig.add_subplot(gs[1, 2])
    create_depth_gradient_map(ax_grad, depth_data, 'Depth Gradient')
    
    # Row 3: Segmentation, Contour, EM Correlation
    ax_seg = fig.add_subplot(gs[2, 0])
    create_depth_segmentation(ax_seg, depth_data, 'Depth Layers')
    
    ax_contour = fig.add_subplot(gs[2, 1])
    create_depth_contour(ax_contour, depth_data, 'Depth Contours')
    
    ax_em = fig.add_subplot(gs[2, 2])
    create_em_spectrum_depth_correlation(ax_em, depth_data, 
                                        'EM Wavelength Penetration')
    
    # Row 4: CDF, Radial Profile, Statistics Table
    ax_cdf = fig.add_subplot(gs[3, 0])
    create_cumulative_depth_distribution(ax_cdf, depth_data, 
                                        'Cumulative Distribution')
    
    ax_radial = fig.add_subplot(gs[3, 1])
    create_depth_radial_profile(ax_radial, depth_data, 'Radial Depth Profile')
    
    ax_table = fig.add_subplot(gs[3, 2])
    create_depth_statistics_table(ax_table, depth_data, 'Statistical Summary')
    
    # Main title
    fig.suptitle('Categorical Depth Analysis from Dual-Membrane Structure',
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path}")


def main():
    """Main processing function"""
    print("=" * 80)
    print("CATEGORICAL DEPTH VISUALIZATION")
    print("=" * 80)
    
    # Find depth NPY file
    npy_path = Path("../maxwell/demo_complete_results/categorical_depth.npy")
    if not npy_path.exists():
        npy_path = Path("maxwell/demo_complete_results/categorical_depth.npy")
    
    if not npy_path.exists():
        print(f"✗ File not found: {npy_path}")
        return
    
    print(f"\nLoading: {npy_path}")
    depth_data = load_depth_data(npy_path)
    
    print(f"  Shape: {depth_data.shape}")
    print(f"  Depth range: [{np.min(depth_data):.6f}, {np.max(depth_data):.6f}]")
    print(f"  Mean depth: {np.mean(depth_data):.6f}")
    print(f"  Std depth: {np.std(depth_data):.6f}")
    
    # Create output directory
    output_dir = Path("categorical_depth_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comprehensive panel
    print("\nGenerating comprehensive panel...")
    create_comprehensive_panel(depth_data, 
                               output_dir / 'categorical_depth_analysis.png')
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput: {output_dir}/categorical_depth_analysis.png")
    print("\nPanel includes:")
    print("  • 3D surface plot showing depth topography")
    print("  • 2D heatmap and gradient map")
    print("  • Statistical distributions (histogram, CDF)")
    print("  • Cross-sections and radial profiles")
    print("  • Depth segmentation into layers")
    print("  • Contour visualization")
    print("  • EM spectrum penetration correlation")
    print("  • Comprehensive statistical table")
    print("=" * 80)


if __name__ == '__main__':
    main()

