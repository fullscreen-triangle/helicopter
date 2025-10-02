# plot_video_tracking_detailed.py
"""
Detailed Video Tracking Analysis
Creates comprehensive tracking analysis with trajectories, velocities, and behavioral patterns
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from scipy import stats
from visualisation_setup import *


def plot_video_tracking_detailed(video_files, output_name="video_tracking_detailed"):
    """
    Create detailed video tracking analysis with trajectory and behavior analysis

    Parameters:
    -----------
    video_files : list of str or Path
        List of JSON file paths for video tracking data
    output_name : str or Path
        Base name for output files
    """

    # Load data
    datasets = [load_json(f) for f in video_files]
    labels = [Path(f).stem.split('_')[0] for f in video_files]

    n_videos = len(datasets)

    # Create figure with optimized layout
    fig = plt.figure(figsize=(17, 4.5 * n_videos))
    gs = gridspec.GridSpec(n_videos, 4, figure=fig, hspace=0.45, wspace=0.4)

    colors = plt.cm.Set3(np.linspace(0.1, 0.9, n_videos))

    for vid_idx, (data, label, color) in enumerate(zip(datasets, labels, colors)):

        # ====================================================================
        # Panel A: Activity Heatmap Over Time
        # ====================================================================
        ax1 = fig.add_subplot(gs[vid_idx, 0])

        if 'activity_over_time' not in data:
            ax1.text(0.5, 0.5, 'No activity data', ha='center', va='center')
            ax1.set_title(f'{label}: Activity Heatmap', fontweight='bold', loc='left')
            ax1.axis('off')
        else:
            activity = np.array(data['activity_over_time'])
            frames = np.arange(len(activity))

            # Create a 2D representation for heatmap effect
            activity_2d = np.tile(activity, (15, 1))

            im = ax1.imshow(activity_2d, cmap='hot', aspect='auto',
                            interpolation='bilinear',
                            extent=[0, len(activity), 0, 1],
                            vmin=0, vmax=np.percentile(activity, 95))

            ax1.set_xlabel('Frame Number', fontsize=9)
            ax1.set_ylabel('Activity\nIntensity', fontsize=9, rotation=0,
                           ha='right', va='center')
            ax1.set_yticks([])
            ax1.set_title(f'{label}: Temporal Activity Heatmap',
                          fontweight='bold', loc='left', fontsize=10)

            # Add colorbar with better formatting
            cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            cbar.set_label('Activity Level', rotation=270, labelpad=18, fontsize=8)
            cbar.ax.tick_params(labelsize=7)

            # Add statistics overlay
            mean_activity = np.mean(activity)
            max_activity = np.max(activity)
            stats_text = f'Mean: {mean_activity:.1f}\nMax: {max_activity:.1f}'
            ax1.text(0.02, 0.98, stats_text,
                     transform=ax1.transAxes, ha='left', va='top',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                               edgecolor='black', alpha=0.9, linewidth=1),
                     fontsize=7, family='monospace', fontweight='bold')

        # ====================================================================
        # Panel B: Velocity Distribution with Statistics
        # ====================================================================
        ax2 = fig.add_subplot(gs[vid_idx, 1])

        if 'velocity_distribution' not in data:
            ax2.text(0.5, 0.5, 'No velocity data', ha='center', va='center')
            ax2.set_title(f'{label}: Velocity Distribution',
                          fontweight='bold', loc='left')
            ax2.axis('off')
        else:
            vel_dist = np.array(data['velocity_distribution'])
            # Remove zeros and extreme outliers
            vel_nonzero = vel_dist[(vel_dist > 0) & (vel_dist < np.percentile(vel_dist, 99))]

            if len(vel_nonzero) > 0:
                # Create histogram
                n, bins, patches = ax2.hist(vel_nonzero, bins=30, color=color,
                                            alpha=0.7, edgecolor='black',
                                            linewidth=0.8, density=True,
                                            label='Distribution')

                # Fit and overlay normal distribution
                mu, sigma = stats.norm.fit(vel_nonzero)
                x = np.linspace(vel_nonzero.min(), vel_nonzero.max(), 100)
                ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-',
                         linewidth=2.5, label=f'Normal fit\n(μ={mu:.1f}, σ={sigma:.1f})',
                         alpha=0.8)

                # Add statistics lines
                mean_vel = np.mean(vel_nonzero)
                median_vel = np.median(vel_nonzero)

                ax2.axvline(mean_vel, color='darkred', linestyle='--',
                            linewidth=2, label=f'Mean: {mean_vel:.2f}',
                            alpha=0.8, zorder=5)
                ax2.axvline(median_vel, color='darkblue', linestyle=':',
                            linewidth=2, label=f'Median: {median_vel:.2f}',
                            alpha=0.8, zorder=5)

                # Add quartile shading
                q1, q3 = np.percentile(vel_nonzero, [25, 75])
                ax2.axvspan(q1, q3, alpha=0.15, color='green',
                            label='IQR', zorder=1)

            ax2.set_xlabel('Velocity (pixels/frame)', fontsize=9)
            ax2.set_ylabel('Probability Density', fontsize=9)
            ax2.set_title(f'{label}: Velocity Distribution & Statistics',
                          fontweight='bold', loc='left', fontsize=10)
            ax2.legend(frameon=False, fontsize=6.5, loc='best')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

            # Add skewness and kurtosis
            if len(vel_nonzero) > 3:
                skew = stats.skew(vel_nonzero)
                kurt = stats.kurtosis(vel_nonzero)
                stats_text = f'Skew: {skew:.2f}\nKurt: {kurt:.2f}'
                ax2.text(0.98, 0.98, stats_text,
                         transform=ax2.transAxes, ha='right', va='top',
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                   edgecolor='gray', alpha=0.9, linewidth=1),
                         fontsize=7, family='monospace')

        # ====================================================================
        # Panel C: Displacement Metrics
        # ====================================================================
        ax3 = fig.add_subplot(gs[vid_idx, 2])

        if 'displacement_metrics' not in data:
            ax3.text(0.5, 0.5, 'No displacement data', ha='center', va='center')
            ax3.set_title(f'{label}: Displacement Metrics',
                          fontweight='bold', loc='left')
            ax3.axis('off')
        else:
            disp_metrics = data['displacement_metrics']

            metric_names = ['Mean\nDisplacement', 'Max\nDisplacement',
                            'Std Dev\nDisplacement', 'Total Path\nLength']

            # Extract values with fallbacks
            metric_values = [
                disp_metrics.get('mean_displacement', 0),
                disp_metrics.get('max_displacement', 0),
                np.sqrt(disp_metrics.get('displacement_variance', 0)),
                disp_metrics.get('total_path_length',
                                 disp_metrics.get('mean_displacement', 0) * 10)
            ]

            x_pos = np.arange(len(metric_names))
            bars = ax3.bar(x_pos, metric_values, color=color, alpha=0.8,
                           edgecolor='black', linewidth=1.2, zorder=3)

            # Color bars by magnitude
            max_val = max(metric_values)
            for bar, val in zip(bars, metric_values):
                bar.set_alpha(0.5 + 0.5 * (val / max_val))

            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(metric_names, fontsize=8)
            ax3.set_ylabel('Distance (pixels)', fontsize=9)
            ax3.set_title(f'{label}: Displacement Metrics Analysis',
                          fontweight='bold', loc='left', fontsize=10)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

            # Add value labels with better positioning
            for bar, val in zip(bars, metric_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{val:.1f}', ha='center', va='bottom',
                         fontsize=8, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                   edgecolor='none', alpha=0.8))

            # Add efficiency metric
            if metric_values[3] > 0:  # Total path length
                efficiency = (metric_values[0] / metric_values[3]) * 100
                ax3.text(0.98, 0.98, f'Efficiency:\n{efficiency:.1f}%',
                         transform=ax3.transAxes, ha='right', va='top',
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                                   edgecolor='orange', alpha=0.9, linewidth=1.5),
                         fontsize=7.5, fontweight='bold')

        # ====================================================================
        # Panel D: Behavior Distribution (Enhanced Pie Chart)
        # ====================================================================
        ax4 = fig.add_subplot(gs[vid_idx, 3])

        if 'behavior_distribution' not in data:
            ax4.text(0.5, 0.5, 'No behavior data', ha='center', va='center')
            ax4.set_title(f'{label}: Behavior Distribution', fontweight='bold')
            ax4.axis('off')
        else:
            behaviors = data['behavior_distribution']
            behavior_names = list(behaviors.keys())
            behavior_values = list(behaviors.values())

            # Filter out zero values
            non_zero = [(n, v) for n, v in zip(behavior_names, behavior_values) if v > 0]

            if non_zero:
                behavior_names_nz, behavior_values_nz = zip(*non_zero)

                # Enhanced color scheme for behaviors
                behavior_colors = {
                    'stationary': '#95a5a6',
                    'migrating': '#3498db',
                    'oscillating': '#e74c3c',
                    'dividing': '#2ecc71',
                    'interacting': '#f39c12',
                    'rotating': '#9b59b6',
                    'moving': '#1abc9c',
                    'paused': '#34495e'
                }

                colors_behav = [behavior_colors.get(b.lower(), '#7f8c8d')
                                for b in behavior_names_nz]

                # Create exploded pie chart for emphasis
                explode = [0.05 if v == max(behavior_values_nz) else 0
                           for v in behavior_values_nz]

                wedges, texts, autotexts = ax4.pie(
                    behavior_values_nz,
                    labels=behavior_names_nz,
                    autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
                    colors=colors_behav,
                    startangle=90,
                    explode=explode,
                    shadow=True,
                    wedgeprops={'edgecolor': 'black', 'linewidth': 1.5,
                                'antialiased': True}
                )

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(8)
                    autotext.set_weight('bold')
                    autotext.set_path_effects([
                        plt.matplotlib.patheffects.withStroke(linewidth=2,
                                                              foreground='black')
                    ])

                for text in texts:
                    text.set_fontsize(8)
                    text.set_weight('bold')

                # Add count information
                total_count = sum(behavior_values_nz)
                ax4.text(0, -1.3, f'Total observations: {total_count}',
                         ha='center', va='top', fontsize=8,
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                   edgecolor='gray', alpha=0.9, linewidth=1))
            else:
                ax4.text(0.5, 0.5, 'No behaviors\ndetected',
                         ha='center', va='center', fontsize=10)
                ax4.axis('off')

            ax4.set_title(f'{label}: Behavioral Pattern Distribution',
                          fontweight='bold', fontsize=10)

    # Add overall figure title
    fig.suptitle('Detailed Video Tracking Analysis: Activity, Velocity & Behavior',
                 fontsize=14, fontweight='bold', y=0.998)

    save_figure(fig, output_name)
    plt.show()


if __name__ == "__main__":
    # Get project root (from src/visualisations/ to root)
    project_root = Path(__file__).parent.parent.parent

    # Define paths
    results_dir = project_root / "results"
    output_dir = project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    files = [
        results_dir / "7199_web_live_cell_comprehensive.json",
        results_dir / "astrosoma-g2s2_VOL_time_lapse_comprehensive.json",
        results_dir / "astrosoma-g3s10_vol_cell_migration_comprehensive.json"
    ]

    # Output path
    output_path = output_dir / "figure6_video_tracking_detailed"

    # Generate figure
    plot_video_tracking_detailed(files, output_path)
