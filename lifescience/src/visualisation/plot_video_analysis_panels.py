# plot_video_analysis_panels.py
"""
Video Analysis Comprehensive Panels
Creates multi-panel figures for video tracking and behavioral analysis
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from visualisation_setup import *


def plot_video_analysis_comprehensive(data_files, output_name="video_analysis"):
    """
    Create comprehensive multi-panel figure for video/tracking data

    Parameters:
    -----------
    data_files : list of str or Path
        List of JSON file paths for video analysis
    output_name : str or Path
        Base name for output files
    """

    # Load all data
    datasets = [load_json(f) for f in data_files]

    # Extract sample names from file paths
    labels = [Path(f).stem.split('_')[0] for f in data_files]

    # Create figure with optimized layout
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.45)

    # Define color palette
    colors = plt.cm.Set3(np.linspace(0.1, 0.9, len(datasets)))

    # ========================================================================
    # Panel A: Tracking Performance Metrics
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0:2])

    metrics = ['tracking_accuracy', 'track_completeness',
               'false_positive_rate', 'false_negative_rate']
    metric_labels = ['Tracking\nAccuracy', 'Track\nCompleteness',
                     'False Positive\nRate', 'False Negative\nRate']
    x = np.arange(len(metrics))
    width = 0.7 / len(datasets)

    for i, (data, label) in enumerate(zip(datasets, labels)):
        values = [data.get(m, 0) for m in metrics]
        offset = (i - len(datasets) / 2 + 0.5) * width
        bars = ax1.bar(x + offset, values, width, label=label,
                       color=colors[i], alpha=0.85, edgecolor='black',
                       linewidth=1, zorder=3)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if height > 0.05:  # Only show if bar is visible
                ax1.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{val:.2f}', ha='center', va='bottom',
                         fontsize=6.5, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=8)
    ax1.set_ylabel('Score', fontsize=9)
    ax1.set_ylim([0, 1.15])
    ax1.set_title('A. Tracking Performance Metrics', fontweight='bold',
                  loc='left', fontsize=11)
    ax1.legend(frameon=False, ncol=len(datasets), fontsize=8, loc='upper right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1,
                alpha=0.4, zorder=1)

    # ========================================================================
    # Panel B: Mean Velocity Comparison
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 2:])

    velocities = [d.get('mean_velocity', 0) for d in datasets]
    velocity_stds = [d.get('velocity_std', 0) for d in datasets]

    y_pos = np.arange(len(labels))
    bars = ax2.barh(y_pos, velocities, xerr=velocity_stds if any(velocity_stds) else None,
                    color=colors, alpha=0.85, edgecolor='black', linewidth=1,
                    error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 4})

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel('Mean Velocity (pixels/frame)', fontsize=9)
    ax2.set_title('B. Mean Velocity Comparison', fontweight='bold',
                  loc='left', fontsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

    # Add value labels
    for bar, val, std in zip(bars, velocities, velocity_stds if any(velocity_stds) else [0] * len(velocities)):
        width = bar.get_width()
        label_text = f'{val:.2f}' if std == 0 else f'{val:.2f}±{std:.2f}'
        ax2.text(width + max(velocities) * 0.02, bar.get_y() + bar.get_height() / 2.,
                 label_text, ha='left', va='center', fontsize=7.5,
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='gray', alpha=0.9, linewidth=0.8))

    # ========================================================================
    # Panels C-E: Behavior Distribution (Pie charts)
    # ========================================================================
    behavior_axes = [fig.add_subplot(gs[1, i]) for i in range(min(4, len(datasets)))]

    for idx, (data, label, ax) in enumerate(zip(datasets, labels, behavior_axes)):
        if 'behavior_distribution' not in data:
            ax.text(0.5, 0.5, 'No behavior\ndata available',
                    ha='center', va='center', fontsize=9)
            ax.set_title(f'{chr(67 + idx)}. {label} Behaviors',
                         fontweight='bold', fontsize=10)
            ax.axis('off')
            continue

        behaviors = data['behavior_distribution']
        behavior_labels = list(behaviors.keys())
        behavior_values = list(behaviors.values())

        # Filter out zero values
        non_zero = [(l, v) for l, v in zip(behavior_labels, behavior_values) if v > 0]

        if non_zero:
            labels_nz, values_nz = zip(*non_zero)
            colors_pie = plt.cm.Pastel1(np.linspace(0, 1, len(labels_nz)))

            wedges, texts, autotexts = ax.pie(values_nz, labels=labels_nz,
                                              autopct='%1.1f%%',
                                              colors=colors_pie,
                                              startangle=90,
                                              textprops={'fontsize': 7.5},
                                              wedgeprops={'edgecolor': 'black',
                                                          'linewidth': 1,
                                                          'antialiased': True})

            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontsize(7.5)
                autotext.set_weight('bold')

            for text in texts:
                text.set_fontsize(7.5)
        else:
            ax.text(0.5, 0.5, 'No behaviors\ndetected',
                    ha='center', va='center', fontsize=9)
            ax.axis('off')

        ax.set_title(f'{chr(67 + idx)}. {label} Behavior Distribution',
                     fontweight='bold', fontsize=10)

    # Hide unused behavior axes
    for i in range(len(datasets), min(4, len(behavior_axes))):
        behavior_axes[i].axis('off')

    # ========================================================================
    # Panels F-H: Activity Over Time
    # ========================================================================
    activity_axes = [fig.add_subplot(gs[2, i]) for i in range(min(3, len(datasets)))]

    for idx, (data, label, ax, color) in enumerate(zip(datasets[:3], labels[:3],
                                                       activity_axes, colors[:3])):
        if 'activity_over_time' not in data:
            ax.text(0.5, 0.5, 'No activity data', ha='center', va='center')
            ax.set_title(f'{chr(70 + idx)}. {label} Activity',
                         fontweight='bold', loc='left', fontsize=10)
            continue

        activity = np.array(data['activity_over_time'])
        frames = np.arange(len(activity))

        # Plot activity with smoothing
        ax.plot(frames, activity, color=color, linewidth=2, alpha=0.8,
                label='Activity', zorder=3)
        ax.fill_between(frames, activity, alpha=0.25, color=color, zorder=2)

        # Add mean line
        mean_activity = np.mean(activity)
        ax.axhline(mean_activity, color='red', linestyle='--',
                   linewidth=1.5, alpha=0.6, label=f'Mean: {mean_activity:.1f}',
                   zorder=4)

        # Mark peak activity frames
        if 'peak_activity_frames' in data:
            peak_frames = data['peak_activity_frames']
            peak_values = [activity[f] for f in peak_frames if f < len(activity)]
            peak_frames_valid = [f for f in peak_frames if f < len(activity)]
            if peak_frames_valid:
                ax.scatter(peak_frames_valid, peak_values, color='red',
                           s=50, zorder=5, marker='*', edgecolors='black',
                           linewidths=1, label=f'{len(peak_frames_valid)} peaks')

        ax.set_xlabel('Frame Number', fontsize=9)
        ax.set_ylabel('Activity Level', fontsize=9)
        ax.set_title(f'{chr(70 + idx)}. {label} Temporal Activity',
                     fontweight='bold', loc='left', fontsize=10)
        ax.legend(frameon=False, fontsize=7, loc='best')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

        # Add statistics box
        if len(activity) > 0:
            stats_text = (f'Max: {np.max(activity):.1f}\n'
                          f'Min: {np.min(activity):.1f}\n'
                          f'SD: {np.std(activity):.1f}')
            ax.text(0.98, 0.98, stats_text,
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                              edgecolor='gray', alpha=0.9, linewidth=1),
                    fontsize=7, family='monospace')

    # ========================================================================
    # Panel I: Velocity Distribution Histogram (combined)
    # ========================================================================
    ax_vel = fig.add_subplot(gs[2, 3])

    for data, label, color in zip(datasets, labels, colors):
        if 'velocity_distribution' not in data:
            continue

        vel_dist = data['velocity_distribution']
        # Remove zeros and outliers for cleaner visualization
        vel_nonzero = [v for v in vel_dist if 0 < v < np.percentile(vel_dist, 99)]

        if vel_nonzero:
            ax_vel.hist(vel_nonzero, bins=25, alpha=0.6, label=label,
                        color=color, edgecolor='black', linewidth=0.8,
                        density=True)

    ax_vel.set_xlabel('Velocity (pixels/frame)', fontsize=9)
    ax_vel.set_ylabel('Probability Density', fontsize=9)
    ax_vel.set_title('I. Velocity Distribution Comparison',
                     fontweight='bold', loc='left', fontsize=10)
    ax_vel.legend(frameon=False, fontsize=7.5, loc='best')
    ax_vel.spines['top'].set_visible(False)
    ax_vel.spines['right'].set_visible(False)
    ax_vel.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Add overall figure title
    fig.suptitle('Comprehensive Video Tracking and Behavioral Analysis',
                 fontsize=14, fontweight='bold', y=0.995)

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
    output_path = output_dir / "figure2_video_tracking_analysis"

    # Generate figure
    plot_video_analysis_comprehensive(files, output_path)
