# plot_video_tracking_overview.py
"""
Video Tracking Overview Figure
Compares tracking performance across multiple videos
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from visualisations_setup import *


def plot_video_tracking_overview(data_files, output_name="figure3_video_tracking_overview"):
    """
    Create comprehensive video tracking overview

    Parameters:
    -----------
    data_files : list of str or Path
        List of JSON file paths for video data
    output_name : str or Path
        Base name for output files
    """

    # Load all datasets
    datasets = [load_json(f) for f in data_files]
    labels = [Path(f).stem.replace('_comprehensive', '').replace('_', ' ').title()
              for f in data_files]
    n_datasets = len(datasets)

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, n_datasets))

    # ========================================================================
    # Panel A: Tracking Accuracy Comparison
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    accuracies = [data.get('tracking_accuracy', 0) for data in datasets]

    bars = ax1.bar(range(n_datasets), accuracies, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5, zorder=3)

    for bar, val in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

    ax1.set_xticks(range(n_datasets))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Tracking Accuracy', fontsize=10)
    ax1.set_title('A. Tracking Accuracy Comparison',
                  fontweight='bold', fontsize=11)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax1.set_ylim([0, 1.1])
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

    # ========================================================================
    # Panel B: Number of Tracks
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    num_tracks = [data.get('num_tracks', 0) for data in datasets]

    bars = ax2.bar(range(n_datasets), num_tracks, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5, zorder=3)

    for bar, val in zip(bars, num_tracks):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(val)}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

    ax2.set_xticks(range(n_datasets))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Number of Tracks', fontsize=10)
    ax2.set_title('B. Track Count Comparison',
                  fontweight='bold', fontsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)

    # ========================================================================
    # Panel C: Mean Velocity Comparison
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])

    mean_velocities = [data.get('mean_velocity', 0) for data in datasets]

    bars = ax3.bar(range(n_datasets), mean_velocities, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5, zorder=3)

    for bar, val in zip(bars, mean_velocities):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{val:.2f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

    ax3.set_xticks(range(n_datasets))
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Mean Velocity (pixels/frame)', fontsize=10)
    ax3.set_title('C. Mean Velocity Comparison',
                  fontweight='bold', fontsize=11)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)

    # ========================================================================
    # Panel D: Velocity Distribution Box Plot
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 0])

    velocity_data = []
    for data in datasets:
        if 'velocity_distribution' in data:
            vel = np.array(data['velocity_distribution'])
            vel_clean = vel[(vel > 0) & (vel < np.percentile(vel, 99))]
            velocity_data.append(vel_clean)
        else:
            velocity_data.append([0])

    bp = ax4.boxplot(velocity_data, labels=[l[:15] for l in labels],
                     patch_artist=True, widths=0.6, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red',
                                    markersize=6, markeredgecolor='black'),
                     medianprops=dict(color='black', linewidth=2),
                     boxprops=dict(linewidth=1.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax4.set_ylabel('Velocity (pixels/frame)', fontsize=10)
    ax4.set_title('D. Velocity Distribution',
                  fontweight='bold', fontsize=11)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    # ========================================================================
    # Panel E: Activity Over Time
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 1])

    for data, label, color in zip(datasets, labels, colors):
        if 'activity_over_time' in data:
            activity = np.array(data['activity_over_time'])
            if len(activity) > 0:
                frames = np.linspace(0, 100, len(activity))
                activity_norm = (activity / np.max(activity)) * 100 if np.max(activity) > 0 else activity
                ax5.plot(frames, activity_norm, linewidth=2.5, label=label[:20],
                         color=color, alpha=0.8)

    ax5.set_xlabel('Time Progress (%)', fontsize=10)
    ax5.set_ylabel('Normalized Activity (%)', fontsize=10)
    ax5.set_title('E. Temporal Activity Patterns',
                  fontweight='bold', fontsize=11)
    ax5.legend(frameon=False, fontsize=8, loc='best')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax5.set_xlim(0, 100)

    # ========================================================================
    # Panel F: Displacement Metrics
    # ========================================================================
    ax6 = fig.add_subplot(gs[1, 2])

    disp_categories = ['Mean', 'Max']
    x_disp = np.arange(len(disp_categories))
    width = 0.8 / n_datasets

    for i, (data, label, color) in enumerate(zip(datasets, labels, colors)):
        if 'displacement_metrics' in data:
            disp = data['displacement_metrics']
            values = [
                disp.get('mean_displacement', 0),
                disp.get('max_displacement', 0)
            ]
            offset = (i - n_datasets / 2 + 0.5) * width
            ax6.bar(x_disp + offset, values, width, label=label[:15],
                    color=color, alpha=0.8, edgecolor='black', linewidth=1)

    ax6.set_xticks(x_disp)
    ax6.set_xticklabels(disp_categories, fontsize=9)
    ax6.set_ylabel('Displacement (pixels)', fontsize=10)
    ax6.set_title('F. Displacement Metrics',
                  fontweight='bold', fontsize=11)
    ax6.legend(frameon=False, fontsize=7, loc='upper left')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # ========================================================================
    # Panel G: Frame Rate and Duration
    # ========================================================================
    ax7 = fig.add_subplot(gs[2, 0])

    fps_values = [data.get('fps', 0) for data in datasets]
    durations = [data.get('duration_seconds', 0) for data in datasets]

    x_pos = np.arange(n_datasets)
    width = 0.35

    bars1 = ax7.bar(x_pos - width / 2, fps_values, width, label='FPS',
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax7_twin = ax7.twinx()
    bars2 = ax7_twin.bar(x_pos + width / 2, durations, width, label='Duration (s)',
                         color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax7.set_xticks(x_pos)
    ax7.set_xticklabels([l[:15] for l in labels], rotation=45, ha='right', fontsize=8)
    ax7.set_ylabel('FPS', fontsize=10, color='#3498db')
    ax7_twin.set_ylabel('Duration (seconds)', fontsize=10, color='#e74c3c')
    ax7.set_title('G. Temporal Properties',
                  fontweight='bold', fontsize=11)
    ax7.tick_params(axis='y', labelcolor='#3498db')
    ax7_twin.tick_params(axis='y', labelcolor='#e74c3c')
    ax7.spines['top'].set_visible(False)
    ax7_twin.spines['top'].set_visible(False)
    ax7.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Combined legend
    lines1, labels1 = ax7.get_legend_handles_labels()
    lines2, labels2 = ax7_twin.get_legend_handles_labels()
    ax7.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8, loc='upper left')

    # ========================================================================
    # Panel H: Processing Efficiency
    # ========================================================================
    ax8 = fig.add_subplot(gs[2, 1])

    proc_times = [data.get('processing_time', 0) for data in datasets]
    frame_counts = [data.get('frame_count', 1) for data in datasets]
    efficiency = [t / f if f > 0 else 0 for t, f in zip(proc_times, frame_counts)]

    bars = ax8.bar(range(n_datasets), efficiency, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5, zorder=3)

    for bar, val in zip(bars, efficiency):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width() / 2., height + 0.002,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=8, fontweight='bold')

    ax8.set_xticks(range(n_datasets))
    ax8.set_xticklabels([l[:15] for l in labels], rotation=45, ha='right', fontsize=8)
    ax8.set_ylabel('Time per Frame (s/frame)', fontsize=10)
    ax8.set_title('H. Processing Efficiency',
                  fontweight='bold', fontsize=11)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    ax8.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)

    # ========================================================================
    # Panel I: Summary Statistics Table
    # ========================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    table_data = []
    for data, label in zip(datasets, labels):
        row = [
            label[:15],
            f"{data.get('num_tracks', 0)}",
            f"{data.get('tracking_accuracy', 0):.3f}",
            f"{data.get('mean_velocity', 0):.2f}",
            f"{data.get('frame_count', 0)}"
        ]
        table_data.append(row)

    headers = ['Video', 'Tracks', 'Accuracy', 'Vel.', 'Frames']

    table = ax9.table(cellText=table_data, colLabels=headers,
                      cellLoc='center', loc='center',
                      colWidths=[0.3, 0.15, 0.2, 0.15, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#2196F3')
        cell.set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_facecolor(colors[i - 1])
            cell.set_alpha(0.3)
            if j == 0:
                cell.set_text_props(weight='bold')

    ax9.set_title('I. Summary Statistics',
                  fontweight='bold', fontsize=11, pad=20)

    # Overall title
    fig.suptitle('Video Tracking Analysis: Comprehensive Overview',
                 fontsize=14, fontweight='bold', y=0.995)

    save_figure(fig, output_name)
    plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / "results"
    output_dir = project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        results_dir / "7199_web_live_cell_comprehensive.json",
        results_dir / "astrosoma-g2s2_VOL_time_lapse_comprehensive.json",
        results_dir / "astrosoma-g3s10_vol_cell_migration_comprehensive.json"
    ]

    output_path = output_dir / "figure3_video_tracking_overview"
    plot_video_tracking_overview(files, output_path)
