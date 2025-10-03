# plot_comparative_summary.py
"""
Comparative Summary Figure
Cross-dataset comparison and benchmarking
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from visualisations_setup import *


def plot_comparative_summary(data_files, output_name="figure5_comparative_summary"):
    """
    Create comparative summary figure

    Parameters:
    -----------
    data_files : list of str or Path
        List of JSON file paths
    output_name : str or Path
        Base name for output files
    """

    # Load all datasets
    datasets = [load_json(f) for f in data_files]
    labels = [Path(f).stem.replace('_comprehensive', '').replace('_', ' ').title()
              for f in data_files]
    n_datasets = len(datasets)

    # Determine analysis type
    analysis_type = datasets[0].get('analysis_type', 'unknown')
    is_video = analysis_type == 'video_analysis'
    is_fluor = analysis_type == 'fluorescence_microscopy'

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, n_datasets))

    # ========================================================================
    # Panel A: Performance Radar Chart
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')

    if is_video:
        metrics = ['Accuracy', 'Velocity', 'Tracks', 'Efficiency']
        for data, label, color in zip(datasets, labels, colors):
            values = [
                data.get('tracking_accuracy', 0),
                min(data.get('mean_velocity', 0) / 50, 1.0),  # Normalize
                min(data.get('num_tracks', 0) / 10, 1.0),  # Normalize
                1 - min(data.get('processing_time', 0) / 5, 1.0)  # Invert & normalize
            ]
            values += values[:1]

            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]

            ax1.plot(angles, values, 'o-', linewidth=2.5, label=label[:15],
                     color=color, markersize=7)
            ax1.fill(angles, values, alpha=0.15, color=color)

        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metrics, fontsize=9)

    elif is_fluor:
        metrics = ['SNR', 'Dice', 'IoU', 'Contrast']
        for data, label, color in zip(datasets, labels, colors):
            channel = data.get('channels', ['unknown'])[0]
            values = [
                min(data.get('signal_to_noise_ratios', {}).get(channel, 0) / 15, 1.0),
                data.get('segmentation_dice', 0) / 0.3,
                data.get('segmentation_iou', 0) / 0.2,
                data.get('contrast_metrics', {}).get(channel, {}).get('michelson_contrast', 0) / 0.5
            ]
            values += values[:1]

            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]

            ax1.plot(angles, values, 'o-', linewidth=2.5, label=label[:15],
                     color=color, markersize=7)
            ax1.fill(angles, values, alpha=0.15, color=color)

        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metrics, fontsize=9)

    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=7)
    ax1.set_title('A. Performance Radar',
                  fontweight='bold', pad=20, fontsize=11)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
               frameon=False, fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ========================================================================
    # Panel B: Normalized Performance Heatmap
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    if is_video:
        metric_names = ['Accuracy', 'Velocity', 'Tracks', 'Frames', 'FPS']
        heatmap_data = []
        for data in datasets:
            row = [
                data.get('tracking_accuracy', 0),
                data.get('mean_velocity', 0) / 50,  # Normalize
                data.get('num_tracks', 0) / 10,  # Normalize
                data.get('frame_count', 0) / 1000,  # Normalize
                data.get('fps', 0) / 30  # Normalize
            ]
            heatmap_data.append(row)

    elif is_fluor:
        metric_names = ['SNR', 'Dice', 'IoU', 'Contrast', 'Accuracy']
        heatmap_data = []
        for data in datasets:
            channel = data.get('channels', ['unknown'])[0]
            row = [
                data.get('signal_to_noise_ratios', {}).get(channel, 0) / 15,
                data.get('segmentation_dice', 0) / 0.3,
                data.get('segmentation_iou', 0) / 0.2,
                data.get('contrast_metrics', {}).get(channel, {}).get('michelson_contrast', 0) / 0.5,
                data.get('pixel_accuracy', 0) / 0.3
            ]
            heatmap_data.append(row)

    heatmap_array = np.array(heatmap_data)
    im = ax2.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax2.set_xticks(np.arange(len(metric_names)))
    ax2.set_yticks(np.arange(n_datasets))
    ax2.set_xticklabels(metric_names, fontsize=9, rotation=45, ha='right')
    ax2.set_yticklabels([l[:15] for l in labels], fontsize=9)

    # Add text annotations
    for i in range(n_datasets):
        for j in range(len(metric_names)):
            text = ax2.text(j, i, f'{heatmap_array[i, j]:.2f}',
                            ha="center", va="center", color="black", fontsize=8, fontweight='bold')

    plt.colorbar(im, ax=ax2, label='Normalized Score')
    ax2.set_title('B. Performance Heatmap',
                  fontweight='bold', fontsize=11)

    # ========================================================================
    # Panel C: Processing Time Comparison
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])

    proc_times = [data.get('processing_time', 0) for data in datasets]

    bars = ax3.barh(range(n_datasets), proc_times, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)

    for i, (bar, val) in enumerate(zip(bars, proc_times)):
        width = bar.get_width()
        ax3.text(width + 0.1, bar.get_y() + bar.get_height() / 2.,
                 f'{val:.2f}s', ha='left', va='center',
                 fontsize=9, fontweight='bold')

    ax3.set_yticks(range(n_datasets))
    ax3.set_yticklabels([l[:20] for l in labels], fontsize=9)
    ax3.set_xlabel('Processing Time (seconds)', fontsize=10)
    ax3.set_title('C. Processing Time Comparison',
                  fontweight='bold', fontsize=11)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

    # ========================================================================
    # Panel D: Quality Score Distribution
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 0])

    if is_video:
        quality_scores = [data.get('tracking_accuracy', 0) for data in datasets]
        ylabel = 'Tracking Accuracy'
    elif is_fluor:
        quality_scores = [data.get('segmentation_dice', 0) for data in datasets]
        ylabel = 'Segmentation Dice Score'

    bars = ax4.bar(range(n_datasets), quality_scores, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5, zorder=3)

    for bar, val in zip(bars, quality_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

    ax4.set_xticks(range(n_datasets))
    ax4.set_xticklabels([l[:15] for l in labels], rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel(ylabel, fontsize=10)
    ax4.set_title('D. Quality Score Comparison',
                  fontweight='bold', fontsize=11)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)

    # ========================================================================
    # Panel E: Efficiency Scatter Plot
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 1])

    if is_video:
        x_values = [data.get('frame_count', 0) for data in datasets]
        y_values = [data.get('processing_time', 0) for data in datasets]
        xlabel = 'Frame Count'
        ylabel = 'Processing Time (s)'
    elif is_fluor:
        x_values = [data.get('image_dimensions', [0, 0, 0])[0] * data.get('image_dimensions', [0, 0, 0])[1] / 1e6
                    for data in datasets]
        y_values = [data.get('processing_time', 0) for data in datasets]
        xlabel = 'Image Size (Megapixels)'
        ylabel = 'Processing Time (s)'

    for i, (x, y, label, color) in enumerate(zip(x_values, y_values, labels, colors)):
        ax5.scatter(x, y, s=300, alpha=0.7, color=color,
                    edgecolors='black', linewidth=2, zorder=3)
        ax5.annotate(label[:10], (x, y), fontsize=8, ha='center', va='center', fontweight='bold')

    ax5.set_xlabel(xlabel, fontsize=10)
    ax5.set_ylabel(ylabel, fontsize=10)
    ax5.set_title('E. Efficiency Analysis',
                  fontweight='bold', fontsize=11)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # ========================================================================
    # Panel F: Relative Performance Bar Chart
    # ========================================================================
    ax6 = fig.add_subplot(gs[1, 2])

    if is_video:
        categories = ['Accuracy', 'Velocity', 'Tracks']
        data_matrix = np.array([
            [data.get('tracking_accuracy', 0) for data in datasets],
            [data.get('mean_velocity', 0) / 50 for data in datasets],
            [data.get('num_tracks', 0) / 10 for data in datasets]
        ])
    elif is_fluor:
        categories = ['SNR', 'Dice', 'IoU']
        data_matrix = np.array([
            [data.get('signal_to_noise_ratios', {}).get(data.get('channels', ['unknown'])[0], 0) / 15
             for data in datasets],
            [data.get('segmentation_dice', 0) / 0.3 for data in datasets],
            [data.get('segmentation_iou', 0) / 0.2 for data in datasets]
        ])

    x = np.arange(len(categories))
    width = 0.8 / n_datasets

    for i, (label, color) in enumerate(zip(labels, colors)):
        offset = (i - n_datasets / 2 + 0.5) * width
        ax6.bar(x + offset, data_matrix[:, i], width, label=label[:15],
                color=color, alpha=0.8, edgecolor='black', linewidth=1)

    ax6.set_xticks(x)
    ax6.set_xticklabels(categories, fontsize=9)
    ax6.set_ylabel('Normalized Score', fontsize=10)
    ax6.set_title('F. Relative Performance',
                  fontweight='bold', fontsize=11)
    ax6.legend(frameon=False, fontsize=7, loc='upper left')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax6.set_ylim([0, 1.2])

    # ========================================================================
    # Panel G: Ranking Matrix
    # ========================================================================
    ax7 = fig.add_subplot(gs[2, 0])

    if is_video:
        rank_metrics = ['Accuracy', 'Velocity', 'Tracks', 'FPS']
        rank_data = []
        for data in datasets:
            rank_data.append([
                data.get('tracking_accuracy', 0),
                data.get('mean_velocity', 0),
                data.get('num_tracks', 0),
                data.get('fps', 0)
            ])
    elif is_fluor:
        rank_metrics = ['SNR', 'Dice', 'IoU', 'Contrast']
        rank_data = []
        for data in datasets:
            channel = data.get('channels', ['unknown'])[0]
            rank_data.append([
                data.get('signal_to_noise_ratios', {}).get(channel, 0),
                data.get('segmentation_dice', 0),
                data.get('segmentation_iou', 0),
                data.get('contrast_metrics', {}).get(channel, {}).get('michelson_contrast', 0)
            ])

    rank_array = np.array(rank_data)

    # Calculate ranks (higher is better)
    ranks = np.zeros_like(rank_array)
    for j in range(rank_array.shape[1]):
        ranks[:, j] = rank_array.shape[0] - np.argsort(np.argsort(rank_array[:, j]))

    im = ax7.imshow(ranks, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=n_datasets)

    ax7.set_xticks(np.arange(len(rank_metrics)))
    ax7.set_yticks(np.arange(n_datasets))
    ax7.set_xticklabels(rank_metrics, fontsize=9, rotation=45, ha='right')
    ax7.set_yticklabels([l[:15] for l in labels], fontsize=9)

    # Add rank annotations
    for i in range(n_datasets):
        for j in range(len(rank_metrics)):
            text = ax7.text(j, i, f'#{int(ranks[i, j])}',
                            ha="center", va="center", color="black",
                            fontsize=9, fontweight='bold')

    plt.colorbar(im, ax=ax7, label='Rank (1=Best)')
    ax7.set_title('G. Performance Ranking',
                  fontweight='bold', fontsize=11)

    # ========================================================================
    # Panel H: Composite Score
    # ========================================================================
    ax8 = fig.add_subplot(gs[2, 1])

    # Calculate composite scores (average of normalized metrics)
    composite_scores = []
    for i in range(n_datasets):
        if is_video:
            score = (
                            datasets[i].get('tracking_accuracy', 0) +
                            min(datasets[i].get('mean_velocity', 0) / 50, 1.0) +
                            min(datasets[i].get('num_tracks', 0) / 10, 1.0) +
                            (1 - min(datasets[i].get('processing_time', 0) / 5, 1.0))
                    ) / 4
        elif is_fluor:
            channel = datasets[i].get('channels', ['unknown'])[0]
            score = (
                            min(datasets[i].get('signal_to_noise_ratios', {}).get(channel, 0) / 15, 1.0) +
                            datasets[i].get('segmentation_dice', 0) / 0.3 +
                            datasets[i].get('segmentation_iou', 0) / 0.2 +
                            datasets[i].get('contrast_metrics', {}).get(channel, {}).get('michelson_contrast', 0) / 0.5
                    ) / 4
        composite_scores.append(score)

    bars = ax8.bar(range(n_datasets), composite_scores, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5, zorder=3)

    for bar, val in zip(bars, composite_scores):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

    ax8.set_xticks(range(n_datasets))
    ax8.set_xticklabels([l[:15] for l in labels], rotation=45, ha='right', fontsize=8)
    ax8.set_ylabel('Composite Score', fontsize=10)
    ax8.set_title('H. Overall Performance Score',
                  fontweight='bold', fontsize=11)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    ax8.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax8.set_ylim([0, 1.1])

    # Highlight best performer
    best_idx = np.argmax(composite_scores)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

    # ========================================================================
    # Panel I: Comprehensive Statistics Table
    # ========================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    if is_video:
        table_data = []
        for data, label, score in zip(datasets, labels, composite_scores):
            row = [
                label[:12],
                f"{data.get('tracking_accuracy', 0):.3f}",
                f"{data.get('mean_velocity', 0):.2f}",
                f"{data.get('num_tracks', 0)}",
                f"{score:.3f}"
            ]
            table_data.append(row)
        headers = ['Dataset', 'Accuracy', 'Velocity', 'Tracks', 'Score']

    elif is_fluor:
        table_data = []
        for data, label, score in zip(datasets, labels, composite_scores):
            channel = data.get('channels', ['unknown'])[0]
            row = [
                label[:12],
                f"{data.get('signal_to_noise_ratios', {}).get(channel, 0):.2f}",
                f"{data.get('segmentation_dice', 0):.3f}",
                f"{data.get('segmentation_iou', 0):.3f}",
                f"{score:.3f}"
            ]
            table_data.append(row)
        headers = ['Dataset', 'SNR', 'Dice', 'IoU', 'Score']

    table = ax9.table(cellText=table_data, colLabels=headers,
                      cellLoc='center', loc='center',
                      colWidths=[0.25, 0.18, 0.18, 0.18, 0.21])

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#9C27B0')
        cell.set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_facecolor(colors[i - 1])
            cell.set_alpha(0.3)
            if j == 0:
                cell.set_text_props(weight='bold')

    # Highlight best performer row
    for j in range(len(headers)):
        cell = table[(best_idx + 1, j)]
        cell.set_edgecolor('gold')
        cell.set_linewidth(3)

    ax9.set_title('I. Summary Statistics',
                  fontweight='bold', fontsize=11, pad=20)

    # Overall title
    analysis_name = "Video Tracking" if is_video else "Fluorescence Microscopy"
    fig.suptitle(f'{analysis_name} Analysis: Comparative Summary',
                 fontsize=14, fontweight='bold', y=0.995)

    save_figure(fig, output_name)
    plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / "results"
    output_dir = project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # For video data
    video_files = [
        results_dir / "7199_web_live_cell_comprehensive.json",
        results_dir / "astrosoma-g2s2_VOL_time_lapse_comprehensive.json",
        results_dir / "astrosoma-g3s10_vol_cell_migration_comprehensive.json"
    ]

    output_path = output_dir / "figure5_video_comparative_summary"
    plot_comparative_summary(video_files, output_path)
