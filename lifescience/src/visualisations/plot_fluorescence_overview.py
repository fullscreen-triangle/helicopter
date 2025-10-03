# plot_fluorescence_overview.py
"""
Multi-Channel Fluorescence Overview Figure
Compares DAPI, GFP, and RFP channels
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from visualisations_setup import *


def plot_fluorescence_overview(data_files, output_name="figure1_fluorescence_overview"):
    """
    Create comprehensive multi-channel fluorescence overview

    Parameters:
    -----------
    data_files : list of str or Path
        List of JSON file paths [dapi, gfp, rfp]
    output_name : str or Path
        Base name for output files
    """

    # Load all datasets
    datasets = [load_json(f) for f in data_files]

    # Extract channel names
    channels = []
    for data in datasets:
        if 'channels' in data and len(data['channels']) > 0:
            channels.append(data['channels'][0].upper())
        else:
            channels.append('Unknown')

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Color scheme for channels
    channel_colors = {
        'DAPI': '#4A90E2',  # Blue
        'GFP': '#50C878',  # Green
        'RFP': '#E74C3C'  # Red
    }
    colors = [channel_colors.get(ch, '#888888') for ch in channels]

    # ========================================================================
    # Panel A: Signal-to-Noise Ratio Comparison
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    snr_values = []
    for data, channel in zip(datasets, channels):
        if 'signal_to_noise_ratios' in data:
            snr = data['signal_to_noise_ratios'].get(channel.lower(), 0)
            snr_values.append(snr)
        else:
            snr_values.append(0)

    bars = ax1.bar(channels, snr_values, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5, zorder=3)

    # Add value labels
    for bar, val in zip(bars, snr_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
                 f'{val:.2f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

    ax1.set_ylabel('Signal-to-Noise Ratio', fontsize=10)
    ax1.set_xlabel('Channel', fontsize=10)
    ax1.set_title('A. Signal-to-Noise Ratio Comparison',
                  fontweight='bold', fontsize=11)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax1.set_ylim([0, max(snr_values) * 1.2])

    # ========================================================================
    # Panel B: Segmentation Quality Metrics
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    metrics = ['Dice Score', 'IoU', 'Pixel Accuracy']
    x = np.arange(len(metrics))
    width = 0.25

    for i, (data, channel, color) in enumerate(zip(datasets, channels, colors)):
        values = [
            data.get('segmentation_dice', 0),
            data.get('segmentation_iou', 0),
            data.get('pixel_accuracy', 0)
        ]
        offset = (i - 1) * width
        bars = ax2.bar(x + offset, values, width, label=channel,
                       color=color, alpha=0.8, edgecolor='black',
                       linewidth=1, zorder=3)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if height > 0.02:
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{val:.2f}', ha='center', va='bottom',
                         fontsize=7, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=9)
    ax2.set_ylabel('Score', fontsize=10)
    ax2.set_title('B. Segmentation Quality Metrics',
                  fontweight='bold', fontsize=11)
    ax2.legend(frameon=False, fontsize=9, loc='upper right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax2.set_ylim([0, 0.35])

    # ========================================================================
    # Panel C: Intensity Distribution Comparison
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])

    for data, channel, color in zip(datasets, channels, colors):
        if 'intensity_measurements' in data:
            stats = data['intensity_measurements']
            channel_key = channel.lower()
            if channel_key in stats:
                mean = stats[channel_key].get('mean', 0)
                std = stats[channel_key].get('std', 0)
                if std == 0:  # Handle case where std is 0
                    std = mean * 0.1  # Use 10% of mean as std

                # Create normal distribution
                x_range = np.linspace(max(0, mean - 3 * std), mean + 3 * std, 100)
                y_dist = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)

                ax3.plot(x_range, y_dist, linewidth=2.5, label=channel,
                         color=color, alpha=0.8)
                ax3.fill_between(x_range, y_dist, alpha=0.2, color=color)

    ax3.set_xlabel('Intensity (a.u.)', fontsize=10)
    ax3.set_ylabel('Probability Density', fontsize=10)
    ax3.set_title('C. Intensity Distribution Comparison',
                  fontweight='bold', fontsize=11)
    ax3.legend(frameon=False, fontsize=9, loc='upper right')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # ========================================================================
    # Panel D: Contrast and Dynamic Range
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 0])

    contrast_values = []
    for data, channel in zip(datasets, channels):
        if 'contrast_metrics' in data:
            contrast = data['contrast_metrics'].get(channel.lower(), {}).get('michelson_contrast', 0)
            contrast_values.append(contrast)
        else:
            contrast_values.append(0)

    bars = ax4.bar(channels, contrast_values, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5, zorder=3)

    for bar, val in zip(bars, contrast_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

    ax4.set_ylabel('Michelson Contrast', fontsize=10)
    ax4.set_xlabel('Channel', fontsize=10)
    ax4.set_title('D. Contrast Comparison',
                  fontweight='bold', fontsize=11)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)

    # ========================================================================
    # Panel E: Spatial Statistics (Mean, Median, Max)
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 1])

    stat_types = ['Mean', 'Median', 'Max']
    x = np.arange(len(stat_types))
    width = 0.25

    for i, (data, channel, color) in enumerate(zip(datasets, channels, colors)):
        if 'intensity_measurements' in data:
            stats = data['intensity_measurements'].get(channel.lower(), {})
            values = [
                stats.get('mean', 0),
                stats.get('percentile_50', stats.get('mean', 0)),  # Use mean as median fallback
                stats.get('max', 0)
            ]
            offset = (i - 1) * width
            ax5.bar(x + offset, values, width, label=channel,
                    color=color, alpha=0.8, edgecolor='black', linewidth=1)

    ax5.set_xticks(x)
    ax5.set_xticklabels(stat_types, fontsize=9)
    ax5.set_ylabel('Intensity (a.u.)', fontsize=10)
    ax5.set_title('E. Intensity Statistics Comparison',
                  fontweight='bold', fontsize=11)
    ax5.legend(frameon=False, fontsize=9, loc='upper left')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # ========================================================================
    # Panel F: Processing Time Comparison
    # ========================================================================
    ax6 = fig.add_subplot(gs[1, 2])

    proc_times = [data.get('processing_time', 0) for data in datasets]

    bars = ax6.bar(channels, proc_times, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5, zorder=3)

    for bar, val in zip(bars, proc_times):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                 f'{val:.2f}s', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

    ax6.set_ylabel('Processing Time (seconds)', fontsize=10)
    ax6.set_xlabel('Channel', fontsize=10)
    ax6.set_title('F. Processing Time Comparison',
                  fontweight='bold', fontsize=11)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)

    # ========================================================================
    # Panel G: Image Dimensions and Resolution
    # ========================================================================
    ax7 = fig.add_subplot(gs[2, 0])

    dimensions = []
    for data in datasets:
        dims = data.get('image_dimensions', [0, 0, 0])
        dimensions.append(dims[0] * dims[1])  # Total pixels

    dimensions_mp = [d / 1e6 for d in dimensions]  # Convert to megapixels

    bars = ax7.bar(channels, dimensions_mp, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5, zorder=3)

    for bar, val in zip(bars, dimensions_mp):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{val:.2f}MP', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

    ax7.set_ylabel('Image Size (Megapixels)', fontsize=10)
    ax7.set_xlabel('Channel', fontsize=10)
    ax7.set_title('G. Image Resolution Comparison',
                  fontweight='bold', fontsize=11)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    ax7.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)

    # ========================================================================
    # Panel H: Quality Metrics Radar Chart
    # ========================================================================
    ax8 = fig.add_subplot(gs[2, 1], projection='polar')

    radar_metrics = ['SNR', 'Dice', 'IoU', 'Contrast']
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    for data, channel, color in zip(datasets, channels, colors):
        snr = data.get('signal_to_noise_ratios', {}).get(channel.lower(), 0)
        dice = data.get('segmentation_dice', 0)
        iou = data.get('segmentation_iou', 0)
        contrast = data.get('contrast_metrics', {}).get(channel.lower(), {}).get('michelson_contrast', 0)

        # Normalize values
        values = [
            min(snr / 15, 1.0),  # Normalize SNR (assume max ~15)
            dice / 0.3,  # Normalize Dice
            iou / 0.2,  # Normalize IoU
            contrast / 0.5  # Normalize contrast
        ]
        values += values[:1]

        ax8.plot(angles, values, 'o-', linewidth=2.5, label=channel,
                 color=color, markersize=7)
        ax8.fill(angles, values, alpha=0.15, color=color)

    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(radar_metrics, fontsize=9)
    ax8.set_ylim(0, 1)
    ax8.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax8.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=7)
    ax8.set_title('H. Multi-Metric Quality Comparison',
                  fontweight='bold', pad=20, fontsize=11)
    ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
               frameon=False, fontsize=9)
    ax8.grid(True, linestyle='--', alpha=0.5)

    # ========================================================================
    # Panel I: Summary Statistics Table
    # ========================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    table_data = []
    for data, channel in zip(datasets, channels):
        dims = data.get('image_dimensions', [0, 0, 0])
        snr = data.get('signal_to_noise_ratios', {}).get(channel.lower(), 0)
        dice = data.get('segmentation_dice', 0)

        row = [
            channel,
            f"{dims[0]}×{dims[1]}",
            f"{snr:.2f}",
            f"{dice:.3f}",
            f"{data.get('processing_time', 0):.2f}s"
        ]
        table_data.append(row)

    headers = ['Channel', 'Dimensions', 'SNR', 'Dice', 'Time']

    table = ax9.table(cellText=table_data, colLabels=headers,
                      cellLoc='center', loc='center',
                      colWidths=[0.2, 0.25, 0.15, 0.2, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
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
    fig.suptitle('Multi-Channel Fluorescence Microscopy Analysis: Comprehensive Overview',
                 fontsize=14, fontweight='bold', y=0.995)

    save_figure(fig, output_name)
    plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / "results"
    output_dir = project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        results_dir / "1585_dapi_comprehensive.json",
        results_dir / "1585_gfp_comprehensive.json",
        results_dir / "10954_rfp_comprehensive.json"
    ]

    output_path = output_dir / "figure1_fluorescence_overview"
    plot_fluorescence_overview(files, output_path)
