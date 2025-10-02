# plot_timeseries_detailed.py
"""
Detailed Time-Series Analysis
Creates comprehensive time-series analysis with trend detection, peak analysis, and dynamics
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import zscore
from pathlib import Path
from visualisation_setup import *


def plot_timeseries_detailed(fluor_files, output_name="timeseries_detailed"):
    """
    Create detailed time-series analysis with trend detection and peak analysis

    Parameters:
    -----------
    fluor_files : list of str or Path
        List of JSON file paths for fluorescence data
    output_name : str or Path
        Base name for output files
    """

    # Load data
    datasets = {}
    for file in fluor_files:
        data = load_json(file)
        channel = data['channels'][0]
        datasets[channel] = data

    channels = list(datasets.keys())
    n_channels = len(channels)

    fig = plt.figure(figsize=(16, 4.5 * n_channels))
    gs = gridspec.GridSpec(n_channels, 4, figure=fig, hspace=0.35, wspace=0.35)

    for ch_idx, channel in enumerate(channels):
        ts_data = datasets[channel]['time_series_data']
        intensities = np.array(ts_data['fluorescence_intensity'])
        background = np.array(ts_data['background_level'])
        snr = np.array(ts_data['signal_to_noise'])
        frames = np.arange(len(intensities))

        color = CHANNEL_COLORS.get(channel, '#333333')

        # Panel A: Raw Signal with Smoothing
        ax1 = fig.add_subplot(gs[ch_idx, 0])

        # Apply Savitzky-Golay filter for smoothing
        window_length = min(11, len(intensities) if len(intensities) % 2 == 1 else len(intensities) - 1)
        if window_length >= 5:
            smoothed = savgol_filter(intensities, window_length, 3)
            ax1.plot(frames, smoothed, color='red', linewidth=2.5,
                     label='Smoothed (Savitzky-Golay)', alpha=0.85, linestyle='-', zorder=3)

        ax1.plot(frames, intensities, color=color, linewidth=1.2,
                 label='Raw signal', alpha=0.5, marker='o', markersize=3,
                 markeredgecolor='black', markeredgewidth=0.3)

        # Calculate and display trend statistics
        mean_intensity = np.mean(intensities)
        std_intensity = np.std(intensities)
        cv = (std_intensity / mean_intensity) * 100  # Coefficient of variation

        ax1.axhline(mean_intensity, color='green', linestyle='--',
                    linewidth=1.5, alpha=0.6, label=f'Mean: {mean_intensity:.1f}')
        ax1.axhspan(mean_intensity - std_intensity, mean_intensity + std_intensity,
                    alpha=0.15, color='green', label=f'±1 SD')

        ax1.set_xlabel('Frame Number', fontsize=9)
        ax1.set_ylabel('Fluorescence Intensity (a.u.)', fontsize=9)
        ax1.set_title(f'{channel.upper()}: Raw Signal & Temporal Trend',
                      fontweight='bold', loc='left', fontsize=10)
        ax1.legend(frameon=False, loc='best', fontsize=7)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(alpha=0.3, linestyle='--', linewidth=0.5)

        # Add statistics box
        stats_text = f'CV: {cv:.1f}%\nRange: {np.ptp(intensities):.1f}'
        ax1.text(0.98, 0.02, stats_text,
                 transform=ax1.transAxes, ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor='gray', alpha=0.9, linewidth=1),
                 fontsize=7, family='monospace')

        # Panel B: Signal-to-Background Ratio
        ax2 = fig.add_subplot(gs[ch_idx, 1])

        sb_ratio = intensities / (background + 1e-6)  # Avoid division by zero
        mean_sb = np.mean(sb_ratio)

        ax2.plot(frames, sb_ratio, color=color, linewidth=2, alpha=0.8, label='S/B Ratio')
        ax2.fill_between(frames, 1, sb_ratio, where=(sb_ratio >= 1),
                         color=color, alpha=0.25, label='Above background')
        ax2.fill_between(frames, sb_ratio, 1, where=(sb_ratio < 1),
                         color='gray', alpha=0.25, label='Below background')

        ax2.axhline(y=1, color='red', linestyle='--', linewidth=2,
                    alpha=0.7, label='Background threshold', zorder=5)
        ax2.axhline(y=mean_sb, color='blue', linestyle=':', linewidth=1.5,
                    alpha=0.6, label=f'Mean S/B: {mean_sb:.2f}')

        ax2.set_xlabel('Frame Number', fontsize=9)
        ax2.set_ylabel('Signal-to-Background Ratio', fontsize=9)
        ax2.set_title(f'{channel.upper()}: Signal-to-Background Dynamics',
                      fontweight='bold', loc='left', fontsize=10)
        ax2.legend(frameon=False, loc='best', fontsize=6.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(alpha=0.3, linestyle='--', linewidth=0.5)

        # Calculate percentage above background
        pct_above = (np.sum(sb_ratio > 1) / len(sb_ratio)) * 100
        ax2.text(0.98, 0.98, f'{pct_above:.1f}% above\nbackground',
                 transform=ax2.transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor='gray', alpha=0.9, linewidth=1),
                 fontsize=7, fontweight='bold')

        # Panel C: Peak Detection
        ax3 = fig.add_subplot(gs[ch_idx, 2])

        # Find peaks with adaptive threshold
        prominence_threshold = np.std(intensities) * 0.5
        peaks, properties = find_peaks(intensities,
                                       prominence=prominence_threshold,
                                       distance=3)  # Minimum distance between peaks

        ax3.plot(frames, intensities, color=color, linewidth=2, alpha=0.7, label='Signal')

        if len(peaks) > 0:
            ax3.plot(peaks, intensities[peaks], "x", color='red',
                     markersize=10, markeredgewidth=2.5, label=f'{len(peaks)} peaks detected',
                     zorder=5)

            # Mark peak regions with vertical lines
            for peak in peaks:
                ax3.axvline(x=peak, color='red', linestyle=':', alpha=0.4, linewidth=1)

            # Calculate peak statistics
            peak_intensities = intensities[peaks]
            mean_peak = np.mean(peak_intensities)

            ax3.axhline(mean_peak, color='orange', linestyle='--',
                        linewidth=1.5, alpha=0.6, label=f'Mean peak: {mean_peak:.1f}')

            # Add peak info
            peak_info = (f'Peaks: {len(peaks)}\n'
                         f'Mean: {mean_peak:.1f}\n'
                         f'Max: {np.max(peak_intensities):.1f}')
        else:
            peak_info = 'No significant\npeaks detected'

        ax3.text(0.02, 0.98, peak_info,
                 transform=ax3.transAxes, ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor='gray', alpha=0.9, linewidth=1),
                 fontsize=7, family='monospace')

        ax3.set_xlabel('Frame Number', fontsize=9)
        ax3.set_ylabel('Fluorescence Intensity (a.u.)', fontsize=9)
        ax3.set_title(f'{channel.upper()}: Peak Detection Analysis',
                      fontweight='bold', loc='left', fontsize=10)
        ax3.legend(frameon=False, loc='upper right', fontsize=7)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.grid(alpha=0.3, linestyle='--', linewidth=0.5)

        # Panel D: Rate of Change (First Derivative)
        ax4 = fig.add_subplot(gs[ch_idx, 3])

        # Calculate rate of change (first derivative)
        rate_of_change = np.diff(intensities)
        rate_frames = frames[:-1]

        # Smooth the rate of change for better visualization
        if len(rate_of_change) >= 5:
            window = min(5, len(rate_of_change) if len(rate_of_change) % 2 == 1 else len(rate_of_change) - 1)
            if window >= 3:
                rate_smoothed = savgol_filter(rate_of_change, window, 2)
                ax4.plot(rate_frames, rate_smoothed, color='black',
                         linewidth=1.5, alpha=0.6, linestyle='--', label='Smoothed')

        ax4.plot(rate_frames, rate_of_change, color=color,
                 linewidth=1.2, alpha=0.5, label='Raw derivative')

        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5, zorder=5)
        ax4.fill_between(rate_frames, 0, rate_of_change,
                         where=(rate_of_change > 0),
                         color='green', alpha=0.35, label='Increasing', interpolate=True)
        ax4.fill_between(rate_frames, 0, rate_of_change,
                         where=(rate_of_change <= 0),
                         color='red', alpha=0.35, label='Decreasing', interpolate=True)

        # Calculate statistics
        pct_increasing = (np.sum(rate_of_change > 0) / len(rate_of_change)) * 100
        max_increase = np.max(rate_of_change)
        max_decrease = np.min(rate_of_change)

        ax4.set_xlabel('Frame Number', fontsize=9)
        ax4.set_ylabel('ΔIntensity / Δframe (a.u./frame)', fontsize=9)
        ax4.set_title(f'{channel.upper()}: Temporal Rate of Change',
                      fontweight='bold', loc='left', fontsize=10)
        ax4.legend(frameon=False, loc='best', fontsize=6.5)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.grid(alpha=0.3, linestyle='--', linewidth=0.5)

        # Add dynamics statistics
        dynamics_text = (f'Increasing: {pct_increasing:.1f}%\n'
                         f'Max Δ+: {max_increase:.2f}\n'
                         f'Max Δ−: {max_decrease:.2f}')
        ax4.text(0.02, 0.98, dynamics_text,
                 transform=ax4.transAxes, ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor='gray', alpha=0.9, linewidth=1),
                 fontsize=7, family='monospace')

    # Add overall figure title
    fig.suptitle('Detailed Time-Series Analysis of Fluorescence Dynamics',
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
        results_dir / "1585_dapi_comprehensive.json",
        results_dir / "1585_gfp_comprehensive.json",
        results_dir / "10954_rfp_comprehensive.json"
    ]

    # Output path
    output_path = output_dir / "figure5_timeseries_detailed"

    # Generate figure
    plot_timeseries_detailed(files, output_path)
