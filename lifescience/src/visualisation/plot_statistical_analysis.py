# plot_statistical_analysis.py
"""
Statistical Analysis Visualization
Creates comprehensive statistical analysis panels with correlation and distribution analysis
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats
from pathlib import Path
from visualisation_setup import *


def plot_statistical_analysis(fluor_files, output_name="statistical_analysis"):
    """
    Create statistical analysis panels with correlation and distribution analysis

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

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

    channels = list(datasets.keys())

    # Panel A: Correlation Matrix
    ax1 = fig.add_subplot(gs[0, :])

    metrics = ['segmentation_dice', 'segmentation_iou', 'pixel_accuracy']
    metric_labels = ['Dice Coefficient', 'IoU Score', 'Pixel Accuracy']

    # Create correlation matrix
    corr_matrix = np.zeros((len(channels), len(metrics)))
    for i, ch in enumerate(channels):
        for j, m in enumerate(metrics):
            corr_matrix[i, j] = datasets[ch][m]

    # Calculate correlation between metrics
    metric_corr = np.corrcoef(corr_matrix.T)

    im = ax1.imshow(metric_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels(metric_labels, fontsize=9)
    ax1.set_yticks(range(len(metrics)))
    ax1.set_yticklabels(metric_labels, fontsize=9)
    ax1.set_title('A. Segmentation Metric Correlation Matrix', fontweight='bold', loc='left', fontsize=11)

    # Add correlation values to heatmap
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            text_color = "white" if abs(metric_corr[i, j]) > 0.5 else "black"
            text = ax1.text(j, i, f'{metric_corr[i, j]:.3f}',
                            ha="center", va="center",
                            color=text_color,
                            fontsize=10, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson Correlation (r)', rotation=270, labelpad=20, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Panels B-D: Distribution Analysis for each channel
    dist_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    for idx, (channel, ax) in enumerate(zip(channels, dist_axes)):
        if idx >= len(channels):
            ax.axis('off')
            continue

        ts_data = datasets[channel]['time_series_data']
        intensities = np.array(ts_data['fluorescence_intensity'])

        # Create histogram
        n, bins, patches = ax.hist(intensities, bins=30, density=True,
                                   alpha=0.7, color=CHANNEL_COLORS.get(channel, '#333333'),
                                   edgecolor='black', linewidth=0.5, label='Observed')

        # Fit normal distribution
        mu, sigma = np.mean(intensities), np.std(intensities)
        x = np.linspace(min(intensities), max(intensities), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2.5,
                label=f'Normal fit', alpha=0.8)

        ax.set_xlabel('Fluorescence Intensity (a.u.)', fontsize=9)
        ax.set_ylabel('Probability Density', fontsize=9)
        ax.set_title(f'{chr(66 + idx)}. {channel.upper()} Intensity Distribution',
                     fontweight='bold', loc='left', fontsize=10)
        ax.legend(frameon=False, fontsize=7, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

        # Calculate statistics
        skewness = stats.skew(intensities)
        kurtosis = stats.kurtosis(intensities)

        # Shapiro-Wilk test for normality
        if len(intensities) <= 5000:  # Shapiro-Wilk has sample size limit
            shapiro_stat, shapiro_p = stats.shapiro(intensities)
            normality_text = f'Shapiro-Wilk p={shapiro_p:.4f}'
        else:
            ks_stat, ks_p = stats.kstest(intensities, 'norm', args=(mu, sigma))
            normality_text = f'KS test p={ks_p:.4f}'

        # Add statistics text box
        stats_text = (f'μ = {mu:.2f}\n'
                      f'σ = {sigma:.2f}\n'
                      f'Skewness = {skewness:.3f}\n'
                      f'Kurtosis = {kurtosis:.3f}\n'
                      f'{normality_text}')

        ax.text(0.97, 0.97, stats_text,
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='gray', alpha=0.9, linewidth=1),
                fontsize=6.5, family='monospace')

    # Panels E-G: Q-Q Plots (Quantile-Quantile plots for normality assessment)
    qq_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]

    for idx, (channel, ax) in enumerate(zip(channels, qq_axes)):
        if idx >= len(channels):
            ax.axis('off')
            continue

        ts_data = datasets[channel]['time_series_data']
        intensities = np.array(ts_data['fluorescence_intensity'])

        # Create Q-Q plot
        stats.probplot(intensities, dist="norm", plot=ax)

        # Style the scatter points
        ax.get_lines()[0].set_marker('o')
        ax.get_lines()[0].set_markersize(5)
        ax.get_lines()[0].set_markerfacecolor(CHANNEL_COLORS.get(channel, '#333333'))
        ax.get_lines()[0].set_markeredgecolor('black')
        ax.get_lines()[0].set_markeredgewidth(0.5)
        ax.get_lines()[0].set_alpha(0.6)
        ax.get_lines()[0].set_linestyle('')

        # Style the reference line
        ax.get_lines()[1].set_color('red')
        ax.get_lines()[1].set_linewidth(2)
        ax.get_lines()[1].set_linestyle('--')
        ax.get_lines()[1].set_alpha(0.8)
        ax.get_lines()[1].set_label('Normal distribution')

        ax.set_xlabel('Theoretical Quantiles', fontsize=9)
        ax.set_ylabel('Sample Quantiles', fontsize=9)
        ax.set_title(f'{chr(69 + idx)}. {channel.upper()} Q-Q Plot (Normality Test)',
                     fontweight='bold', loc='left', fontsize=10)
        ax.legend(frameon=False, fontsize=7, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

        # Calculate R² for Q-Q plot fit
        theoretical_quantiles = ax.get_lines()[0].get_xdata()
        sample_quantiles = ax.get_lines()[0].get_ydata()

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            theoretical_quantiles, sample_quantiles)
        r_squared = r_value ** 2

        # Add R² text
        ax.text(0.05, 0.95, f'R² = {r_squared:.4f}',
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='gray', alpha=0.9, linewidth=1),
                fontsize=7, fontweight='bold')

    # Add overall figure title
    fig.suptitle('Statistical Analysis of Fluorescence Microscopy Data',
                 fontsize=13, fontweight='bold', y=0.995)

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
    output_path = output_dir / "figure4_statistical_analysis"

    # Generate figure
    plot_statistical_analysis(files, output_path)
