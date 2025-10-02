# plot_fluorescence_panels.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from visualisation_setup import *

def plot_fluorescence_comprehensive(data_files, output_name="fluorescence_analysis"):
    """
    Create comprehensive multi-panel figure for fluorescence microscopy data
    
    Parameters:
    -----------
    data_files : list of str or Path
        List of JSON file paths
    output_name : str or Path
        Base name for output files
    """
    
    # Convert all paths to Path objects
    data_files = [Path(f) for f in data_files]
    output_name = Path(output_name)
    
    # Verify all files exist
    for file in data_files:
        if not file.exists():
            raise FileNotFoundError(f"Data file not found: {file}")
    
    # Load all data
    datasets = {}
    for file in data_files:
        data = load_json(file)
        channel = data['channels'][0]
        datasets[channel] = data
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)
    
    # Panel A: Signal-to-Noise Ratios Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    channels = list(datasets.keys())
    snr_values = [datasets[ch]['signal_to_noise_ratios'][ch] for ch in channels]
    colors = [CHANNEL_COLORS.get(ch, '#333333') for ch in channels]
    
    bars = ax1.bar(range(len(channels)), snr_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_xticks(range(len(channels)))
    ax1.set_xticklabels([ch.upper() for ch in channels])
    ax1.set_ylabel('Signal-to-Noise Ratio')
    ax1.set_title('A. Signal-to-Noise Comparison', fontweight='bold', loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bar, val in zip(bars, snr_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    
    # Panel B: Segmentation Quality Metrics
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ['segmentation_dice', 'segmentation_iou', 'pixel_accuracy']
    metric_labels = ['Dice', 'IoU', 'Pixel Acc.']
    x = np.arange(len(metrics))
    width = 0.8 / len(channels)
    
    for i, channel in enumerate(channels):
        values = [datasets[channel][m] for m in metrics]
        offset = (i - len(channels)/2 + 0.5) * width
        ax2.bar(x + offset, values, width, label=channel.upper(), 
                color=CHANNEL_COLORS.get(channel, '#333333'), alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_labels)
    ax2.set_ylabel('Score')
    ax2.set_title('B. Segmentation Quality', fontweight='bold', loc='left')
    ax2.legend(frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim([0, 1])
    
    # Panel C: Intensity Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    for channel in channels:
        intensity_data = datasets[channel]['intensity_measurements'][channel]
        positions = [intensity_data['min'], intensity_data['percentile_25'], 
                    intensity_data['mean'], intensity_data['percentile_75'], 
                    intensity_data['max']]
        
        ax3.plot(['Min', 'Q1', 'Mean', 'Q3', 'Max'], positions, 
                marker='o', linewidth=2, markersize=6, 
                label=channel.upper(), color=CHANNEL_COLORS.get(channel, '#333333'))
    
    ax3.set_ylabel('Intensity (a.u.)')
    ax3.set_title('C. Intensity Distribution', fontweight='bold', loc='left')
    ax3.legend(frameon=False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel D-F: Time Series Data (one per channel)
    time_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    
    for idx, (channel, ax) in enumerate(zip(channels, time_axes)):
        if idx >= len(channels):
            ax.axis('off')
            continue
            
        ts_data = datasets[channel]['time_series_data']
        frames = np.arange(len(ts_data['fluorescence_intensity']))
        
        # Plot fluorescence intensity
        ax.plot(frames, ts_data['fluorescence_intensity'], 
               color=CHANNEL_COLORS.get(channel, '#333333'), 
               linewidth=1.5, label='Signal', alpha=0.8)
        
        # Plot background level
        ax.plot(frames, ts_data['background_level'], 
               color='gray', linewidth=1, linestyle='--', 
               label='Background', alpha=0.6)
        
        ax.fill_between(frames, ts_data['background_level'], 
                        ts_data['fluorescence_intensity'],
                        color=CHANNEL_COLORS.get(channel, '#333333'), 
                        alpha=0.2)
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title(f'{chr(68+idx)}. {channel.upper()} Time Series', 
                    fontweight='bold', loc='left')
        ax.legend(frameon=False, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Panel G-I: Signal-to-Noise Over Time
    snr_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    
    for idx, (channel, ax) in enumerate(zip(channels, snr_axes)):
        if idx >= len(channels):
            ax.axis('off')
            continue
            
        ts_data = datasets[channel]['time_series_data']
        frames = np.arange(len(ts_data['signal_to_noise']))
        
        ax.plot(frames, ts_data['signal_to_noise'], 
               color=CHANNEL_COLORS.get(channel, '#333333'), 
               linewidth=1.5, alpha=0.8)
        
        # Add mean line
        mean_snr = np.mean(ts_data['signal_to_noise'])
        ax.axhline(mean_snr, color='red', linestyle='--', 
                  linewidth=1, alpha=0.5, label=f'Mean: {mean_snr:.2f}')
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('SNR')
        ax.set_title(f'{chr(71+idx)}. {channel.upper()} SNR Dynamics', 
                    fontweight='bold', loc='left')
        ax.legend(frameon=False, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    save_figure(fig, output_name)
    plt.show()

# Example usage
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
    output_path = output_dir / "figure1_fluorescence_analysis"
    
    # Generate figure
    plot_fluorescence_comprehensive(files, output_path)
