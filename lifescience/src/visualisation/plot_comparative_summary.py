# plot_comparative_summary.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from visualization_setup import *

def plot_comparative_summary(fluor_files, video_files, output_name="comparative_summary"):
    """
    Create a comprehensive summary figure comparing all analyses
    """
    
    # Load fluorescence data
    fluor_data = {}
    for file in fluor_files:
        data = load_json(file)
        channel = data['channels'][0]
        fluor_data[channel] = data
    
    # Load video data
    video_data = [load_json(f) for f in video_files]
    video_labels = [f.split('_')[0] for f in video_files]
    
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Processing Time Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    all_times = [fluor_data[ch]['processing_time'] for ch in fluor_data] + \
                [vd['processing_time'] for vd in video_data]
    all_labels = [f"{ch.upper()}\n(Fluor)" for ch in fluor_data] + \
                 [f"{lbl}\n(Video)" for lbl in video_labels]
    
    colors_proc = ['#4A90E2']*len(fluor_data) + ['#E74C3C']*len(video_data)
    bars = ax1.bar(range(len(all_times)), all_times, color=colors_proc, 
                   alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_xticks(range(len(all_times)))
    ax1.set_xticklabels(all_labels, fontsize=7)
    ax1.set_ylabel('Processing Time (s)')
    ax1.set_title('A. Processing Time', fontweight='bold', loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Panel B: Image/Video Dimensions
    ax2 = fig.add_subplot(gs[0, 1])
    fluor_dims = [f"{d['image_dimensions'][0]}×{d['image_dimensions'][1]}" 
                  for d in fluor_data.values()]
    video_dims = [f"{d['frame_dimensions'][0]}×{d['frame_dimensions'][1]}" 
                  for d in video_data]
    
    ax2.text(0.5, 0.9, 'Fluorescence Images:', ha='center', va='top', 
            fontweight='bold', transform=ax2.transAxes)
    y_pos = 0.75
    for ch, dim in zip(fluor_data.keys(), fluor_dims):
        ax2.text(0.5, y_pos, f"{ch.upper()}: {dim}", ha='center', va='top',
                transform=ax2.transAxes, fontsize=8)
        y_pos -= 0.15
    
    ax2.text(0.5, y_pos-0.1, 'Video Frames:', ha='center', va='top',
            fontweight='bold', transform=ax2.transAxes)
    y_pos -= 0.25
    for lbl, dim in zip(video_labels, video_dims):
        ax2.text(0.5, y_pos, f"{lbl}: {dim}", ha='center', va='top',
                transform=ax2.transAxes, fontsize=8)
        y_pos -= 0.15
    
    ax2.set_title('B. Data Dimensions', fontweight='bold', loc='left')
    ax2.axis('off')
    
    # Panel C: Quality Metrics Heatmap (Fluorescence)
    ax3 = fig.add_subplot(gs[0, 2:])
    channels = list(fluor_data.keys())
    metrics = ['segmentation_dice', 'segmentation_iou', 'pixel_accuracy']
    metric_labels = ['Dice Score', 'IoU', 'Pixel Accuracy']
    
    heatmap_data = np.array([[fluor_data[ch][m] for m in metrics] 
                             for ch in channels])
    
    im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax3.set_xticks(range(len(metrics)))
    ax3.set_xticklabels(metric_labels)
    ax3.set_yticks(range(len(channels)))
    ax3.set_yticklabels([ch.upper() for ch in channels])
    ax3.set_title('C. Fluorescence Quality Metrics', fontweight='bold', loc='left')
    
    # Add text annotations
    for i in range(len(channels)):
        for j in range(len(metrics)):
            text = ax3.text(j, i, f'{heatmap_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Score', rotation=270, labelpad=15)
    
    # Panel D: SNR Trends
    ax4 = fig.add_subplot(gs[1, :2])
    for channel in channels:
        ts_data = fluor_data[channel]['time_series_data']
        frames = np.arange(len(ts_data['signal_to_noise']))
        ax4.plot(frames, ts_data['signal_to_noise'], 
                linewidth=1.5, label=channel.upper(), 
                color=CHANNEL_COLORS.get(channel, '#333333'), alpha=0.8)
    
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Signal-to-Noise Ratio')
    ax4.set_title('D. SNR Temporal Dynamics', fontweight='bold', loc='left')
    ax4.legend(frameon=False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel E: Video Tracking Summary
    ax5 = fig.add_subplot(gs[1, 2:])
    metrics_video = ['num_tracks', 'mean_velocity', 'mean_displacement']
    metric_labels_video = ['# Tracks', 'Mean Velocity', 'Mean Displacement']
    
    x = np.arange(len(metrics_video))
    width = 0.8 / len(video_data)
    colors_video = plt.cm.Set2(np.linspace(0, 1, len(video_data)))
    
    for i, (data, label) in enumerate(zip(video_data, video_labels)):
        values = [data['num_tracks'], data['mean_velocity'], 
                 data['displacement_metrics']['mean_displacement']]
        # Normalize for visualization
        values_norm = [v / max([vd[m] if m != 'mean_displacement' else vd['displacement_metrics']['mean_displacement'] 
                               for vd in video_data]) 
                      for v, m in zip(values, ['num_tracks', 'mean_velocity', 'mean_displacement'])]
        
        offset = (i - len(video_data)/2 + 0.5) * width
        bars = ax5.bar(x + offset, values_norm, width, label=label,
                      color=colors_video[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=6)
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(metric_labels_video)
    ax5.set_ylabel('Normalized Value')
    ax5.set_title('E. Video Tracking Summary', fontweight='bold', loc='left')
    ax5.legend(frameon=False, ncol=len(video_data))
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    save_figure(fig, output_name)
    plt.show()

# Example usage
if __name__ == "__main__":
    fluor_files = [
        '1585_dapi_comprehensive.json',
        '1585_gfp_comprehensive.json',
        '10954_rfp_comprehensive.json'
    ]
    
    video_files = [
        '7199_web_live_cell_comprehensive.json',
        'astrosoma-g2s2_VOL_time_lapse_comprehensive.json',
        'astrosoma-g3s10_vol_cell_migration_comprehensive.json'
    ]
    
    plot_comparative_summary(fluor_files, video_files, "figure3_comparative_summary")
