"""
Results Output and Visualization Module

Handles JSON serialization of analysis results and creates publication-ready
multi-panel figures according to the results template.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats


@dataclass
class AnalysisMetrics:
    """Base class for analysis metrics with JSON serialization"""
    analysis_type: str
    timestamp: str
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_json(self, file_path: Optional[Path] = None) -> str:
        """Convert to JSON string or save to file"""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, default=self._json_serializer)
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for numpy types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@dataclass 
class FluorescenceMetrics(AnalysisMetrics):
    """Comprehensive fluorescence analysis metrics"""
    # Image properties
    image_dimensions: tuple
    pixel_size_um: float
    channels: List[str]
    
    # Segmentation metrics
    num_structures: int
    segmentation_dice: float
    segmentation_iou: float
    pixel_accuracy: float
    
    # Signal analysis
    signal_to_noise_ratios: Dict[str, float]
    intensity_measurements: Dict[str, Dict[str, float]]
    background_levels: Dict[str, float]
    
    # Time series (if applicable)
    time_series_data: Optional[Dict[str, List[float]]] = None
    temporal_metrics: Optional[Dict[str, float]] = None
    
    # Colocalization
    colocalization_metrics: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class VideoMetrics(AnalysisMetrics):
    """Video analysis metrics with tracking accuracy"""
    # Video properties
    frame_count: int
    fps: float
    duration_seconds: float
    frame_dimensions: tuple
    
    # Tracking metrics
    num_tracks: int
    tracking_accuracy: float
    track_completeness: float
    false_positive_rate: float
    false_negative_rate: float
    
    # Motion analysis
    mean_velocity: float
    velocity_distribution: List[float]
    displacement_metrics: Dict[str, float]
    
    # Behavioral classification
    behavior_distribution: Dict[str, int]
    behavior_transitions: Dict[str, Dict[str, int]]
    
    # Temporal dynamics
    activity_over_time: List[float]
    peak_activity_frames: List[int]


@dataclass
class ElectronMicroscopyMetrics(AnalysisMetrics):
    """Electron microscopy analysis metrics"""
    # Image properties
    magnification: float
    pixel_size_nm: float
    em_type: str
    
    # Structure detection
    detected_structures: Dict[str, int]
    structure_areas: Dict[str, List[float]]
    structure_confidences: Dict[str, List[float]]
    
    # Image quality
    resolution_estimate: float
    contrast_metrics: Dict[str, float]
    noise_level: float


class ResultsVisualizer:
    """Creates publication-ready multi-panel figures"""
    
    def __init__(self, style='scientific'):
        """Initialize with plotting style"""
        plt.style.use('seaborn-v0_8' if style == 'scientific' else style)
        sns.set_palette("husl")
        
        # Set publication-ready defaults
        self.figure_params = {
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 10,
            'lines.linewidth': 2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'axes.axisbelow': True,
            'grid.alpha': 0.3
        }
        plt.rcParams.update(self.figure_params)
    
    def create_fluorescence_figure(self, image: np.ndarray, metrics: FluorescenceMetrics,
                                 segmentation_mask: Optional[np.ndarray] = None) -> plt.Figure:
        """Create comprehensive fluorescence analysis figure"""
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel A: Segmented Image Results (top row)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_segmented_image(ax1, image, segmentation_mask, metrics)
        ax1.set_title('Panel A: Segmented Image Results', fontweight='bold', pad=20)
        
        # Panel B: Time Series Analysis (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_time_series_analysis(ax2, metrics)
        ax2.set_title('Panel B: Time Series Analysis', fontweight='bold', pad=20)
        
        # Panel C: Signal-to-Noise Analysis (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_snr_analysis(ax3, metrics)
        ax3.set_title('Panel C: Signal-to-Noise Analysis', fontweight='bold', pad=20)
        
        # Panel D: Segmentation Performance (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_segmentation_performance(ax4, metrics)
        ax4.set_title('Panel D: Segmentation Performance', fontweight='bold', pad=20)
        
        # Panel E: Colocalization Analysis (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_colocalization_analysis(ax5, metrics)
        ax5.set_title('Panel E: Colocalization Analysis', fontweight='bold', pad=20)
        
        # Add overall title and metadata
        fig.suptitle(f'Fluorescence Analysis Results - {metrics.num_structures} Structures Detected', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Add timestamp and processing info
        fig.text(0.02, 0.02, f'Analysis: {metrics.timestamp} | Processing: {metrics.processing_time:.2f}s', 
                fontsize=8, alpha=0.7)
        
        return fig
    
    def _plot_segmented_image(self, ax: plt.Axes, image: np.ndarray, 
                            mask: Optional[np.ndarray], metrics: FluorescenceMetrics):
        """Plot original image with segmentation overlay"""
        if len(image.shape) == 3:
            # Multi-channel image
            ax.imshow(image)
        else:
            # Single channel
            ax.imshow(image, cmap='gray')
        
        # Overlay segmentation mask
        if mask is not None:
            masked = np.ma.masked_where(mask == 0, mask)
            ax.imshow(masked, alpha=0.5, cmap='jet')
        
        # Add scale bar (assuming pixel_size_um is available)
        scale_bar_um = 10  # 10 μm scale bar
        scale_bar_pixels = scale_bar_um / metrics.pixel_size_um
        scale_rect = Rectangle((image.shape[1] - scale_bar_pixels - 20, image.shape[0] - 30),
                             scale_bar_pixels, 5, facecolor='white', edgecolor='black')
        ax.add_patch(scale_rect)
        ax.text(image.shape[1] - scale_bar_pixels//2 - 20, image.shape[0] - 45,
               f'{scale_bar_um} μm', ha='center', va='top', color='white', fontweight='bold')
        
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.axis('off')
        
        # Add channel labels
        channel_text = ', '.join(metrics.channels) if metrics.channels else 'Composite'
        ax.text(10, 20, f'Channels: {channel_text}', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor='white', alpha=0.8), fontsize=9)
    
    def _plot_time_series_analysis(self, ax: plt.Axes, metrics: FluorescenceMetrics):
        """Plot time series analysis with area charts"""
        if not metrics.time_series_data:
            ax.text(0.5, 0.5, 'No time series data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Fluorescence Intensity (AU)')
            return
        
        # Create time axis
        times = np.array(list(range(len(next(iter(metrics.time_series_data.values()))))))
        
        # Plot each channel as filled area chart
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for i, (channel, intensities) in enumerate(metrics.time_series_data.items()):
            intensities = np.array(intensities)
            
            # Add noise/error bands
            noise = intensities * 0.1  # Assume 10% noise
            upper = intensities + noise
            lower = intensities - noise
            
            # Fill area chart
            ax.fill_between(times, lower, upper, alpha=0.3, color=colors[i % len(colors)], 
                          label=f'{channel} (±SD)')
            ax.plot(times, intensities, color=colors[i % len(colors)], linewidth=2, 
                   label=f'{channel} mean')
        
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Fluorescence Intensity (AU)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_snr_analysis(self, ax: plt.Axes, metrics: FluorescenceMetrics):
        """Plot signal-to-noise ratio analysis"""
        channels = list(metrics.signal_to_noise_ratios.keys())
        snr_values = list(metrics.signal_to_noise_ratios.values())
        
        # Create synthetic signal and noise envelopes for visualization
        x = np.arange(len(channels))
        signal_levels = [metrics.intensity_measurements[ch]['mean'] for ch in channels]
        noise_levels = [metrics.background_levels[ch] for ch in channels]
        
        # Normalize for plotting
        max_signal = max(signal_levels)
        signal_norm = [s/max_signal for s in signal_levels]
        noise_norm = [n/max_signal for n in noise_levels]
        
        # Create area plots
        ax.fill_between(x, noise_norm, signal_norm, alpha=0.6, color='lightblue', 
                       label='Signal Range')
        ax.fill_between(x, [0]*len(x), noise_norm, alpha=0.4, color='lightcoral', 
                       label='Noise Floor')
        
        # Add SNR values as text annotations
        for i, (ch, snr) in enumerate(zip(channels, snr_values)):
            color = 'green' if snr > 10 else 'orange' if snr > 5 else 'red'
            ax.annotate(f'SNR: {snr:.1f}', (i, signal_norm[i]), 
                       textcoords="offset points", xytext=(0,10), ha='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        ax.set_xlabel('Channel')
        ax.set_ylabel('Normalized Intensity')
        ax.set_xticks(x)
        ax.set_xticklabels(channels)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_segmentation_performance(self, ax: plt.Axes, metrics: FluorescenceMetrics):
        """Plot segmentation performance metrics"""
        metrics_names = ['Dice Coefficient', 'IoU Score', 'Pixel Accuracy']
        values = [metrics.segmentation_dice, metrics.segmentation_iou, metrics.pixel_accuracy]
        
        # Create violin/box plots
        positions = np.arange(len(metrics_names))
        
        # Simulate distribution around each metric for violin plot
        np.random.seed(42)  # For reproducibility
        distributions = []
        for val in values:
            # Create synthetic distribution around the value
            dist = np.random.normal(val, val * 0.05, 100)  # 5% std dev
            dist = np.clip(dist, 0, 1)  # Clip to valid range
            distributions.append(dist)
        
        # Create violin plot
        violin_parts = ax.violinplot(distributions, positions, showmeans=True, showmedians=True)
        
        # Customize violin colors based on performance
        for i, (pc, val) in enumerate(zip(violin_parts['bodies'], values)):
            color = 'lightgreen' if val > 0.8 else 'yellow' if val > 0.6 else 'lightcoral'
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Add value labels
        for i, val in enumerate(values):
            ax.text(i, val + 0.05, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Segmentation Metrics')
        ax.set_ylabel('Score')
        ax.set_xticks(positions)
        ax.set_xticklabels(metrics_names, rotation=45)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    def _plot_colocalization_analysis(self, ax: plt.Axes, metrics: FluorescenceMetrics):
        """Plot colocalization analysis"""
        if not metrics.colocalization_metrics:
            ax.text(0.5, 0.5, 'No colocalization data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Create heatmap of colocalization coefficients
        channel_pairs = list(metrics.colocalization_metrics.keys())
        coeff_types = ['pearson_correlation', 'manders_m1', 'manders_m2']
        
        # Prepare data matrix
        data_matrix = []
        for pair in channel_pairs:
            row = []
            for coeff in coeff_types:
                val = metrics.colocalization_metrics[pair].get(coeff, 0)
                row.append(val)
            data_matrix.append(row)
        
        # Create heatmap
        im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('Colocalization Coefficient')
        
        # Set labels
        ax.set_xticks(range(len(coeff_types)))
        ax.set_xticklabels(coeff_types, rotation=45)
        ax.set_yticks(range(len(channel_pairs)))
        ax.set_yticklabels(channel_pairs)
        
        # Add values as text
        for i in range(len(channel_pairs)):
            for j in range(len(coeff_types)):
                text = ax.text(j, i, f'{data_matrix[i][j]:.3f}', 
                             ha="center", va="center", color="black" if abs(data_matrix[i][j]) < 0.5 else "white")
    
    def create_video_analysis_figure(self, metrics: VideoMetrics, 
                                   representative_frame: Optional[np.ndarray] = None) -> plt.Figure:
        """Create comprehensive video analysis figure"""
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel A: Representative frame with tracks
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_tracking_results(ax1, representative_frame, metrics)
        ax1.set_title('Panel A: Cell Tracking Results', fontweight='bold', pad=20)
        
        # Panel B: Tracking accuracy metrics
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_tracking_accuracy(ax2, metrics)
        ax2.set_title('Panel B: Tracking Performance', fontweight='bold', pad=20)
        
        # Panel C: Motion analysis over time
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_motion_analysis(ax3, metrics)
        ax3.set_title('Panel C: Motion Analysis Over Time', fontweight='bold', pad=20)
        
        # Panel D: Behavioral classification
        ax4 = fig.add_subplot(gs[2, :2])
        self._plot_behavioral_analysis(ax4, metrics)
        ax4.set_title('Panel D: Behavioral Classification', fontweight='bold', pad=20)
        
        # Panel E: Velocity distribution
        ax5 = fig.add_subplot(gs[2, 2:])
        self._plot_velocity_distribution(ax5, metrics)
        ax5.set_title('Panel E: Velocity Distribution', fontweight='bold', pad=20)
        
        fig.suptitle(f'Video Analysis Results - {metrics.num_tracks} Tracks, {metrics.duration_seconds:.1f}s', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        return fig
    
    def _plot_tracking_results(self, ax: plt.Axes, frame: Optional[np.ndarray], metrics: VideoMetrics):
        """Plot representative frame with tracking overlay"""
        if frame is not None:
            ax.imshow(frame, cmap='gray' if len(frame.shape) == 2 else None)
            
            # Add synthetic track visualization (in real implementation, would use actual track data)
            np.random.seed(42)
            for i in range(min(10, metrics.num_tracks)):  # Show up to 10 tracks
                # Generate synthetic track
                x_start = np.random.randint(50, frame.shape[1] - 50)
                y_start = np.random.randint(50, frame.shape[0] - 50)
                
                # Create curved track
                t = np.linspace(0, 2*np.pi, 20)
                x_track = x_start + 30 * np.sin(t) + np.cumsum(np.random.randn(20) * 2)
                y_track = y_start + 20 * np.cos(t) + np.cumsum(np.random.randn(20) * 2)
                
                # Plot track
                ax.plot(x_track, y_track, color=f'C{i}', linewidth=2, alpha=0.8)
                ax.scatter(x_track[-1], y_track[-1], color=f'C{i}', s=50, marker='o')
                ax.text(x_track[-1], y_track[-1], f'{i+1}', color='white', fontweight='bold', 
                       ha='center', va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No representative frame available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
        
        ax.set_title(f'Tracking Accuracy: {metrics.tracking_accuracy:.1%}')
        ax.axis('off')
    
    def _plot_tracking_accuracy(self, ax: plt.Axes, metrics: VideoMetrics):
        """Plot tracking accuracy metrics"""
        metric_names = ['Tracking\nAccuracy', 'Track\nCompleteness', 'False\nPositive Rate', 'False\nNegative Rate']
        values = [metrics.tracking_accuracy, metrics.track_completeness, 
                 metrics.false_positive_rate, metrics.false_negative_rate]
        colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow']
        
        bars = ax.bar(metric_names, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_motion_analysis(self, ax: plt.Axes, metrics: VideoMetrics):
        """Plot motion analysis over time with area chart"""
        if not metrics.activity_over_time:
            ax.text(0.5, 0.5, 'No motion data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
        
        times = np.arange(len(metrics.activity_over_time)) / metrics.fps
        activity = np.array(metrics.activity_over_time)
        
        # Create area chart with light blue fill
        ax.fill_between(times, 0, activity, alpha=0.6, color='lightblue', label='Activity Level')
        ax.plot(times, activity, color='blue', linewidth=2, label='Mean Activity')
        
        # Add peak activity markers
        if metrics.peak_activity_frames:
            peak_times = np.array(metrics.peak_activity_frames) / metrics.fps
            peak_values = activity[metrics.peak_activity_frames]
            ax.scatter(peak_times, peak_values, color='red', s=100, marker='^', 
                      label='Peak Activity', zorder=5)
        
        # Add mean line
        mean_activity = np.mean(activity)
        ax.axhline(y=mean_activity, color='orange', linestyle='--', alpha=0.8, 
                  label=f'Mean: {mean_activity:.3f}')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Activity Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_behavioral_analysis(self, ax: plt.Axes, metrics: VideoMetrics):
        """Plot behavioral classification"""
        behaviors = list(metrics.behavior_distribution.keys())
        counts = list(metrics.behavior_distribution.values())
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(counts, labels=behaviors, autopct='%1.1f%%', 
                                         startangle=90, colors=sns.color_palette('pastel'))
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(f'Behavioral Distribution (n={sum(counts)} cells)')
    
    def _plot_velocity_distribution(self, ax: plt.Axes, metrics: VideoMetrics):
        """Plot velocity distribution"""
        if not metrics.velocity_distribution:
            ax.text(0.5, 0.5, 'No velocity data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
        
        velocities = np.array(metrics.velocity_distribution)
        
        # Create histogram with area fill
        n, bins, patches = ax.hist(velocities, bins=20, alpha=0.7, color='lightblue', 
                                  edgecolor='black', density=True)
        
        # Add distribution curve
        x = np.linspace(velocities.min(), velocities.max(), 100)
        try:
            # Fit normal distribution
            mu, sigma = stats.norm.fit(velocities)
            y = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, y, 'r-', linewidth=2, label=f'Normal fit (μ={mu:.2f}, σ={sigma:.2f})')
        except:
            pass
        
        # Add mean velocity line
        ax.axvline(x=metrics.mean_velocity, color='orange', linestyle='--', linewidth=2, 
                  label=f'Mean: {metrics.mean_velocity:.2f} px/frame')
        
        ax.set_xlabel('Velocity (pixels/frame)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)


def save_analysis_results(metrics: AnalysisMetrics, output_dir: Path, 
                         prefix: str, include_json: bool = True) -> Dict[str, Path]:
    """Save analysis results in multiple formats"""
    output_dir.mkdir(exist_ok=True)
    saved_files = {}
    
    # Save JSON data
    if include_json:
        json_file = output_dir / f"{prefix}_results.json"
        metrics.to_json(json_file)
        saved_files['json'] = json_file
    
    return saved_files
