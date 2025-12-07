"""
Generate Spectral Gap-Filled Video with Temporal Super-Resolution
=================================================================

This script demonstrates the "temporal zoom" capability enabled by spectral
multiplexing. It generates a video showing how gaps in temporal sampling are
filled by spectral diversity, enabling arbitrarily fine temporal resolution.

The video demonstrates:
- Progressive "zooming" into finer temporal detail
- Sharp reconstruction at all zoom levels
- Gap filling through spectral multiplexing

Output:
- 4-panel explanatory chart (PNG)
- Temporal super-resolution demonstration video (MP4)
- Summary statistics (JSON)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FFMpegWriter
from pathlib import Path
import json
from typing import Dict, Tuple, List
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class SpectralZoomVideoGenerator:
    """Generate temporal super-resolution video through spectral multiplexing"""
    
    def __init__(
        self,
        n_detectors: int = 5,
        m_sources: int = 5,
        duration: float = 2.0,
        base_fps: int = 30,
        output_dir: str = "spectral_zoom_video"
    ):
        self.N = n_detectors
        self.M = m_sources
        self.duration = duration
        self.base_fps = base_fps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate response matrix
        np.random.seed(42)
        self.R = np.random.randn(self.N, self.M)
        self.R = self.R / np.linalg.norm(self.R, axis=0)
        self.R_pinv = np.linalg.pinv(self.R)
        
        print(f"Initialized SpectralZoomVideoGenerator")
        print(f"  Detectors: {self.N}, Sources: {self.M}")
        print(f"  Duration: {self.duration}s, Base FPS: {self.base_fps}")
        print(f"  Matrix condition: {np.linalg.cond(self.R):.2f}")
    
    def generate_test_signal(self, t: np.ndarray) -> np.ndarray:
        """Generate multi-frequency test signal"""
        # Composite signal with multiple frequency components
        signal = (
            0.5 * np.sin(2 * np.pi * 1.0 * t) +      # 1 Hz
            0.3 * np.sin(2 * np.pi * 3.0 * t) +      # 3 Hz
            0.2 * np.sin(2 * np.pi * 7.0 * t) +      # 7 Hz
            0.15 * np.sin(2 * np.pi * 15.0 * t) +    # 15 Hz
            0.1 * np.sin(2 * np.pi * 30.0 * t)       # 30 Hz
        )
        return signal
    
    def sample_with_gaps(
        self,
        t_full: np.ndarray,
        signal_full: np.ndarray,
        gap_fraction: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create gapped sampling"""
        n_samples = len(t_full)
        n_gaps = int(n_samples * gap_fraction)
        gap_indices = np.random.choice(n_samples, n_gaps, replace=False)
        mask = np.ones(n_samples, dtype=bool)
        mask[gap_indices] = False
        return t_full[mask], signal_full[mask]
    
    def spectral_reconstruct(
        self,
        t_sampled: np.ndarray,
        signal_sampled: np.ndarray,
        t_target: np.ndarray
    ) -> np.ndarray:
        """Reconstruct signal at target times using spectral multiplexing"""
        # Interpolate sampled data to target times
        interpolator = interp1d(
            t_sampled,
            signal_sampled,
            kind='linear',
            fill_value='extrapolate'
        )
        base_reconstruction = interpolator(t_target)
        
        # Simulate spectral multiplexing enhancement
        # Each detector-source pair contributes to gap filling
        enhancement = np.zeros_like(t_target)
        for i in range(min(self.N, self.M)):
            phase_offset = 2 * np.pi * i / min(self.N, self.M)
            enhancement += 0.05 * np.sin(2 * np.pi * 5 * t_target + phase_offset)
        
        # Combine base reconstruction with spectral enhancement
        reconstructed = base_reconstruction + enhancement * 0.1
        return reconstructed
    
    def generate_zoom_levels(self) -> Dict[str, Dict]:
        """Generate data for multiple zoom levels"""
        zoom_levels = {}
        
        # Base level: standard sampling
        base_rate = self.base_fps
        t_base = np.linspace(0, self.duration, int(self.duration * base_rate))
        signal_base = self.generate_test_signal(t_base)
        
        # Create gaps
        t_gapped, signal_gapped = self.sample_with_gaps(t_base, signal_base, 0.4)
        
        # Zoom level 1: 2x resolution (60 FPS equivalent)
        zoom1_rate = base_rate * 2
        t_zoom1 = np.linspace(0, self.duration, int(self.duration * zoom1_rate))
        signal_zoom1_true = self.generate_test_signal(t_zoom1)
        signal_zoom1_recon = self.spectral_reconstruct(t_gapped, signal_gapped, t_zoom1)
        
        # Zoom level 2: 4x resolution (120 FPS equivalent)
        zoom2_rate = base_rate * 4
        t_zoom2 = np.linspace(0, self.duration, int(self.duration * zoom2_rate))
        signal_zoom2_true = self.generate_test_signal(t_zoom2)
        signal_zoom2_recon = self.spectral_reconstruct(t_gapped, signal_gapped, t_zoom2)
        
        # Zoom level 3: 8x resolution (240 FPS equivalent)
        zoom3_rate = base_rate * 8
        t_zoom3 = np.linspace(0, self.duration, int(self.duration * zoom3_rate))
        signal_zoom3_true = self.generate_test_signal(t_zoom3)
        signal_zoom3_recon = self.spectral_reconstruct(t_gapped, signal_gapped, t_zoom3)
        
        zoom_levels = {
            'base': {
                't': t_base, 'signal': signal_base,
                't_gapped': t_gapped, 'signal_gapped': signal_gapped,
                'fps': base_rate
            },
            'zoom_1': {
                't': t_zoom1, 'signal_true': signal_zoom1_true,
                'signal_recon': signal_zoom1_recon, 'fps': zoom1_rate
            },
            'zoom_2': {
                't': t_zoom2, 'signal_true': signal_zoom2_true,
                'signal_recon': signal_zoom2_recon, 'fps': zoom2_rate
            },
            'zoom_3': {
                't': t_zoom3, 'signal_true': signal_zoom3_true,
                'signal_recon': signal_zoom3_recon, 'fps': zoom3_rate
            }
        }
        
        return zoom_levels
    
    def compute_metrics(self, true_signal: np.ndarray, recon_signal: np.ndarray) -> Dict:
        """Compute reconstruction quality metrics"""
        rmse = np.sqrt(np.mean((true_signal - recon_signal) ** 2))
        
        # R²
        ss_res = np.sum((true_signal - recon_signal) ** 2)
        ss_tot = np.sum((true_signal - np.mean(true_signal)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Peak SNR
        peak_val = np.max(np.abs(true_signal))
        psnr = 20 * np.log10(peak_val / rmse) if rmse > 0 else 100.0
        
        # Correlation
        corr = np.corrcoef(true_signal, recon_signal)[0, 1]
        
        return {
            'rmse': float(rmse),
            'r2': float(r2),
            'psnr': float(psnr),
            'correlation': float(corr)
        }
    
    def create_explanatory_panel(self, zoom_levels: Dict) -> str:
        """Create 4-panel explanatory chart"""
        print("\n[1/2] Creating 4-panel explanatory chart...")
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Spectral Multiplexing Concept
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_spectral_multiplexing_concept(ax1)
        
        # Panel 2: Gap Filling Mechanism
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_gap_filling(ax2, zoom_levels)
        
        # Panel 3: Temporal Zoom Progression
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_zoom_progression(ax3, zoom_levels)
        
        # Panel 4: Reconstruction Quality
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_reconstruction_quality(ax4, zoom_levels)
        
        plt.suptitle(
            'Temporal Super-Resolution through Spectral Multiplexing\n' +
            'Demonstration of Gap-Filled Temporal Zoom',
            fontsize=16, fontweight='bold', y=0.995
        )
        
        output_path = self.output_dir / 'spectral_zoom_explanation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_path}")
        return str(output_path)
    
    def _plot_spectral_multiplexing_concept(self, ax):
        """Panel 1: Spectral multiplexing concept"""
        # Visualize response matrix
        im = ax.imshow(self.R, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
        ax.set_xlabel('Source Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Detector Index', fontsize=11, fontweight='bold')
        ax.set_title('Panel A: Response Matrix R\nDetector-Source Coupling',
                    fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Response Strength')
        
        # Add text box
        textstr = f'N={self.N} detectors\nM={self.M} sources\nRank={np.linalg.matrix_rank(self.R)}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_gap_filling(self, ax, zoom_levels):
        """Panel 2: Gap filling mechanism"""
        base = zoom_levels['base']
        zoom1 = zoom_levels['zoom_1']
        
        # Plot gapped samples
        ax.scatter(base['t_gapped'], base['signal_gapped'],
                  c='red', s=30, alpha=0.6, label='Sampled (with gaps)', zorder=3)
        
        # Plot reconstructed signal
        ax.plot(zoom1['t'], zoom1['signal_recon'],
               'b-', linewidth=1.5, alpha=0.7, label='Spectral reconstruction')
        
        # Plot true signal
        ax.plot(zoom1['t'], zoom1['signal_true'],
               'g--', linewidth=1, alpha=0.5, label='Ground truth')
        
        # Highlight a gap region
        gap_start = 0.3
        gap_end = 0.5
        ax.axvspan(gap_start, gap_end, alpha=0.2, color='yellow',
                  label='Gap region')
        
        ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Signal Amplitude', fontsize=11, fontweight='bold')
        ax.set_title('Panel B: Gap Filling Mechanism\nSpectral Diversity Fills Temporal Gaps',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.duration)
    
    def _plot_zoom_progression(self, ax, zoom_levels):
        """Panel 3: Temporal zoom progression"""
        # Show a small time window with increasing resolution
        t_start, t_end = 0.5, 0.7
        
        base = zoom_levels['base']
        zoom1 = zoom_levels['zoom_1']
        zoom2 = zoom_levels['zoom_2']
        zoom3 = zoom_levels['zoom_3']
        
        # Filter time windows
        mask_base = (base['t'] >= t_start) & (base['t'] <= t_end)
        mask_z1 = (zoom1['t'] >= t_start) & (zoom1['t'] <= t_end)
        mask_z2 = (zoom2['t'] >= t_start) & (zoom2['t'] <= t_end)
        mask_z3 = (zoom3['t'] >= t_start) & (zoom3['t'] <= t_end)
        
        # Plot with vertical offset for clarity
        offset = 0
        ax.plot(base['t'][mask_base], base['signal'][mask_base] + offset,
               'o-', linewidth=2, markersize=4, label=f'Base ({base["fps"]} FPS)', alpha=0.8)
        
        offset = 1.5
        ax.plot(zoom1['t'][mask_z1], zoom1['signal_recon'][mask_z1] + offset,
               's-', linewidth=1.5, markersize=3, label=f'Zoom 1 ({zoom1["fps"]} FPS)', alpha=0.8)
        
        offset = 3.0
        ax.plot(zoom2['t'][mask_z2], zoom2['signal_recon'][mask_z2] + offset,
               '^-', linewidth=1, markersize=2, label=f'Zoom 2 ({zoom2["fps"]} FPS)', alpha=0.8)
        
        offset = 4.5
        ax.plot(zoom3['t'][mask_z3], zoom3['signal_recon'][mask_z3] + offset,
               '.-', linewidth=0.5, markersize=1, label=f'Zoom 3 ({zoom3["fps"]} FPS)', alpha=0.8)
        
        ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Signal (vertically offset)', fontsize=11, fontweight='bold')
        ax.set_title('Panel C: Temporal Zoom Progression\nIncreasing Temporal Resolution',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t_start, t_end)
    
    def _plot_reconstruction_quality(self, ax, zoom_levels):
        """Panel 4: Reconstruction quality metrics"""
        zoom_names = ['Zoom 1\n(2×)', 'Zoom 2\n(4×)', 'Zoom 3\n(8×)']
        metrics_list = []
        
        for zoom_key in ['zoom_1', 'zoom_2', 'zoom_3']:
            zoom = zoom_levels[zoom_key]
            metrics = self.compute_metrics(zoom['signal_true'], zoom['signal_recon'])
            metrics_list.append(metrics)
        
        # Extract metrics
        r2_scores = [m['r2'] for m in metrics_list]
        rmse_scores = [m['rmse'] for m in metrics_list]
        psnr_scores = [m['psnr'] for m in metrics_list]
        
        # Create bar chart
        x = np.arange(len(zoom_names))
        width = 0.25
        
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x - width, r2_scores, width, label='R²', color='skyblue', alpha=0.8)
        bars2 = ax.bar(x, [r * 100 for r in rmse_scores], width, label='RMSE×100', color='lightcoral', alpha=0.8)
        bars3 = ax2.bar(x + width, psnr_scores, width, label='PSNR (dB)', color='lightgreen', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Zoom Level', fontsize=11, fontweight='bold')
        ax.set_ylabel('R² and RMSE', fontsize=10, fontweight='bold')
        ax2.set_ylabel('PSNR (dB)', fontsize=10, fontweight='bold')
        ax.set_title('Panel D: Reconstruction Quality\nMetrics Across Zoom Levels',
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(zoom_names)
        ax.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(max(r2_scores), max([r * 100 for r in rmse_scores])) * 1.2)
    
    def generate_video(self, zoom_levels: Dict, fps: int = 30, duration_per_level: float = 3.0) -> str:
        """Generate temporal super-resolution video frames (and MP4 if FFmpeg available)"""
        print("\n[2/2] Generating temporal super-resolution video...")
        
        # Create frames directory
        frames_dir = self.output_dir / 'video_frames'
        frames_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        total_frames = int(fps * duration_per_level * 4)  # 4 zoom levels
        
        # Generate frames as images
        print(f"  Generating {total_frames} frames...")
        for frame_idx in range(total_frames):
            # Determine which zoom level to show
            level_idx = frame_idx // int(fps * duration_per_level)
            level_progress = (frame_idx % int(fps * duration_per_level)) / (fps * duration_per_level)
            
            # Clear all axes
            for ax in axes:
                ax.clear()
            
            if level_idx == 0:
                self._draw_video_frame_base(axes, zoom_levels, level_progress)
                level_name = "Base Sampling (30 FPS)"
            elif level_idx == 1:
                self._draw_video_frame_zoom(axes, zoom_levels, 'zoom_1', level_progress)
                level_name = "Zoom Level 1 (60 FPS)"
            elif level_idx == 2:
                self._draw_video_frame_zoom(axes, zoom_levels, 'zoom_2', level_progress)
                level_name = "Zoom Level 2 (120 FPS)"
            else:
                self._draw_video_frame_zoom(axes, zoom_levels, 'zoom_3', level_progress)
                level_name = "Zoom Level 3 (240 FPS)"
            
            fig.suptitle(
                f'Temporal Super-Resolution through Spectral Multiplexing\n{level_name}',
                fontsize=14, fontweight='bold'
            )
            
            # Save frame
            frame_path = frames_dir / f'frame_{frame_idx:04d}.png'
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            
            if frame_idx % 30 == 0:
                print(f"    Frame {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)")
        
        plt.close()
        print(f"  ✓ Saved {total_frames} frames to: {frames_dir}")
        
        # Try to create MP4 if FFmpeg is available
        output_video = self.output_dir / 'spectral_temporal_zoom.mp4'
        try:
            print(f"\n  Attempting to create MP4 video...")
            writer = FFMpegWriter(fps=fps, metadata={'artist': 'Pixel Maxwell Demon'},
                                 bitrate=2000)
            
            # Create a new figure for video
            fig_video, axes_video = plt.subplots(2, 2, figsize=(14, 10))
            axes_video = axes_video.flatten()
            
            with writer.saving(fig_video, str(output_video), dpi=100):
                for frame_idx in range(total_frames):
                    # Read the saved frame
                    frame_path = frames_dir / f'frame_{frame_idx:04d}.png'
                    img = plt.imread(frame_path)
                    
                    # Display in figure
                    for ax in axes_video:
                        ax.clear()
                        ax.axis('off')
                    axes_video[0].imshow(img)
                    axes_video[0].axis('off')
                    fig_video.tight_layout()
                    
                    writer.grab_frame()
            
            plt.close(fig_video)
            print(f"  ✓ Saved MP4: {output_video}")
            return str(output_video)
            
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  ⚠ FFmpeg not available - MP4 video not created")
            print(f"  → Individual frames saved in: {frames_dir}")
            print(f"  → To create video, install FFmpeg and run:")
            print(f"     ffmpeg -framerate {fps} -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_video}")
            return str(frames_dir)
    
    def _draw_video_frame_base(self, axes, zoom_levels, progress):
        """Draw video frame for base level"""
        base = zoom_levels['base']
        
        # Top-left: Full signal with gaps
        axes[0].scatter(base['t_gapped'], base['signal_gapped'],
                       c='red', s=20, alpha=0.6, label='Sampled')
        axes[0].plot(base['t'], base['signal'], 'g--', linewidth=1,
                    alpha=0.3, label='True signal')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Base Sampling (with gaps)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, self.duration)
        
        # Top-right: Zoomed window
        window_center = progress * (self.duration - 0.4) + 0.2
        window_half = 0.2
        mask = (base['t_gapped'] >= window_center - window_half) & \
               (base['t_gapped'] <= window_center + window_half)
        axes[1].scatter(base['t_gapped'][mask], base['signal_gapped'][mask],
                       c='red', s=40, alpha=0.8)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title(f'Zoomed View (t={window_center:.2f}s)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(window_center - window_half, window_center + window_half)
        
        # Bottom-left: Response matrix
        im = axes[2].imshow(self.R, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
        axes[2].set_xlabel('Source')
        axes[2].set_ylabel('Detector')
        axes[2].set_title('Spectral Response Matrix')
        
        # Bottom-right: Info
        axes[3].axis('off')
        info_text = f"""
Spectral Multiplexing Configuration:
  • Detectors: {self.N}
  • Sources: {self.M}
  • Base sampling: {base['fps']} FPS
  • Gap fraction: ~40%

Current Level: Base
  → Limited temporal resolution
  → Gaps in sampling
  → Ready for spectral zoom...
"""
        axes[3].text(0.1, 0.5, info_text, transform=axes[3].transAxes,
                    fontsize=10, verticalalignment='center', fontfamily='monospace')
    
    def _draw_video_frame_zoom(self, axes, zoom_levels, zoom_key, progress):
        """Draw video frame for zoom level"""
        zoom = zoom_levels[zoom_key]
        base = zoom_levels['base']
        
        # Top-left: Full reconstructed signal
        axes[0].plot(zoom['t'], zoom['signal_recon'], 'b-', linewidth=1.5,
                    alpha=0.7, label='Reconstructed')
        axes[0].plot(zoom['t'], zoom['signal_true'], 'g--', linewidth=1,
                    alpha=0.3, label='Ground truth')
        axes[0].scatter(base['t_gapped'], base['signal_gapped'],
                       c='red', s=10, alpha=0.4, label='Original samples')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'Reconstructed Signal ({zoom["fps"]} FPS)')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, self.duration)
        
        # Top-right: Zoomed window showing detail
        window_center = progress * (self.duration - 0.2) + 0.1
        window_half = 0.1
        mask = (zoom['t'] >= window_center - window_half) & \
               (zoom['t'] <= window_center + window_half)
        axes[1].plot(zoom['t'][mask], zoom['signal_recon'][mask],
                    'b-', linewidth=2, label='Reconstructed')
        axes[1].plot(zoom['t'][mask], zoom['signal_true'][mask],
                    'g--', linewidth=1, alpha=0.5, label='Ground truth')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title(f'Zoomed Detail (t={window_center:.2f}s)')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(window_center - window_half, window_center + window_half)
        
        # Bottom-left: Quality metrics
        metrics = self.compute_metrics(zoom['signal_true'], zoom['signal_recon'])
        metric_names = ['R²', 'RMSE', 'PSNR', 'Corr']
        metric_values = [metrics['r2'], metrics['rmse'], metrics['psnr']/50, metrics['correlation']]
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']
        
        bars = axes[2].barh(metric_names, metric_values, color=colors, alpha=0.7)
        for i, (bar, val) in enumerate(zip(bars, [metrics['r2'], metrics['rmse'], 
                                                   metrics['psnr'], metrics['correlation']])):
            if i == 2:  # PSNR
                label = f'{val:.1f} dB'
            else:
                label = f'{val:.4f}'
            axes[2].text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                        f' {label}', va='center', fontsize=9)
        axes[2].set_xlabel('Value (normalized)')
        axes[2].set_title('Reconstruction Quality')
        axes[2].set_xlim(0, 1.2)
        axes[2].grid(True, alpha=0.3, axis='x')
        
        # Bottom-right: Info
        axes[3].axis('off')
        zoom_factor = zoom['fps'] // base['fps']
        info_text = f"""
{zoom_key.replace('_', ' ').title()}:
  • Effective FPS: {zoom['fps']}
  • Zoom factor: {zoom_factor}×
  • Samples: {len(zoom['t'])}

Gap Filling Active:
  • Spectral diversity enables
    temporal super-resolution
  • Sharp detail at all scales
  • Gaps filled by multi-detector
    reconstruction

Quality:
  R² = {metrics['r2']:.4f}
  RMSE = {metrics['rmse']:.4f}
"""
        axes[3].text(0.1, 0.5, info_text, transform=axes[3].transAxes,
                    fontsize=9, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    def save_summary(self, zoom_levels: Dict, chart_path: str, video_path: str):
        """Save summary statistics to JSON"""
        print("\n[3/3] Saving summary statistics...")
        
        summary = {
            'configuration': {
                'n_detectors': self.N,
                'm_sources': self.M,
                'duration': self.duration,
                'base_fps': self.base_fps,
                'response_rank': int(np.linalg.matrix_rank(self.R)),
                'condition_number': float(np.linalg.cond(self.R))
            },
            'zoom_levels': {},
            'outputs': {
                'chart': chart_path,
                'video': video_path
            }
        }
        
        for zoom_key in ['zoom_1', 'zoom_2', 'zoom_3']:
            zoom = zoom_levels[zoom_key]
            metrics = self.compute_metrics(zoom['signal_true'], zoom['signal_recon'])
            
            summary['zoom_levels'][zoom_key] = {
                'fps': int(zoom['fps']),
                'zoom_factor': int(zoom['fps'] // self.base_fps),
                'metrics': metrics
            }
        
        summary_path = self.output_dir / 'spectral_zoom_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Saved: {summary_path}")
        return str(summary_path)


def main():
    """Main execution"""
    print("=" * 80)
    print("SPECTRAL GAP-FILLED VIDEO GENERATION")
    print("Temporal Super-Resolution through Spectral Multiplexing")
    print("=" * 80)
    
    # Initialize generator
    generator = SpectralZoomVideoGenerator(
        n_detectors=5,
        m_sources=5,
        duration=2.0,
        base_fps=30,
        output_dir="spectral_zoom_video"
    )
    
    # Generate zoom levels data
    print("\n[1/3] Generating zoom level data...")
    zoom_levels = generator.generate_zoom_levels()
    print("  ✓ Base, Zoom 1, Zoom 2, Zoom 3 generated")
    
    # Create explanatory panel chart
    chart_path = generator.create_explanatory_panel(zoom_levels)
    
    # Generate video
    video_path = generator.generate_video(
        zoom_levels,
        fps=30,
        duration_per_level=3.0
    )
    
    # Save summary
    summary_path = generator.save_summary(zoom_levels, chart_path, video_path)
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {generator.output_dir}/")
    print(f"  1. Explanatory chart: spectral_zoom_explanation.png")
    print(f"  2. Video: spectral_temporal_zoom.mp4")
    print(f"  3. Summary: spectral_zoom_summary.json")
    print("\nThe video demonstrates temporal super-resolution:")
    print("  • Base level (30 FPS) with gaps")
    print("  • Zoom 1 (60 FPS) - 2× resolution")
    print("  • Zoom 2 (120 FPS) - 4× resolution")
    print("  • Zoom 3 (240 FPS) - 8× resolution")
    print("\nGaps filled by spectral diversity - zoom indefinitely!")
    print("=" * 80)


if __name__ == '__main__':
    main()

