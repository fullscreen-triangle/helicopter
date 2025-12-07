#!/usr/bin/env python3
"""
Comprehensive Signal Processing Visualization for Virtual Imaging Results

Creates detailed panel charts for each wavelength with:
- Circular phase histograms
- Power spectrum (2D FFT)
- Frequency spectrum
- Peak detection
- Radial profiles
- Phase maps
- Statistical analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from scipy import signal, ndimage
from scipy.fft import fft2, fftshift, fftfreq
from scipy.signal import find_peaks
from skimage import feature, filters
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class SignalProcessingAnalyzer:
    """Comprehensive signal processing analysis for images"""
    
    def __init__(self, image: np.ndarray, name: str):
        self.image = self._normalize_image(image)
        self.name = name
        self.h, self.w = self.image.shape[:2]
        
        # Convert to grayscale if needed
        if len(self.image.shape) == 3:
            self.gray = np.mean(self.image, axis=2)
        else:
            self.gray = self.image.copy()
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range"""
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        return img.astype(np.float32)
    
    def compute_2d_fft(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 2D FFT and power spectrum"""
        # Apply window to reduce edge effects
        window_h = signal.windows.hann(self.h)
        window_w = signal.windows.hann(self.w)
        window_2d = np.outer(window_h, window_w)
        
        windowed = self.gray * window_2d
        
        # Compute FFT
        fft_result = fft2(windowed)
        fft_shifted = fftshift(fft_result)
        
        # Power spectrum
        power_spectrum = np.abs(fft_shifted) ** 2
        
        # Log scale for visualization
        power_spectrum_log = np.log10(power_spectrum + 1)
        
        return fft_shifted, power_spectrum_log
    
    def compute_radial_profile(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute radial average profile"""
        center_y, center_x = np.array(data.shape) // 2
        y, x = np.indices(data.shape)
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = r.astype(int)
        
        # Radial binning
        max_r = min(center_x, center_y)
        radial_bins = np.arange(0, max_r)
        radial_profile = np.zeros(len(radial_bins))
        
        for i, radius in enumerate(radial_bins):
            mask = (r >= radius) & (r < radius + 1)
            if np.sum(mask) > 0:
                radial_profile[i] = np.mean(data[mask])
        
        return radial_bins, radial_profile
    
    def compute_phase_map(self) -> np.ndarray:
        """Compute phase map using Hilbert transform"""
        # Apply Hilbert transform along both axes
        analytic_signal = signal.hilbert(self.gray, axis=0)
        phase_y = np.angle(analytic_signal)
        
        analytic_signal = signal.hilbert(self.gray, axis=1)
        phase_x = np.angle(analytic_signal)
        
        # Combined phase
        phase_combined = np.arctan2(phase_y, phase_x)
        
        return phase_combined
    
    def detect_edges(self) -> np.ndarray:
        """Detect edges using Canny"""
        sigma = 2.0
        edges = feature.canny(self.gray, sigma=sigma)
        return edges
    
    def extract_peaks_1d(self) -> Dict:
        """Extract peaks from 1D profiles"""
        # Horizontal profile (average along columns)
        h_profile = np.mean(self.gray, axis=0)
        h_peaks, h_properties = find_peaks(h_profile, prominence=0.01)
        
        # Vertical profile (average along rows)
        v_profile = np.mean(self.gray, axis=1)
        v_peaks, v_properties = find_peaks(v_profile, prominence=0.01)
        
        return {
            'horizontal': {'profile': h_profile, 'peaks': h_peaks, 'properties': h_properties},
            'vertical': {'profile': v_profile, 'peaks': v_peaks, 'properties': v_properties}
        }
    
    def compute_autocorrelation(self) -> np.ndarray:
        """Compute 2D autocorrelation"""
        # FFT-based autocorrelation
        fft = fft2(self.gray)
        power = np.abs(fft) ** 2
        autocorr = np.real(fftshift(np.fft.ifft2(power)))
        
        # Normalize
        autocorr = autocorr / autocorr.max()
        
        return autocorr
    
    def compute_gradient_statistics(self) -> Dict:
        """Compute gradient magnitude and direction statistics"""
        # Sobel gradients
        grad_x = filters.sobel_h(self.gray)
        grad_y = filters.sobel_v(self.gray)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        return {
            'magnitude': magnitude,
            'direction': direction,
            'mean_mag': np.mean(magnitude),
            'std_mag': np.std(magnitude),
            'max_mag': np.max(magnitude)
        }
    
    def compute_frequency_bands(self, fft_data: np.ndarray) -> Dict:
        """Analyze frequency bands (low, mid, high)"""
        power = np.abs(fft_data) ** 2
        center_y, center_x = np.array(power.shape) // 2
        y, x = np.indices(power.shape)
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        max_r = min(center_x, center_y)
        
        # Define bands
        low_band = r < max_r * 0.2
        mid_band = (r >= max_r * 0.2) & (r < max_r * 0.6)
        high_band = r >= max_r * 0.6
        
        return {
            'low': np.sum(power[low_band]) / np.sum(power),
            'mid': np.sum(power[mid_band]) / np.sum(power),
            'high': np.sum(power[high_band]) / np.sum(power)
        }


def create_circular_phase_histogram(phase_data: np.ndarray, ax: plt.Axes, title: str):
    """Create circular histogram for phase distribution"""
    # Flatten phase data
    phase_flat = phase_data.flatten()
    
    # Create bins
    n_bins = 36  # 10-degree bins
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _ = np.histogram(phase_flat, bins=bins)
    
    # Normalize
    hist = hist / hist.max()
    
    # Circular plot
    theta = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
    width = 2 * np.pi / n_bins
    
    ax = plt.subplot(111, projection='polar')
    bars = ax.bar(theta, hist, width=width, bottom=0.0, alpha=0.8, edgecolor='black')
    
    # Color bars by angle
    cm = plt.cm.hsv
    for bar, angle in zip(bars, theta):
        bar.set_facecolor(cm((angle + np.pi) / (2 * np.pi)))
    
    ax.set_title(title, pad=20, fontweight='bold')
    ax.set_ylim(0, 1.1)
    
    return ax


def create_comprehensive_panel(analyzer: SignalProcessingAnalyzer, output_path: Path):
    """Create comprehensive 4×4 panel chart with signal processing visualizations"""
    
    print(f"  Creating comprehensive panel for: {analyzer.name}")
    
    # Compute all analyses
    fft_data, power_spectrum = analyzer.compute_2d_fft()
    phase_map = analyzer.compute_phase_map()
    edges = analyzer.detect_edges()
    peaks_1d = analyzer.extract_peaks_1d()
    autocorr = analyzer.compute_autocorrelation()
    gradient_stats = analyzer.compute_gradient_statistics()
    freq_bands = analyzer.compute_frequency_bands(fft_data)
    
    # Radial profiles
    radial_bins_power, radial_profile_power = analyzer.compute_radial_profile(power_spectrum)
    radial_bins_autocorr, radial_profile_autocorr = analyzer.compute_radial_profile(autocorr)
    
    # Create 4×4 figure
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
    
    # Row 1: Original, Phase Map, Power Spectrum, Edges
    # 1. Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(analyzer.gray, cmap='gray')
    ax1.set_title(f'Original Image\n{analyzer.name}', fontweight='bold', fontsize=11)
    ax1.axis('off')
    
    # 2. Phase Map
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(phase_map, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax2.set_title('Phase Map\n(Hilbert Transform)', fontweight='bold', fontsize=11)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Phase (rad)')
    
    # 3. Power Spectrum
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(power_spectrum, cmap='hot', origin='lower')
    ax3.set_title('2D Power Spectrum\n(Log Scale)', fontweight='bold', fontsize=11)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Log Power')
    
    # 4. Edge Detection
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(edges, cmap='gray')
    ax4.set_title('Edge Detection\n(Canny)', fontweight='bold', fontsize=11)
    ax4.axis('off')
    
    # Row 2: Circular Phase Histogram, Gradient Magnitude, Autocorrelation, Frequency Bands
    # 5. Circular Phase Histogram
    ax5 = fig.add_subplot(gs[1, 0], projection='polar')
    phase_flat = phase_map.flatten()
    n_bins = 36
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _ = np.histogram(phase_flat, bins=bins)
    hist = hist / hist.max()
    theta = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
    width = 2 * np.pi / n_bins
    bars = ax5.bar(theta, hist, width=width, bottom=0.0, alpha=0.8, edgecolor='black')
    cm = plt.cm.hsv
    for bar, angle in zip(bars, theta):
        bar.set_facecolor(cm((angle + np.pi) / (2 * np.pi)))
    ax5.set_title('Circular Phase Histogram', pad=20, fontweight='bold', fontsize=11)
    ax5.set_ylim(0, 1.1)
    
    # 6. Gradient Magnitude
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(gradient_stats['magnitude'], cmap='viridis')
    ax6.set_title(f"Gradient Magnitude\nMean={gradient_stats['mean_mag']:.3f}", 
                  fontweight='bold', fontsize=11)
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04, label='Magnitude')
    
    # 7. Autocorrelation
    ax7 = fig.add_subplot(gs[1, 2])
    # Show central region
    center_h, center_w = autocorr.shape[0] // 2, autocorr.shape[1] // 2
    crop_size = min(200, center_h, center_w)
    autocorr_crop = autocorr[center_h-crop_size:center_h+crop_size, 
                             center_w-crop_size:center_w+crop_size]
    im7 = ax7.imshow(autocorr_crop, cmap='RdBu_r')
    ax7.set_title('2D Autocorrelation\n(Central Region)', fontweight='bold', fontsize=11)
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04, label='Correlation')
    
    # 8. Frequency Band Distribution
    ax8 = fig.add_subplot(gs[1, 3])
    bands = ['Low', 'Mid', 'High']
    values = [freq_bands['low'], freq_bands['mid'], freq_bands['high']]
    colors_bands = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax8.bar(bands, values, color=colors_bands, alpha=0.8, edgecolor='black', linewidth=2)
    ax8.set_ylabel('Power Fraction', fontsize=10, fontweight='bold')
    ax8.set_title('Frequency Band Distribution', fontweight='bold', fontsize=11)
    ax8.set_ylim(0, max(values) * 1.2)
    ax8.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Row 3: Horizontal Profile, Vertical Profile, Radial Profile (Power), Radial Profile (Autocorr)
    # 9. Horizontal Profile with Peaks
    ax9 = fig.add_subplot(gs[2, 0])
    h_profile = peaks_1d['horizontal']['profile']
    h_peaks = peaks_1d['horizontal']['peaks']
    ax9.plot(h_profile, 'b-', linewidth=1.5, label='Profile')
    ax9.plot(h_peaks, h_profile[h_peaks], 'ro', markersize=6, label=f'Peaks ({len(h_peaks)})')
    ax9.set_xlabel('Pixel X', fontweight='bold')
    ax9.set_ylabel('Intensity', fontweight='bold')
    ax9.set_title('Horizontal Profile & Peaks', fontweight='bold', fontsize=11)
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)
    
    # 10. Vertical Profile with Peaks
    ax10 = fig.add_subplot(gs[2, 1])
    v_profile = peaks_1d['vertical']['profile']
    v_peaks = peaks_1d['vertical']['peaks']
    ax10.plot(v_profile, 'g-', linewidth=1.5, label='Profile')
    ax10.plot(v_peaks, v_profile[v_peaks], 'ro', markersize=6, label=f'Peaks ({len(v_peaks)})')
    ax10.set_xlabel('Pixel Y', fontweight='bold')
    ax10.set_ylabel('Intensity', fontweight='bold')
    ax10.set_title('Vertical Profile & Peaks', fontweight='bold', fontsize=11)
    ax10.legend(fontsize=9)
    ax10.grid(True, alpha=0.3)
    
    # 11. Radial Profile (Power Spectrum)
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.plot(radial_bins_power, radial_profile_power, 'r-', linewidth=2)
    ax11.set_xlabel('Radial Frequency (pixels)', fontweight='bold')
    ax11.set_ylabel('Log Power', fontweight='bold')
    ax11.set_title('Radial Power Profile', fontweight='bold', fontsize=11)
    ax11.grid(True, alpha=0.3)
    ax11.set_xlim(0, len(radial_bins_power))
    
    # 12. Radial Profile (Autocorrelation)
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.plot(radial_bins_autocorr, radial_profile_autocorr, 'purple', linewidth=2)
    ax12.set_xlabel('Radius (pixels)', fontweight='bold')
    ax12.set_ylabel('Correlation', fontweight='bold')
    ax12.set_title('Radial Autocorrelation Profile', fontweight='bold', fontsize=11)
    ax12.grid(True, alpha=0.3)
    ax12.set_xlim(0, len(radial_bins_autocorr))
    
    # Row 4: Intensity Histogram, Phase Histogram, Gradient Direction, Statistics Summary
    # 13. Intensity Histogram
    ax13 = fig.add_subplot(gs[3, 0])
    ax13.hist(analyzer.gray.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax13.set_xlabel('Intensity', fontweight='bold')
    ax13.set_ylabel('Count', fontweight='bold')
    ax13.set_title('Intensity Distribution', fontweight='bold', fontsize=11)
    ax13.grid(axis='y', alpha=0.3)
    
    # 14. Phase Histogram (Linear)
    ax14 = fig.add_subplot(gs[3, 1])
    ax14.hist(phase_map.flatten(), bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax14.set_xlabel('Phase (radians)', fontweight='bold')
    ax14.set_ylabel('Count', fontweight='bold')
    ax14.set_title('Phase Distribution', fontweight='bold', fontsize=11)
    ax14.grid(axis='y', alpha=0.3)
    
    # 15. Gradient Direction (2D histogram)
    ax15 = fig.add_subplot(gs[3, 2])
    im15 = ax15.imshow(gradient_stats['direction'], cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax15.set_title('Gradient Direction Map', fontweight='bold', fontsize=11)
    ax15.axis('off')
    plt.colorbar(im15, ax=ax15, fraction=0.046, pad=0.04, label='Angle (rad)')
    
    # 16. Statistics Summary
    ax16 = fig.add_subplot(gs[3, 3])
    ax16.axis('off')
    
    # Compute statistics
    stats_text = f"""
    STATISTICAL SUMMARY
    {'='*30}
    
    Intensity:
      Mean: {np.mean(analyzer.gray):.4f}
      Std:  {np.std(analyzer.gray):.4f}
      Min:  {np.min(analyzer.gray):.4f}
      Max:  {np.max(analyzer.gray):.4f}
    
    Gradient:
      Mean: {gradient_stats['mean_mag']:.4f}
      Std:  {gradient_stats['std_mag']:.4f}
      Max:  {gradient_stats['max_mag']:.4f}
    
    Peaks:
      Horizontal: {len(peaks_1d['horizontal']['peaks'])}
      Vertical:   {len(peaks_1d['vertical']['peaks'])}
    
    Frequency Bands:
      Low:  {freq_bands['low']:.3f}
      Mid:  {freq_bands['mid']:.3f}
      High: {freq_bands['high']:.3f}
    
    Phase:
      Mean: {np.mean(phase_map):.4f}
      Std:  {np.std(phase_map):.4f}
    """
    
    ax16.text(0.05, 0.95, stats_text, transform=ax16.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Main title
    fig.suptitle(f'Signal Processing Analysis: {analyzer.name}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {output_path}")


def process_virtual_imaging_results():
    """Main processing function"""
    print("=" * 80)
    print("VIRTUAL IMAGING SIGNAL PROCESSING VISUALIZATION")
    print("=" * 80)
    print()
    
    # Find virtual_imaging_results directory
    base_dir = Path("virtual_imaging_results")
    if not base_dir.exists():
        print(f"✗ Directory not found: {base_dir}")
        return
    
    # Find all .npy files
    npy_files = list(base_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} .npy files")
    
    if len(npy_files) == 0:
        print("✗ No .npy files found!")
        return
    
    # Group by wavelength/type
    groups = {}
    for npy_file in npy_files:
        # Parse filename (e.g., "virtual_650nm.npy", "virtual_fluorescence_561nm.npy")
        name = npy_file.stem
        
        # Extract wavelength or modality
        if 'wavelength' in name or 'nm' in name:
            # Extract wavelength number
            import re
            match = re.search(r'(\d+)nm', name)
            if match:
                key = f"{match.group(1)}nm"
            else:
                key = name
        else:
            key = name
        
        if key not in groups:
            groups[key] = []
        groups[key].append(npy_file)
    
    print(f"\nGrouped into {len(groups)} categories:")
    for key, files in groups.items():
        print(f"  • {key}: {len(files)} file(s)")
    
    # Create output directory
    output_dir = Path("signal_processing_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each group
    print(f"\nProcessing signal analysis...")
    all_stats = {}
    
    for key, files in groups.items():
        print(f"\n[{key}]")
        
        for npy_file in files:
            try:
                # Load data
                data = np.load(npy_file)
                
                # Create analyzer
                analyzer = SignalProcessingAnalyzer(data, npy_file.stem)
                
                # Create comprehensive panel
                output_path = output_dir / f"{npy_file.stem}_signal_analysis.png"
                create_comprehensive_panel(analyzer, output_path)
                
                # Store stats
                fft_data, power_spectrum = analyzer.compute_2d_fft()
                gradient_stats = analyzer.compute_gradient_statistics()
                freq_bands = analyzer.compute_frequency_bands(fft_data)
                
                all_stats[npy_file.stem] = {
                    'intensity_mean': float(np.mean(analyzer.gray)),
                    'intensity_std': float(np.std(analyzer.gray)),
                    'gradient_mean': float(gradient_stats['mean_mag']),
                    'gradient_std': float(gradient_stats['std_mag']),
                    'freq_low': float(freq_bands['low']),
                    'freq_mid': float(freq_bands['mid']),
                    'freq_high': float(freq_bands['high'])
                }
                
            except Exception as e:
                print(f"    ✗ Error processing {npy_file.name}: {e}")
    
    # Save statistics
    stats_file = output_dir / "analysis_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n✓ Saved statistics: {stats_file}")
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}/")
    print(f"Generated {len(all_stats)} comprehensive analysis panels")
    print("\nEach panel includes:")
    print("  • Original image & phase map")
    print("  • 2D power spectrum & edge detection")
    print("  • Circular phase histogram")
    print("  • Gradient magnitude & autocorrelation")
    print("  • Frequency band distribution")
    print("  • Horizontal/vertical profiles with peak detection")
    print("  • Radial profiles (power & autocorrelation)")
    print("  • Statistical histograms & summary")
    print("=" * 80)


if __name__ == '__main__':
    process_virtual_imaging_results()

