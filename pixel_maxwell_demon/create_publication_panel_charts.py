#!/usr/bin/env python3
"""
Create comprehensive publication-quality panel charts from NPY visualizations

Combines individual NPY files into informative multi-panel visualizations with:
- Radar charts for multi-dimensional comparisons
- Polar phase distributions
- Statistical heatmaps
- Comparative bar charts
- Correlation matrices
- Temporal evolution plots
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
import matplotlib.cm as cm
from pathlib import Path as PathLib
from typing import Dict, List, Tuple, Optional
import json
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from collections import defaultdict
import re


class MultiModalAnalyzer:
    """Analyze and compare multiple detector/modality results"""
    
    def __init__(self, npy_dir: PathLib):
        self.npy_dir = npy_dir
        self.data_cache = {}
        self.metadata = {}
    
    def load_all_npy_files(self, search_dirs: Optional[List[PathLib]] = None) -> Dict[str, np.ndarray]:
        """Load all NPY files from multiple directories and organize by category"""
        if search_dirs is None:
            search_dirs = [self.npy_dir]
        
        # Collect all NPY files from all directories
        npy_files = []
        for search_dir in search_dirs:
            if search_dir.exists():
                npy_files.extend(list(search_dir.glob("**/*.npy")))
        
        print(f"  Total NPY files found: {len(npy_files)}")
        
        organized = defaultdict(list)
        
        for npy_file in npy_files:
            try:
                data = np.load(npy_file)
                
                # Parse filename for metadata
                name = npy_file.stem
                
                # Categorize by detector/experiment type
                if 'photodiode' in name.lower():
                    category = 'photodiode'
                elif 'raman' in name.lower():
                    category = 'raman'
                elif 'mass_spec' in name.lower() or 'mass' in name.lower():
                    category = 'mass_spec'
                elif 'interferometer' in name.lower():
                    category = 'interferometer'
                elif 'thermometer' in name.lower() or 'temperature' in name.lower():
                    category = 'thermometer'
                elif 'barometer' in name.lower() or 'pressure' in name.lower():
                    category = 'barometer'
                elif 'hygrometer' in name.lower() or 'humidity' in name.lower():
                    category = 'hygrometer'
                elif 'ir' in name.lower() and 'detector' in name.lower():
                    category = 'ir_detector'
                elif 'dual' in name.lower() or 'membrane' in name.lower():
                    category = 'dual_membrane'
                elif 'virtual' in name.lower():
                    category = 'virtual_imaging'
                else:
                    category = 'other'
                
                organized[category].append({
                    'name': name,
                    'data': data,
                    'path': npy_file
                })
                
                self.data_cache[name] = data
                
            except Exception as e:
                print(f"  Warning: Could not load {npy_file.name}: {e}")
        
        return dict(organized)
    
    def compute_statistics(self, data: np.ndarray) -> Dict:
        """Compute comprehensive statistics for data"""
        import warnings
        flat_data = data.flatten()
        
        # Suppress warnings for nearly constant data
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            skew_val = stats.skew(flat_data)
            kurt_val = stats.kurtosis(flat_data)
        
        return {
            'mean': float(np.mean(flat_data)),
            'std': float(np.std(flat_data)),
            'min': float(np.min(flat_data)),
            'max': float(np.max(flat_data)),
            'median': float(np.median(flat_data)),
            'q25': float(np.percentile(flat_data, 25)),
            'q75': float(np.percentile(flat_data, 75)),
            'skewness': 0.0 if not np.isfinite(skew_val) else float(skew_val),
            'kurtosis': 0.0 if not np.isfinite(kurt_val) else float(kurt_val),
            'entropy': float(stats.entropy(np.histogram(flat_data, bins=50)[0] + 1e-10))
        }


def create_detector_comparison_panel(organized_data: Dict, output_path: PathLib):
    """
    Create comprehensive detector comparison panel
    
    Layout: 4×4 grid showing different comparison methods
    """
    print("  Creating detector comparison panel...")
    
    # Get all detector categories
    detector_categories = [k for k in organized_data.keys() 
                          if k not in ['other', 'virtual_imaging', 'dual_membrane']]
    
    if len(detector_categories) == 0:
        print("    No detector data found")
        return
    
    # Compute statistics for each detector
    detector_stats = {}
    for category in detector_categories:
        if len(organized_data[category]) > 0:
            # Use first file from each category
            data = organized_data[category][0]['data']
            analyzer = MultiModalAnalyzer(PathLib('.'))
            detector_stats[category] = analyzer.compute_statistics(data)
    
    if len(detector_stats) == 0:
        print("    No statistics computed")
        return
    
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.35)
    
    # Panel 1: Radar chart (multi-dimensional comparison)
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    create_radar_chart(detector_stats, ax1, 'Detector Performance Profile')
    
    # Panel 2: Statistical heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    create_statistical_heatmap(detector_stats, ax2, 'Statistical Fingerprints')
    
    # Panel 3: Distribution comparison (violin plot)
    ax3 = fig.add_subplot(gs[0, 2])
    create_distribution_comparison(organized_data, detector_categories, ax3, 
                                   'Value Distributions')
    
    # Panel 4: Correlation matrix
    ax4 = fig.add_subplot(gs[0, 3])
    create_correlation_matrix(detector_stats, ax4, 'Metric Correlations')
    
    # Panel 5: Mean comparison bar chart
    ax5 = fig.add_subplot(gs[1, 0])
    create_metric_bars(detector_stats, 'mean', ax5, 'Mean Values')
    
    # Panel 6: Std comparison
    ax6 = fig.add_subplot(gs[1, 1])
    create_metric_bars(detector_stats, 'std', ax6, 'Standard Deviations')
    
    # Panel 7: Entropy comparison
    ax7 = fig.add_subplot(gs[1, 2])
    create_metric_bars(detector_stats, 'entropy', ax7, 'Information Entropy')
    
    # Panel 8: Dynamic range
    ax8 = fig.add_subplot(gs[1, 3])
    create_dynamic_range_plot(detector_stats, ax8, 'Dynamic Range')
    
    # Panel 9: Polar phase distribution (combined)
    ax9 = fig.add_subplot(gs[2, 0], projection='polar')
    create_polar_phase_combined(organized_data, detector_categories, ax9,
                                'Phase Distributions')
    
    # Panel 10: Skewness-Kurtosis scatter
    ax10 = fig.add_subplot(gs[2, 1])
    create_skewness_kurtosis_plot(detector_stats, ax10, 'Distribution Shape')
    
    # Panel 11: Hierarchical clustering
    ax11 = fig.add_subplot(gs[2, 2])
    create_dendrogram_plot(detector_stats, ax11, 'Similarity Clustering')
    
    # Panel 12: PCA projection
    ax12 = fig.add_subplot(gs[2, 3])
    create_pca_projection(detector_stats, ax12, 'Principal Components')
    
    # Panel 13: Box plots
    ax13 = fig.add_subplot(gs[3, 0])
    create_box_plots(organized_data, detector_categories, ax13, 'Quartile Analysis')
    
    # Panel 14: Cumulative distributions
    ax14 = fig.add_subplot(gs[3, 1])
    create_cumulative_distributions(organized_data, detector_categories, ax14,
                                   'Cumulative Distribution Functions')
    
    # Panel 15: Normalized comparison
    ax15 = fig.add_subplot(gs[3, 2])
    create_normalized_comparison(detector_stats, ax15, 'Normalized Metrics')
    
    # Panel 16: Summary table
    ax16 = fig.add_subplot(gs[3, 3])
    create_summary_table(detector_stats, ax16, 'Quantitative Summary')
    
    # Main title
    fig.suptitle('Multi-Detector Comparison Analysis', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path}")


def create_radar_chart(detector_stats: Dict, ax: plt.Axes, title: str):
    """Create radar chart for multi-dimensional comparison"""
    categories = list(detector_stats.keys())
    if len(categories) == 0:
        return
    
    # Select metrics for radar chart
    metrics = ['mean', 'std', 'entropy', 'skewness', 'kurtosis']
    
    # Normalize metrics to 0-1 range
    normalized_data = {}
    for metric in metrics:
        values = [detector_stats[cat].get(metric, 0) for cat in categories]
        if max(values) > min(values):
            normalized = [(v - min(values)) / (max(values) - min(values)) 
                         for v in values]
        else:
            normalized = [0.5] * len(values)
        normalized_data[metric] = normalized
    
    # Number of variables
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Plot for each detector
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    
    for idx, category in enumerate(categories):
        values = [normalized_data[m][idx] for m in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=category, 
               color=colors[idx], alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax.grid(True, alpha=0.3)


def create_statistical_heatmap(detector_stats: Dict, ax: plt.Axes, title: str):
    """Create heatmap of statistical metrics"""
    categories = list(detector_stats.keys())
    metrics = ['mean', 'std', 'median', 'entropy', 'skewness', 'kurtosis']
    
    # Build data matrix
    data_matrix = []
    for metric in metrics:
        row = [detector_stats[cat].get(metric, 0) for cat in categories]
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Normalize each row
    for i in range(len(metrics)):
        row = data_matrix[i, :]
        if row.max() > row.min():
            data_matrix[i, :] = (row - row.min()) / (row.max() - row.min())
    
    # Plot heatmap
    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([c.replace('_', '\n') for c in categories], 
                       fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(metrics, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Value', fontsize=8)
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(categories)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black" if data_matrix[i, j] > 0.5 else "white",
                         fontsize=7)


def create_distribution_comparison(organized_data: Dict, categories: List[str], 
                                   ax: plt.Axes, title: str):
    """Create violin plot comparing distributions"""
    data_for_plot = []
    labels = []
    
    for cat in categories:
        if cat in organized_data and len(organized_data[cat]) > 0:
            data = organized_data[cat][0]['data'].flatten()
            # Sample if too large
            if len(data) > 10000:
                data = np.random.choice(data, 10000, replace=False)
            data_for_plot.append(data)
            labels.append(cat.replace('_', '\n'))
    
    if len(data_for_plot) > 0:
        parts = ax.violinplot(data_for_plot, showmeans=True, showmedians=True)
        
        # Color the violins
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_for_plot)))
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Value', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)


def create_correlation_matrix(detector_stats: Dict, ax: plt.Axes, title: str):
    """Create correlation matrix between metrics"""
    metrics = ['mean', 'std', 'entropy', 'skewness', 'kurtosis', 'median']
    
    # Build data matrix
    data_matrix = []
    for cat in detector_stats.keys():
        row = [detector_stats[cat].get(m, 0) for m in metrics]
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Compute correlation
    if data_matrix.shape[0] > 1:
        corr_matrix = np.corrcoef(data_matrix.T)
    else:
        corr_matrix = np.ones((len(metrics), len(metrics)))
    
    # Plot
    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(metrics, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation', fontsize=8)
    
    # Add values
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                         ha="center", va="center", 
                         color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                         fontsize=7)


def create_metric_bars(detector_stats: Dict, metric: str, ax: plt.Axes, title: str):
    """Create bar chart for specific metric"""
    categories = list(detector_stats.keys())
    values = [detector_stats[cat].get(metric, 0) for cat in categories]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(categories)))
    bars = ax.bar(range(len(categories)), values, color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([c.replace('_', '\n') for c in categories], 
                       fontsize=8, rotation=45, ha='right')
    ax.set_ylabel(metric.capitalize(), fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom', fontsize=7)


def create_dynamic_range_plot(detector_stats: Dict, ax: plt.Axes, title: str):
    """Plot dynamic range (max - min) for each detector"""
    categories = list(detector_stats.keys())
    ranges = [detector_stats[cat]['max'] - detector_stats[cat]['min'] 
             for cat in categories]
    mins = [detector_stats[cat]['min'] for cat in categories]
    maxs = [detector_stats[cat]['max'] for cat in categories]
    
    x = np.arange(len(categories))
    
    # Plot as error bars showing range
    ax.errorbar(x, [(mi + ma) / 2 for mi, ma in zip(mins, maxs)],
               yerr=[(ma - mi) / 2 for mi, ma in zip(mins, maxs)],
               fmt='o', markersize=8, capsize=5, capthick=2,
               color='steelblue', ecolor='coral', linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in categories],
                       fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Value', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)


def create_polar_phase_combined(organized_data: Dict, categories: List[str],
                                ax: plt.Axes, title: str):
    """Create combined polar phase distribution"""
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    
    for idx, cat in enumerate(categories):
        if cat in organized_data and len(organized_data[cat]) > 0:
            data = organized_data[cat][0]['data'].flatten()
            
            # Convert to phase (angle)
            phase = np.angle(data + 1j * np.roll(data, 1))
            
            # Histogram
            bins = np.linspace(-np.pi, np.pi, 37)
            hist, _ = np.histogram(phase, bins=bins)
            hist = hist / hist.max()
            
            theta = np.linspace(-np.pi, np.pi, 36, endpoint=False)
            width = 2 * np.pi / 36
            
            ax.bar(theta, hist, width=width, bottom=0.0, alpha=0.6,
                  color=colors[idx], label=cat, edgecolor='black', linewidth=0.5)
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=7)


def create_skewness_kurtosis_plot(detector_stats: Dict, ax: plt.Axes, title: str):
    """Scatter plot of skewness vs kurtosis"""
    categories = list(detector_stats.keys())
    skewness = [detector_stats[cat]['skewness'] for cat in categories]
    kurtosis = [detector_stats[cat]['kurtosis'] for cat in categories]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    
    for idx, cat in enumerate(categories):
        ax.scatter(skewness[idx], kurtosis[idx], s=200, alpha=0.7,
                  color=colors[idx], edgecolor='black', linewidth=2,
                  label=cat)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Skewness', fontsize=10, fontweight='bold')
    ax.set_ylabel('Kurtosis', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)


def create_dendrogram_plot(detector_stats: Dict, ax: plt.Axes, title: str):
    """Create hierarchical clustering dendrogram"""
    categories = list(detector_stats.keys())
    metrics = ['mean', 'std', 'entropy', 'skewness', 'kurtosis']
    
    # Build feature matrix
    data_matrix = []
    for cat in categories:
        row = [detector_stats[cat].get(m, 0) for m in metrics]
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix, dtype=float)
    
    # Replace non-finite values
    data_matrix = np.nan_to_num(data_matrix, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Normalize
    for i in range(data_matrix.shape[1]):
        col = data_matrix[:, i]
        col_range = col.max() - col.min()
        if col_range > 1e-10:
            data_matrix[:, i] = (col - col.min()) / col_range
        else:
            data_matrix[:, i] = 0.0
    
    # Hierarchical clustering
    if len(categories) > 1:
        try:
            linkage_matrix = linkage(data_matrix, method='ward')
            dendrogram(linkage_matrix, labels=categories, ax=ax,
                      orientation='right', leaf_font_size=8)
            ax.set_xlabel('Distance', fontsize=10, fontweight='bold')
        except (ValueError, FloatingPointError) as e:
            ax.text(0.5, 0.5, f'Clustering unavailable\n(data too similar)',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title(title, fontsize=11, fontweight='bold')


def create_pca_projection(detector_stats: Dict, ax: plt.Axes, title: str):
    """PCA projection of detectors"""
    from sklearn.decomposition import PCA
    
    categories = list(detector_stats.keys())
    metrics = ['mean', 'std', 'entropy', 'skewness', 'kurtosis', 'median']
    
    # Build feature matrix
    data_matrix = []
    for cat in categories:
        row = [detector_stats[cat].get(m, 0) for m in metrics]
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix, dtype=float)
    
    # Replace non-finite values
    data_matrix = np.nan_to_num(data_matrix, nan=0.0, posinf=1.0, neginf=0.0)
    
    if len(categories) >= 2:
        try:
            # Standardize
            std = data_matrix.std(axis=0)
            std[std < 1e-10] = 1.0  # Avoid division by zero
            data_matrix = (data_matrix - data_matrix.mean(axis=0)) / std
            
            # PCA
            pca = PCA(n_components=2)
            transformed = pca.fit_transform(data_matrix)
            
            colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
            
            for idx, cat in enumerate(categories):
                ax.scatter(transformed[idx, 0], transformed[idx, 1], 
                          s=200, alpha=0.7, color=colors[idx],
                          edgecolor='black', linewidth=2, label=cat)
                ax.annotate(cat, (transformed[idx, 0], transformed[idx, 1]),
                           fontsize=7, ha='center', va='bottom')
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                         fontsize=10, fontweight='bold')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                         fontsize=10, fontweight='bold')
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        except (ValueError, np.linalg.LinAlgError) as e:
            ax.text(0.5, 0.5, f'PCA unavailable\n(insufficient variance)',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_title(title, fontsize=11, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Need ≥2 detectors\nfor PCA',
               ha='center', va='center', transform=ax.transAxes,
               fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')


def create_box_plots(organized_data: Dict, categories: List[str],
                    ax: plt.Axes, title: str):
    """Create box plots for quartile analysis"""
    data_for_plot = []
    labels = []
    
    for cat in categories:
        if cat in organized_data and len(organized_data[cat]) > 0:
            data = organized_data[cat][0]['data'].flatten()
            if len(data) > 10000:
                data = np.random.choice(data, 10000, replace=False)
            data_for_plot.append(data)
            labels.append(cat.replace('_', '\n'))
    
    if len(data_for_plot) > 0:
        bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True,
                       showfliers=False)
        
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(data_for_plot)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax.set_ylabel('Value', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)


def create_cumulative_distributions(organized_data: Dict, categories: List[str],
                                   ax: plt.Axes, title: str):
    """Plot cumulative distribution functions"""
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    
    for idx, cat in enumerate(categories):
        if cat in organized_data and len(organized_data[cat]) > 0:
            data = organized_data[cat][0]['data'].flatten()
            data_sorted = np.sort(data)
            cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
            
            ax.plot(data_sorted, cdf, linewidth=2, alpha=0.7,
                   color=colors[idx], label=cat)
    
    ax.set_xlabel('Value', fontsize=10, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)


def create_normalized_comparison(detector_stats: Dict, ax: plt.Axes, title: str):
    """Normalized multi-metric comparison"""
    categories = list(detector_stats.keys())
    metrics = ['mean', 'std', 'entropy', 'skewness']
    
    # Normalize all metrics to 0-1
    normalized = {}
    for metric in metrics:
        values = [detector_stats[cat].get(metric, 0) for cat in categories]
        if max(values) > min(values):
            norm_values = [(v - min(values)) / (max(values) - min(values)) for v in values]
        else:
            norm_values = [0.5] * len(values)
        normalized[metric] = norm_values
    
    x = np.arange(len(categories))
    width = 0.2
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, metric in enumerate(metrics):
        offset = (idx - len(metrics)/2 + 0.5) * width
        ax.bar(x + offset, normalized[metric], width, label=metric,
              color=colors[idx], alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in categories],
                       fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Normalized Value', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', alpha=0.3)


def create_summary_table(detector_stats: Dict, ax: plt.Axes, title: str):
    """Create summary statistics table"""
    ax.axis('off')
    
    categories = list(detector_stats.keys())
    metrics = ['mean', 'std', 'entropy']
    
    # Build table data
    table_data = []
    for cat in categories:
        row = [cat.replace('_', ' ').capitalize()]
        for metric in metrics:
            val = detector_stats[cat].get(metric, 0)
            row.append(f'{val:.4f}')
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Detector'] + [m.capitalize() for m in metrics],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(metrics) + 1):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(categories) + 1):
        for j in range(len(metrics) + 1):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=20)


def main():
    """Main processing function"""
    print("=" * 80)
    print("PUBLICATION PANEL CHARTS GENERATOR")
    print("=" * 80)
    
    # Find NPY files in multiple possible directories
    search_dirs = [
        PathLib("npy_visualizations"),
        PathLib("../maxwell/demo_complete_results"),
        PathLib("maxwell/demo_complete_results"),
        PathLib("../maxwell/multi_modal_validation"),
        PathLib("maxwell/multi_modal_validation"),
        PathLib("virtual_imaging_results"),
        PathLib("multi_modal_motion_picture"),
        PathLib("motion_picture_validation")
    ]
    
    # Find which directories exist and have NPY files
    valid_dirs = []
    for search_dir in search_dirs:
        if search_dir.exists():
            npy_count = len(list(search_dir.glob("**/*.npy")))
            if npy_count > 0:
                valid_dirs.append(search_dir)
                print(f"  Found {npy_count} NPY files in: {search_dir}")
    
    if not valid_dirs:
        print("✗ No NPY files found in any of the search directories:")
        for d in search_dirs:
            print(f"    - {d}")
        return
    
    # Use the first valid directory (or combine all)
    npy_dir = valid_dirs[0]
    print(f"\n  Using directory: {npy_dir}")
    
    # Load and organize data
    print("\nLoading NPY files from all valid directories...")
    analyzer = MultiModalAnalyzer(npy_dir)
    organized_data = analyzer.load_all_npy_files(valid_dirs)
    
    print(f"  Found {sum(len(v) for v in organized_data.values())} NPY files")
    print(f"  Organized into {len(organized_data)} categories:")
    for category, files in organized_data.items():
        print(f"    • {category}: {len(files)} file(s)")
    
    # Create output directory
    output_dir = PathLib("publication_panels")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive detector comparison panel
    print("\nGenerating comprehensive panels...")
    create_detector_comparison_panel(organized_data, 
                                    output_dir / 'detector_comparison_panel.png')
    
    print("\n" + "=" * 80)
    print("PANEL GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}/")
    print("  - detector_comparison_panel.png (4×4 comprehensive analysis)")
    print("\nEach panel includes:")
    print("  • Radar charts (multi-dimensional profiles)")
    print("  • Statistical heatmaps")
    print("  • Polar phase distributions")
    print("  • Correlation matrices")
    print("  • Distribution comparisons")
    print("  • PCA projections")
    print("  • Hierarchical clustering")
    print("  • And 9 more visualization types!")
    print("=" * 80)


if __name__ == '__main__':
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("Installing scikit-learn for PCA...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'scikit-learn'])
        from sklearn.decomposition import PCA
    
    main()

