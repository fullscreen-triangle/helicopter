"""
Visualization Panel Generator.

Creates publication-quality figure panels for PTRM validation results.
Each panel contains 4+ charts with at least one 3D visualization.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

# Matplotlib configuration for publication quality
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# Style configuration
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'quaternary': '#C73E1D',
    'success': '#3A7D44',
    'warning': '#F0A202',
    'S_k': '#E63946',
    'S_t': '#457B9D',
    'S_e': '#2A9D8F',
}


class ValidationPanelGenerator:
    """
    Generate visualization panels for validation experiments.

    Each panel contains multiple charts including 3D visualizations.
    """

    def __init__(self, output_dir: str = 'panels'):
        """
        Initialize generator.

        Args:
            output_dir: Directory for saving panels
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_panel_1_partition_coordinates(self,
                                                partition_data: Dict,
                                                capacity_test: Dict,
                                                entropy_data: Dict,
                                                distribution_3d: Dict) -> str:
        """
        Panel 1: Partition Coordinate Analysis

        Charts:
        1. Observed vs theoretical C(n) = 2n² distribution
        2. Partition state entropy over (n, ℓ) space (heatmap)
        3. 3D scatter: (n, ℓ, m) distribution
        4. Cumulative capacity comparison
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Panel 1: Partition Coordinate Validation\n'
                     'Testing C(n) = 2n² Capacity Theorem', fontsize=14, fontweight='bold')

        # Chart 1: Observed vs Theoretical Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        n_values = sorted(capacity_test.get('observed_distribution', {}).keys())
        if n_values:
            observed = [capacity_test['observed_distribution'].get(n, 0) for n in n_values]
            expected = [capacity_test['expected_distribution'].get(n, 0) for n in n_values]

            x = np.arange(len(n_values))
            width = 0.35

            bars1 = ax1.bar(x - width/2, observed, width, label='Observed',
                           color=COLORS['primary'], alpha=0.8)
            bars2 = ax1.bar(x + width/2, expected, width, label='Expected (C(n)=2n²)',
                           color=COLORS['secondary'], alpha=0.8)

            ax1.set_xlabel('Principal Quantum Number n')
            ax1.set_ylabel('Count')
            ax1.set_title(f'Partition Distribution (χ²={capacity_test.get("chi2", 0):.2f}, '
                         f'p={capacity_test.get("p_value", 0):.3f})')
            ax1.set_xticks(x)
            ax1.set_xticklabels(n_values)
            ax1.legend()

        # Chart 2: Entropy Heatmap over (n, ℓ)
        ax2 = fig.add_subplot(gs[0, 1])
        if distribution_3d and len(distribution_3d.get('n', [])) > 0:
            # Create 2D histogram for (n, ℓ)
            n_arr = distribution_3d['n']
            ell_arr = distribution_3d['ell']
            counts = distribution_3d['counts']

            # Create heatmap
            n_max = max(n_arr) + 1
            ell_max = max(ell_arr) + 1
            heatmap = np.zeros((ell_max, n_max))

            for n, ell, c in zip(n_arr, ell_arr, counts):
                heatmap[ell, n] = c

            im = ax2.imshow(heatmap, cmap='viridis', aspect='auto', origin='lower')
            plt.colorbar(im, ax=ax2, label='Count')
            ax2.set_xlabel('Principal n')
            ax2.set_ylabel('Angular ℓ')
            ax2.set_title('Partition State Occupancy')

        # Chart 3: 3D Scatter (n, ℓ, m)
        ax3 = fig.add_subplot(gs[1, 0], projection='3d')
        if distribution_3d and len(distribution_3d.get('n', [])) > 0:
            n_arr = distribution_3d['n']
            ell_arr = distribution_3d['ell']
            m_arr = distribution_3d['m']
            counts = distribution_3d['counts']

            # Normalize counts for size
            sizes = (counts / counts.max()) * 200 + 20

            scatter = ax3.scatter(n_arr, ell_arr, m_arr,
                                  c=counts, cmap='plasma',
                                  s=sizes, alpha=0.7, edgecolors='white', linewidth=0.5)

            ax3.set_xlabel('n (Principal)')
            ax3.set_ylabel('ℓ (Angular)')
            ax3.set_zlabel('m (Magnetic)')
            ax3.set_title('3D Partition Space Distribution')
            plt.colorbar(scatter, ax=ax3, shrink=0.5, label='Count')

        # Chart 4: Cumulative Capacity
        ax4 = fig.add_subplot(gs[1, 1])
        if n_values:
            # Theoretical cumulative capacity
            n_range = np.arange(1, max(n_values) + 1)
            theoretical_cumulative = np.cumsum([2*n**2 for n in n_range])

            # Observed cumulative
            observed_cumulative = np.cumsum([capacity_test['observed_distribution'].get(n, 0)
                                             for n in n_range])

            # Normalize for comparison
            if theoretical_cumulative[-1] > 0:
                theoretical_norm = theoretical_cumulative / theoretical_cumulative[-1]
            else:
                theoretical_norm = theoretical_cumulative

            if observed_cumulative[-1] > 0:
                observed_norm = observed_cumulative / observed_cumulative[-1]
            else:
                observed_norm = observed_cumulative

            ax4.plot(n_range, theoretical_norm, 'o-', color=COLORS['secondary'],
                    linewidth=2, markersize=8, label='Theory: C(n)=2n²')
            ax4.plot(n_range, observed_norm, 's-', color=COLORS['primary'],
                    linewidth=2, markersize=8, label='Observed')

            ax4.fill_between(n_range, theoretical_norm, observed_norm,
                            alpha=0.3, color=COLORS['tertiary'])

            ax4.set_xlabel('Principal Quantum Number n')
            ax4.set_ylabel('Cumulative Fraction')
            ax4.set_title('Cumulative Capacity Comparison')
            ax4.legend()

        # Add validation status
        status = '✓ VALIDATED' if capacity_test.get('validated', False) else '✗ NOT VALIDATED'
        fig.text(0.99, 0.01, status, ha='right', va='bottom',
                fontsize=12, fontweight='bold',
                color=COLORS['success'] if capacity_test.get('validated', False) else COLORS['quaternary'])

        # Save
        output_path = self.output_dir / 'panel_1_partition_coordinates.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return str(output_path)

    def generate_panel_2_s_entropy(self,
                                    trajectory_data: Dict,
                                    conservation_test: Dict,
                                    phase_space: np.ndarray) -> str:
        """
        Panel 2: S-Entropy Analysis

        Charts:
        1. S-entropy components over time (stacked area)
        2. S_total conservation check
        3. 3D phase space trajectory (S_k, S_t, S_e)
        4. Component correlation matrix
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Panel 2: S-Entropy Trajectory Analysis\n'
                     'Testing Conservation: S_k + S_t + S_e = constant', fontsize=14, fontweight='bold')

        t = trajectory_data.get('t', np.array([]))
        S_k = trajectory_data.get('S_k', np.array([]))
        S_t = trajectory_data.get('S_t', np.array([]))
        S_e = trajectory_data.get('S_e', np.array([]))

        # Chart 1: Stacked Area Plot
        ax1 = fig.add_subplot(gs[0, 0])
        if len(t) > 0:
            ax1.stackplot(t, S_k, S_t, S_e,
                         labels=['S_k (Knowledge)', 'S_t (Temporal)', 'S_e (Evolution)'],
                         colors=[COLORS['S_k'], COLORS['S_t'], COLORS['S_e']],
                         alpha=0.8)
            ax1.set_xlabel('Time Index')
            ax1.set_ylabel('S-Entropy (normalized)')
            ax1.set_title('S-Entropy Components Over Time')
            ax1.legend(loc='upper right')
            ax1.set_ylim(0, 1.1)

        # Chart 2: S_total Conservation
        ax2 = fig.add_subplot(gs[0, 1])
        if len(t) > 0:
            S_total = trajectory_data.get('S_total', S_k + S_t + S_e)

            ax2.plot(t, S_total, 'o-', color=COLORS['primary'],
                    linewidth=2, markersize=6, label='S_total')
            ax2.axhline(y=1.0, color=COLORS['secondary'], linestyle='--',
                       linewidth=2, label='Expected (1.0)')

            # Confidence band
            mean_val = np.mean(S_total)
            std_val = np.std(S_total)
            ax2.fill_between(t, mean_val - 2*std_val, mean_val + 2*std_val,
                            alpha=0.2, color=COLORS['tertiary'])

            ax2.set_xlabel('Time Index')
            ax2.set_ylabel('S_total')
            ax2.set_title(f'Conservation Check (CV={conservation_test.get("cv", 0):.4f})')
            ax2.legend()
            ax2.set_ylim(0.8, 1.2)

        # Chart 3: 3D Phase Space Trajectory
        ax3 = fig.add_subplot(gs[1, 0], projection='3d')
        if len(phase_space) > 0:
            # Color by time
            colors = cm.viridis(np.linspace(0, 1, len(phase_space)))

            ax3.scatter(phase_space[:, 0], phase_space[:, 1], phase_space[:, 2],
                       c=np.arange(len(phase_space)), cmap='viridis',
                       s=50, alpha=0.8, edgecolors='white', linewidth=0.5)

            # Plot trajectory line
            ax3.plot(phase_space[:, 0], phase_space[:, 1], phase_space[:, 2],
                    '-', color=COLORS['primary'], alpha=0.5, linewidth=1)

            # Mark start and end
            ax3.scatter(*phase_space[0], color='green', s=100, marker='^', label='Start')
            ax3.scatter(*phase_space[-1], color='red', s=100, marker='v', label='End')

            ax3.set_xlabel('S_k')
            ax3.set_ylabel('S_t')
            ax3.set_zlabel('S_e')
            ax3.set_title('S-Entropy Phase Space Trajectory')
            ax3.legend()

        # Chart 4: Component Correlation
        ax4 = fig.add_subplot(gs[1, 1])
        if len(S_k) > 1:
            components = np.array([S_k, S_t, S_e])
            corr_matrix = np.corrcoef(components)

            im = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax4, label='Correlation')

            ax4.set_xticks([0, 1, 2])
            ax4.set_yticks([0, 1, 2])
            ax4.set_xticklabels(['S_k', 'S_t', 'S_e'])
            ax4.set_yticklabels(['S_k', 'S_t', 'S_e'])
            ax4.set_title('Component Correlation Matrix')

            # Add correlation values
            for i in range(3):
                for j in range(3):
                    ax4.text(j, i, f'{corr_matrix[i, j]:.2f}',
                            ha='center', va='center',
                            color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

        # Validation status
        status = '✓ VALIDATED' if conservation_test.get('validated', False) else '✗ NOT VALIDATED'
        fig.text(0.99, 0.01, status, ha='right', va='bottom',
                fontsize=12, fontweight='bold',
                color=COLORS['success'] if conservation_test.get('validated', False) else COLORS['quaternary'])

        output_path = self.output_dir / 'panel_2_s_entropy.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return str(output_path)

    def generate_panel_3_sequential_exclusion(self,
                                               exclusion_data: Dict,
                                               curve_data: Dict,
                                               resolution_data: Dict) -> str:
        """
        Panel 3: Sequential Exclusion Analysis

        Charts:
        1. Cumulative exclusion curve (log scale)
        2. Individual modality contributions
        3. 3D correlation surface
        4. Resolution enhancement comparison
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Panel 3: Sequential Exclusion Validation\n'
                     'Testing N_12 = N_0 × ∏ε_i → 1', fontsize=14, fontweight='bold')

        # Chart 1: Cumulative Exclusion (log scale)
        ax1 = fig.add_subplot(gs[0, 0])
        modality_idx = curve_data.get('modality_index', np.array([]))
        log_N = curve_data.get('log_N_remaining', np.array([]))

        if len(modality_idx) > 0:
            ax1.semilogy(modality_idx, 10**log_N, 'o-', color=COLORS['primary'],
                        linewidth=2, markersize=10)

            # Add modality labels
            names = curve_data.get('names', [])
            for i, (x, y, name) in enumerate(zip(modality_idx, 10**log_N, names)):
                ax1.annotate(name.split()[0], (x, y), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=8, rotation=45)

            ax1.set_xlabel('Number of Modalities')
            ax1.set_ylabel('N_remaining (log scale)')
            ax1.set_title(f'Configuration Space Reduction\n'
                         f'(N_0={exclusion_data.get("N_0", 0):.2e} → '
                         f'N_final={exclusion_data.get("N_final", 0):.2e})')

        # Chart 2: Individual Exclusion Factors
        ax2 = fig.add_subplot(gs[0, 1])
        epsilons = curve_data.get('epsilon', np.array([]))
        names = curve_data.get('names', [])

        if len(epsilons) > 0:
            x_pos = np.arange(len(epsilons))
            bars = ax2.bar(x_pos, epsilons, color=COLORS['secondary'], alpha=0.8)

            # Color bars by value
            for bar, eps in zip(bars, epsilons):
                bar.set_color(cm.RdYlGn(eps))

            ax2.set_xlabel('Modality')
            ax2.set_ylabel('Exclusion Factor ε')
            ax2.set_title('Individual Modality Contributions')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([n.split()[0] for n in names], rotation=45, ha='right')
            ax2.axhline(y=np.mean(epsilons), color=COLORS['quaternary'],
                       linestyle='--', label=f'Mean: {np.mean(epsilons):.3f}')
            ax2.legend()

        # Chart 3: 3D Correlation Surface
        ax3 = fig.add_subplot(gs[1, 0], projection='3d')
        corr_matrix = resolution_data.get('correlation_matrix', [])

        if len(corr_matrix) > 0:
            corr_arr = np.array(corr_matrix)
            n = len(corr_arr)

            # Create mesh
            x = np.arange(n)
            y = np.arange(n)
            X, Y = np.meshgrid(x, y)

            surf = ax3.plot_surface(X, Y, corr_arr, cmap='coolwarm',
                                    linewidth=0, antialiased=True, alpha=0.8)

            ax3.set_xlabel('Modality i')
            ax3.set_ylabel('Modality j')
            ax3.set_zlabel('Correlation ρ_ij')
            ax3.set_title(f'Inter-Modality Correlations\n(Σρ = {resolution_data.get("sum_rho", 0):.2f})')
            plt.colorbar(surf, ax=ax3, shrink=0.5)

        # Chart 4: Resolution Enhancement
        ax4 = fig.add_subplot(gs[1, 1])
        if resolution_data:
            categories = ['Single\nModality', 'Independent\n(K^{-1/2})', 'Correlated\n(exp(-Σρ))']
            values = [
                1.0,  # Baseline
                resolution_data.get('delta_x_independent', 1.0),
                resolution_data.get('delta_x_correlated', 1.0)
            ]

            bars = ax4.bar(categories, values,
                          color=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary']])

            ax4.set_ylabel('Relative Resolution Δx/Δx_1')
            ax4.set_title(f'Resolution Enhancement\n'
                         f'({resolution_data.get("enhancement_correlated", 1):.1f}× improvement)')
            ax4.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

            # Add value labels
            for bar, val in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        # Validation status
        status = '✓ VALIDATED' if exclusion_data.get('validated', False) else '✗ NOT VALIDATED'
        fig.text(0.99, 0.01, status, ha='right', va='bottom',
                fontsize=12, fontweight='bold',
                color=COLORS['success'] if exclusion_data.get('validated', False) else COLORS['quaternary'])

        output_path = self.output_dir / 'panel_3_sequential_exclusion.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return str(output_path)

    def generate_panel_4_reaction_localization(self,
                                                signal_maps: Dict[str, np.ndarray],
                                                peak_data: Dict[str, np.ndarray],
                                                consensus_data: Dict,
                                                resolution_results: Dict) -> str:
        """
        Panel 4: Multimodal Reaction Localization

        Charts:
        1. Multi-modality signal comparison (grid)
        2. Consensus detection overlay
        3. 3D surface of combined signal
        4. Resolution enhancement by modality count
        """
        fig = plt.figure(figsize=(14, 12))

        # Use custom layout
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
                               height_ratios=[1, 1, 1.2])

        fig.suptitle('Panel 4: Multimodal Reaction Localization\n'
                     'Testing Intersection Theorem', fontsize=14, fontweight='bold')

        # Charts 1-6: Individual modality signals (2x3 grid)
        modality_names = list(signal_maps.keys())[:6]

        for i, name in enumerate(modality_names):
            row, col = i // 3, i % 3
            ax = fig.add_subplot(gs[row, col])

            signal = signal_maps[name]
            im = ax.imshow(signal, cmap='hot', aspect='equal')

            # Overlay peaks
            if name in peak_data and len(peak_data[name]) > 0:
                peaks = peak_data[name]
                ax.scatter(peaks[:, 1], peaks[:, 0], c='cyan', s=30,
                          marker='x', linewidths=1)

            ax.set_title(f'{name}', fontsize=10)
            ax.axis('off')

        # Chart 7: 3D Combined Signal Surface
        ax_3d = fig.add_subplot(gs[2, 0], projection='3d')

        # Combine signals
        if signal_maps:
            combined = np.zeros_like(list(signal_maps.values())[0])
            for signal in signal_maps.values():
                if signal.shape == combined.shape:
                    combined += signal / signal.max() if signal.max() > 0 else signal

            # Downsample for 3D plotting
            step = max(1, combined.shape[0] // 50)
            combined_ds = combined[::step, ::step]

            y = np.arange(combined_ds.shape[0])
            x = np.arange(combined_ds.shape[1])
            X, Y = np.meshgrid(x, y)

            surf = ax_3d.plot_surface(X, Y, combined_ds, cmap='viridis',
                                      linewidth=0, antialiased=True, alpha=0.8)

            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Combined Signal')
            ax_3d.set_title('3D Combined Signal')

        # Chart 8: Consensus Overlay
        ax_consensus = fig.add_subplot(gs[2, 1])

        if signal_maps:
            # Show one signal as background
            bg_signal = list(signal_maps.values())[0]
            ax_consensus.imshow(bg_signal, cmap='gray', aspect='equal')

            # Overlay consensus detections
            positions = consensus_data.get('positions', np.array([]).reshape(0, 2))
            n_modalities = consensus_data.get('n_modalities', np.array([]))

            if len(positions) > 0:
                scatter = ax_consensus.scatter(positions[:, 1], positions[:, 0],
                                               c=n_modalities, cmap='plasma',
                                               s=100, marker='*', edgecolors='white',
                                               linewidth=1)
                plt.colorbar(scatter, ax=ax_consensus, label='# Modalities')

            ax_consensus.set_title(f'Consensus Detections (n={len(positions)})')
            ax_consensus.axis('off')

        # Chart 9: Resolution Enhancement Curve
        ax_res = fig.add_subplot(gs[2, 2])

        # Simulated resolution vs modality count
        n_modalities = np.arange(1, 7)
        delta_r_independent = 1.0 / np.sqrt(n_modalities)
        delta_r_correlated = np.exp(-0.3 * n_modalities)  # Assume ρ=0.3

        ax_res.plot(n_modalities, delta_r_independent, 'o-',
                   color=COLORS['primary'], linewidth=2, markersize=8,
                   label='Independent (K^{-1/2})')
        ax_res.plot(n_modalities, delta_r_correlated, 's-',
                   color=COLORS['secondary'], linewidth=2, markersize=8,
                   label='Correlated (exp(-Σρ))')

        ax_res.set_xlabel('Number of Modalities')
        ax_res.set_ylabel('Relative Resolution Δx/Δx_1')
        ax_res.set_title('Resolution Enhancement')
        ax_res.legend()
        ax_res.set_ylim(0, 1.1)

        # Add enhancement annotation
        final_enhancement = 1.0 / delta_r_correlated[-1]
        ax_res.annotate(f'{final_enhancement:.1f}× enhancement',
                       xy=(6, delta_r_correlated[-1]),
                       xytext=(4, 0.3),
                       arrowprops=dict(arrowstyle='->', color=COLORS['secondary']),
                       fontsize=10, color=COLORS['secondary'])

        # Validation status
        status = '✓ VALIDATED' if resolution_results.get('validated', False) else '✗ NOT VALIDATED'
        fig.text(0.99, 0.01, status, ha='right', va='bottom',
                fontsize=12, fontweight='bold',
                color=COLORS['success'] if resolution_results.get('validated', False) else COLORS['quaternary'])

        output_path = self.output_dir / 'panel_4_reaction_localization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return str(output_path)

    def generate_summary_panel(self,
                                all_results: Dict,
                                dataset_stats: Dict) -> str:
        """
        Summary Panel: Overview of all validation results.

        Charts:
        1. Validation status matrix
        2. Key metrics comparison
        3. 3D summary space
        4. Prediction vs observation scatter
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Summary Panel: PTRM Validation Results\n'
                     f'BBBC039 Dataset ({dataset_stats.get("n_images", 0)} images, '
                     f'{dataset_stats.get("total_nuclei", 0)} nuclei)', fontsize=14, fontweight='bold')

        # Chart 1: Validation Status Matrix
        ax1 = fig.add_subplot(gs[0, 0])

        experiments = ['Partition\nC(n)=2n²', 'S-Entropy\nConservation',
                      'Sequential\nExclusion', 'Reaction\nLocalization']
        validated = [
            all_results.get('partition', {}).get('validated', False),
            all_results.get('s_entropy', {}).get('validated', False),
            all_results.get('exclusion', {}).get('validated', False),
            all_results.get('localization', {}).get('validated', False)
        ]

        colors = [COLORS['success'] if v else COLORS['quaternary'] for v in validated]
        bars = ax1.barh(experiments, [1]*4, color=colors, alpha=0.8)

        ax1.set_xlim(0, 1.5)
        ax1.set_xlabel('')
        ax1.set_title('Validation Status')

        # Add status text
        for i, (bar, v) in enumerate(zip(bars, validated)):
            status_text = '✓ PASS' if v else '✗ FAIL'
            ax1.text(1.1, bar.get_y() + bar.get_height()/2,
                    status_text, va='center', fontsize=11, fontweight='bold',
                    color=COLORS['success'] if v else COLORS['quaternary'])

        ax1.set_xticks([])

        # Chart 2: Key Metrics
        ax2 = fig.add_subplot(gs[0, 1])

        metrics = {
            'χ² p-value': all_results.get('partition', {}).get('p_value', 0),
            'S-entropy CV': all_results.get('s_entropy', {}).get('cv', 0),
            'Reduction Factor': np.log10(all_results.get('exclusion', {}).get('reduction_factor', 1) + 1),
            'Resolution Enh.': all_results.get('localization', {}).get('enhancement_factor', 1)
        }

        x = np.arange(len(metrics))
        bars = ax2.bar(x, list(metrics.values()), color=COLORS['primary'], alpha=0.8)

        ax2.set_xticks(x)
        ax2.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        ax2.set_ylabel('Value')
        ax2.set_title('Key Validation Metrics')

        # Add value labels
        for bar, val in zip(bars, metrics.values()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        # Chart 3: 3D Summary Space
        ax3 = fig.add_subplot(gs[1, 0], projection='3d')

        # Create summary visualization
        # X: Partition accuracy, Y: Entropy conservation, Z: Resolution enhancement
        x = [all_results.get('partition', {}).get('p_value', 0)]
        y = [1 - all_results.get('s_entropy', {}).get('cv', 1)]  # Invert CV
        z = [all_results.get('localization', {}).get('enhancement_factor', 1)]

        ax3.scatter(x, y, z, c=COLORS['primary'], s=200, marker='*')

        # Add target region
        theta = np.linspace(0, 2*np.pi, 100)
        ax3.plot(0.05*np.cos(theta) + 0.5, 0.05*np.sin(theta) + 0.95,
                [2]*100, 'g--', alpha=0.5, label='Target region')

        ax3.set_xlabel('p-value')
        ax3.set_ylabel('1 - CV')
        ax3.set_zlabel('Enhancement')
        ax3.set_title('3D Validation Summary')

        # Chart 4: Theory vs Observation
        ax4 = fig.add_subplot(gs[1, 1])

        # Collect all theory vs observation pairs
        theory = []
        observed = []
        labels = []

        # From partition capacity
        partition_data = all_results.get('partition', {})
        if partition_data.get('expected_distribution') and partition_data.get('observed_distribution'):
            for n in partition_data['expected_distribution']:
                theory.append(partition_data['expected_distribution'][n])
                observed.append(partition_data['observed_distribution'].get(int(n), 0))
                labels.append(f'C({n})')

        if theory:
            ax4.scatter(theory, observed, c=COLORS['primary'], s=50, alpha=0.7)

            # Perfect agreement line
            max_val = max(max(theory), max(observed))
            ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect agreement')

            # Compute R²
            if len(theory) > 1:
                correlation = np.corrcoef(theory, observed)[0, 1]
                r_squared = correlation ** 2
                ax4.text(0.05, 0.95, f'R² = {r_squared:.3f}',
                        transform=ax4.transAxes, fontsize=11,
                        verticalalignment='top')

            ax4.set_xlabel('Theoretical Prediction')
            ax4.set_ylabel('Observed Value')
            ax4.set_title('Theory vs Observation')
            ax4.legend()

        # Overall validation count
        n_validated = sum(validated)
        n_total = len(validated)
        fig.text(0.99, 0.01, f'Overall: {n_validated}/{n_total} experiments validated',
                ha='right', va='bottom', fontsize=12, fontweight='bold',
                color=COLORS['success'] if n_validated >= 3 else COLORS['warning'])

        output_path = self.output_dir / 'panel_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return str(output_path)
