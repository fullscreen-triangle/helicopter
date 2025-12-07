"""
Visualization tools for HCCC algorithm results.

Visualizes:
- Processing sequence overlay on image
- Network BMD structure
- Convergence metrics
- Hardware stream coherence
"""

import numpy as np
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if TYPE_CHECKING:
    from ..vision.bmd import NetworkBMD, BMDState
    from ..regions import Region


class ResultVisualizer:
    """
    Visualize HCCC algorithm results.

    Creates publication-quality figures demonstrating:
    - Region processing order
    - Network BMD growth
    - Convergence behavior
    - Hardware grounding
    """

    def __init__(self):
        """Initialize visualizer."""
        pass

    def visualize_processing_sequence(
        self,
        image: np.ndarray,
        regions: List['Region'],
        processing_sequence: List[str],
        save_path: Optional[str] = None
    ):
        """
        Visualize region processing sequence overlaid on image.

        Args:
            image: Original image
            regions: List of all regions
            processing_sequence: Order of processing
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Show image
        ax.imshow(image)

        # Create region dictionary
        region_dict = {r.id: r for r in regions}

        # Color map for processing order
        n_processed = len(processing_sequence)
        cmap = plt.cm.viridis

        # Overlay regions in processing order
        for i, region_id in enumerate(processing_sequence):
            if region_id not in region_dict:
                continue

            region = region_dict[region_id]
            color = cmap(i / max(n_processed - 1, 1))

            # Draw bounding box
            x_min, y_min, x_max, y_max = region.bbox
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)

            # Add processing order number
            centroid = region.get_centroid()
            ax.text(
                centroid[1], centroid[0],
                str(i + 1),
                color='white',
                fontsize=10,
                weight='bold',
                ha='center',
                va='center',
                bbox=dict(boxstyle='circle', facecolor=color, alpha=0.8)
            )

        ax.set_title(f'Processing Sequence ({n_processed} regions)', fontsize=14)
        ax.axis('off')

        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=0, vmax=n_processed)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Processing Step', rotation=270, labelpad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved processing sequence visualization to {save_path}")
        else:
            plt.show()

    def visualize_network_growth(
        self,
        richness_history: List[float],
        ambiguity_history: List[float],
        divergence_history: List[float],
        save_path: Optional[str] = None
    ):
        """
        Visualize network BMD growth metrics.

        Args:
            richness_history: Richness over iterations
            ambiguity_history: Ambiguity over iterations
            divergence_history: Divergence over iterations
            save_path: Optional save path
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        iterations = np.arange(len(richness_history))

        # Richness growth (log scale)
        axes[0, 0].semilogy(iterations, richness_history, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Iteration', fontsize=12)
        axes[0, 0].set_ylabel('Network Richness R(β)', fontsize=12)
        axes[0, 0].set_title('Categorical Richness Growth', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)

        # Ambiguity reduction
        axes[0, 1].plot(iterations, ambiguity_history, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Iteration', fontsize=12)
        axes[0, 1].set_ylabel('Ambiguity A(β, R)', fontsize=12)
        axes[0, 1].set_title('Ambiguity Evolution', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)

        # Stream divergence
        axes[1, 0].plot(iterations, divergence_history, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Iteration', fontsize=12)
        axes[1, 0].set_ylabel('Stream Divergence D', fontsize=12)
        axes[1, 0].set_title('Hardware Stream Coherence', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)

        # Combined view
        ax2 = axes[1, 1]
        ax2_twin = ax2.twinx()

        ax2.plot(iterations, ambiguity_history, 'r-', label='Ambiguity', linewidth=2)
        ax2_twin.plot(iterations, divergence_history, 'g--', label='Divergence', linewidth=2)

        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Ambiguity', color='r', fontsize=12)
        ax2_twin.set_ylabel('Divergence', color='g', fontsize=12)
        ax2.set_title('Dual Objective Balance', fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved network growth visualization to {save_path}")
        else:
            plt.show()

    def visualize_hierarchical_structure(
        self,
        network_bmd: 'NetworkBMD',
        save_path: Optional[str] = None
    ):
        """
        Visualize hierarchical compound BMD structure.

        Args:
            network_bmd: Final network BMD
            save_path: Optional save path
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Count compounds by order
        compound_counts = {}
        for order in range(2, 6):
            compounds = network_bmd.get_compound_bmds_by_order(order)
            compound_counts[order] = len(compounds)

        orders = list(compound_counts.keys())
        counts = list(compound_counts.values())

        # Bar plot
        bars = ax.bar(orders, counts, color='steelblue', alpha=0.7, edgecolor='black')

        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{int(count)}',
                ha='center',
                va='bottom',
                fontsize=12,
                weight='bold'
            )

        ax.set_xlabel('Compound Order', fontsize=14)
        ax.set_ylabel('Number of Compounds', fontsize=14)
        ax.set_title('Hierarchical BMD Network Structure', fontsize=16)
        ax.set_xticks(orders)
        ax.set_xticklabels([f'Order {o}' for o in orders])
        ax.grid(True, axis='y', alpha=0.3)

        # Add summary text
        total_compounds = sum(counts)
        summary_text = (
            f"Total Regions: {len(network_bmd.region_bmds)}\n"
            f"Total Compounds: {total_compounds}\n"
            f"Network Richness: {network_bmd.network_categorical_richness():.2e}"
        )

        ax.text(
            0.98, 0.97,
            summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved hierarchical structure visualization to {save_path}")
        else:
            plt.show()

    def visualize_complete_results(
        self,
        image: np.ndarray,
        regions: List['Region'],
        results: Dict[str, Any],
        save_dir: Optional[str] = None
    ):
        """
        Create complete visualization suite for results.

        Args:
            image: Original image
            regions: All regions
            results: Results dictionary from HCCC algorithm
            save_dir: Optional directory to save figures
        """
        # Processing sequence
        self.visualize_processing_sequence(
            image,
            regions,
            results['processing_sequence'],
            save_path=f"{save_dir}/processing_sequence.png" if save_dir else None
        )

        # Network growth
        self.visualize_network_growth(
            results['network_richness_history'],
            results['ambiguity_history'],
            results['stream_divergence_history'],
            save_path=f"{save_dir}/network_growth.png" if save_dir else None
        )

        # Hierarchical structure
        self.visualize_hierarchical_structure(
            results['network_bmd_final'],
            save_path=f"{save_dir}/hierarchical_structure.png" if save_dir else None
        )

        print(f"Generated complete visualization suite")
