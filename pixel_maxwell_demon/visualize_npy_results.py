#!/usr/bin/env python3
"""
Visualize NPY Results: Create Panel Charts from Experiment Results
==================================================================

This script finds all NPY files from experiments and creates comprehensive
panel chart visualizations.

Usage:
    python visualize_npy_results.py [--search-dir DIR] [--output-dir DIR] [--detailed]

Author: Kundai Sachikonye
Date: 2024
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
from typing import List, Dict, Tuple
import json

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

def find_all_npy_files(directory: str = '.') -> Dict[str, List[Path]]:
    """
    Find all NPY files organized by experiment directory.
    
    Returns:
        Dictionary mapping experiment directory to list of NPY files
    """
    base_path = Path(directory)
    npy_files = {}
    
    for npy_file in base_path.rglob('*.npy'):
        # Get the experiment directory (parent of NPY file)
        exp_dir = npy_file.parent.name
        
        if exp_dir not in npy_files:
            npy_files[exp_dir] = []
        
        npy_files[exp_dir].append(npy_file)
    
    return npy_files


def load_npy_file(file_path: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load NPY file and extract metadata.
    
    Returns:
        (array, metadata_dict)
    """
    array = np.load(file_path)
    
    metadata = {
        'filename': file_path.name,
        'shape': array.shape,
        'dtype': str(array.dtype),
        'min': float(np.min(array)),
        'max': float(np.max(array)),
        'mean': float(np.mean(array)),
        'std': float(np.std(array))
    }
    
    return array, metadata


def create_panel_chart(
    arrays: List[Tuple[np.ndarray, str]],
    output_path: Path,
    title: str = "Experiment Results",
    colormap: str = 'viridis'
):
    """
    Create panel chart from multiple arrays.
    
    Args:
        arrays: List of (array, label) tuples
        output_path: Where to save the figure
        title: Overall title
        colormap: Colormap to use
    """
    n_arrays = len(arrays)
    
    # Determine grid size
    if n_arrays <= 3:
        n_rows, n_cols = 1, n_arrays
    elif n_arrays <= 6:
        n_rows, n_cols = 2, 3
    elif n_arrays <= 9:
        n_rows, n_cols = 3, 3
    elif n_arrays <= 12:
        n_rows, n_cols = 3, 4
    else:
        n_rows = int(np.ceil(np.sqrt(n_arrays)))
        n_cols = int(np.ceil(n_arrays / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    # Handle single subplot case
    if n_arrays == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_arrays > 1 else axes
    
    for idx, (array, label) in enumerate(arrays):
        ax = axes[idx] if n_arrays > 1 else axes[0]
        
        # Determine if this is a special type based on label
        if 'phase' in label.lower():
            cmap = 'twilight'
        elif 'red' in label.lower() or '650' in label:
            cmap = 'Reds'
        elif 'blue' in label.lower() or '450' in label:
            cmap = 'Blues'
        elif 'fluor' in label.lower():
            cmap = 'hot'
        elif 'dark' in label.lower():
            cmap = 'gray'
        elif 'depth' in label.lower():
            cmap = 'turbo'
        else:
            cmap = colormap
        
        # Plot
        im = ax.imshow(array, cmap=cmap)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add statistics
        stats_text = f"Range: [{array.min():.3f}, {array.max():.3f}]\nMean: {array.mean():.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Hide unused subplots
    for idx in range(len(arrays), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Panel chart saved: {output_path}")


def visualize_experiment_results(exp_dir: Path, npy_files: List[Path], output_dir: Path):
    """
    Visualize all NPY files from one experiment.
    
    Args:
        exp_dir: Experiment directory name
        npy_files: List of NPY files from this experiment
        output_dir: Where to save visualizations
    """
    print(f"\nVisualizing experiment: {exp_dir}")
    print(f"  Found {len(npy_files)} NPY files")
    
    # Load all arrays
    arrays = []
    metadata_all = {}
    
    for npy_file in sorted(npy_files):
        try:
            array, metadata = load_npy_file(npy_file)
            
            # Create nice label from filename
            label = npy_file.stem.replace('_', ' ').replace('-', ' ').title()
            
            arrays.append((array, label))
            metadata_all[npy_file.stem] = metadata
            
            print(f"  ✓ Loaded: {npy_file.name} - {array.shape}")
            
        except Exception as e:
            print(f"  ✗ Failed to load {npy_file.name}: {e}")
    
    if not arrays:
        print(f"  ✗ No valid arrays loaded")
        return
    
    # Create panel chart
    panel_output = output_dir / f'{exp_dir}_panel_chart.png'
    create_panel_chart(
        arrays,
        panel_output,
        title=f'{exp_dir.replace("_", " ").title()} - Panel Chart'
    )
    
    # Save metadata
    metadata_output = output_dir / f'{exp_dir}_metadata.json'
    with open(metadata_output, 'w') as f:
        json.dump(metadata_all, f, indent=2)
    
    print(f"  ✓ Metadata saved: {metadata_output}")
    
    # Create individual detailed views if many arrays
    if len(arrays) > 6:
        print(f"  Creating individual detailed views...")
        detail_dir = output_dir / f'{exp_dir}_detailed'
        detail_dir.mkdir(parents=True, exist_ok=True)
        
        for array, label in arrays:
            detail_output = detail_dir / f'{label.replace(" ", "_").lower()}.png'
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Determine colormap
            if 'phase' in label.lower():
                cmap = 'twilight'
            elif 'depth' in label.lower():
                cmap = 'turbo'
            else:
                cmap = 'viridis'
            
            im = ax.imshow(array, cmap=cmap)
            ax.set_title(label, fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Add detailed statistics
            stats = [
                f"Shape: {array.shape}",
                f"Min: {array.min():.6f}",
                f"Max: {array.max():.6f}",
                f"Mean: {array.mean():.6f}",
                f"Std: {array.std():.6f}",
                f"Median: {np.median(array):.6f}"
            ]
            
            stats_text = '\n'.join(stats)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            fig.tight_layout()
            fig.savefig(detail_output, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        print(f"  ✓ Detailed views saved to: {detail_dir}")


def create_comparison_chart(all_experiments: Dict[str, List[Path]], output_dir: Path):
    """
    Create comparison chart across all experiments.
    
    Args:
        all_experiments: Dictionary of experiment -> NPY files
        output_dir: Where to save visualization
    """
    print(f"\nCreating cross-experiment comparison...")
    
    # Collect one representative image from each experiment
    representatives = []
    
    for exp_name, npy_files in sorted(all_experiments.items()):
        if not npy_files:
            continue
        
        # Try to find a depth map or first available
        depth_file = None
        for npy_file in npy_files:
            if 'depth' in npy_file.stem.lower():
                depth_file = npy_file
                break
        
        if depth_file is None:
            depth_file = npy_files[0]
        
        try:
            array, _ = load_npy_file(depth_file)
            label = f"{exp_name}\n{depth_file.stem}"
            representatives.append((array, label))
        except:
            continue
    
    if representatives:
        comparison_output = output_dir / 'cross_experiment_comparison.png'
        create_panel_chart(
            representatives,
            comparison_output,
            title='Cross-Experiment Comparison',
            colormap='viridis'
        )


def main():
    parser = argparse.ArgumentParser(
        description='Visualize NPY Results as Panel Charts'
    )
    parser.add_argument(
        '--search-dir',
        type=str,
        default='.',
        help='Directory to search for NPY files (default: current)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='npy_visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Create detailed individual views for each array'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("  NPY RESULTS VISUALIZER")
    print("  Creating Panel Charts from Experiment Data")
    print("="*80)
    
    # Find all NPY files
    search_path = Path(args.search_dir)
    print(f"\nSearching for NPY files in: {search_path.absolute()}")
    
    if not search_path.exists():
        print(f"✗ Directory not found: {search_path}")
        print(f"\nTip: Make sure the directory exists or try:")
        print(f"  python visualize_npy_results.py --search-dir ../maxwell")
        print(f"  python visualize_npy_results.py --search-dir .")
        return 1
    
    all_experiments = find_all_npy_files(args.search_dir)
    
    if not all_experiments:
        print("✗ No NPY files found!")
        print(f"\nSearched in: {search_path.absolute()}")
        print(f"\nTip: Generate NPY files first by running:")
        print(f"  python demo_virtual_imaging.py ../maxwell/public/1585.jpg")
        print(f"  python validate_life_sciences_multi_modal.py --max-images 3")
        print(f"\nThen visualize with:")
        print(f"  python visualize_npy_results.py --search-dir .")
        return 1
    
    total_files = sum(len(files) for files in all_experiments.values())
    print(f"✓ Found {len(all_experiments)} experiments with {total_files} total NPY files")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Visualize each experiment
    for exp_name, npy_files in sorted(all_experiments.items()):
        visualize_experiment_results(exp_name, npy_files, output_path)
    
    # Create cross-experiment comparison
    create_comparison_chart(all_experiments, output_path)
    
    # Summary
    print("\n" + "="*80)
    print("  VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\n✓ Processed {len(all_experiments)} experiments")
    print(f"✓ Created {len(all_experiments)} panel charts")
    print(f"✓ All results saved to: {output_path}")
    
    # List what was created
    print("\nGenerated files:")
    for item in sorted(output_path.iterdir()):
        if item.is_file():
            print(f"  • {item.name}")
        elif item.is_dir():
            n_files = len(list(item.iterdir()))
            print(f"  • {item.name}/ ({n_files} files)")
    
    print("="*80 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

