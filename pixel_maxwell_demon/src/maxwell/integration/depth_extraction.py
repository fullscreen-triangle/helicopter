"""
Depth Extraction from Membrane Thickness
========================================

Utilities for extracting and visualizing categorical depth from
dual-membrane pixel demon thickness measurements.

Author: Kundai Sachikonye
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import cv2


class DepthExtractor:
    """
    Extract and process categorical depth from membrane thickness.
    
    Depth = |S_k^(front) - S_k^(back)|
    
    For phase conjugate: S_k^(back) = -S_k^(front)
    Therefore: depth = 2|S_k^(front)|
    """
    
    def __init__(
        self,
        normalize: bool = True,
        smoothing_sigma: float = 1.0
    ):
        """
        Initialize depth extractor.
        
        Args:
            normalize: Normalize depth to [0, 1] range
            smoothing_sigma: Gaussian smoothing sigma (0 = no smoothing)
        """
        self.normalize = normalize
        self.smoothing_sigma = smoothing_sigma
    
    def extract(
        self,
        depth_map: np.ndarray,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract and process depth map.
        
        Args:
            depth_map: Raw membrane thickness map (H, W)
            min_depth: Minimum depth for normalization (auto if None)
            max_depth: Maximum depth for normalization (auto if None)
        
        Returns:
            Processed depth map (H, W)
        """
        # Apply smoothing if requested
        if self.smoothing_sigma > 0:
            depth_map = cv2.GaussianBlur(
                depth_map.astype(np.float32),
                (0, 0),
                self.smoothing_sigma
            )
        
        # Normalize if requested
        if self.normalize:
            if min_depth is None:
                min_depth = depth_map.min()
            if max_depth is None:
                max_depth = depth_map.max()
            
            # Avoid division by zero
            if max_depth - min_depth < 1e-10:
                depth_map = np.zeros_like(depth_map)
            else:
                depth_map = (depth_map - min_depth) / (max_depth - min_depth)
                depth_map = np.clip(depth_map, 0.0, 1.0)
        
        return depth_map
    
    def visualize_depth(
        self,
        depth_map: np.ndarray,
        colormap: str = 'turbo',
        show_colorbar: bool = True,
        title: str = 'Categorical Depth from Membrane Thickness'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize depth map with colormap.
        
        Args:
            depth_map: Depth map (H, W)
            colormap: Matplotlib colormap name
            show_colorbar: Show colorbar
            title: Figure title
        
        Returns:
            Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(depth_map, cmap=colormap)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Categorical Depth', fontsize=12)
        
        plt.tight_layout()
        
        return fig, ax
    
    def create_3d_visualization(
        self,
        depth_map: np.ndarray,
        image: Optional[np.ndarray] = None,
        elevation: float = 30,
        azimuth: float = 45
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create 3D surface visualization of depth.
        
        Args:
            depth_map: Depth map (H, W)
            image: Optional RGB image for texture mapping
            elevation: View elevation angle
            azimuth: View azimuth angle
        
        Returns:
            Figure and 3D axes
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        h, w = depth_map.shape
        
        # Create meshgrid
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        
        # Create figure
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        if image is not None:
            # Use image as texture
            surf = ax.plot_surface(
                X, Y, depth_map,
                facecolors=image / 255.0,
                rstride=1,
                cstride=1,
                shade=False
            )
        else:
            # Use colormap
            surf = ax.plot_surface(
                X, Y, depth_map,
                cmap='turbo',
                rstride=1,
                cstride=1,
                alpha=0.8
            )
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set labels
        ax.set_xlabel('X (pixels)', fontsize=10)
        ax.set_ylabel('Y (pixels)', fontsize=10)
        ax.set_zlabel('Categorical Depth', fontsize=10)
        ax.set_title('3D Categorical Depth Surface', fontsize=14, fontweight='bold')
        
        # Set view angle
        ax.view_init(elev=elevation, azim=azimuth)
        
        plt.tight_layout()
        
        return fig, ax
    
    def compute_depth_statistics(
        self,
        depth_map: np.ndarray
    ) -> dict:
        """
        Compute statistics on depth map.
        
        Args:
            depth_map: Depth map (H, W)
        
        Returns:
            Dictionary with statistics
        """
        return {
            'mean': np.mean(depth_map),
            'std': np.std(depth_map),
            'min': np.min(depth_map),
            'max': np.max(depth_map),
            'median': np.median(depth_map),
            'percentile_25': np.percentile(depth_map, 25),
            'percentile_75': np.percentile(depth_map, 75)
        }
    
    def create_depth_histogram(
        self,
        depth_map: np.ndarray,
        bins: int = 50
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create histogram of depth values.
        
        Args:
            depth_map: Depth map (H, W)
            bins: Number of histogram bins
        
        Returns:
            Figure and axes
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(depth_map.flatten(), bins=bins, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Categorical Depth', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Depth Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats = self.compute_depth_statistics(depth_map)
        stats_text = f"Mean: {stats['mean']:.4f}\nStd: {stats['std']:.4f}\nMedian: {stats['median']:.4f}"
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10
        )
        
        plt.tight_layout()
        
        return fig, ax
    
    def export_depth_map(
        self,
        depth_map: np.ndarray,
        output_path: str,
        format: str = 'npy'
    ):
        """
        Export depth map to file.
        
        Args:
            depth_map: Depth map (H, W)
            output_path: Output file path
            format: 'npy', 'png', or 'exr'
        """
        if format == 'npy':
            np.save(output_path, depth_map)
        elif format == 'png':
            # Normalize to uint16 for precision
            depth_uint16 = (depth_map * 65535).astype(np.uint16)
            cv2.imwrite(output_path, depth_uint16)
        elif format == 'exr':
            try:
                import OpenEXR
                import Imath
                
                # Convert to float32
                depth_float32 = depth_map.astype(np.float32)
                
                # Create EXR file
                header = OpenEXR.Header(depth_map.shape[1], depth_map.shape[0])
                header['channels'] = {'Z': Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))}
                
                exr = OpenEXR.OutputFile(output_path, header)
                exr.writePixels({'Z': depth_float32.tobytes()})
                exr.close()
            except ImportError:
                print("Warning: OpenEXR not installed. Falling back to NPY format.")
                np.save(output_path.replace('.exr', '.npy'), depth_map)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Depth map exported to: {output_path}")

