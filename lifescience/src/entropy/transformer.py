"""
S-Entropy Transformer - Core transformation engine for biological images
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

from .coordinates import SEntropyCoordinates, BiologicalContext

logger = logging.getLogger(__name__)


class SEntropyTransformer:
    """Core S-entropy coordinate transformation engine"""
    
    def __init__(self, biological_context: BiologicalContext = BiologicalContext.CELLULAR):
        self.biological_context = biological_context
        self.transformation_history = []
        logger.info(f"Initialized S-entropy transformer for {biological_context.value}")
    
    def transform(self, image: np.ndarray) -> SEntropyCoordinates:
        """Transform image to S-entropy coordinates"""
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.float32)
        
        # Normalize
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        
        # Analyze structural complexity
        structural = self._analyze_structural_complexity(gray)
        
        # Analyze functional activity (gradient magnitude)
        functional = self._analyze_functional_activity(gray)
        
        # Analyze morphological diversity (texture variation)
        morphological = self._analyze_morphological_diversity(gray)
        
        # Analyze temporal dynamics (single image - estimate potential)
        temporal = self._analyze_temporal_dynamics(gray)
        
        coordinates = SEntropyCoordinates(
            structural=structural,
            functional=functional,
            morphological=morphological,
            temporal=temporal,
            biological_context=self.biological_context,
            confidence=0.8  # Simplified confidence
        )
        
        # Store in history
        self.transformation_history.append({
            'coordinates': coordinates,
            'image_shape': image.shape,
            'biological_context': self.biological_context.value
        })
        
        logger.info(f"Transformed image to coordinates: {coordinates.to_array()}")
        return coordinates
    
    def _analyze_structural_complexity(self, gray: np.ndarray) -> float:
        """Analyze structural complexity of image"""
        # Edge density as measure of structure
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        structural = edge_density * 2 - 1  # Normalize to [-1, 1]
        return np.clip(structural, -1, 1)
    
    def _analyze_functional_activity(self, gray: np.ndarray) -> float:
        """Analyze functional activity indicators"""
        # Gradient magnitude as activity indicator
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        functional = np.clip(np.mean(gradient_magnitude) * 4 - 1, -1, 1)
        return functional
    
    def _analyze_morphological_diversity(self, gray: np.ndarray) -> float:
        """Analyze morphological diversity"""
        # Texture variation using Laplacian
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        morphological = np.clip(laplacian_var / 100.0 * 2 - 1, -1, 1)
        return morphological
    
    def _analyze_temporal_dynamics(self, gray: np.ndarray) -> float:
        """Estimate temporal dynamics potential from single image"""
        # Use asymmetry as proxy for dynamic potential
        left_half = gray[:, :gray.shape[1]//2]
        right_half = np.fliplr(gray[:, gray.shape[1]//2:])
        asymmetry = np.mean(np.abs(left_half - right_half[:, :left_half.shape[1]]))
        temporal = np.clip(asymmetry * 8 - 1, -1, 1)
        return temporal
    
    def batch_transform(self, images: List[np.ndarray]) -> List[SEntropyCoordinates]:
        """Transform multiple images"""
        return [self.transform(img) for img in images]
    
    def analyze_trajectory(self, coordinates_sequence: List[SEntropyCoordinates]) -> Dict[str, Any]:
        """Analyze trajectory of S-entropy coordinates over time"""
        if len(coordinates_sequence) < 2:
            return {'error': 'Need at least 2 coordinates for trajectory analysis'}
        
        # Convert to array for analysis
        coord_array = np.array([coords.to_array() for coords in coordinates_sequence])
        
        # Calculate trajectory properties
        trajectory_length = 0.0
        for i in range(1, len(coord_array)):
            displacement = coord_array[i] - coord_array[i-1]
            distance = np.linalg.norm(displacement)
            trajectory_length += distance
        
        # Trajectory statistics
        analysis = {
            'total_length': trajectory_length,
            'final_position': coord_array[-1],
            'position_variance': np.var(coord_array, axis=0),
            'dominant_motion_dimension': np.argmax(np.var(coord_array, axis=0))
        }
        
        # Biological interpretation
        dominant_dim_names = ['structural', 'functional', 'morphological', 'temporal']
        dominant_motion = dominant_dim_names[analysis['dominant_motion_dimension']]
        
        analysis['biological_interpretation'] = f"Primary changes in {dominant_motion} dimension"
        
        return analysis
    
    def visualize_coordinates(self, coordinates: Union[SEntropyCoordinates, List[SEntropyCoordinates]], 
                            save_path: Optional[str] = None) -> plt.Figure:
        """Visualize S-entropy coordinates"""
        if isinstance(coordinates, SEntropyCoordinates):
            coordinates = [coordinates]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('S-Entropy Coordinate Analysis', fontsize=16)
        
        # Extract data
        coord_arrays = [coord.to_array() for coord in coordinates]
        confidences = [coord.confidence for coord in coordinates]
        
        dimension_names = ['Structural', 'Functional', 'Morphological', 'Temporal']
        
        if len(coordinates) == 1:
            # Single coordinate visualization
            coord = coord_arrays[0]
            
            # Radar chart
            angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()
            angles += angles[:1]  # Close the circle
            coord_plot = coord.tolist() + [coord[0]]
            
            axes[0, 0].plot(angles, coord_plot, 'o-', linewidth=2, label='Coordinates')
            axes[0, 0].fill(angles, coord_plot, alpha=0.25)
            axes[0, 0].set_xticks(angles[:-1])
            axes[0, 0].set_xticklabels(dimension_names)
            axes[0, 0].set_ylim(-1, 1)
            axes[0, 0].set_title('S-Entropy Radar Chart')
            axes[0, 0].grid(True)
            
            # Bar chart
            axes[0, 1].bar(dimension_names, coord)
            axes[0, 1].set_ylim(-1, 1)
            axes[0, 1].set_title('Coordinate Values')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Interpretation text
            interpretation = coordinates[0].biological_interpretation()
            axes[1, 0].text(0.1, 0.5, interpretation, wrap=True, fontsize=10,
                           verticalalignment='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Biological Interpretation')
            axes[1, 0].axis('off')
            
            # Confidence
            axes[1, 1].bar(['Confidence'], [confidences[0]])
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title('Analysis Confidence')
            
        else:
            # Multiple coordinates - trajectory visualization
            coord_matrix = np.array(coord_arrays)
            
            # Time series
            for i, dim_name in enumerate(dimension_names):
                axes[0, 0].plot(range(len(coord_matrix)), coord_matrix[:, i], 'o-', label=dim_name)
            axes[0, 0].set_xlabel('Time Point')
            axes[0, 0].set_ylabel('Coordinate Value')
            axes[0, 0].set_title('Coordinate Evolution')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 3D trajectory (first 3 dimensions)
            ax_3d = fig.add_subplot(2, 2, 2, projection='3d')
            ax_3d.plot(coord_matrix[:, 0], coord_matrix[:, 1], coord_matrix[:, 2], 'o-')
            ax_3d.set_xlabel('Structural')
            ax_3d.set_ylabel('Functional')
            ax_3d.set_zlabel('Morphological')
            ax_3d.set_title('3D Coordinate Trajectory')
            
            # Confidence evolution
            axes[1, 0].plot(range(len(confidences)), confidences, 'o-', color='red')
            axes[1, 0].set_xlabel('Time Point')
            axes[1, 0].set_ylabel('Confidence')
            axes[1, 0].set_title('Analysis Confidence Over Time')
            axes[1, 0].grid(True)
            
            # Coordinate variance
            coord_var = np.var(coord_matrix, axis=0)
            axes[1, 1].bar(dimension_names, coord_var)
            axes[1, 1].set_title('Coordinate Variability')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Coordinate visualization saved to {save_path}")
        
        return fig
