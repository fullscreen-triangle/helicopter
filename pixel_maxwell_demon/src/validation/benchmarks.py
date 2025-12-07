"""
Benchmark datasets and tests for HCCC algorithm.

Provides:
- Synthetic test images with known ground truth
- Standard dataset loaders
- Performance benchmarking utilities
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from skimage import data, color, transform


class BenchmarkSuite:
    """
    Benchmark suite for HCCC algorithm validation.

    Includes:
    - Synthetic images with known structure
    - Standard vision datasets
    - Performance metrics
    """

    def __init__(self):
        """Initialize benchmark suite."""
        pass

    def generate_synthetic_image(
        self,
        image_type: str = 'geometric',
        size: Tuple[int, int] = (256, 256),
        **kwargs
    ) -> np.ndarray:
        """
        Generate synthetic test image.

        Args:
            image_type: Type of synthetic image
                - 'geometric': Simple geometric shapes
                - 'gradient': Color gradients
                - 'texture': Texture patterns
                - 'random': Random noise with structure
            size: Image size (H, W)
            **kwargs: Type-specific parameters

        Returns:
            Synthetic image (H x W x 3)
        """
        if image_type == 'geometric':
            return self._generate_geometric(size, **kwargs)
        elif image_type == 'gradient':
            return self._generate_gradient(size, **kwargs)
        elif image_type == 'texture':
            return self._generate_texture(size, **kwargs)
        elif image_type == 'random':
            return self._generate_random_structured(size, **kwargs)
        else:
            raise ValueError(f"Unknown image type: {image_type}")

    def _generate_geometric(
        self,
        size: Tuple[int, int],
        n_shapes: int = 5
    ) -> np.ndarray:
        """Generate image with geometric shapes."""
        H, W = size
        image = np.ones((H, W, 3)) * 255  # White background

        for i in range(n_shapes):
            # Random shape type
            shape_type = np.random.choice(['circle', 'square', 'triangle'])

            # Random position and size
            cx = np.random.randint(W // 4, 3 * W // 4)
            cy = np.random.randint(H // 4, 3 * H // 4)
            radius = np.random.randint(20, 60)

            # Random color
            color = np.random.randint(0, 256, 3)

            # Draw shape
            if shape_type == 'circle':
                y, x = np.ogrid[:H, :W]
                mask = ((x - cx)**2 + (y - cy)**2) <= radius**2
                image[mask] = color

        return image.astype(np.uint8)

    def _generate_gradient(
        self,
        size: Tuple[int, int],
        direction: str = 'horizontal'
    ) -> np.ndarray:
        """Generate color gradient image."""
        H, W = size

        if direction == 'horizontal':
            gradient = np.linspace(0, 255, W)
            image = np.tile(gradient, (H, 1))
        elif direction == 'vertical':
            gradient = np.linspace(0, 255, H)
            image = np.tile(gradient[:, np.newaxis], (1, W))
        elif direction == 'radial':
            y, x = np.ogrid[:H, :W]
            cy, cx = H // 2, W // 2
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            image = (distance / distance.max()) * 255

        # Convert to RGB
        image_rgb = np.stack([image] * 3, axis=-1)

        return image_rgb.astype(np.uint8)

    def _generate_texture(
        self,
        size: Tuple[int, int],
        frequency: float = 0.1
    ) -> np.ndarray:
        """Generate texture pattern."""
        H, W = size

        # Generate checkerboard-like texture
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        pattern = (
            np.sin(frequency * x) * np.cos(frequency * y) +
            np.cos(frequency * x * 1.5) * np.sin(frequency * y * 1.5)
        )

        # Normalize to [0, 255]
        pattern = ((pattern - pattern.min()) /
                  (pattern.max() - pattern.min() + 1e-10) * 255)

        # Convert to RGB
        image = np.stack([pattern] * 3, axis=-1)

        return image.astype(np.uint8)

    def _generate_random_structured(
        self,
        size: Tuple[int, int],
        n_regions: int = 10
    ) -> np.ndarray:
        """Generate random but structured image."""
        from scipy.ndimage import gaussian_filter

        H, W = size

        # Random regions
        image = np.zeros((H, W, 3))

        for i in range(n_regions):
            # Random region mask
            region_mask = np.random.rand(H, W) > 0.7
            region_mask = gaussian_filter(region_mask.astype(float), sigma=10) > 0.3

            # Random color
            color = np.random.rand(3) * 255

            image[region_mask] = color

        # Smooth
        for c in range(3):
            image[:, :, c] = gaussian_filter(image[:, :, c], sigma=2)

        return image.astype(np.uint8)

    def load_standard_image(
        self,
        name: str = 'astronaut'
    ) -> np.ndarray:
        """
        Load standard test image from scikit-image.

        Args:
            name: Image name ('astronaut', 'camera', 'cat', etc.)

        Returns:
            Image array
        """
        if name == 'astronaut':
            image = data.astronaut()
        elif name == 'camera':
            image = data.camera()
            image = color.gray2rgb(image)
        elif name == 'cat':
            image = data.chelsea()
        elif name == 'coffee':
            image = data.coffee()
        else:
            # Fallback
            image = data.astronaut()

        return image

    def benchmark_algorithm(
        self,
        algorithm,
        test_images: List[np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Benchmark algorithm on test images.

        Args:
            algorithm: HCCC algorithm instance
            test_images: List of test images
            **kwargs: Algorithm parameters

        Returns:
            Benchmark results dictionary
        """
        results = {
            'n_images': len(test_images),
            'per_image': [],
            'aggregate': {}
        }

        import time

        for i, image in enumerate(test_images):
            print(f"Processing image {i+1}/{len(test_images)}...")

            start_time = time.time()

            try:
                result = algorithm.process_image(image, **kwargs)

                elapsed_time = time.time() - start_time

                image_result = {
                    'success': True,
                    'time': elapsed_time,
                    'iterations': result['convergence_step'],
                    'regions_processed': result['regions_processed'],
                    'final_richness': result['network_bmd_final'].network_categorical_richness(),
                    'converged': result['metrics'].get('quality_score', 0) > 0.5
                }
            except Exception as e:
                elapsed_time = time.time() - start_time
                image_result = {
                    'success': False,
                    'time': elapsed_time,
                    'error': str(e)
                }

            results['per_image'].append(image_result)

        # Aggregate metrics
        successful = [r for r in results['per_image'] if r['success']]

        if successful:
            results['aggregate'] = {
                'success_rate': len(successful) / len(test_images),
                'mean_time': np.mean([r['time'] for r in successful]),
                'mean_iterations': np.mean([r['iterations'] for r in successful]),
                'convergence_rate': np.mean([r['converged'] for r in successful])
            }

        return results
