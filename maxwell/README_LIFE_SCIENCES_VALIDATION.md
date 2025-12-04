# Life Sciences Image Validation

This document describes how to use the dual-membrane HCCC framework to validate on life sciences images in the `maxwell/public/` directory.

## Available Images

The `public/` directory contains various life sciences images including:
- Microscopy images (`.jpg`, `.tif`)
- Medical imaging data
- Biological samples
- Cellular structures at multiple scales

## Quick Start

### 1. Validate All Images

Process all images in the public directory:

```bash
cd maxwell
python validate_with_life_sciences_images.py
```

This will:
- Process all life sciences images in `public/`
- Generate depth maps for each image
- Create comprehensive visualizations
- Produce aggregate statistics
- Save results to `life_sciences_validation/`

### 2. Validate Specific Number of Images

To test on a limited number of images first:

```bash
python validate_with_life_sciences_images.py --max-images 3
```

### 3. Custom Parameters

Adjust processing parameters:

```bash
python validate_with_life_sciences_images.py \
    --n-segments 100 \
    --cascade-depth 20 \
    --output-dir my_validation_results
```

## Output Structure

After running validation, you'll find:

```
life_sciences_validation/
├── complete_results.json          # Aggregate statistics
├── summary_report.png              # Visual summary across all images
│
├── 10954/                          # Results for image 10954.jpg
│   ├── results.json               # Processing metrics
│   ├── depth_map.npy              # Raw depth data
│   ├── depth_map.png              # Depth visualization
│   ├── depth_histogram.png        # Depth distribution
│   ├── comparison.png             # Original vs depth vs 3D
│   └── convergence.png            # Algorithm convergence
│
├── 1585/                           # Results for image 1585.jpg
│   └── ...
│
└── ...                             # More image results
```

## Results Interpretation

### Per-Image Results

Each image produces:

1. **Depth Map** (`depth_map.png`): 
   - Categorical depth extracted from membrane thickness
   - Colormap: Red = far, Blue = near
   - Range: Normalized [0, 1]

2. **3D Visualization** (`comparison.png`):
   - Original image
   - 2D depth map
   - 3D surface with texture

3. **Processing Metrics** (`results.json`):
   ```json
   {
     "processing_time": 45.3,
     "total_iterations": 42,
     "converged": true,
     "final_richness": 1234.567,
     "final_stream_coherence": 0.8543,
     "energy_dissipation": 2.85e-18,
     "depth_statistics": {
       "mean": 0.456,
       "std": 0.123,
       "min": 0.001,
       "max": 0.999
     }
   }
   ```

### Aggregate Statistics

The `summary_report.png` shows:
- **Processing Time Distribution**: How long each image takes
- **Network Richness Distribution**: Categorical information gained
- **Stream Coherence Distribution**: Hardware stream alignment
- **Convergence Iterations**: How many iterations until convergence
- **Energy Dissipation Distribution**: Thermodynamic cost
- **Depth by Image**: Mean categorical depth for each image

The `complete_results.json` includes:
```json
{
  "aggregate_statistics": {
    "total_images": 12,
    "successful": 12,
    "failed": 0,
    "success_rate": 1.0,
    "average_processing_time": 43.2,
    "average_iterations": 38.5,
    "average_richness": 1189.34,
    "average_coherence": 0.8421,
    "average_energy": 2.67e-18,
    "convergence_rate": 0.917
  }
}
```

## Expected Performance

For typical life sciences images (512×512 to 1024×1024):

| Metric | Typical Value | Notes |
|--------|---------------|-------|
| Processing time | 30-60 seconds | Depends on n_segments |
| Iterations | 30-50 | Usually converges well before max |
| Richness | 1000-2000 | Scales with image complexity |
| Stream coherence | 0.7-0.9 | Higher = better hardware alignment |
| Energy | ~10⁻¹⁸ J | Within thermodynamic bounds |
| Convergence rate | >90% | Most images converge successfully |

## Validation Checks

The framework automatically validates:

1. **Conjugate Relationship**: S_k^(back) = -S_k^(front)
   - Expected: Perfect anti-correlation (r = -1.000)
   
2. **Zero-Backaction**: Categorical queries transfer zero momentum
   - Expected: ⟨Δp⟩ = 0
   
3. **O(N³) Cascade Scaling**: Information gain scales cubically
   - Expected: I_N ∝ N³
   - For N=10: 385× enhancement
   
4. **Energy Dissipation**: Thermodynamically consistent
   - Expected: E ≥ k_B T ln(2) per bit
   - At 298K: ≥2.85×10⁻²¹ J per bit
   
5. **Convergence**: Monotonic richness growth
   - Expected: Δ richness → 0

## Common Use Cases

### 1. Microscopy Image Analysis

```python
from maxwell.integration import DualMembraneHCCCAlgorithm
import cv2

# Load microscopy image
image = cv2.imread('public/10954.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process with fine-grained segmentation
algorithm = DualMembraneHCCCAlgorithm(
    max_iterations=100,
    use_cascade=True,
    cascade_depth=10
)

result = algorithm.process_image(image, n_segments=200)

# Depth reveals cellular structure
import numpy as np
np.save('cellular_depth.npy', result.depth_map)
```

### 2. Medical Image Depth Extraction

```python
# For medical imaging, use higher cascade depth
algorithm = DualMembraneHCCCAlgorithm(
    cascade_depth=20,  # More information gain
    lambda_stream=0.7  # Stronger hardware coherence
)

result = algorithm.process_image(medical_image, n_segments=150)
```

### 3. Multi-Image Batch Processing

```python
from pathlib import Path
from maxwell.integration import DualMembraneHCCCAlgorithm

algorithm = DualMembraneHCCCAlgorithm()

for image_path in Path('public').glob('*.jpg'):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    result = algorithm.process_image(image, n_segments=50)
    
    # Save results
    output_dir = Path('batch_results') / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'depth.npy', result.depth_map)
```

## Troubleshooting

### Image Loading Fails

```python
# Check if image is valid
image = cv2.imread('public/image.jpg')
if image is None:
    print("Failed to load image")
    # Try different format
    from PIL import Image
    image = np.array(Image.open('public/image.tif'))
```

### Processing Too Slow

```python
# Reduce segments for faster processing
algorithm = DualMembraneHCCCAlgorithm()
result = algorithm.process_image(image, n_segments=25)  # Faster
```

### Memory Issues

```python
# For large images, downsample first
import cv2

# Resize to max dimension of 1024
h, w = image.shape[:2]
if max(h, w) > 1024:
    scale = 1024 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    image = cv2.resize(image, (new_w, new_h))
```

## Scientific Validation

The life sciences images provide excellent validation because:

1. **Multi-Scale Structure**: Cellular images have depth at multiple scales
2. **Known Ground Truth**: Some microscopy images have known depth information
3. **Biological Relevance**: Tests framework on real biological systems
4. **Diverse Samples**: Different imaging modalities and biological structures
5. **Real-World Data**: Non-synthetic, authentic scientific images

## Citation

If you use this framework for life sciences image analysis, please cite:

```bibtex
@article{mataranyika2025dualmembrane,
  title={Hardware-Constrained Categorical Computer Vision via Dual-Membrane Pixel Maxwell Demons},
  author={Mataranyika, Kundai Sachikonye},
  journal={[Journal Name]},
  year={2025}
}
```

## Contact

For questions or issues with life sciences image validation:
- Open an issue on GitHub
- Email: [your email]
- Documentation: See `maxwell/src/maxwell/integration/README.md`

---

**Status**: Ready for validation
**Last Updated**: December 2024
**Framework Version**: 1.0.0

