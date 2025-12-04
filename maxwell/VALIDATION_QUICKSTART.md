# Validation Quick Start Guide

## Run Validation on Life Sciences Images in 3 Commands

### 1. Navigate to Maxwell Directory
```bash
cd maxwell
```

### 2. Run Validation on Life Sciences Images
```bash
python validate_with_life_sciences_images.py --max-images 5
```

This will:
- Process 5 images from `public/` directory
- Generate depth maps for each
- Create comprehensive visualizations
- Produce aggregate statistics
- Save everything to `life_sciences_validation/`

Expected time: ~3-5 minutes for 5 images

### 3. View Results
```bash
# On Windows
explorer life_sciences_validation

# On Mac
open life_sciences_validation

# On Linux
xdg-open life_sciences_validation
```

## What You'll See

### Per-Image Results
Each image gets its own folder with:
- **depth_map.png** - Beautiful categorical depth visualization
- **comparison.png** - Original vs depth vs 3D surface
- **depth_histogram.png** - Depth distribution
- **convergence.png** - Algorithm convergence
- **results.json** - All numerical metrics

### Aggregate Results
- **summary_report.png** - Visual summary across all images
- **complete_results.json** - Comprehensive statistics

## Expected Output

```
life_sciences_validation/
├── summary_report.png          ← Start here! Visual summary
├── complete_results.json       ← All statistics
│
├── 10954/                      ← First image
│   ├── comparison.png         ← See original + depth + 3D
│   ├── depth_map.png
│   ├── depth_histogram.png
│   ├── convergence.png
│   └── results.json
│
├── 1585/                       ← Second image
│   └── ...
│
└── ... (more images)
```

## Validate Specific Image

To process just one image with custom parameters:

```bash
python demo_dual_hccc.py public/10954.jpg \
    --n-segments 100 \
    --cascade-depth 20 \
    --validate \
    --output-dir my_results
```

## Validation Metrics to Check

Look for these in the results:

1. **Success Rate**: Should be >90%
2. **Average Richness**: Typically 1000-2000
3. **Stream Coherence**: Should be 0.7-0.9
4. **Energy Dissipation**: ~10⁻¹⁸ J (thermodynamically consistent)
5. **Convergence Rate**: >90% of images should converge

## Troubleshooting

### Python Not Found
```bash
python3 validate_with_life_sciences_images.py --max-images 5
```

### Module Import Errors
```bash
cd maxwell
pip install -r requirements.txt
python validate_with_life_sciences_images.py --max-images 5
```

### Memory Issues
Process fewer images or reduce segments:
```bash
python validate_with_life_sciences_images.py --max-images 2 --n-segments 25
```

## Advanced Options

### Process All Images
```bash
python validate_with_life_sciences_images.py
```
(No `--max-images` argument processes all images in `public/`)

### Higher Precision
```bash
python validate_with_life_sciences_images.py \
    --n-segments 200 \
    --cascade-depth 30 \
    --max-images 3
```

### Custom Output Directory
```bash
python validate_with_life_sciences_images.py \
    --output-dir my_validation_results \
    --max-images 5
```

## What the Validation Tests

1. ✅ **Conjugate Relationship**: S_k^(back) = -S_k^(front)
2. ✅ **Zero-Backaction**: Categorical queries transfer zero momentum
3. ✅ **O(N³) Cascade**: Information scales cubically
4. ✅ **Energy Bounds**: Thermodynamically consistent (Landauer's principle)
5. ✅ **Convergence**: Monotonic richness growth
6. ✅ **Depth Extraction**: Multi-scale categorical depth

## Quick Python Test

```python
# In Python interpreter or Jupyter notebook
import sys
from pathlib import Path
sys.path.insert(0, str(Path('maxwell/src')))

from maxwell.integration import DualMembraneHCCCAlgorithm
import cv2

# Load one image
image = cv2.imread('maxwell/public/10954.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process
algorithm = DualMembraneHCCCAlgorithm()
result = algorithm.process_image(image, n_segments=50)

# Check results
print(f"✓ Final richness: {result.final_richness:.2f}")
print(f"✓ Stream coherence: {result.final_stream_coherence:.2f}")
print(f"✓ Converged: {result.converged}")
print(f"✓ Depth range: [{result.depth_map.min():.3f}, {result.depth_map.max():.3f}]")
```

## Next Steps After Validation

1. Review `life_sciences_validation/summary_report.png`
2. Check individual image results in subfolders
3. Read `complete_results.json` for detailed metrics
4. See `COMPLETE_FRAMEWORK_GUIDE.md` for full documentation
5. Read the scientific paper in `publication/`

## Support

- **Documentation**: See `maxwell/COMPLETE_FRAMEWORK_GUIDE.md`
- **Integration Details**: See `maxwell/src/maxwell/integration/README.md`
- **Implementation**: See `maxwell/IMPLEMENTATION_SUMMARY.md`

---

**Time Required**: 5-10 minutes for complete validation

**Output Size**: ~50-100 MB (images + depth maps + visualizations)

**Success Criteria**: >90% images processed successfully with consistent metrics

**Status**: ✅ Ready to validate groundbreaking work!

---

*Let's validate this framework on real life sciences data!* 🔬✨

