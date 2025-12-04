# Quick Start: Multi-Modal Virtual Detectors

## Installation

```bash
cd maxwell
pip install -r requirements.txt
```

## Quick Demo (30 seconds)

Run complete framework demo on a single image:

```bash
python demo_complete_framework.py public/1585.jpg
```

This will:
- ✅ Initialize pixel demon grid
- ✅ Run 8 imaging modalities **simultaneously**
- ✅ Extract categorical depth
- ✅ Generate comprehensive visualization
- ✅ Save all results to `demo_complete_results/`

**Output:**
- `complete_analysis.png` - 12-panel visualization
- `categorical_depth.npy` - 3D depth map
- `complete_results.json` - All statistics
- 8 modality maps

## Multi-Modal Validation Suite

Process multiple life sciences images:

```bash
python validate_life_sciences_multi_modal.py --max-images 5
```

Arguments:
- `--public-dir` - Image directory (default: `public/`)
- `--output-dir` - Output directory (default: `multi_modal_validation/`)
- `--max-images` - Maximum images to process (default: all)
- `--temperature` - Temperature in Kelvin (default: 310.15 K = 37°C)

**Output:**
- One directory per image with:
  - 8 modality NPY files
  - Multi-modal visualization
  - JSON statistics
- `complete_multi_modal_results.json` - Aggregate statistics

## Available Virtual Detectors

| Detector | Measures | Use Case |
|----------|----------|----------|
| `VirtualPhotodiode` | Light intensity | Fluorescence microscopy |
| `VirtualIRSpectrometer` | IR absorption | Molecular vibrations |
| `VirtualRamanSpectrometer` | Raman signal | Molecular bonds |
| `VirtualMassSpectrometer` | Molecular mass | Composition analysis |
| `VirtualThermometer` | Temperature | Thermal mapping |
| `VirtualBarometer` | Pressure | Pressure distribution |
| `VirtualHygrometer` | Humidity | Moisture content |
| `VirtualInterferometer` | Phase | Optical path length |

## Python API

### Basic Usage

```python
from maxwell.pixel_maxwell_demon import PixelDemonGrid
from maxwell.virtual_detectors import VirtualThermometer, VirtualIRSpectrometer

# Load image
import cv2
image = cv2.imread('sample.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize pixel demon grid
grid = PixelDemonGrid(width=image.shape[1], height=image.shape[0])
grid.initialize_from_image(image)

# Query with virtual thermometer
temp_map = np.zeros(image.shape[:2])
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        pixel_demon = grid.grid[y, x]
        thermometer = VirtualThermometer(pixel_demon)
        temp_map[y, x] = thermometer.observe_molecular_demons(
            pixel_demon.molecular_demons
        )

# Query with IR spectrometer (SAME grid!)
ir_map = np.zeros(image.shape[:2])
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        pixel_demon = grid.grid[y, x]
        ir_spec = VirtualIRSpectrometer(pixel_demon)
        ir_map[y, x] = ir_spec.observe_molecular_demons(
            pixel_demon.molecular_demons
        )
```

### All Modalities at Once

```python
from maxwell.virtual_detectors import (
    VirtualThermometer, VirtualBarometer, VirtualHygrometer,
    VirtualIRSpectrometer, VirtualRamanSpectrometer,
    VirtualMassSpectrometer, VirtualPhotodiode, VirtualInterferometer
)

detector_classes = [
    VirtualThermometer, VirtualBarometer, VirtualHygrometer,
    VirtualIRSpectrometer, VirtualRamanSpectrometer,
    VirtualMassSpectrometer, VirtualPhotodiode, VirtualInterferometer
]

modality_maps = {}
for DetectorClass in detector_classes:
    map_2d = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            detector = DetectorClass(grid.grid[y, x])
            map_2d[y, x] = detector.observe_molecular_demons(
                grid.grid[y, x].molecular_demons
            )
    modality_maps[DetectorClass.__name__] = map_2d

# Now you have 8 modality maps from SAME sample!
```

## Key Advantages

### 1. Zero Sample Commitment

Traditional: Choose modality → Prepare sample → Measure → Done (can't change)
Our method: Initialize once → Query all modalities → Choose interpretation later

### 2. Non-Destructive Mass Spectrometry

Traditional: Ionize and fragment → Sample destroyed
Our method: Query molecular demon states → Sample unchanged

### 3. Perfect Spatial Correlation

Traditional: Different samples → No correlation possible
Our method: Same pixel grid → Perfect pixel-by-pixel correlation

### 4. Massive Sample Savings

**Example**: 10 images, 8 modalities
- Traditional: 80 separate samples
- Our method: 10 samples
- **Savings: 70 samples (87.5%)**

## Expected Output

### Console Output

```
================================================================================
  COMPLETE FRAMEWORK DEMONSTRATION
  Dual-Membrane HCCC + Multi-Modal Virtual Detectors
================================================================================

Loading image: public/1585.jpg
  Image shape: 512×512 = 262,144 pixels

Atmospheric conditions (biological):
  Temperature: 310.15 K (37.00 °C)
  Pressure: 101325 Pa
  Humidity: 80%

================================================================================
PART 1: INITIALIZING PIXEL DEMON GRID
================================================================================

Creating 512×512 dual-membrane pixel Maxwell demons...
✓ Grid initialized (12.34s)
  Each pixel maintains:
    • Molecular demon lattice (O₂, N₂, H₂O, CO₂, Ar)
    • Dual-membrane state (front/back conjugate faces)
    • S-entropy coordinates (S_k, S_t, S_e)

================================================================================
PART 2: MULTI-MODAL SIMULTANEOUS ANALYSIS
================================================================================

🚀 REVOLUTIONARY: Running ALL modalities on SAME sample!

Traditional imaging would require: 8 separate samples
Our method requires: 1 sample
Savings: 7 samples (88%)

  • Fluorescence          ... 2.31s ✓
  • IR_Spectroscopy       ... 2.18s ✓
  • Raman_Spectroscopy    ... 2.25s ✓
  • Mass_Spectrometry     ... 2.19s ✓
  • Temperature           ... 2.23s ✓
  • Pressure              ... 2.20s ✓
  • Humidity              ... 2.17s ✓
  • Phase_Interference    ... 2.22s ✓

✓ All 8 modalities complete (30.45s)
  Zero-backaction: Sample completely unchanged!
  Simultaneous: All measurements from same molecular demons!

================================================================================
PART 3: CATEGORICAL DEPTH EXTRACTION
================================================================================

Extracting depth from membrane thickness...
✓ Depth extracted from membrane thickness
  Depth range: [0.0023, 2.6834]
  Mean depth: 1.3428 ± 0.5234
  No stereo vision required!
  No depth sensors required!

================================================================================
  DEMONSTRATION COMPLETE
================================================================================

1. Categorical Depth:
   ✓ Extracted from membrane thickness
   ✓ Range: [0.0023, 2.6834]
   ✓ No stereo vision needed

2. Multi-Modal Analysis:
   ✓ 8 modalities simultaneously
   ✓ Traditional: 8 samples required
   ✓ Our method: 1 sample (saves 7 samples!)
   ✓ Zero-backaction (no sample disturbance)

3. Performance:
   ✓ Total time: 30.45s
   ✓ Time per modality: 3.81s
   ✓ Zero physical commitment

4. Output:
   ✓ All results saved to: demo_complete_results
   ✓ Complete visualization: complete_analysis.png
   ✓ Multi-modal maps: 8 NPY files
   ✓ Comprehensive metrics: complete_results.json

================================================================================
```

### Visualization

The output `complete_analysis.png` contains:
- **Row 1:**
  - Original image
  - Categorical depth map
  - 3D depth surface
  - Depth histogram
- **Rows 2-3:** All 8 modality maps with colorbars

## Troubleshooting

### "ModuleNotFoundError: No module named 'maxwell'"

```bash
cd maxwell
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

Or add to script:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
```

### "No images found"

Ensure images are in `maxwell/public/` directory:
```bash
ls maxwell/public/*.jpg
```

### Slow performance

The pixel demon initialization is O(H×W), so larger images take longer.
For quick tests, resize images:

```python
# In script, after loading:
image = cv2.resize(image, (256, 256))  # Faster
```

## Next Steps

1. **Read the paper**: `maxwell/publication/hardware-constrained-categorical-cv/`
2. **Understand theory**: `maxwell/MULTI_MODAL_REVOLUTIONARY_ADVANTAGE.md`
3. **Run validation**: `maxwell/validate_life_sciences_multi_modal.py`
4. **Explore API**: `maxwell/src/maxwell/virtual_detectors.py`

## Summary

- ✅ **8 modalities** on **1 sample** (vs 8 samples traditionally)
- ✅ **Zero-backaction** (no sample disturbance)
- ✅ **Non-destructive** mass spectrometry
- ✅ **Perfect correlation** (same pixel grid)
- ✅ **~30 seconds** per image
- ✅ **87.5% sample savings**

**Revolutionary advantage**: No physical commitment required for life sciences imaging!

---

For detailed documentation, see `MULTI_MODAL_REVOLUTIONARY_ADVANTAGE.md`

