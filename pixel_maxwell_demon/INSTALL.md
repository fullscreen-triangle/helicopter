# Installation Instructions

## Method 1: Automated Install (Recommended)

Run the automated installer:

```bash
cd pixel_maxwell_demon
python install_and_test.py
```

This will:
1. Reinstall the package
2. Test all imports
3. Create quick-start scripts

## Method 2: Manual Install

```bash
cd pixel_maxwell_demon

# Install the package
pip install -e .

# Test imports
python test_imports.py
```

## Verify Installation

After installation, you should be able to import:

```python
# Core modules
from maxwell.pixel_maxwell_demon import PixelMaxwellDemon, SEntropyCoordinates
from maxwell.simple_pixel_grid import PixelDemonGrid
from maxwell.dual_membrane_pixel_demon import DualMembranePixelDemon, DualState

# Auxiliary modules
from maxwell.categorical_light_sources import Color
from maxwell.live_cell_imaging import LiveCellMicroscope
from maxwell.virtual_detectors import VirtualDetector
from maxwell.harmonic_coincidence import HarmonicCoincidenceNetwork
from maxwell.reflectance_cascade import ReflectanceCascade
```

## Running Analysis Scripts

### Virtual Imaging Demo

```bash
python demo_virtual_imaging.py ../maxwell/public/1585.jpg
```

**Output:** `virtual_imaging_results/` with 5 virtual images and visualization

### Visualize NPY Files

```bash
python visualize_npy_results.py --search-dir ../maxwell
```

**Output:** `npy_visualizations/` with panel charts

### Life Sciences Validation

```bash
python validate_life_sciences_multi_modal.py --max-images 5
```

**Output:** `multi_modal_validation/` with detector maps

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:

```bash
# Reinstall with dependencies
pip install -e . --force-reinstall

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Linux/Mac
$env:PYTHONPATH += ";$(pwd)\src"              # Windows PowerShell
```

### Missing Dependencies

```bash
pip install -r requirements.txt
```

### Package Not Found

Make sure you're in the `pixel_maxwell_demon` directory:

```bash
pwd  # Should end with /pixel_maxwell_demon
ls   # Should show: src/, setup.py, pyproject.toml, README.md
```

## Quick Start After Installation

```bash
# Run all three analysis pipelines
python demo_virtual_imaging.py ../maxwell/public/1585.jpg
python visualize_npy_results.py --search-dir .
python validate_life_sciences_multi_modal.py --max-images 3
python visualize_npy_results.py --search-dir multi_modal_validation --detailed
```

## Directory Structure

After successful installation:

```
pixel_maxwell_demon/
├── src/maxwell/              # Installed Python package
├── demo_virtual_imaging.py   # Virtual imaging script
├── visualize_npy_results.py  # NPY visualization script
├── validate_life_sciences_multi_modal.py  # Validation script
├── run_demo.py              # Quick demo runner (created by installer)
├── run_visualize.py         # Quick visualize runner (created by installer)
├── run_validate.py          # Quick validate runner (created by installer)
└── [results directories]    # Created by scripts
```

## Verification Checklist

- [ ] Package installed: `pip list | grep pixel-maxwell-demon`
- [ ] Imports work: `python test_imports.py` (all tests pass)
- [ ] Demo runs: `python demo_virtual_imaging.py ../maxwell/public/1585.jpg`
- [ ] Visualizer works: `python visualize_npy_results.py --search-dir .`
- [ ] Validation works: `python validate_life_sciences_multi_modal.py --max-images 1`

## Getting Help

If problems persist:

1. Check Python version: `python --version` (need ≥ 3.8)
2. Check working directory: `pwd`
3. List installed packages: `pip list`
4. Read error messages carefully
5. Check QUICKSTART.md for usage examples

