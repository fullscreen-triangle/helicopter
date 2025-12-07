# Quick Start Guide

## Installation

```bash
cd pixel_maxwell_demon
pip install -e .
```

## Running Analysis Scripts

### 1. Virtual Imaging Demo

```bash
# From pixel_maxwell_demon directory
python demo_virtual_imaging.py ../maxwell/public/1585.jpg

# Expected output:
# - Creates virtual_imaging_results/ directory
# - Generates 5 virtual images (.npy files)
# - Creates panel chart visualization
# - Runtime: ~30 seconds
```

### 2. Visualize Existing NPY Files

```bash
# Visualize NPY files in maxwell directory
python visualize_npy_results.py --search-dir ../maxwell

# Or visualize locally generated files
python visualize_npy_results.py --search-dir .

# Expected output:
# - Creates npy_visualizations/ directory
# - Panel charts for each experiment
# - Cross-experiment comparison
# - Metadata JSON files
```

### 3. Validate with Life Sciences Images

```bash
# Process all images in maxwell/public
python validate_life_sciences_multi_modal.py

# Or process limited number
python validate_life_sciences_multi_modal.py --max-images 3

# Expected output:
# - Creates multi_modal_validation/ directory
# - 8 detector maps per image (Photodiode, IR, Raman, Mass Spec, etc.)
# - Multi-modal comparison charts
# - Aggregate statistics
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:

```bash
# Make sure you're in the pixel_maxwell_demon directory
cd pixel_maxwell_demon

# Re-install in editable mode
pip install -e .

# Or add to PYTHONPATH temporarily
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Linux/Mac
$env:PYTHONPATH += ";$(pwd)\src"              # Windows PowerShell
```

### Image Path Issues

Scripts expect images in `../maxwell/public/`:

```bash
# Check if images exist
ls ../maxwell/public/*.jpg

# Or provide absolute path
python demo_virtual_imaging.py /absolute/path/to/image.jpg
```

### Missing Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install specific missing package
pip install numpy matplotlib opencv-python
```

## Expected Results

### Virtual Imaging

After running `demo_virtual_imaging.py`, you should see:

```
================================================================================
  VIRTUAL IMAGING DEMONSTRATION
  Capture ONCE, Query MULTIPLE Ways!
================================================================================

Loading image: ../maxwell/public/1585.jpg
  Image: 1024×1024

Initializing pixel demon grid...
  ✓ Grid initialized with categorical coordinates

================================================================================
SCENARIO 1: WAVELENGTH SHIFTING
================================================================================
  Generating virtual image: 550nm → 650nm
    ✓ Virtual image generated (no re-imaging required!)
  
  Generating virtual image: 550nm → 450nm
    ✓ Virtual image generated (no re-imaging required!)

... (continues for all 4 scenarios)

================================================================================
  DEMONSTRATION COMPLETE
================================================================================

Virtual Images Generated:
  1. ✓ 650nm (red) - from 550nm
  2. ✓ 450nm (blue) - from 550nm
  3. ✓ Dark-field - from bright-field
  4. ✓ Fluorescence 561nm - from 488nm
  5. ✓ Phase contrast - from amplitude (dual-membrane!)

Traditional vs Virtual:
  Traditional: 7 separate captures, > 4 hours
  Virtual: 1 capture, < 30 seconds
  Savings: 6 captures (85.7%), > 3.5 hours

Results saved to: virtual_imaging_results
================================================================================
```

### NPY Visualization

After running `visualize_npy_results.py`:

```
================================================================================
  NPY RESULTS VISUALIZER
  Creating Panel Charts from Experiment Data
================================================================================

Searching for NPY files in: ../maxwell
✓ Found 3 experiments with 24 total NPY files

Visualizing experiment: virtual_imaging_results
  Found 5 NPY files
  ✓ Loaded: virtual_650nm.npy - (1024, 1024)
  ✓ Loaded: virtual_450nm.npy - (1024, 1024)
  ✓ Loaded: virtual_darkfield.npy - (1024, 1024)
  ✓ Loaded: virtual_fluorescence_561nm.npy - (1024, 1024)
  ✓ Loaded: virtual_phase_contrast.npy - (1024, 1024)
  ✓ Panel chart saved: npy_visualizations/virtual_imaging_results_panel_chart.png
  ✓ Metadata saved: npy_visualizations/virtual_imaging_results_metadata.json

... (continues for other experiments)

================================================================================
  VISUALIZATION COMPLETE
================================================================================

✓ Processed 3 experiments
✓ Created 3 panel charts
✓ All results saved to: npy_visualizations

Generated files:
  • cross_experiment_comparison.png
  • virtual_imaging_results_metadata.json
  • virtual_imaging_results_panel_chart.png
  ...
================================================================================
```

## Next Steps

1. **Explore Results**: Open the generated PNG files to see your visualizations
2. **Adjust Parameters**: Edit scripts to try different wavelengths, angles, etc.
3. **Process Your Data**: Replace image paths with your own microscopy images
4. **Read Theory**: Check `../maxwell/publication/` for scientific papers

## Common Workflows

### Full Analysis Pipeline

```bash
# 1. Generate virtual images
python demo_virtual_imaging.py ../maxwell/public/1585.jpg

# 2. Visualize all results
python visualize_npy_results.py --search-dir . --output-dir all_viz

# 3. Validate with multiple images
python validate_life_sciences_multi_modal.py --max-images 5

# 4. Visualize validation results
python visualize_npy_results.py --search-dir multi_modal_validation --detailed
```

### Batch Processing

```bash
# Process multiple images
for img in ../maxwell/public/*.jpg; do
    python demo_virtual_imaging.py "$img" --output-dir "results/$(basename $img .jpg)"
done

# Visualize all results
python visualize_npy_results.py --search-dir results --output-dir batch_viz
```

## Help

For more information:

```bash
# Script-specific help
python demo_virtual_imaging.py --help
python visualize_npy_results.py --help
python validate_life_sciences_multi_modal.py --help

# Or check the README
cat README.md
```

## Support

If you encounter issues:
1. Check this guide's Troubleshooting section
2. Verify dependencies: `pip list | grep -E "(numpy|matplotlib|opencv)"`
3. Check Python version: `python --version` (should be ≥ 3.8)
4. Ensure you're in the correct directory: `pwd` should end with `pixel_maxwell_demon`

