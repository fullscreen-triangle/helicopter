# How to Run Analysis Scripts

## Prerequisites

Make sure you've installed the package:

```bash
cd pixel_maxwell_demon
python install_and_test.py
```

## Script 1: Virtual Imaging Demo

### Purpose
Generate virtual images at different wavelengths/modalities from a single capture.

### Usage

```bash
python demo_virtual_imaging.py IMAGE_PATH [--output-dir DIR]
```

### Example

```bash
python demo_virtual_imaging.py ../maxwell/public/1585.jpg
```

### What It Does

1. Loads the input image
2. Creates a pixel demon grid
3. Generates 5 virtual images:
   - 650nm (red wavelength)
   - 450nm (blue wavelength)
   - Dark-field illumination
   - Fluorescence at 561nm excitation
   - Phase contrast from amplitude

### Output

```
virtual_imaging_results/
├── virtual_imaging_demo.png           # 3×3 panel visualization
├── virtual_650nm.npy                  # Red-shifted image
├── virtual_450nm.npy                  # Blue-shifted image
├── virtual_darkfield.npy              # Dark-field image
├── virtual_fluorescence_561nm.npy     # Virtual fluorescence
├── virtual_phase_contrast.npy         # Phase from amplitude
└── virtual_imaging_results.json       # Metadata
```

### Time
~30-60 seconds depending on image size

---

## Script 2: Visualize NPY Results

### Purpose
Create panel chart visualizations from all `.npy` files in a directory tree.

### Usage

```bash
python visualize_npy_results.py [OPTIONS]
```

### Options

- `--search-dir DIR`: Directory to search (default: current directory)
- `--output-dir DIR`: Output directory (default: `npy_visualizations`)
- `--detailed`: Create detailed individual views for each array

### Examples

```bash
# Visualize all NPY files in maxwell directory
python visualize_npy_results.py --search-dir ../maxwell

# Visualize local results
python visualize_npy_results.py --search-dir .

# Create detailed views
python visualize_npy_results.py --search-dir multi_modal_validation --detailed
```

### What It Does

1. Recursively finds all `.npy` files
2. Groups by experiment directory
3. Creates panel charts with auto-selected colormaps
4. Generates metadata JSON files
5. Creates cross-experiment comparison

### Output

```
npy_visualizations/
├── experiment1_panel_chart.png
├── experiment1_metadata.json
├── experiment2_panel_chart.png
├── experiment2_metadata.json
├── cross_experiment_comparison.png
└── experiment1_detailed/      # If --detailed flag used
    ├── array1.png
    ├── array2.png
    └── ...
```

### Time
~10-30 seconds depending on number of NPY files

---

## Script 3: Life Sciences Validation

### Purpose
Validate the framework with real microscopy images using multiple virtual detectors.

### Usage

```bash
python validate_life_sciences_multi_modal.py [OPTIONS]
```

### Options

- `--max-images N`: Maximum number of images to process (default: all)
- `--image-dir DIR`: Image directory (default: `../maxwell/public`)
- `--output-dir DIR`: Output directory (default: `multi_modal_validation`)

### Examples

```bash
# Process all images
python validate_life_sciences_multi_modal.py

# Process only 3 images (for testing)
python validate_life_sciences_multi_modal.py --max-images 3

# Process images from custom directory
python validate_life_sciences_multi_modal.py --image-dir /path/to/images
```

### What It Does

1. Loads life sciences images
2. Creates pixel demon grid for each
3. Applies 8 virtual detectors:
   - Photodiode (amplitude)
   - IR Spectrometer
   - Raman Spectrometer
   - Mass Spectrometer
   - Thermometer
   - Barometer
   - Hygrometer
   - Interferometer (phase)
4. Generates multi-modal comparison charts
5. Computes aggregate statistics

### Output

```
multi_modal_validation/
├── image1/
│   ├── Photodiode_map.npy
│   ├── IR_Spectrometer_map.npy
│   ├── Raman_Spectrometer_map.npy
│   ├── Mass_Spectrometer_map.npy
│   ├── Thermometer_map.npy
│   ├── Barometer_map.npy
│   ├── Hygrometer_map.npy
│   ├── Interferometer_map.npy
│   ├── multi_modal_comparison.png
│   └── multi_modal_results.json
├── image2/
│   └── ...
└── complete_multi_modal_results.json
```

### Time
~1-2 minutes per image (depends on resolution)

---

## Complete Analysis Pipeline

Run all three scripts in sequence:

```bash
#!/bin/bash

# Step 1: Generate virtual images
echo "Step 1: Virtual Imaging Demo..."
python demo_virtual_imaging.py ../maxwell/public/1585.jpg

# Step 2: Visualize virtual imaging results
echo "Step 2: Visualize Virtual Imaging Results..."
python visualize_npy_results.py --search-dir virtual_imaging_results

# Step 3: Run life sciences validation
echo "Step 3: Life Sciences Validation..."
python validate_life_sciences_multi_modal.py --max-images 3

# Step 4: Visualize validation results with detail
echo "Step 4: Visualize Validation Results..."
python visualize_npy_results.py --search-dir multi_modal_validation --detailed --output-dir validation_viz

echo "Complete! Check the output directories for results."
```

Save this as `run_full_analysis.sh` (Linux/Mac) or `run_full_analysis.bat` (Windows).

---

## Tips

### Speed Up Processing

For quick testing, use smaller images or limit the number:

```bash
# Test with 1 small image
python validate_life_sciences_multi_modal.py --max-images 1

# Process only part of an image (edit script to add downsampling)
```

### View Results

```bash
# On Linux/Mac
open virtual_imaging_results/virtual_imaging_demo.png
open npy_visualizations/cross_experiment_comparison.png

# On Windows
start virtual_imaging_results\virtual_imaging_demo.png
start npy_visualizations\cross_experiment_comparison.png
```

### Batch Processing

Process multiple images:

```bash
for img in ../maxwell/public/*.jpg; do
    echo "Processing: $img"
    python demo_virtual_imaging.py "$img" --output-dir "results/$(basename $img .jpg)"
done

# Then visualize all at once
python visualize_npy_results.py --search-dir results --output-dir batch_viz
```

---

## Troubleshooting

### Script Won't Run

```bash
# Check you're in the right directory
pwd  # Should end with pixel_maxwell_demon

# Check Python version
python --version  # Should be ≥ 3.8

# Reinstall if needed
python install_and_test.py
```

### Import Errors

```bash
# Make sure package is installed
pip list | grep pixel-maxwell-demon

# Reinstall
pip install -e . --force-reinstall
```

### Out of Memory

```bash
# Process fewer images
python validate_life_sciences_multi_modal.py --max-images 1

# Or reduce image size (edit scripts to add downsampling)
```

### No Output

Check that output directories have write permissions:

```bash
ls -la  # Check permissions
mkdir -p results  # Create manually if needed
```

---

## What's Next?

After running the analysis:

1. **Examine visualizations**: Open the PNG files
2. **Check metadata**: Read the JSON files for statistics
3. **Compare results**: Use the cross-experiment comparison charts
4. **Process your data**: Replace image paths with your own images
5. **Publish results**: Use the visualizations in your papers/presentations

