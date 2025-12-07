# Scripts Fixed - Ready to Run

## Problems Fixed

### 1. `demo_virtual_imaging.py`
**Problem:** Used wrong constructor arguments `shape=(h, w)` 
**Fix:** Changed to `width=w, height=h`

### 2. `validate_life_sciences_multi_modal.py`  
**Problems:**
- Default directory was `'public'` instead of `'../maxwell/public'`
- Crashed with KeyError when no images found
- No helpful error messages

**Fixes:**
- Changed default to `'../maxwell/public'`
- Added fallback directory search
- Added safe `aggregate.get('success_rate', 0)` access
- Added helpful error messages when no images found

### 3. `visualize_npy_results.py`
**Problem:** No helpful message when directory doesn't exist or is empty

**Fix:** Added directory existence check and helpful tips

## How to Use Fixed Scripts

### Quick Test

Run all three scripts in sequence:

```bash
cd pixel_maxwell_demon
python test_all_scripts.py
```

This will:
1. Find a test image
2. Run virtual imaging demo
3. Run life sciences validation (1 image)
4. Visualize the results
5. Report pass/fail for each

### Individual Scripts

#### 1. Virtual Imaging Demo

```bash
python demo_virtual_imaging.py ../maxwell/public/1585.jpg
```

**Expected output:**
- `virtual_imaging_results/` directory
- 5 NPY files (650nm, 450nm, darkfield, fluorescence, phase)
- 1 visualization PNG
- 1 JSON metadata file

**Time:** ~30-60 seconds

#### 2. Life Sciences Validation

```bash
# Process 3 images
python validate_life_sciences_multi_modal.py --max-images 3

# Process all images
python validate_life_sciences_multi_modal.py

# Custom image directory
python validate_life_sciences_multi_modal.py --public-dir /path/to/images
```

**Expected output:**
- `multi_modal_validation/` directory
- Subdirectory for each image
- 8 NPY files per image (one per detector)
- Comparison PNG per image
- JSON metadata

**Time:** ~1-2 minutes per image

#### 3. NPY Visualization

```bash
# Visualize virtual imaging results
python visualize_npy_results.py --search-dir virtual_imaging_results

# Visualize validation results with detail
python visualize_npy_results.py --search-dir multi_modal_validation --detailed

# Visualize everything
python visualize_npy_results.py --search-dir . --output-dir all_viz
```

**Expected output:**
- `npy_visualizations/` directory (or custom via --output-dir)
- Panel chart PNG for each experiment
- Metadata JSON for each experiment
- Cross-experiment comparison PNG
- Detailed views if --detailed flag used

**Time:** ~10-30 seconds

## Complete Analysis Workflow

```bash
#!/bin/bash
# save as: run_complete_analysis.sh

cd pixel_maxwell_demon

echo "Step 1: Virtual Imaging Demo..."
python demo_virtual_imaging.py ../maxwell/public/1585.jpg

echo -e "\nStep 2: Life Sciences Validation (3 images)..."
python validate_life_sciences_multi_modal.py --max-images 3

echo -e "\nStep 3: Visualize All Results..."
python visualize_npy_results.py --search-dir . --detailed --output-dir complete_viz

echo -e "\nDone! Check these directories:"
echo "  - virtual_imaging_results/"
echo "  - multi_modal_validation/"
echo "  - complete_viz/"
```

Make executable and run:
```bash
chmod +x run_complete_analysis.sh
./run_complete_analysis.sh
```

## Verification Checklist

- [x] Package installed: `pip list | grep pixel-maxwell-demon`
- [x] Imports work: `python test_imports.py`
- [x] Demo runs: `python demo_virtual_imaging.py ../maxwell/public/1585.jpg`
- [x] Validation finds images: checks `../maxwell/public/` exists
- [x] Validation runs: `python validate_life_sciences_multi_modal.py --max-images 1`
- [x] Visualizer works: `python visualize_npy_results.py --search-dir virtual_imaging_results`

## Expected File Structure After Running

```
pixel_maxwell_demon/
├── virtual_imaging_results/
│   ├── virtual_imaging_demo.png
│   ├── virtual_650nm.npy
│   ├── virtual_450nm.npy
│   ├── virtual_darkfield.npy
│   ├── virtual_fluorescence_561nm.npy
│   ├── virtual_phase_contrast.npy
│   └── virtual_imaging_results.json
│
├── multi_modal_validation/
│   ├── 1585/
│   │   ├── Photodiode_map.npy
│   │   ├── IR_Spectrometer_map.npy
│   │   ├── Raman_Spectrometer_map.npy
│   │   ├── Mass_Spectrometer_map.npy
│   │   ├── Thermometer_map.npy
│   │   ├── Barometer_map.npy
│   │   ├── Hygrometer_map.npy
│   │   ├── Interferometer_map.npy
│   │   ├── multi_modal_comparison.png
│   │   └── multi_modal_results.json
│   └── complete_multi_modal_results.json
│
└── npy_visualizations/ (or complete_viz/)
    ├── virtual_imaging_results_panel_chart.png
    ├── virtual_imaging_results_metadata.json
    ├── 1585_panel_chart.png
    ├── 1585_metadata.json
    ├── cross_experiment_comparison.png
    └── virtual_imaging_results_detailed/ (if --detailed)
        ├── virtual_650nm.png
        ├── virtual_450nm.png
        └── ...
```

## Troubleshooting

### "Could not load image"
- Check file exists: `ls ../maxwell/public/1585.jpg`
- Try absolute path: `python demo_virtual_imaging.py /full/path/to/image.jpg`

### "No images found"
- Check directory: `ls ../maxwell/public/*.jpg`
- Specify path: `python validate_life_sciences_multi_modal.py --public-dir /path/to/images`

### "No NPY files found"
- Run demo/validation first to generate NPY files
- Check search directory: `ls -la virtual_imaging_results/`

### "Import errors"
- Reinstall package: `python install_and_test.py`
- Check installation: `pip list | grep pixel-maxwell-demon`

### "Out of memory"
- Process fewer/smaller images: `--max-images 1`
- Close other applications
- Downscale images before processing

## Next Steps

After successful runs:

1. **View Results:** Open PNG files to see visualizations
2. **Check Metadata:** Read JSON files for statistics
3. **Process Your Data:** Replace paths with your own images
4. **Batch Process:** Use loops to process multiple images
5. **Publish:** Use visualizations in papers/presentations

## Support

If scripts still don't work:

1. Run: `python test_all_scripts.py` for automated testing
2. Check: `test_imports.py` for import issues
3. Read: `INSTALL.md` for installation help
4. See: `RUN_ANALYSIS.md` for detailed usage

All scripts are now fixed and ready to use! 🚀

