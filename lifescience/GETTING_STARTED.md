# 🚁 Helicopter Life Science Framework - Getting Started

Welcome to the Helicopter Life Science Framework! This guide will help you set up and run analysis on your microscopy data in just a few simple steps.

## 📦 Quick Setup

### 1. Install Dependencies

```bash
cd lifescience
python setup.py
```

This will:

- Check your Python version (3.7+ required)
- Install missing packages automatically
- Test module imports
- Set up the results directory

### 2. Add Your Data

Place your microscopy files in the `lifescience/public/` directory:

- **Images**: `.jpg`, `.png`, `.tif`, `.bmp`
- **Videos**: `.mp4`, `.avi`, `.mov`, `.mpg`
- **Archives**: `.zip`, `.tar`, `.gz`

### 3. Configure Analysis

Edit `config.py` to point to your data:

```python
MICROSCOPY_IMAGES = {
    'my_cells': DATA_DIR / "my_cell_image.jpg",
    'tissue_sample': DATA_DIR / "tissue.tif",
    # Add more images...
}

MICROSCOPY_VIDEOS = {
    'live_cells': DATA_DIR / "time_lapse.mp4",
    # Add more videos...
}
```

## 🧪 Testing Your Setup

Run a quick test to verify everything works:

```bash
python demo_quick_test.py
```

This will:

- Test all module imports
- Load your data files
- Run basic analysis
- Report any issues

## 🚀 Running Analysis

### Complete Analysis (All Modules)

```bash
python demo_all_modules.py
```

Analyzes your data with all 6 framework modules:

- Gas Molecular Dynamics
- S-Entropy Coordinate Transformation
- Fluorescence Microscopy Analysis
- Electron Microscopy Analysis
- Video Processing & Cell Tracking
- Meta-Information Extraction

### Focused Analysis (Single Module)

```bash
python demo_fluorescence.py    # Fluorescence microscopy only
python demo_video.py          # Video analysis only
```

## 📊 Understanding Results

Results are saved in `lifescience/results/`:

- **Visualizations**: PNG files with analysis plots
- **Console Output**: Detailed numerical results

### Gas Molecular Analysis

- Protein folding quality assessment
- Binding site identification
- Molecular dynamics simulation

### S-Entropy Analysis

- 4D semantic coordinates: [Structural, Functional, Morphological, Temporal]
- Confidence metrics
- Biological context mapping

### Fluorescence Analysis

- Multi-channel colocalization
- Signal-to-noise ratios
- Structure quantification

### Electron Microscopy Analysis

- Ultrastructure identification
- Organelle detection
- High-resolution analysis

### Video Analysis

- Cell tracking and migration
- Behavioral classification
- Temporal dynamics

### Meta-Information Analysis

- Information content assessment
- Compression potential analysis
- Semantic density metrics

## ⚙️ Configuration Options

### Analysis Parameters

Adjust analysis behavior in `config.py`:

```python
ANALYSIS_PARAMS = {
    'gas_molecular': {
        'evolution_steps': 1000,
        'structure_type': 'folded'
    },
    'entropy': {
        'biological_context': 'cellular'  # cellular, tissue, molecular
    },
    'fluorescence': {
        'channel': 'gfp',  # dapi, gfp, rfp, fitc
        'background_subtraction': True
    },
    'electron_microscopy': {
        'em_type': 'tem',  # sem, tem, cryo_em
        'target_structures': ['mitochondria', 'vesicles']
    },
    'video': {
        'video_type': 'live_cell',  # live_cell, time_lapse, calcium_imaging
        'frame_interval': 1.0
    }
}
```

### Visualization Settings

```python
SAVE_FIGURES = True    # Save plots to files
SHOW_FIGURES = False   # Display plots interactively
```

## 🔬 Data Types Supported

| Data Type                   | File Formats  | Analysis Modules              |
| --------------------------- | ------------- | ----------------------------- |
| **Fluorescence Microscopy** | JPG, PNG, TIF | Fluorescence, S-Entropy, Meta |
| **Electron Microscopy**     | TIF, PNG      | Electron, Gas Molecular, Meta |
| **Live Cell Videos**        | MP4, AVI, MOV | Video, S-Entropy              |
| **Time-lapse**              | MP4, MPG      | Video, Gas Molecular          |
| **Multi-modal**             | All formats   | All modules                   |

## 📈 Example Workflow

1. **Setup**: `python setup.py`
2. **Test**: `python demo_quick_test.py`
3. **Configure**: Edit paths in `config.py`
4. **Analyze**: `python demo_all_modules.py`
5. **Review**: Check `results/` directory
6. **Refine**: Adjust parameters and re-run

## 🎯 Best Practices

### For Fluorescence Analysis

- Use appropriate channel settings (DAPI, GFP, RFP, FITC)
- Enable background subtraction for better SNR
- Multi-channel analysis reveals protein interactions

### For Video Analysis

- Match video_type to your experiment:
  - `live_cell`: General cell tracking
  - `time_lapse`: Developmental processes
  - `cell_migration`: Directional movement analysis
- Higher frame rates improve tracking accuracy

### For Electron Microscopy

- Specify target structures for focused analysis
- Use appropriate EM type (SEM, TEM, cryo-EM)
- High-resolution images yield better results

### For Optimization

- Start with quick_test to identify issues
- Process subset of data first
- Adjust parameters based on initial results

## 🛠️ Troubleshooting

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check module structure
python -c "import sys; sys.path.append('.'); from src.gas import BiologicalGasAnalyzer; print('✅ Success')"
```

### Data Loading Issues

- Check file paths in `config.py`
- Verify file formats are supported
- Ensure files aren't corrupted

### Analysis Failures

- Check console output for specific errors
- Try with smaller dataset first
- Verify analysis parameters are appropriate

### Memory Issues

- Reduce image resolution
- Process fewer frames for videos
- Use `max_frames` parameter in video analysis

## 📚 Advanced Usage

### Custom Analysis Pipeline

```python
# Create your own analysis script
from src.gas import BiologicalGasAnalyzer
from src.entropy import SEntropyTransformer

analyzer = BiologicalGasAnalyzer()
transformer = SEntropyTransformer()

# Your custom analysis logic here...
```

### Batch Processing

```python
# Process multiple files automatically
from pathlib import Path
from config import ensure_output_dir

for image_file in Path("public").glob("*.jpg"):
    # Run analysis on each file
    pass
```

## 🤝 Support

If you encounter issues:

1. Check this guide first
2. Run `demo_quick_test.py` to diagnose problems
3. Verify your data files and configuration
4. Check the console output for specific error messages

---

**🎉 You're ready to analyze your microscopy data with the Helicopter Life Science Framework!**

Start with `python demo_quick_test.py` and then move to `python demo_all_modules.py` for complete analysis.
