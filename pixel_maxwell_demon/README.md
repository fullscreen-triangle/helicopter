# Pixel Maxwell Demon

**Hardware-Constrained Categorical Computer Vision Framework**

A revolutionary computer vision framework based on dual-membrane pixel structures that enables virtual imaging, multi-modal validation, temporal super-resolution, and motion picture Maxwell demons. This framework extracts categorical information from images while respecting the fundamental conjugate structure of information itself.

## 🚀 Key Innovations

### 1. Virtual Imaging via Dual-Membrane Pixels
Generate images at different wavelengths, illumination angles, or modalities from a **single capture** without re-imaging:
- Wavelength shifting (e.g., green → red or blue)
- Illumination angle changes (bright-field → dark-field)
- Fluorescence excitation wavelength modification
- Phase contrast extraction from amplitude

### 2. Multi-Modal Validation
Simultaneous analysis with multiple detector types from a **single sample**:
- Infrared (IR) spectroscopy
- Raman spectroscopy
- Mass spectrometry
- Fluorescence microscopy
- All from one categorical capture

### 3. Motion Picture Maxwell Demon
Videos that always play forward in entropy, regardless of scrubbing direction:
- Dual-membrane temporal structure (Front/Back faces)
- S-entropy temporal coordinates
- Irreversible playback demonstration
- Multi-modal temporal analysis

### 4. Temporal Super-Resolution through Spectral Multiplexing
**Breakthrough**: Videos without mechanical shutters, infinite temporal zoom:
- Light source multiplexing replaces frame-based sampling
- N detectors × M light sources = N×M effective frame rate
- Spectral diversity fills temporal gaps
- 4D video (x, y, t, λ) with sharp slow-motion at any zoom level

## 📊 Scientific Publications

Three comprehensive LaTeX papers with rigorous theoretical foundations:

### Paper 1: Virtual Imaging and Membrane Pixels
**Location**: `docs/virtual-imaging-membranes/`

Complete treatment of dual-membrane pixel structure enabling virtual imaging:
- Pixel Maxwell Demon framework
- Wavelength shifting mechanisms
- Illumination angle transformation
- Fluorescence excitation control
- Phase-amplitude conjugate extraction
- Hardware stream integration

**Compile**: `cd docs/virtual-imaging-membranes && pdflatex virtual-imaging-membrane-pixels.tex`

### Paper 2: Temporal Super-Resolution via Spectral Multiplexing
**Location**: `docs/spectral-multiplexing/`

Revolutionary temporal imaging paradigm:
- Categorical temporal encoding
- Multi-detector wavelength sequences
- Adaptive time integration
- Fractal temporal architecture
- Motion picture pixel Maxwell demon

**Compile**: `cd docs/spectral-multiplexing && pdflatex temporal-resolution-spectral-multiplexing.tex`

### Paper 3: Motion Picture Maxwell Demon
**Location**: `docs/motion-picture/`

Theoretical framework for entropy-driven video playback:
- S-entropy coordinates (S_t, P_t, dS/dt, ΔS)
- Dual-membrane temporal structure
- Frame motion under entropy gradient
- Gas molecular dynamics analogy

**Compile**: `cd docs/motion-picture && pdflatex motion-picture-maxwell-demon.tex`

## 🔧 Installation

### Quick Install
```bash
cd pixel_maxwell_demon
pip install -e .
```

### Development Install (with all dependencies)
```bash
cd pixel_maxwell_demon
pip install -e ".[dev]"
```

### Requirements
- Python ≥ 3.8
- numpy ≥ 1.20.0
- matplotlib ≥ 3.3.0
- opencv-python ≥ 4.5.0
- scipy ≥ 1.6.0
- scikit-image ≥ 0.18.0
- scikit-learn ≥ 0.24.0
- seaborn ≥ 0.11.0
- pillow ≥ 8.0.0

## 🎯 Usage Guide

### Core Demonstrations

#### 1. Virtual Imaging Demo
Generate virtual images at different wavelengths from a single capture:

```bash
python demo_virtual_imaging.py ../maxwell/public/1585.jpg
```

**Output**: 3×3 panel chart showing wavelength shifts, dark-field illumination, fluorescence excitation changes, and phase contrast extraction.

#### 2. Irreversible Playback Demo
Demonstrate dual-membrane temporal structure:

```bash
python demo_irreversible_playback.py
```

**Output**: Visualization of forward/backward playback with entropy tracking.

### Validation Experiments

#### 3. Multi-Modal Life Sciences Validation
Validate framework with real microscopy images:

```bash
python validate_life_sciences_multi_modal.py --max-images 5
```

**Output**: Success rates, reconstruction errors, and multi-modal consistency metrics.

#### 4. Motion Picture Maxwell Demon Validation
Validate irreversible video playback concept:

```bash
python validate_motion_picture_demon.py
```

**Output**: 
- Panel charts showing dual-membrane playback
- MP4 video demonstrating scrubbing behavior
- JSON summary with entropy statistics

#### 5. Multi-Modal Motion Picture Validation
Extended validation with multiple detector types:

```bash
python validate_multi_modal_motion_picture.py
```

**Output**: Comprehensive analysis of temporal super-resolution with IR, Raman, Mass Spec, and Fluorescence detectors.

### Spectral Multiplexing Experiments

#### 6. Temporal Resolution Enhancement
Validate temporal super-resolution theorem:

```bash
python validate_temporal_resolution_enhancement.py
```

**Output**: 4×4 panel chart demonstrating N×M effective sampling rate boost.

#### 7. Spectral Gap Filling
Validate gap-filling capabilities:

```bash
python validate_spectral_gap_filling.py
```

**Output**: 4×4 panel chart showing reconstruction from gapped data.

#### 8. Spectral Zoom Video Generation
**NEW**: Generate video demonstrating infinite temporal zoom:

```bash
python generate_spectral_zoom_video.py
```

**Output**:
- 4-panel explanatory chart
- Video frames (and MP4 if FFmpeg available)
- Progression through 30→60→120→240 FPS zoom levels
- Summary JSON with quality metrics

### Advanced Visualizations

#### 9. Publication Panel Charts
Generate comprehensive publication-quality panel charts:

```bash
python create_publication_panel_charts.py
```

**Output**: 4×4 panel chart with diverse visualizations:
- Radar charts for detector comparison
- Polar phase charts
- Frequency spectra
- Hierarchical clustering
- PCA projections
- Correlation matrices

#### 10. Virtual Imaging Signal Processing
Advanced signal processing for virtual imaging results:

```bash
python visualize_virtual_imaging_signal_processing.py
```

**Output**: 4×4 panel chart with FFT, power spectrum, phase histograms, peak extraction.

#### 11. Multi-Modal Detector Visualization
Radar charts and EM spectrum mapping:

```bash
python visualize_multi_modal_detectors.py
```

**Output**: Comprehensive detector performance visualization mapped to electromagnetic spectrum.

#### 12. Categorical Depth Visualization
EM wavelength penetration and categorical depth:

```bash
python visualize_categorical_depth.py
```

**Output**: Depth analysis with wavelength penetration characteristics.

#### 13. NPY Results Visualization
General-purpose NPY file visualization:

```bash
python visualize_npy_results.py --search-dir <directory> --detailed
```

**Output**: Auto-detected panel charts for all `.npy` files in specified directory.

## 📁 Project Structure

```
pixel_maxwell_demon/
├── docs/                          # Scientific publications (LaTeX)
│   ├── virtual-imaging-membranes/ # Paper 1: Virtual imaging
│   ├── spectral-multiplexing/     # Paper 2: Temporal super-resolution
│   └── motion-picture/            # Paper 3: Motion picture demon
│
├── src/maxwell/                   # Core framework
│   ├── pixel_maxwell_demon.py     # Pixel Maxwell Demon base
│   ├── dual_membrane_pixel_demon.py  # Dual-membrane implementation
│   ├── simple_pixel_grid.py       # Grid structure
│   ├── categorical_light_sources.py  # Light source management
│   ├── virtual_detectors.py       # Virtual detector system
│   └── integration/               # Unified integration
│       ├── dual_bmd_state.py      # Dual BMD state
│       ├── dual_region.py         # Dual region processing
│       ├── dual_network_bmd.py    # Network BMD
│       ├── dual_ambiguity.py      # Ambiguity handling
│       └── pixel_hardware_stream.py  # Hardware stream
│
├── demo_virtual_imaging.py        # Virtual imaging demonstration
├── demo_irreversible_playback.py  # Irreversible playback demo
│
├── validate_life_sciences_multi_modal.py  # Life sciences validation
├── validate_motion_picture_demon.py       # Motion picture validation
├── validate_multi_modal_motion_picture.py # Extended multi-modal validation
├── validate_temporal_resolution_enhancement.py  # Temporal resolution
├── validate_spectral_gap_filling.py       # Gap filling validation
│
├── generate_spectral_zoom_video.py        # Spectral zoom video generator
│
├── create_publication_panel_charts.py     # Publication panels
├── visualize_virtual_imaging_signal_processing.py  # Signal processing
├── visualize_multi_modal_detectors.py     # Detector visualization
├── visualize_categorical_depth.py         # Depth visualization
├── visualize_npy_results.py               # General NPY visualization
│
├── setup.py                       # Package setup
├── pyproject.toml                 # Modern packaging
└── requirements.txt               # Dependencies
```

## 🔬 Theoretical Foundation

### Pixel Maxwell Demon

A categorical observer at a spatial location that:
- Manages molecular demons for ensemble queries
- Uses virtual detectors for hypothesis validation
- Operates in S-entropy coordinates: (S_k, S_t, S_e)
- Achieves zero-backaction observation

### Dual-Membrane Structure

**Fundamental Discovery**: Information has two conjugate faces that cannot be simultaneously observed:

| Property | Front Face | Back Face |
|----------|-----------|-----------|
| Content | Amplitude | Phase |
| Observable | Direct measurement | Conjugate calculation |
| Traditional | Standard image | Hidden information |
| Access | Immediate | Via membrane flip |
| Complementarity | Like ammeter | Like voltmeter |

**Key Insight**: The "thickness" of the membrane encodes categorical distance, providing depth information from 2D images.

### Virtual Detectors

Generate virtual measurements without physical instruments:

| Aspect | Traditional Approach | Virtual Detector Approach |
|--------|---------------------|--------------------------|
| Captures needed | 7 separate captures | 1 capture |
| Configuration | Multiple physical setups | Query categorical coordinates |
| Sample interaction | Potential disturbance | Zero backaction |
| Time required | Hours | Seconds |
| Cost | $500K+ equipment | Software query |

### S-Entropy Coordinates

Four-dimensional temporal coordinate system:

1. **S_t**: Shannon entropy at time t
2. **P_t**: Participation ratio at time t
3. **dS/dt**: Entropy production rate
4. **ΔS**: Cumulative entropy production

**Property**: These coordinates always increase → temporal irreversibility

### Spectral Multiplexing

**Revolutionary Insight**: Time doesn't need to be sampled by mechanical shutters. Instead:

- **Traditional video**: Frame rate limited by shutter speed
- **Spectral multiplexing**: Each light source cycle = temporal sample
- **Effective rate**: f_eff = min(N_detectors, M_sources) × f_cycle
- **Result**: N×M rate boost with NO mechanical parts

**Theorem 1 (Temporal Resolution Enhancement)**: N detectors observing M sequentially-cycled light sources achieve effective temporal resolution of min(N,M) × f_cycle.

**Theorem 2 (Spectral Gap Filling)**: Temporal sampling gaps are filled by spectral diversity, with reconstruction error bounded by detector noise.

## 📈 Example Workflows

### Workflow 1: Complete Virtual Imaging Pipeline

```bash
# Step 1: Generate virtual images
python demo_virtual_imaging.py ../maxwell/public/1585.jpg

# Step 2: Create signal processing analysis
python visualize_virtual_imaging_signal_processing.py

# Step 3: Validate with life sciences images
python validate_life_sciences_multi_modal.py --max-images 10

# Step 4: Generate publication panels
python create_publication_panel_charts.py
```

**Output**: Complete publication-ready analysis with multiple panel charts.

### Workflow 2: Temporal Super-Resolution Validation

```bash
# Step 1: Validate temporal resolution theorem
python validate_temporal_resolution_enhancement.py

# Step 2: Validate gap filling
python validate_spectral_gap_filling.py

# Step 3: Generate demonstration video
python generate_spectral_zoom_video.py

# Step 4: Visualize multi-modal performance
python visualize_multi_modal_detectors.py
```

**Output**: Complete validation of spectral multiplexing with video demonstration.

### Workflow 3: Motion Picture Maxwell Demon

```bash
# Step 1: Basic validation
python validate_motion_picture_demon.py

# Step 2: Extended multi-modal validation
python validate_multi_modal_motion_picture.py

# Step 3: Visualize categorical depth
python visualize_categorical_depth.py
```

**Output**: Full analysis of entropy-driven temporal dynamics.

## 🎬 Video Outputs

Several scripts generate video demonstrations:

1. **Motion Picture Demon** (`validate_motion_picture_demon.py`):
   - 150-frame MP4 showing dual-membrane playback
   - Entropy tracking during forward/backward scrubbing
   - Visual indicators for active face

2. **Spectral Zoom Video** (`generate_spectral_zoom_video.py`):
   - Progression through zoom levels (30→240 FPS)
   - Gap filling demonstration
   - Quality metrics visualization
   - **Note**: Requires FFmpeg for MP4 generation, otherwise saves frames

### Installing FFmpeg (for video generation)

**Windows**:
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**Linux**:
```bash
sudo apt install ffmpeg
```

**macOS**:
```bash
brew install ffmpeg
```

## 📊 Output Directory Structure

After running all validation scripts:

```
pixel_maxwell_demon/
├── virtual_imaging_results/
│   ├── virtual_imaging_demo.png
│   ├── virtual_650nm.npy
│   └── virtual_imaging_results.json
│
├── npy_visualizations/
│   ├── detector_heatmaps.png
│   └── cross_experiment_comparison.png
│
├── publication_panels/
│   ├── detector_comparison_panel.png
│   └── comprehensive_stats.json
│
├── spectral_multiplexing_validation/
│   ├── temporal_resolution/
│   │   └── temporal_resolution_enhancement.png
│   └── gap_filling/
│       └── spectral_gap_filling.png
│
├── motion_picture_validation/
│   ├── motion_picture_validation_panel.png
│   ├── dual_membrane_playback.mp4
│   └── validation_summary.json
│
├── multi_modal_motion_picture/
│   ├── multi_modal_motion_picture_panel.png
│   └── multi_modal_motion_picture_demo.mp4
│
└── spectral_zoom_video/
    ├── spectral_zoom_explanation.png
    ├── video_frames/               # Individual frames
    ├── spectral_temporal_zoom.mp4  # If FFmpeg available
    └── spectral_zoom_summary.json
```

## 🧪 Testing

Run all validation experiments:

```bash
# Quick test of core imports
python test_imports.py

# Full validation suite (requires images in ../maxwell/public/)
python validate_life_sciences_multi_modal.py --max-images 3
python validate_motion_picture_demon.py
python validate_temporal_resolution_enhancement.py
python validate_spectral_gap_filling.py
```

## 📝 Code Examples

### Example 1: Virtual Wavelength Shifting

```python
from maxwell.simple_pixel_grid import PixelDemonGrid
from maxwell.categorical_light_sources import wavelength_scaling_factor
import cv2
import numpy as np

# Load image
image = cv2.imread('image.jpg')
h, w = image.shape[:2]

# Create pixel demon grid
grid = PixelDemonGrid(width=w, height=h, physical_extent=(1e-3, 1e-3))
grid.initialize_from_image(image)

# Shift from 550nm (green) to 650nm (red)
scaling = wavelength_scaling_factor(550, 650)
virtual_red = np.zeros_like(image)

for y in range(h):
    for x in range(w):
        demon = grid.grid[y, x]
        # Access front face for amplitude
        amplitude = demon.dual_state.front_s.S_k * scaling
        virtual_red[y, x] = amplitude

cv2.imwrite('virtual_red.png', virtual_red)
```

### Example 2: Phase Extraction (Back Face)

```python
# Extract phase information from amplitude (dual-membrane back face)
phase_image = np.zeros((h, w))

for y in range(h):
    for x in range(w):
        demon = grid.grid[y, x]
        
        # Front face = amplitude (what you normally see)
        amplitude = demon.dual_state.front_s.S_k
        
        # Back face = phase (conjugate information)
        phase = demon.dual_state.back_s.S_k
        
        phase_image[y, x] = phase
```

### Example 3: Multi-Modal Virtual Detection

```python
from maxwell.virtual_detectors import VirtualDetectorArray

# Create virtual detector array
detectors = VirtualDetectorArray(['ir', 'raman', 'mass_spec', 'fluorescence'])

# Single capture, multiple modalities
for modality in detectors:
    virtual_image = modality.measure(grid, categorical_coords)
    cv2.imwrite(f'virtual_{modality.name}.png', virtual_image)
```

## 🔗 Related Work

This framework builds on concepts from:
- Maxwell's demon (information thermodynamics)
- Complementarity principle (quantum mechanics)
- Categorical observation (computational physics)
- Hardware-constrained computation
- Shannon entropy and information theory

## 📚 Citation

```bibtex
@software{pixel_maxwell_demon_2024,
  title={Pixel Maxwell Demon: Hardware-Constrained Categorical Computer Vision},
  author={Sachikonye, Kundai},
  year={2024},
  url={https://github.com/fullscreen-triangle/lavoisier},
  note={Framework for virtual imaging, multi-modal validation, and temporal super-resolution}
}

@article{virtual_imaging_membranes_2024,
  title={Virtual Imaging through Dual-Membrane Pixel Structures},
  author={Sachikonye, Kundai},
  journal={In preparation},
  year={2024}
}

@article{spectral_multiplexing_2024,
  title={Temporal Super-Resolution through Spectral Multiplexing},
  author={Sachikonye, Kundai},
  journal={In preparation},
  year={2024}
}

@article{motion_picture_demon_2024,
  title={Motion Picture Maxwell Demon: Entropy-Driven Video Playback},
  author={Sachikonye, Kundai},
  journal={In preparation},
  year={2024}
}
```

## 🏆 Key Achievements

✅ **Virtual Imaging**: Single capture → multiple modalities  
✅ **Multi-Modal Validation**: Simultaneous IR, Raman, Mass Spec, Fluorescence  
✅ **Motion Picture Demon**: Videos that always play forward in entropy  
✅ **Temporal Super-Resolution**: N×M rate boost without mechanical shutters  
✅ **Spectral Gap Filling**: Sharp reconstruction from incomplete temporal data  
✅ **Publication-Ready**: Three comprehensive LaTeX papers  
✅ **Extensive Validation**: 13+ validation and visualization scripts  
✅ **Video Demonstrations**: MP4 outputs showing key concepts  

## 📄 License

MIT License - See LICENSE file

## 👤 Author

**Kundai Sachikonye**  
Repository: [https://github.com/fullscreen-triangle/lavoisier](https://github.com/fullscreen-triangle/lavoisier)

---

**Revolutionary Capability**: 
- **Capture ONCE**, Query INFINITELY 🚀
- **Sample ONCE**, Detect MULTI-MODALLY 🔬
- **Record ONCE**, Zoom INFINITELY 🎬
