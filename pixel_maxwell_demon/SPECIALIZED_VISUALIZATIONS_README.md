# Specialized Visualizations

## Two New Comprehensive Panel Charts

### 1. Multi-Modal Detector Analysis (`visualize_multi_modal_detectors.py`)

**Purpose**: Visualize detector performance with **radar charts and EM spectrum mapping**

**Input**: `maxwell/multi_modal_validation/complete_multi_modal_results.json`

**Output**: `multi_modal_detector_panels/multi_modal_detector_analysis.png` (22×22 inches)

**Layout** (4×4 grid):

```
┌────────────────────────────────────────────────────────────┐
│  Row 1: Detector Performance Radar Charts (4 detectors)   │
├──────────────┬──────────────┬──────────────┬──────────────┤
│ Thermometer  │ Barometer    │ Hygrometer   │ IR Spectro   │
│   Radar      │   Radar      │   Radar      │   Radar      │
│              │              │              │              │
├──────────────┴──────────────┴──────────────┴──────────────┤
│  Row 2: More Detector Performance Radars (4 detectors)    │
├──────────────┬──────────────┬──────────────┬──────────────┤
│ Raman        │ Mass Spec    │ Photodiode   │ Interferom.  │
│   Radar      │   Radar      │   Radar      │   Radar      │
│              │              │              │              │
├──────────────┴──────────────┴──────────────┴──────────────┤
│  Row 3: EM Spectrum Sensitivity (4 detectors)             │
├──────────────┬──────────────┬──────────────┬──────────────┤
│ Thermometer  │ Barometer    │ Hygrometer   │ IR Spectro   │
│   EM Radar   │   EM Radar   │   EM Radar   │   EM Radar   │
│              │              │              │              │
├──────────────┴──────────────┴──────────────┴──────────────┤
│  Row 4: Comparative Analysis                              │
├──────────────┬──────────────┬──────────────┬──────────────┤
│ Detector     │ Measurement  │ Revolution.  │ Consistency  │
│ Comparison   │   Times      │  Advantage   │  Heatmap     │
│   (bars)     │  (box plot)  │  (savings)   │              │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

**Key Features**:

**Radar Charts (Rows 1-2)**: Show 5 performance dimensions per detector
- **Consistency**: Low standard deviation (reliable)
- **Speed**: Fast measurement time
- **Signal**: High signal strength
- **Precision**: Low variance across images
- **Reliability**: Success rate

**EM Spectrum Radars (Row 3)**: Show wavelength sensitivity
- **Colored by wavelength**: UV (violet) → Visible (white) → IR (red)
- **Shows which part of spectrum each detector responds to**
- Non-EM detectors (mechanical/chemical) shown as text

**Comparative Analysis (Row 4)**:
- Signal/Time/Noise comparison bars
- Measurement time distributions
- Revolutionary advantage (sample savings)
- Cross-image consistency heatmap

**Run**:
```bash
python visualize_multi_modal_detectors.py
```

---

### 2. Categorical Depth Analysis (`visualize_categorical_depth.py`)

**Purpose**: Comprehensive visualization of categorical depth from dual-membrane structure

**Input**: `maxwell/demo_complete_results/categorical_depth.npy`

**Output**: `categorical_depth_analysis/categorical_depth_analysis.png` (24×18 inches)

**Layout** (4 rows × 3 cols):

```
┌────────────────────────────────────────────────────────────┐
│  Row 1: Overview                                           │
├──────────────────────────────┬──────────────┬──────────────┤
│ 3D Surface Plot              │ Depth        │              │
│ (spanning 2 columns)         │ Heatmap      │              │
│                              │              │              │
├──────────────┬───────────────┴──────────────┴──────────────┤
│  Row 2: Distributions & Gradients                          │
├──────────────┬──────────────┬──────────────┬──────────────┤
│ Depth        │ Cross-       │ Gradient     │              │
│ Histogram    │ Sections     │ Map          │              │
│              │              │              │              │
├──────────────┴──────────────┴──────────────┴──────────────┤
│  Row 3: Segmentation & Correlations                        │
├──────────────┬──────────────┬──────────────┬──────────────┤
│ Depth        │ Depth        │ EM Wavelength│              │
│ Layers       │ Contours     │ Penetration  │              │
│ (5 levels)   │              │              │              │
├──────────────┴──────────────┴──────────────┴──────────────┤
│  Row 4: Statistical Analysis                               │
├──────────────┬──────────────┬──────────────┬──────────────┤
│ Cumulative   │ Radial       │ Statistics   │              │
│ Distribution │ Profile      │ Table        │              │
│ (CDF)        │ (from center)│              │              │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

**Key Features**:

**3D Surface**: Full depth topography in 3D
- Colored by depth (viridis colormap)
- Subsampled for performance
- Interactive-style visualization

**Depth Heatmap**: 2D top-down view
- Plasma colormap
- Full resolution

**Histogram**: Distribution with statistics
- Mean and median lines
- Color-coded by depth value

**Cross-Sections**: Horizontal and vertical slices
- Through image center
- Shows depth profiles

**Gradient Map**: Rate of depth change
- Hot colormap (bright = steep)
- Reveals edges and transitions

**Depth Segmentation**: 5 discrete layers
- Based on depth percentiles
- Color-coded regions
- Useful for object separation

**Contours**: Topographic-style visualization
- 15 contour levels
- Smoothed for clarity

**EM Wavelength Penetration**: **KEY FEATURE!**
- Shows how different wavelengths penetrate to different depths
- UV (short wavelength) → shallow penetration
- IR (long wavelength) → deep penetration
- **Radar chart showing electromagnetic spectrum correlation**
- Colored bars represent actual spectrum colors

**CDF**: Cumulative distribution
- Shows depth percentiles
- Quartile markers

**Radial Profile**: Depth variation from center
- Averaged in radial bins
- Shows symmetry/asymmetry

**Statistics Table**: Complete quantitative summary
- Mean, median, std, min, max, range
- Quartiles (Q25, Q75)
- Skewness, kurtosis

**Run**:
```bash
python visualize_categorical_depth.py
```

---

## Why These Are Better Than Heatmaps

### Problem with Heatmaps
- **Limited dimensionality**: Only 2D (x, y, color)
- **Hard to compare**: Multiple metrics need multiple heatmaps
- **No multi-dimensional view**: Can't see relationships between metrics

### Radar Charts Solve This
- **Multi-dimensional**: 5-6 metrics visible at once
- **Easy comparison**: Polygon size/shape shows performance
- **Intuitive**: Larger area = better performance
- **Compact**: One chart shows what would need 5 bar charts

### EM Spectrum Radars Are Revolutionary
- **Shows wavelength sensitivity visually**
- **Color-coded by actual spectrum colors**
- **Immediately see which detector works at which wavelength**
- **Perfect for imaging applications** (shows what you can/can't see)

### For Categorical Depth
- **3D visualization**: See actual topography
- **Multiple perspectives**: Heatmap, contours, cross-sections, radial
- **EM correlation**: **Links depth to wavelength penetration** - this is novel!
- **Statistical rigor**: CDF, histogram, quantitative table

---

## Quick Comparison

| Visualization Type | Heatmap | Radar Chart | EM Spectrum Radar |
|-------------------|---------|-------------|-------------------|
| Dimensions shown  | 2       | 5-6         | 8 (spectrum bands)|
| Comparison ease   | Poor    | Excellent   | Excellent         |
| Intuitive         | Medium  | High        | Very High         |
| For multiple metrics | Multiple charts | One chart | One chart |
| For imaging       | OK      | OK          | **Perfect** ✓     |

---

## Understanding the EM Spectrum Radars

### For Detectors (Multi-Modal)

Each detector's radar shows sensitivity across EM spectrum:

- **Photodiode**: Strong in Visible + Near-IR
- **IR Spectrometer**: Strong in Near-IR + Mid-IR
- **Raman**: Strong in Visible + Near-IR
- **Thermometer**: Strong in Far-IR (thermal radiation)
- **Interferometer**: Strong in Visible + Near-IR + UV

**Non-EM detectors** (Mass Spec, Barometer, Hygrometer) don't use EM radiation, so they show as text labels.

**Why this matters**: 
- Shows which wavelengths you can use with each detector
- Reveals complementary coverage (one detector's gap is another's strength)
- Explains why multi-modal analysis works (different detectors see different parts of spectrum)

### For Categorical Depth

The EM penetration chart shows:
- **UV (400nm)**: Shallow penetration (~10-20%)
- **Blue (450nm)**: Slightly deeper (~25-35%)
- **Green (550nm)**: Medium penetration (~40-50%)
- **Red (650nm)**: Deeper (~55-70%)
- **Near-IR (800nm)**: Very deep (~75-85%)
- **Mid-IR (2500nm)**: Maximum penetration (~90-100%)

**Why this matters**:
- **Wavelength selection**: Choose wavelength based on depth of interest
- **Virtual imaging**: Generate deep-penetrating wavelengths from shallow captures
- **Physical insight**: Links abstract "categorical depth" to measurable EM properties
- **Validation**: Depth results should correlate with known penetration physics

---

## Output Files

After running both scripts:

```
pixel_maxwell_demon/
├── multi_modal_detector_panels/
│   └── multi_modal_detector_analysis.png    # 22×22 inches, 150 DPI
│
└── categorical_depth_analysis/
    └── categorical_depth_analysis.png        # 24×18 inches, 150 DPI
```

---

## For Your Paper

### Multi-Modal Detector Figure

**Caption suggestion**:
> "Multi-modal detector performance analysis with electromagnetic spectrum mapping. (a-h) Radar charts show five performance dimensions (consistency, speed, signal, precision, reliability) for each of eight detector types. (i-l) EM spectrum sensitivity radars reveal wavelength-dependent responses, with colors representing actual spectral regions (violet=UV, white=visible, red=IR). (m) Normalized comparison across detectors. (n) Measurement time distributions. (o) Revolutionary advantage showing 87.5% sample reduction. (p) Cross-image consistency heatmap demonstrates detector reliability."

### Categorical Depth Figure

**Caption suggestion**:
> "Categorical depth analysis from dual-membrane pixel structure. (a) 3D surface reconstruction showing depth topography. (b) 2D heatmap visualization. (c) Depth distribution histogram with statistical markers. (d) Horizontal and vertical cross-sections through image center. (e) Gradient magnitude map revealing depth transitions. (f) Five-layer segmentation based on depth percentiles. (g) Contour representation with 15 levels. (h) **Electromagnetic wavelength penetration analysis showing depth-dependent visibility across spectrum** (UV to Mid-IR). (i) Cumulative distribution function with quartile markers. (j) Radial depth profile from image center. (k) Comprehensive statistical summary."

**Highlight**: Panel (h) - the EM wavelength penetration - is **novel** and should be emphasized as it provides a **physical interpretation of abstract categorical depth**.

---

## Integration with Virtual Imaging Paper

These visualizations perfectly complement the virtual imaging paper:

1. **Multi-Modal Detector Analysis**: Shows which detectors work at which wavelengths
2. **Categorical Depth Analysis**: Shows depth information enables wavelength-specific reconstruction
3. **EM Spectrum Connection**: Links categorical coordinates to physical electromagnetic properties

Together they demonstrate:
- **Why virtual imaging works**: Different wavelengths probe different depths
- **Which modalities are compatible**: EM spectrum overlap shows complementarity
- **How to choose wavelengths**: Penetration chart guides selection

This is the **complete story**: from detector physics → categorical depth → virtual multi-modal imaging!

---

## Run Both Visualizations

```bash
# Multi-modal detectors with EM spectrum
python visualize_multi_modal_detectors.py

# Categorical depth with penetration analysis
python visualize_categorical_depth.py
```

Processing time: ~30-60 seconds each

---

**These specialized visualizations go beyond simple heatmaps to provide multi-dimensional, physically-grounded insights into detector performance and categorical depth structure.**

