# Publication Panel Charts

## Overview

This script combines all individual NPY files from `npy_visualizations/` into comprehensive, publication-quality panel charts with **16 different visualization types** in a 4×4 grid.

## What It Generates

### Detector Comparison Panel (4×4 grid)

A single comprehensive chart showing all detector types compared across multiple dimensions:

```
┌──────────────────────────────────────────────────────────┐
│  Row 1: Comparative Visualizations                       │
├──────────────┬──────────────┬──────────────┬─────────────┤
│ 1. Radar     │ 2. Stat      │ 3. Violin    │ 4. Corr     │
│    Chart     │    Heatmap   │    Plot      │    Matrix   │
│              │              │              │             │
├──────────────┼──────────────┼──────────────┼─────────────┤
│  Row 2: Metric-Specific Analysis                         │
├──────────────┬──────────────┬──────────────┬─────────────┤
│ 5. Mean      │ 6. Std Dev   │ 7. Entropy   │ 8. Dynamic  │
│    Bars      │    Bars      │    Bars      │    Range    │
│              │              │              │             │
├──────────────┼──────────────┼──────────────┼─────────────┤
│  Row 3: Advanced Analysis                                │
├──────────────┬──────────────┬──────────────┬─────────────┤
│ 9. Polar     │ 10. Skew-    │ 11. Hierar-  │ 12. PCA     │
│    Phase     │     Kurtosis │     chical   │     Project │
│              │              │     Cluster  │             │
├──────────────┼──────────────┼──────────────┼─────────────┤
│  Row 4: Statistical Summaries                            │
├──────────────┬──────────────┬──────────────┬─────────────┤
│ 13. Box      │ 14. Cumu-    │ 15. Normal-  │ 16. Summary │
│     Plots    │     lative   │     ized     │     Table   │
│              │     CDF      │     Compare  │             │
└──────────────┴──────────────┴──────────────┴─────────────┘
```

## Visualization Types Explained

### Row 1: Comparative Visualizations

**1. Radar Chart** (Polar spider web)
- Multi-dimensional performance profile
- Shows: mean, std, entropy, skewness, kurtosis
- Each detector = one polygon
- Larger area = higher overall performance

**2. Statistical Heatmap**
- Color-coded fingerprint of each detector
- Rows = metrics, Columns = detectors
- Red = high, Yellow = low
- Numbers show normalized values

**3. Violin Plot**
- Shows full distribution shape
- Width = probability density
- Inner lines = quartiles
- Reveals distribution characteristics

**4. Correlation Matrix**
- Shows relationships between metrics
- Red = positive correlation
- Blue = negative correlation
- Numbers = correlation coefficients

### Row 2: Metric-Specific Analysis

**5. Mean Values Bar Chart**
- Direct comparison of average values
- Color-coded by detector
- Value labels on top of each bar

**6. Standard Deviation Bar Chart**
- Variability comparison
- Higher = more spread in values
- Important for noise assessment

**7. Entropy Bar Chart**
- Information content
- Higher = more complex/random
- Lower = more structured/predictable

**8. Dynamic Range Plot**
- Shows min-max range with error bars
- Center dot = midpoint
- Error bars = full range
- Larger range = wider dynamic range

### Row 3: Advanced Analysis

**9. Polar Phase Distribution**
- Circular histogram showing phase angles
- Combined view of all detectors
- Different colors = different detectors
- Angular position = phase value

**10. Skewness-Kurtosis Scatter**
- Distribution shape characterization
- X-axis: Skewness (asymmetry)
- Y-axis: Kurtosis (tailedness)
- Each point = one detector
- Quadrants indicate distribution type

**11. Hierarchical Clustering Dendrogram**
- Shows similarity between detectors
- Tree structure based on all metrics
- Closer branches = more similar
- Distance = dissimilarity measure

**12. PCA Projection**
- 2D projection of high-dimensional data
- PC1 & PC2 = principal components
- Percentages = variance explained
- Closer points = similar detectors

### Row 4: Statistical Summaries

**13. Box Plots**
- Quartile analysis for each detector
- Box = 25th to 75th percentile
- Line in box = median
- Whiskers = data range

**14. Cumulative Distribution Functions (CDF)**
- Shows probability ≤ value
- Steep slope = concentrated values
- Flat slope = spread values
- Different colors = different detectors

**15. Normalized Comparison**
- All metrics scaled to 0-1
- Grouped bars for direct comparison
- Different colors = different metrics
- Easy to spot high/low performers

**16. Summary Table**
- Quantitative values for key metrics
- Mean, Std, Entropy
- Precise numbers for reference
- Alternating row colors for readability

## Running the Script

```bash
cd pixel_maxwell_demon
python create_publication_panel_charts.py
```

### Requirements

The script will auto-install if needed:
- numpy
- matplotlib
- scipy
- seaborn
- scikit-learn

### Expected Output

```
publication_panels/
└── detector_comparison_panel.png    # 4×4 comprehensive panel (20×20 inches, 150 DPI)
```

### Processing Time

- Small dataset (<100 NPY files): 30-60 seconds
- Medium dataset (100-500 files): 1-3 minutes
- Large dataset (>500 files): 3-10 minutes

## What It Detects

The script automatically categorizes NPY files by detector type:

- **Photodiode**: Standard optical detector
- **IR Detector**: Infrared sensor
- **Raman**: Raman spectroscopy
- **Mass Spec**: Mass spectrometry
- **Thermometer**: Temperature sensor
- **Barometer**: Pressure sensor
- **Hygrometer**: Humidity sensor
- **Interferometer**: Interference-based detector
- **Dual Membrane**: Dual-membrane structures
- **Virtual Imaging**: Virtual imaging results

## Interpretation Guide

### For Publication Figures

**Choose visualizations based on message**:

1. **Show overall performance**: Use Radar Chart (#1)
2. **Compare specific metrics**: Use Bar Charts (#5-7)
3. **Show relationships**: Use Correlation Matrix (#4)
4. **Display distributions**: Use Violin Plot (#3) or Box Plot (#13)
5. **Demonstrate similarity**: Use Dendrogram (#11) or PCA (#12)
6. **Provide exact values**: Use Summary Table (#16)

### Reading the Panel

**High performance indicators**:
- Radar chart: Large area
- Heatmap: More red colors
- Bar charts: Higher bars
- Dynamic range: Wider error bars
- PCA: Distinct positioning

**Low noise indicators**:
- Lower standard deviation (panel #6)
- Narrower box plots (panel #13)
- Steeper CDF curves (panel #14)

**Similar detectors**:
- Close in PCA projection (panel #12)
- Near branches in dendrogram (panel #11)
- Similar colors in heatmap (panel #2)

## Customization

### Add More Visualization Types

Edit `create_detector_comparison_panel()` to add panels:

```python
# Panel 17: Your custom visualization
ax17 = fig.add_subplot(gs[4, 0])  # 5th row, 1st column
create_your_custom_plot(data, ax17, 'Your Title')
```

### Change Color Schemes

Modify colormaps:
```python
# Current
colors = plt.cm.viridis(...)

# Options
colors = plt.cm.plasma(...)    # Purple-orange
colors = plt.cm.coolwarm(...)  # Blue-red
colors = plt.cm.Set3(...)      # Pastel colors
```

### Adjust Layout

Change grid size:
```python
# Current: 4×4
gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.35)

# Make 5×5
gs = fig.add_gridspec(5, 5, hspace=0.4, wspace=0.35)
```

## Advanced Features

### Multi-Category Panels

The script can generate separate panels for:
- Detector comparisons (default)
- Temporal evolution (if time-series data)
- Wavelength dependencies (if spectral data)
- Resolution comparisons (if multi-resolution)

### Statistical Tests

Built-in calculations include:
- Pearson correlations
- Hierarchical clustering (Ward's method)
- Principal Component Analysis
- Distribution moments (mean, std, skewness, kurtosis)
- Shannon entropy

### Export Options

Current: PNG at 150 DPI

For higher quality:
```python
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Publication quality
plt.savefig(output_path, dpi=600, bbox_inches='tight')  # Print quality
```

For vector graphics:
```python
plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')  # PDF
plt.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')  # SVG
```

## Troubleshooting

### "No detector data found"
**Cause**: No NPY files matching detector patterns  
**Solution**: Check file naming (should contain detector type keywords)

### "Need ≥2 detectors for PCA"
**Cause**: Only one detector category found  
**Solution**: This is informational, not an error. PCA requires multiple categories.

### Memory errors
**Cause**: Too many large NPY files  
**Solution**: Script auto-samples large datasets (>10,000 points)

### Import errors
**Cause**: Missing dependencies  
**Solution**: Script auto-installs scikit-learn. For others:
```bash
pip install numpy matplotlib scipy seaborn scikit-learn
```

## Integration with Publications

### LaTeX Integration

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=\textwidth]{publication_panels/detector_comparison_panel.png}
  \caption{Comprehensive multi-detector comparison across 16 analytical dimensions.
           (a) Radar chart shows performance profiles. (b) Statistical heatmap reveals
           detector fingerprints. (c-p) Additional panels show distribution comparisons,
           correlation structures, and clustering relationships.}
  \label{fig:detector_comparison}
\end{figure}
```

### Refer to specific panels

```latex
As shown in the radar chart (Fig.~\ref{fig:detector_comparison}a), the photodiode
detector exhibits balanced performance across all metrics. The hierarchical clustering
(Fig.~\ref{fig:detector_comparison}k) reveals three distinct detector families...
```

## Performance Metrics Definitions

**Mean**: Average value across all measurements  
**Std**: Standard deviation (variability measure)  
**Median**: 50th percentile value  
**Entropy**: Shannon entropy (information content)  
**Skewness**: Distribution asymmetry (negative = left tail, positive = right tail)  
**Kurtosis**: Distribution tailedness (high = heavy tails, low = light tails)  
**Dynamic Range**: Maximum - Minimum value  

## Citation

If using these visualizations in publications:

```bibtex
@software{sachikonye2024publication_panels,
  title={Comprehensive Multi-Modal Visualization Suite for Detector Comparison},
  author={Sachikonye, Kundai},
  year={2024},
  note={Part of Pixel Maxwell Demon Framework}
}
```

---

**This comprehensive panel chart provides 16 different perspectives on your detector data, making it easy to identify patterns, relationships, and differences that would be invisible in individual plots.**

