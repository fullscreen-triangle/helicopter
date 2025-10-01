# Multi-panel Figure

## Panel A: Segmented Image Results

- Top row: Original fluorescence image (3-channel composite or individual channels)
- Bottom row: Segmentation overlay with color-coded regions of interest
- Include: Scale bar, channel labels (DAPI, GFP, RFP, etc.), timestamp for video frames

## Panel B: Time Series Analysis

- Primary Y-axis: Fluorescence intensity (AU)
- X-axis: Time (seconds/minutes/frames)
- Three colored lines for each channel
- Include error bands (±SD or SEM)
- Legend with channel names and colors

# Second Panel

## Panel A: Signal-to-Noise Analysis

- Area plot showing signal envelope (upper bound)
- Area plot showing noise floor (lower bound)
- Filled area between them representing SNR margin
- Include SNR ratio values as text annotations
- Color gradient from red (low SNR) to green (high SNR)

## Panel B: Segmentation Performance

- Combined violin/box plots:
- Dice coefficients, IoU scores, pixel accuracy
- One plot per metric, grouped by cell type/structure
- Include median lines, quartiles, and outliers

# Third Panel

Panel A: Classification Performance
Tri-panel layout:

Left: ROC curves (multiple classes on same plot)
Center: Confusion matrix heatmap with percentages
Right: Precision-Recall curves

Panel B : Measurement Validation
Bland-Altman plot with:

Mean difference line
95% confidence intervals
Color-coding by cell type or experimental condition
