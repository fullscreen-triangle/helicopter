# Dodecapartite Virtual Microscopy - Experimental Validation

This directory contains experimental validation of the dodecapartite framework using standard microscopy benchmark datasets.

## Structure

```
experimental_validation/
├── datasets/              # Dataset downloaders and loaders
│   ├── download_bbbc.py  # BBBC dataset downloader
│   └── __init__.py
│
├── dodecapartite_implementation/  # Core framework implementation
│   ├── modality_01_optical.py   # Modality 1: Optical processing
│   └── sequential_exclusion.py  # Core algorithm (Section 9)
│
├── validation/            # Validation metrics
│   └── segmentation_metrics.py  # Dice, IoU, Hausdorff, etc.
│
└── experiments/           # Experiment runners
    └── exp1_optical_to_structure.py  # Experiment 1: BBBC038
```

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy scipy scikit-image matplotlib pillow tifffile opencv-python
```

### 2. Run Experiment 1

```bash
cd maxwell/experimental_validation
python experiments/exp1_optical_to_structure.py
```

This will:
- Download BBBC038 dataset (Kaggle 2018 nuclei segmentation)
- Process 10 images with Modality 1 (optical)
- Predict nucleus segmentation
- Compare to ground truth
- Generate metrics and plots

### 3. View Results

Results are saved to `results/exp1/`:
- `results.json`: All metrics and predictions
- `dice_distribution.png`: Histogram of Dice scores

## Experiments

### Experiment 1: Optical → Structure (BBBC038)

**Goal:** Test if optical microscopy alone can predict nucleus segmentation.

**Dataset:** BBBC038 (670+ images, Kaggle 2018)

**Method:**
1. Load brightfield images
2. Apply Modality 1 (optical processing)
3. Extract initial segmentation
4. Apply sequential exclusion algorithm
5. Compare predicted masks to ground truth

**Success Metric:** Dice coefficient > 0.7

**Run:**
```python
from experiments.exp1_optical_to_structure import OpticalToStructureExperiment

exp = OpticalToStructureExperiment()
results, analysis = exp.run(num_images=10)
```

## Datasets

### BBBC038: Kaggle 2018 Nuclei Segmentation

- **Content:** 670+ microscopy images
- **Modalities:** Brightfield, fluorescence
- **Ground Truth:** Nucleus segmentation masks
- **Size:** ~500 MB
- **Download:** Automatic via `BBBCDownloader`

### BBBC018: Multi-Modal Cells

- **Content:** 50,000+ images
- **Modalities:** Phase contrast + 3 fluorescence channels (DAPI, Tubulin, Actin)
- **Ground Truth:** Cell segmentation
- **Size:** ~2 GB
- **Use Case:** Multi-modal channel prediction

## Validation Metrics

All experiments compute:

- **Dice Coefficient:** Overlap measure (2|A∩B| / |A|+|B|)
- **IoU (Jaccard):** Intersection over Union
- **Pixel Accuracy:** Fraction of correct pixels
- **Hausdorff Distance:** Boundary distance
- **Per-Object Metrics:** Detection and matching statistics

## Next Steps

1. **Experiment 2:** Multi-modal channel prediction (BBBC018)
2. **Experiment 3:** Resolution enhancement validation
3. **Experiment 4:** Partial → Complete structure (core test)

## Citation

If using this validation suite, please cite:

- BBBC datasets: [Caicedo et al., Nature Methods, 2019](https://www.nature.com/articles/s41592-019-0612-7)
- Dodecapartite framework: [Your paper citation]
