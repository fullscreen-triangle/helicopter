# HCCC Algorithm Quick Start Guide

## Installation

```bash
# Install dependencies
pip install numpy scipy scikit-image matplotlib networkx
```

## Minimal Example

```python
from maxwell.src.vision.bmd import HardwareBMDStream
from maxwell.src.algorithm import HCCCAlgorithm
from maxwell.src.validation import BenchmarkSuite

# 1. Create hardware stream (mock for demo)
hardware_stream = HardwareBMDStream()

# 2. Create algorithm
hccc = HCCCAlgorithm(
    hardware_stream=hardware_stream,
    lambda_stream=0.5,         # Balance ambiguity vs coherence
    coherence_threshold=1.0,   # Convergence threshold
    max_iterations=100         # Maximum processing steps
)

# 3. Load test image
benchmark = BenchmarkSuite()
image = benchmark.generate_synthetic_image('geometric')

# 4. Process image
results = hccc.process_image(image, segmentation_method='slic')

# 5. View results
print(f"Converged in {results['convergence_step']} iterations")
print(f"Processed {results['regions_processed']} regions")
print(f"Network richness: {results['interpretation']['network_richness']:.2e}")
```

## Understanding the Results

```python
results = {
    'network_bmd_final': NetworkBMD,          # Final network state
    'processing_sequence': [region_ids],      # Processing order
    'ambiguity_history': [ambiguity_values],  # A(β, R) over time
    'stream_divergence_history': [div_values],# D_stream over time
    'network_richness_history': [R_values],   # R(β) growth
    'convergence_step': int,                  # Iterations to convergence
    'interpretation': {                       # High-level interpretation
        'n_regions_processed': int,
        'network_richness': float,
        'phase_quality': float,
        'n_compounds': int,
        'hierarchical_structure': {...}
    },
    'metrics': {                              # Performance metrics
        'final_ambiguity': float,
        'mean_ambiguity': float,
        'final_divergence': float,
        'final_richness': float,
        'richness_growth': float
    }
}
```

## Key Parameters

### HCCCAlgorithm Parameters

- **lambda_stream** (0.0-1.0): Balance between ambiguity and stream coherence
  - Lower: Prioritize ambiguity (explore categorical richness)
  - Higher: Prioritize coherence (stay grounded in hardware)
  - Default: 0.5

- **coherence_threshold**: Target ambiguity for convergence
  - Lower: Stricter convergence (more precise)
  - Higher: Looser convergence (faster)
  - Default: 1.0

- **max_iterations**: Maximum processing steps
  - Prevents infinite loops
  - Default: 1000

- **allow_revisitation**: Allow regions to be reprocessed
  - True: Can revisit if ambiguity increases
  - False: Process each region once
  - Default: True

### Segmentation Parameters

```python
results = hccc.process_image(
    image,
    segmentation_method='slic',  # 'slic', 'felzenszwalb', 'watershed', 'hierarchical'
    segmentation_params={
        'n_segments': 50,         # Number of regions
        'compactness': 10.0,      # Balance color-space vs image-space
        'sigma': 1.0              # Gaussian smoothing
    }
)
```

## Visualization

```python
from maxwell.src.validation import ResultVisualizer

visualizer = ResultVisualizer()

# Visualize processing sequence
visualizer.visualize_processing_sequence(
    image,
    regions,
    results['processing_sequence'],
    save_path='processing_sequence.png'
)

# Visualize network growth
visualizer.visualize_network_growth(
    results['network_richness_history'],
    results['ambiguity_history'],
    results['stream_divergence_history'],
    save_path='network_growth.png'
)

# Visualize hierarchical structure
visualizer.visualize_hierarchical_structure(
    results['network_bmd_final'],
    save_path='hierarchical_structure.png'
)
```

## Validation

```python
from maxwell.src.validation import BiologicalValidator, PhysicalValidator

# Biological validation
bio_validator = BiologicalValidator()
bio_results = bio_validator.comprehensive_biological_validation(
    network_bmd=results['network_bmd_final'],
    hardware_stream=hardware_stream.get_stream_state(),
    divergence_history=results['stream_divergence_history'],
    richness_history=results['network_richness_history']
)

print(f"Biological validation: {bio_results['overall']['validated']}")

# Physical validation
phys_validator = PhysicalValidator()
phys_results = phys_validator.comprehensive_physical_validation(
    initial_bmd=hardware_stream.get_stream_state(),
    final_network_bmd=results['network_bmd_final'],
    hardware_stream=hardware_stream
)

print(f"Physical validation: {phys_results['overall']['validated']}")
```

## Advanced: Custom Hardware Sensors

```python
from maxwell.src.instruments import DisplayDemon, NetworkSensor, AcousticSensor

# Initialize real hardware sensors
display = DisplayDemon(display_device='/dev/fb0')
network = NetworkSensor()
acoustic = AcousticSensor(device_id=0)

# Create hardware stream with real sensors
hardware_stream = HardwareBMDStream(
    display_sensor=display,
    network_sensor=network,
    acoustic_sensor=acoustic
)

# Use in algorithm
hccc = HCCCAlgorithm(hardware_stream=hardware_stream)
```

## Troubleshooting

### Slow Convergence

- **Increase lambda_stream**: Prioritize stream coherence
- **Reduce n_segments**: Fewer regions to process
- **Increase coherence_threshold**: Looser convergence criterion

### High Stream Divergence

- **Increase lambda_stream**: Enforce hardware grounding
- **Check hardware sensors**: Ensure measurements are valid
- **Reduce ambiguity weight**: Balance dual objective

### Memory Issues

- **Reduce max_compound_order**: Limit hierarchical depth
- **Prune compounds**: Call `network_bmd.prune_low_richness_compounds()`
- **Process smaller images**: Reduce resolution or region count

## Theory References

- **HCCC Paper**: `maxwell/publication/hardware-constrained-categorical-completion.tex`
- **S-Entropy Framework**: `docs/categories/st-stellas-categories.tex`
- **Consciousness Theory**: `docs/categories/categorical-completion-consiousness.tex`

## Support

For issues or questions:

1. Check the comprehensive demo: `maxwell/demo_hccc_vision.py`
2. Read the implementation guide: `maxwell/README_IMPLEMENTATION.md`
3. Review the algorithm proposal: `maxwell/ALGORITHM_IMPLEMENTATION_PROPOSAL.md`
