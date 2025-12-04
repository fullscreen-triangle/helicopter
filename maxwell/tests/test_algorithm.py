"""
Tests for HCCC algorithm.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vision.bmd import HardwareBMDStream
from algorithm import HCCCAlgorithm
from validation import BenchmarkSuite


def test_algorithm_initialization():
    """Test algorithm initialization."""
    hardware_stream = HardwareBMDStream()

    hccc = HCCCAlgorithm(
        hardware_stream=hardware_stream,
        lambda_stream=0.5,
        coherence_threshold=1.0
    )

    assert hccc.lambda_stream == 0.5
    assert hccc.A_coherence == 1.0


def test_algorithm_processing():
    """Test algorithm processing on synthetic image."""
    # Create algorithm
    hardware_stream = HardwareBMDStream()
    hccc = HCCCAlgorithm(
        hardware_stream=hardware_stream,
        max_iterations=10  # Limit for test
    )

    # Generate test image
    benchmark = BenchmarkSuite()
    image = benchmark.generate_synthetic_image('geometric', size=(128, 128))

    # Process
    results = hccc.process_image(
        image,
        segmentation_method='slic',
        segmentation_params={'n_segments': 10}
    )

    # Check results
    assert 'network_bmd_final' in results
    assert 'convergence_step' in results
    assert 'processing_sequence' in results
    assert results['convergence_step'] > 0


def test_algorithm_metrics():
    """Test algorithm produces valid metrics."""
    hardware_stream = HardwareBMDStream()
    hccc = HCCCAlgorithm(hardware_stream=hardware_stream, max_iterations=5)

    benchmark = BenchmarkSuite()
    image = benchmark.generate_synthetic_image('gradient', size=(64, 64))

    results = hccc.process_image(image, segmentation_params={'n_segments': 5})

    metrics = results['metrics']

    assert 'final_ambiguity' in metrics
    assert 'final_divergence' in metrics
    assert 'final_richness' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
