"""
Tests for BMD state representations.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vision.bmd import BMDState, OscillatoryHole, PhaseStructure


def test_oscillatory_hole():
    """Test OscillatoryHole creation."""
    hole = OscillatoryHole(
        required_frequency=40.0,
        required_phase=np.pi/2,
        required_amplitude=1.0,
        n_configurations=int(1e6)
    )

    assert hole.required_frequency == 40.0
    assert hole.n_configurations == int(1e6)


def test_phase_structure():
    """Test PhaseStructure creation and quality."""
    phases = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4])
    frequencies = np.array([10, 20, 30, 40])
    coherence = np.eye(4) * 0.9

    phase_struct = PhaseStructure(phases, frequencies, coherence)

    assert len(phase_struct.phases) == 4
    quality = phase_struct.phase_lock_quality()
    assert 0 <= quality <= 1


def test_bmd_state_creation():
    """Test BMD state creation."""
    holes = [
        OscillatoryHole(40.0, 0.0, 1.0, int(1e6)),
        OscillatoryHole(80.0, np.pi, 0.5, int(1e6))
    ]

    phase = PhaseStructure(
        phases=np.array([0, np.pi/4]),
        frequencies=np.array([40, 80]),
        coherence=np.eye(2) * 0.8
    )

    bmd = BMDState(
        categorical_state='test_state',
        oscillatory_holes=holes,
        phase_structure=phase
    )

    assert bmd.c_current == 'test_state'
    assert len(bmd.holes) == 2

    richness = bmd.categorical_richness()
    assert richness > 0


def test_bmd_serialization():
    """Test BMD to_dict and from_dict."""
    holes = [OscillatoryHole(40.0, 0.0, 1.0, int(1e6))]
    phase = PhaseStructure(
        phases=np.array([0]),
        frequencies=np.array([40]),
        coherence=np.array([[1.0]])
    )

    bmd = BMDState('test', holes, phase)

    # Serialize
    data = bmd.to_dict()

    # Deserialize
    bmd2 = BMDState.from_dict(data)

    assert bmd2.c_current == bmd.c_current
    assert len(bmd2.holes) == len(bmd.holes)


def test_hardware_bmd_measurement():
    """Test creating BMD from hardware measurement."""
    hardware_data = {
        'n_modes': 5,
        'frequencies': np.logspace(0, 3, 5),
        'coherence': 0.8,
        'timestamp': '2024-01-01'
    }

    bmd = BMDState.from_hardware_measurement(hardware_data, 'test_device')

    assert 'hardware' in bmd.metadata['source']
    assert bmd.metadata['device'] == 'test_device'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
