"""
Hardware-based instruments for gas molecule/ion simulation.

Maps computer hardware sensors to physical gas properties:
- Accelerometer → molecular velocities, vibrations
- Magnetometer → magnetic fields, ion behavior
- Thermal → temperature, diffusion coefficients
- Timing → phase coherence, oscillatory precision
- etc.

Enables hardware-in-the-loop validation of gas dynamics simulations.
"""

try:
    from .accelerometer import AccelerometerSensor
except ImportError:
    from .acceleratometer import AccelerometerSensor  # Fallback for typo in filename
from .magnetometer import MagnetometerSensor
from .thermal import ThermalSensor
from .electromagnetic import EMFieldSensor
from .optical import OpticalSensor
from .acoustic import AcousticSensor
try:
    from .capacitive import CapacitiveSensor
except ImportError:
    from .capacitative import CapacitiveSensor  # Fallback for typo in filename
from .timing import TimingSensor
from .computational import ComputationalSensor
from .network import NetworkSensor
from .storage import StorageSensor
from .sensor_fusion import HardwareSensorFusion
from .hardware_mapping import HardwareToMolecularMapper
from .binding_affinity import (
    PhaseLockedInteraction,
    MoleculeHemoglobinBinding,
    compare_hemoglobin_binding,
)
from .categorical_toxicity import (
    MolecularCategoricalStates,
    compare_categorical_richness,
)
from .oxygen_categorical_time import (
    CategoricalTimeState,
    CellularTemporalClock,
    demonstrate_oxygen_categorical_time,
)

__all__ = [
    'AccelerometerSensor',
    'MagnetometerSensor',
    'ThermalSensor',
    'EMFieldSensor',
    'OpticalSensor',
    'AcousticSensor',
    'CapacitiveSensor',
    'TimingSensor',
    'ComputationalSensor',
    'NetworkSensor',
    'StorageSensor',
    'HardwareSensorFusion',
    'HardwareToMolecularMapper',
    'PhaseLockedInteraction',
    'MoleculeHemoglobinBinding',
    'compare_hemoglobin_binding',
    'MolecularCategoricalStates',
    'compare_categorical_richness',
    'CategoricalTimeState',
    'CellularTemporalClock',
    'demonstrate_oxygen_categorical_time',
]


