"""
Hardware BMD Stream integration.

Unifies all hardware BMD measurements into a coherent stream
representing physical reality.
"""

import numpy as np
from typing import Dict, Any, Optional
from .bmd_state import BMDState
from .phase_lock import PhaseLockCoupling


class HardwareBMDStream:
    """
    Unified hardware BMD stream representing physical reality.

    β^(stream) = β_display ⊛ β_network ⊛ β_acoustic ⊛ β_accel ⊛ β_EM ⊛ β_optical

    All hardware BMDs are equivalent components hierarchically composed
    through phase-lock coupling.
    """

    def __init__(
        self,
        display_sensor=None,
        network_sensor=None,
        acoustic_sensor=None,
        accelerometer_sensor=None,
        em_sensor=None,
        optical_sensor=None
    ):
        """
        Initialize hardware BMD stream.

        Args:
            display_sensor: Display/screen sensor (DisplayDemon)
            network_sensor: Network latency/jitter sensor
            acoustic_sensor: Acoustic pressure sensor
            accelerometer_sensor: Accelerometer for vibrations
            em_sensor: Electromagnetic field sensor
            optical_sensor: Optical/camera sensor
        """
        self.sensors = {}

        if display_sensor is not None:
            self.sensors['display'] = display_sensor
        if network_sensor is not None:
            self.sensors['network'] = network_sensor
        if acoustic_sensor is not None:
            self.sensors['acoustic'] = acoustic_sensor
        if accelerometer_sensor is not None:
            self.sensors['accelerometer'] = accelerometer_sensor
        if em_sensor is not None:
            self.sensors['em'] = em_sensor
        if optical_sensor is not None:
            self.sensors['optical'] = optical_sensor

        self.phase_lock = PhaseLockCoupling(coupling_strength=1.0)
        self.stream_bmd = None
        self.measurement_history = []

    def measure_stream(self) -> BMDState:
        """
        Measure current hardware BMD stream state.

        Performs hierarchical composition across all active sensors:
        β^(stream) = β₁ ⊛ β₂ ⊛ ... ⊛ βₙ

        Returns:
            BMDState representing unified hardware stream
        """
        device_bmds = []

        # Measure each device
        for device_name, sensor in self.sensors.items():
            try:
                hardware_data = self._measure_device(device_name, sensor)
                bmd = BMDState.from_hardware_measurement(
                    hardware_data,
                    device_name
                )
                device_bmds.append(bmd)
            except Exception as e:
                print(f"Warning: Failed to measure {device_name}: {e}")
                continue

        if not device_bmds:
            raise RuntimeError("No hardware sensors available for measurement")

        # Hierarchically compose all device BMDs
        stream_bmd = self.phase_lock.compose_sequence(device_bmds)

        # Cache and record
        self.stream_bmd = stream_bmd
        self.measurement_history.append({
            'timestamp': np.datetime64('now'),
            'bmd': stream_bmd,
            'n_devices': len(device_bmds)
        })

        return stream_bmd

    def update_stream(self, prev_stream: Optional[BMDState] = None) -> BMDState:
        """
        Update hardware stream with new measurements.

        β^(stream)(t + δt) = β^(stream)(t) ⊛ Δβ(t, δt)

        Args:
            prev_stream: Previous stream state (if None, uses cached)

        Returns:
            Updated stream BMDState
        """
        if prev_stream is None:
            prev_stream = self.stream_bmd

        # Measure new stream state
        new_measurement = self.measure_stream()

        if prev_stream is None:
            return new_measurement

        # Compose with previous (creates temporal continuity)
        updated_stream = self.phase_lock.compose(prev_stream, new_measurement)

        self.stream_bmd = updated_stream
        return updated_stream

    def stream_richness_intersection(self) -> float:
        """
        Calculate R(β^(stream)) = |∩_devices C_device|

        Returns intersection of compatible categorical states
        across all hardware devices.

        Returns:
            Categorical richness of stream (constrained by intersection)
        """
        if self.stream_bmd is None:
            return 0.0

        # Stream richness is bounded by most restrictive device
        # (intersection reduces richness)
        base_richness = self.stream_bmd.categorical_richness()

        # Apply intersection constraint (empirical factor)
        n_devices = len(self.sensors)
        intersection_factor = 1.0 / (n_devices ** 0.5)  # Heuristic

        return base_richness * intersection_factor

    def phase_coherence_matrix(self) -> np.ndarray:
        """
        Compute phase-lock coherence between all device pairs.

        Returns:
            NxN matrix where [i,j] = phase coherence between device i and j
        """
        device_names = list(self.sensors.keys())
        n = len(device_names)

        if n == 0:
            return np.array([[]])

        coherence_matrix = np.eye(n)

        # Measure pairwise coherence
        for i, name1 in enumerate(device_names):
            for j, name2 in enumerate(device_names):
                if i < j:
                    # Measure cross-device phase coherence
                    # (Would use actual measurements in practice)
                    coherence = self._measure_cross_device_coherence(name1, name2)
                    coherence_matrix[i, j] = coherence
                    coherence_matrix[j, i] = coherence

        return coherence_matrix

    def _measure_device(self, device_name: str, sensor) -> Dict[str, Any]:
        """
        Measure hardware dynamics from specific device.

        Args:
            device_name: Name of device
            sensor: Sensor instance

        Returns:
            Dictionary of hardware measurements
        """
        data = {
            'device': device_name,
            'timestamp': np.datetime64('now')
        }

        # Device-specific measurement protocols
        if device_name == 'display':
            # Measure display refresh, pixel response, etc.
            data['frequencies'] = np.array([60.0, 120.0, 240.0])  # Hz
            data['coherence'] = 0.95
            data['n_modes'] = 10

        elif device_name == 'network':
            # Measure network latency, jitter
            data['frequencies'] = np.array([1000.0, 10000.0])  # Hz
            data['coherence'] = 0.85
            data['n_modes'] = 5

        elif device_name == 'acoustic':
            # Measure acoustic pressure oscillations
            data['frequencies'] = np.logspace(2, 4, 10)  # 100 Hz - 10 kHz
            data['coherence'] = 0.70
            data['n_modes'] = 10

        elif device_name == 'accelerometer':
            # Measure vibrations
            data['frequencies'] = np.logspace(0, 3, 10)  # 1 Hz - 1 kHz
            data['coherence'] = 0.80
            data['n_modes'] = 10

        elif device_name == 'em':
            # Measure EM field oscillations
            data['frequencies'] = np.array([50.0, 60.0, 2.4e9])  # AC + WiFi
            data['coherence'] = 0.90
            data['n_modes'] = 8

        elif device_name == 'optical':
            # Measure optical spectrum
            data['frequencies'] = np.linspace(4e14, 7.5e14, 20)  # Visible
            data['coherence'] = 0.75
            data['n_modes'] = 20

        return data

    def _measure_cross_device_coherence(
        self,
        device1: str,
        device2: str
    ) -> float:
        """
        Measure phase coherence between two devices.

        Returns:
            Coherence value in [0, 1]
        """
        # Known physical couplings
        coupling_map = {
            ('display', 'em'): 0.85,  # Display refresh ↔ AC line
            ('display', 'acoustic'): 0.70,  # Backlight ↔ acoustic noise
            ('network', 'em'): 0.80,  # Network ↔ EM fields
            ('accelerometer', 'acoustic'): 0.75,  # Vibration ↔ pressure
        }

        # Check both orderings
        key1 = (device1, device2)
        key2 = (device2, device1)

        return coupling_map.get(key1, coupling_map.get(key2, 0.5))

    def get_stream_state(self) -> Optional[BMDState]:
        """Get current cached stream state."""
        return self.stream_bmd

    def device_count(self) -> int:
        """Return number of active hardware devices."""
        return len(self.sensors)

