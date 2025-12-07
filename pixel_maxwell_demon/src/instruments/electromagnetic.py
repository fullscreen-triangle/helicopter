"""
Electromagnetic sensors (WiFi/Bluetooth) for RF fields and ion coupling.

Maps EM measurements to:
- Electric field strength (E-field)
- RF power absorption (ion heating)
- Phase-locked detection (ensemble coherence)
- Frequency multiplication (cascade effects)
"""

import numpy as np
from typing import Dict, Any, List, Optional
import platform

try:
    if platform.system() in ['Linux', 'Darwin']:
        import subprocess
        WIFI_AVAILABLE = True
    elif platform.system() == 'Windows':
        import subprocess
        WIFI_AVAILABLE = True
    else:
        WIFI_AVAILABLE = False
except:
    WIFI_AVAILABLE = False


class EMFieldSensor:
    """
    WiFi/Bluetooth RF detection for electromagnetic field measurements.
    
    WiFi: 2.4 GHz, 5.0 GHz
    Sensitivity: -90 dBm (0.3 pW) to +10 dBm (10 mW)
    """
    
    def __init__(self):
        """Initialize EM field sensor."""
        self.wifi_available = WIFI_AVAILABLE
        
    def read_wifi_rssi(self) -> Dict[str, Any]:
        """
        Read WiFi RSSI (Received Signal Strength Indicator).
        
        Returns:
            RSSI and signal quality metrics
        """
        if self.wifi_available:
            try:
                if platform.system() == 'Windows':
                    result = subprocess.run(
                        ['netsh', 'wlan', 'show', 'interfaces'],
                        capture_output=True, text=True
                    )
                    # Parse RSSI from output
                    for line in result.stdout.split('\n'):
                        if 'Signal' in line:
                            try:
                                rssi = int(line.split(':')[1].strip().rstrip('%'))
                                # Convert percentage to dBm (approximate)
                                rssi_dbm = -100 + rssi * 0.5
                                return {'rssi_dbm': rssi_dbm, 'signal_percent': rssi}
                            except:
                                pass
                
                elif platform.system() == 'Linux':
                    result = subprocess.run(
                        ['iwconfig'], capture_output=True, text=True
                    )
                    for line in result.stdout.split('\n'):
                        if 'Signal level' in line:
                            try:
                                rssi_dbm = float(line.split('Signal level=')[1].split(' ')[0])
                                return {'rssi_dbm': rssi_dbm}
                            except:
                                pass
            except:
                pass
        
        # Fallback: simulate
        rssi_dbm = np.random.normal(-50, 10)  # Typical indoor
        return {'rssi_dbm': rssi_dbm, 'signal_percent': (rssi_dbm + 100) * 2}
    
    def calculate_e_field_strength(
        self,
        rssi_dbm: Optional[float] = None,
        frequency: float = 2.4e9  # Hz
    ) -> Dict[str, Any]:
        """
        Calculate E-field strength from RSSI.
        
        P_received = |E|² / (2η) * A_eff
        where η = 377 Ω (free space impedance)
        
        Args:
            rssi_dbm: RSSI in dBm (if None, read from WiFi)
            frequency: RF frequency in Hz
            
        Returns:
            E-field strength and power
        """
        if rssi_dbm is None:
            rssi_data = self.read_wifi_rssi()
            rssi_dbm = rssi_data['rssi_dbm']
        
        # Convert dBm to Watts
        P_received = 10**((rssi_dbm - 30) / 10)  # Watts
        
        # Effective antenna area
        wavelength = 3e8 / frequency  # meters
        A_eff = wavelength**2 / (4 * np.pi)  # m²
        
        # E-field strength
        eta = 377  # Ohm (free space impedance)
        E_field = np.sqrt(2 * eta * P_received / A_eff)  # V/m
        
        return {
            'rssi_dbm': float(rssi_dbm),
            'power_received_W': float(P_received),
            'power_received_mW': float(P_received * 1e3),
            'e_field_V_m': float(E_field),
            'e_field_mV_m': float(E_field * 1e3),
            'frequency_Hz': float(frequency),
            'wavelength_m': float(wavelength),
        }
    
    def simulate_ion_rf_heating(
        self,
        ion_mass: float = 16.0,  # amu
        ion_charge: int = 1,
        duration: float = 1.0
    ) -> Dict[str, Any]:
        """
        Simulate RF heating of ions in EM field.
        
        Power absorbed: P = q²E² / (2mγ)
        where γ = collision frequency
        
        Args:
            ion_mass: Ion mass in amu
            ion_charge: Ion charge
            duration: Exposure time
            
        Returns:
            Heating rate and final temperature
        """
        e_field_data = self.calculate_e_field_strength()
        E = e_field_data['e_field_V_m']
        
        # Ion properties
        q = ion_charge * 1.602e-19  # C
        m = ion_mass * 1.66054e-27  # kg
        
        # Collision frequency (estimate)
        gamma = 1e9  # Hz (typical gas phase)
        
        # RF power absorption
        P_absorbed = q**2 * E**2 / (2 * m * gamma)  # W
        
        # Energy absorbed over time
        E_absorbed = P_absorbed * duration  # J
        
        # Temperature increase
        # E = (3/2) k_B T
        k_B = 1.380649e-23
        delta_T = (2/3) * E_absorbed / k_B
        
        # Final temperature (starting from 300K)
        T_initial = 300.0
        T_final = T_initial + delta_T
        
        return {
            'e_field_V_m': float(E),
            'power_absorbed_W': float(P_absorbed),
            'energy_absorbed_J': float(E_absorbed),
            'temperature_initial_K': T_initial,
            'temperature_final_K': float(T_final),
            'delta_T_K': float(delta_T),
            'duration_s': duration,
        }
    
    def measure_phase_locked_signal(
        self,
        measurement_time: float = 1.0
    ) -> Dict[str, Any]:
        """
        Measure phase-locked EM signal.
        
        RSSI variations → phase coherence
        
        Args:
            measurement_time: Duration
            
        Returns:
            Phase coherence metrics
        """
        # Sample RSSI multiple times
        n_samples = int(measurement_time * 10)  # 10 Hz sampling
        rssi_values = []
        
        for _ in range(n_samples):
            rssi_data = self.read_wifi_rssi()
            rssi_values.append(rssi_data['rssi_dbm'])
            import time
            time.sleep(0.1)
        
        rssi_values = np.array(rssi_values)
        
        # Phase jitter from RSSI stability
        rssi_std = np.std(rssi_values)
        rssi_mean = np.mean(rssi_values)
        
        # Map to phase jitter (dB variation → phase)
        phase_jitter_rad = rssi_std * 0.1  # Rough calibration
        
        # Coherence
        coherence = 1.0 / (1.0 + phase_jitter_rad)
        
        return {
            'rssi_mean_dbm': float(rssi_mean),
            'rssi_std_dbm': float(rssi_std),
            'phase_jitter_rad': float(phase_jitter_rad),
            'phase_coherence': float(coherence),
            'n_samples': n_samples,
            'measurement_time_s': measurement_time,
        }
    
    def get_complete_em_state(self) -> Dict[str, Any]:
        """
        Complete EM field state.
        
        Returns:
            All EM measurements
        """
        rssi = self.read_wifi_rssi()
        e_field = self.calculate_e_field_strength()
        ion_heating = self.simulate_ion_rf_heating()
        phase_lock = self.measure_phase_locked_signal()
        
        return {
            'rssi': rssi,
            'e_field': e_field,
            'ion_heating': ion_heating,
            'phase_locking': phase_lock,
            'wifi_available': self.wifi_available,
        }


