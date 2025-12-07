"""
Thermal sensors (CPU/GPU) for temperature and diffusion measurements.

Maps thermal readings to:
- Gas temperature (T)
- Thermal velocities (Maxwell-Boltzmann)
- Diffusion coefficients (D)
- Heat capacity (Cv)
- Thermal conductivity (κ)
"""

import numpy as np
from typing import Dict, Any, List, Optional
import psutil
import platform

try:
    if platform.system() == 'Linux':
        import glob
        THERMAL_PATHS = glob.glob('/sys/class/thermal/thermal_zone*/temp')
    HARDWARE_AVAILABLE = True
except:
    HARDWARE_AVAILABLE = False


class ThermalSensor:
    """
    CPU/GPU temperature sensors for gas thermal properties.
    
    Temperature readings → molecular kinetic energy
    Thermal gradients → diffusion coefficients
    Cooling/heating rates → heat capacity
    """
    
    def __init__(self):
        """Initialize thermal sensor."""
        self.hardware_available = HARDWARE_AVAILABLE
        self.baseline_temp = self._measure_baseline()
        
    def _measure_baseline(self) -> float:
        """Measure baseline system temperature."""
        temps = self.read_all_temperatures()
        return np.mean(list(temps.values())) if temps else 50.0
    
    def read_all_temperatures(self) -> Dict[str, float]:
        """
        Read all available temperature sensors.
        
        Returns:
            Dictionary of sensor_name: temperature_celsius
        """
        temps = {}
        
        try:
            # Try psutil first
            if hasattr(psutil, 'sensors_temperatures'):
                sensors = psutil.sensors_temperatures()
                for name, entries in sensors.items():
                    for i, entry in enumerate(entries):
                        temps[f"{name}_{i}"] = entry.current
        except:
            pass
        
        # Linux sysfs
        if platform.system() == 'Linux' and self.hardware_available:
            try:
                for i, path in enumerate(THERMAL_PATHS):
                    with open(path, 'r') as f:
                        temp_milliC = int(f.read().strip())
                        temps[f'thermal_zone_{i}'] = temp_milliC / 1000.0
            except:
                pass
        
        # Fallback: simulate with CPU usage
        if not temps:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            temps['cpu_simulated'] = 40.0 + cpu_percent * 0.5
        
        return temps
    
    def map_to_gas_temperature(
        self,
        sensor_temp_C: Optional[float] = None,
        scaling_factor: float = 5.0
    ) -> Dict[str, Any]:
        """
        Map hardware temperature to gas temperature.
        
        Hardware: 30-90°C range
        Gas: Map to 100-500 K range
        
        Args:
            sensor_temp_C: Sensor temperature (if None, read from hardware)
            scaling_factor: Mapping calibration factor
            
        Returns:
            Gas temperature and related properties
        """
        if sensor_temp_C is None:
            temps = self.read_all_temperatures()
            sensor_temp_C = np.mean(list(temps.values()))
        
        # Map to gas temperature (Kelvin)
        # Linear mapping: 40°C → 300K, 80°C → 500K
        T_gas_K = 273.15 + (sensor_temp_C - 20) * scaling_factor
        T_gas_K = max(100, min(T_gas_K, 1000))  # Clamp to reasonable range
        
        # Thermal velocity (O2)
        k_B = 1.380649e-23  # J/K
        m_O2 = 32 * 1.66054e-27  # kg
        v_thermal = np.sqrt(3 * k_B * T_gas_K / m_O2)
        
        # Speed of sound
        gamma = 1.4  # O2 diatomic
        v_sound = np.sqrt(gamma * k_B * T_gas_K / m_O2)
        
        return {
            'sensor_temperature_C': float(sensor_temp_C),
            'gas_temperature_K': float(T_gas_K),
            'gas_temperature_C': float(T_gas_K - 273.15),
            'thermal_velocity_m_s': float(v_thermal),
            'speed_of_sound_m_s': float(v_sound),
            'thermal_energy_J': float(1.5 * k_B * T_gas_K),
        }
    
    def measure_diffusion_coefficient(
        self,
        measurement_time: float = 10.0
    ) -> Dict[str, Any]:
        """
        Estimate diffusion coefficient from temperature measurements.
        
        D ∝ T^(3/2) (kinetic theory)
        
        Args:
            measurement_time: Duration to track temperature
            
        Returns:
            Diffusion coefficient estimate
        """
        # Track temperature over time
        temperatures = []
        times = []
        
        n_samples = int(measurement_time / 0.5)
        for i in range(n_samples):
            temp_data = self.map_to_gas_temperature()
            temperatures.append(temp_data['gas_temperature_K'])
            times.append(i * 0.5)
            
            if i < n_samples - 1:
                import time
                time.sleep(0.5)
        
        T_mean = np.mean(temperatures)
        T_std = np.std(temperatures)
        
        # Diffusion coefficient (Chapman-Enskog)
        # D = (3/16) * sqrt(πkT/m) / (nπd²)
        k_B = 1.380649e-23
        m = 32 * 1.66054e-27
        d = 3.5e-10  # O2 diameter
        n = 2.5e25  # Number density at STP
        
        D = (3/16) * np.sqrt(np.pi * k_B * T_mean / m) / (n * np.pi * d**2)
        
        # Thermal conductivity (κ)
        kappa = (75/64) * k_B * np.sqrt(k_B * T_mean / (np.pi * m)) / (np.pi * d**2)
        
        return {
            'mean_temperature_K': float(T_mean),
            'temperature_std_K': float(T_std),
            'diffusion_coefficient_m2_s': float(D),
            'thermal_conductivity_W_mK': float(kappa),
            'measurement_time_s': measurement_time,
            'n_samples': len(temperatures),
        }
    
    def measure_heat_capacity(
        self,
        heating_time: float = 5.0
    ) -> Dict[str, Any]:
        """
        Estimate heat capacity from CPU load-induced heating.
        
        Apply computational load → measure ΔT/Δt
        C_v = Q / ΔT
        
        Args:
            heating_time: Time to apply CPU load
            
        Returns:
            Heat capacity estimate
        """
        # Initial temperature
        T_initial = self.map_to_gas_temperature()['gas_temperature_K']
        
        # Apply CPU load (simulate heating)
        import time
        cpu_stress = []
        for _ in range(int(heating_time)):
            # Create CPU load
            _ = [i**2 for i in range(10000)]
            time.sleep(0.1)
        
        # Final temperature
        T_final = self.map_to_gas_temperature()['gas_temperature_K']
        delta_T = T_final - T_initial
        
        # Heat input (assume proportional to time)
        Q = 10.0 * heating_time  # Watts * seconds
        
        # Heat capacity
        C_v = Q / (delta_T + 1e-10)
        
        # For gas: C_v = (f/2) * N * k_B
        # where f = degrees of freedom (5 for O2 diatomic)
        k_B = 1.380649e-23
        N_particles = 1e23  # Arbitrary scale
        C_v_theoretical = (5/2) * N_particles * k_B
        
        return {
            'initial_temperature_K': float(T_initial),
            'final_temperature_K': float(T_final),
            'delta_T_K': float(delta_T),
            'heat_input_J': float(Q),
            'heat_capacity_measured_J_K': float(C_v),
            'heat_capacity_theoretical_J_K': float(C_v_theoretical),
            'heating_time_s': heating_time,
        }
    
    def measure_thermal_gradients(self) -> Dict[str, Any]:
        """
        Measure spatial thermal gradients across multiple sensors.
        
        Gradient → Fick's law diffusion simulation
        
        Returns:
            Thermal gradient information
        """
        temps = self.read_all_temperatures()
        
        if len(temps) < 2:
            return {
                'n_sensors': len(temps),
                'gradient_available': False,
            }
        
        # Convert to array
        temp_values = np.array(list(temps.values()))
        sensor_names = list(temps.keys())
        
        # Calculate gradient (assume sensors equally spaced)
        grad_T = np.gradient(temp_values)
        max_gradient = np.max(np.abs(grad_T))
        
        return {
            'n_sensors': len(temps),
            'sensor_names': sensor_names,
            'temperatures_C': temp_values.tolist(),
            'gradient_C': grad_T.tolist(),
            'max_gradient_C': float(max_gradient),
            'gradient_available': True,
        }
    
    def get_complete_thermal_state(self) -> Dict[str, Any]:
        """
        Complete thermal state from all sensors.
        
        Returns:
            All thermal measurements
        """
        all_temps = self.read_all_temperatures()
        gas_temp = self.map_to_gas_temperature()
        diffusion = self.measure_diffusion_coefficient(measurement_time=2.0)
        gradients = self.measure_thermal_gradients()
        
        return {
            'all_temperatures_C': all_temps,
            'gas_properties': gas_temp,
            'diffusion': diffusion,
            'thermal_gradients': gradients,
            'hardware_available': self.hardware_available,
        }


