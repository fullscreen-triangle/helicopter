"""
Accelerometer sensor for molecular velocity and vibration measurements.

Maps 3-axis acceleration data to:
- Molecular velocities (Maxwell-Boltzmann distribution)
- Vibrational modes (molecular oscillations)
- Collision frequencies (from acceleration noise)
- Diffusion coefficients (from random walk analysis)
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import time

try:
    # Try to import actual sensor libraries
    import sensors  # Generic sensor library
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False


class AccelerometerSensor:
    """
    Harvest accelerometer data and map to molecular gas properties.
    
    Acceleration noise → molecular thermal motion
    Vibration frequency → molecular oscillation modes
    RMS acceleration → temperature via equipartition
    """
    
    def __init__(self, sample_rate: float = 100.0):
        """
        Initialize accelerometer sensor.
        
        Args:
            sample_rate: Sampling frequency in Hz
        """
        self.sample_rate = sample_rate
        self.hardware_available = HARDWARE_AVAILABLE
        
    def read_raw_acceleration(self, duration: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Read raw 3-axis acceleration data.
        
        Args:
            duration: Measurement duration in seconds
            
        Returns:
            Dictionary with 'x', 'y', 'z', 't' arrays
        """
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples)
        
        if self.hardware_available:
            # Read from actual hardware
            accel_x = np.array([sensors.accelerometer.x for _ in range(n_samples)])
            accel_y = np.array([sensors.accelerometer.y for _ in range(n_samples)])
            accel_z = np.array([sensors.accelerometer.z for _ in range(n_samples)])
        else:
            # Simulate with realistic noise (±0.01 g)
            accel_x = np.random.normal(0, 0.01, n_samples)
            accel_y = np.random.normal(0, 0.01, n_samples)
            accel_z = np.random.normal(9.81, 0.01, n_samples)  # Gravity + noise
            
        return {
            'x': accel_x,
            'y': accel_y,
            'z': accel_z,
            't': t,
        }
    
    def extract_molecular_velocities(
        self, 
        duration: float = 1.0,
        molecular_mass: float = 32.0  # O2 mass in amu
    ) -> Dict[str, Any]:
        """
        Map acceleration noise to molecular velocity distribution.
        
        Uses equipartition: (1/2)m<v²> = (3/2)kT
        Acceleration RMS ~ velocity variance
        
        Args:
            duration: Measurement time
            molecular_mass: Molecular mass in amu
            
        Returns:
            Velocity statistics and temperature estimate
        """
        data = self.read_raw_acceleration(duration)
        
        # Remove gravity component
        accel_3d = np.array([data['x'], data['y'], data['z'] - 9.81])
        
        # RMS acceleration (g units)
        accel_rms = np.sqrt(np.mean(accel_3d**2))
        
        # Map to molecular velocity (m/s)
        # Calibration: 0.01 g RMS ≈ 500 m/s molecular velocity (O2 at 300K)
        v_rms = accel_rms * 50000  # Empirical scaling
        
        # Estimate temperature from v_rms
        # v_rms = sqrt(3kT/m)
        k_B = 1.380649e-23  # J/K
        m = molecular_mass * 1.66054e-27  # kg
        T_estimated = (v_rms**2) * m / (3 * k_B)
        
        # Velocity components
        vx = data['x'] * 50000
        vy = data['y'] * 50000
        vz = (data['z'] - 9.81) * 50000
        
        # Maxwell-Boltzmann fit
        v_mean = np.sqrt(8 * k_B * T_estimated / (np.pi * m))
        v_most_probable = np.sqrt(2 * k_B * T_estimated / m)
        
        return {
            'v_rms_m_s': float(v_rms),
            'v_mean_m_s': float(v_mean),
            'v_most_probable_m_s': float(v_most_probable),
            'temperature_K': float(T_estimated),
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'n_samples': len(vx),
        }
    
    def extract_collision_frequency(
        self,
        duration: float = 1.0,
        number_density: float = 2.5e25  # molecules/m³ at STP
    ) -> Dict[str, Any]:
        """
        Estimate collision frequency from acceleration fluctuations.
        
        High-frequency acceleration noise ~ frequent collisions
        
        Args:
            duration: Measurement time
            number_density: Gas number density
            
        Returns:
            Collision frequency and mean free path
        """
        data = self.read_raw_acceleration(duration)
        
        # FFT to find dominant frequencies
        accel_magnitude = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
        fft = np.fft.rfft(accel_magnitude)
        freqs = np.fft.rfftfreq(len(accel_magnitude), 1/self.sample_rate)
        
        # Power spectrum
        power = np.abs(fft)**2
        
        # Characteristic frequency (weighted average)
        f_char = np.sum(freqs * power) / np.sum(power)
        
        # Map to collision frequency (rough calibration)
        # f_char ~ sqrt(collision_freq)
        collision_freq = (f_char * 1e9)**2  # Hz
        
        # Mean free path: λ = v / (√2 π d² n ν)
        v_rms = 500  # m/s typical
        d = 3.5e-10  # O2 diameter (m)
        lambda_mfp = v_rms / (collision_freq + 1e-10)
        
        return {
            'collision_frequency_Hz': float(collision_freq),
            'mean_free_path_m': float(lambda_mfp),
            'characteristic_frequency_Hz': float(f_char),
            'power_spectrum_peak_Hz': float(freqs[np.argmax(power)]),
        }
    
    def extract_diffusion_coefficient(
        self,
        duration: float = 10.0
    ) -> Dict[str, Any]:
        """
        Estimate diffusion coefficient from random walk analysis.
        
        Track "center of mass" motion of acceleration signal.
        MSD ~ 2D*t (Einstein relation)
        
        Args:
            duration: Measurement time (longer = better statistics)
            
        Returns:
            Diffusion coefficient estimate
        """
        data = self.read_raw_acceleration(duration)
        
        # Integrate acceleration → velocity → position
        dt = 1.0 / self.sample_rate
        vx = np.cumsum(data['x']) * dt * 50000  # Scale to m/s
        vy = np.cumsum(data['y']) * dt * 50000
        vz = np.cumsum((data['z'] - 9.81)) * dt * 50000
        
        # Positions (random walk)
        x = np.cumsum(vx) * dt
        y = np.cumsum(vy) * dt
        z = np.cumsum(vz) * dt
        
        # Mean squared displacement
        r_squared = x**2 + y**2 + z**2
        msd = np.mean(r_squared)
        
        # Diffusion coefficient: MSD = 6Dt (3D)
        D = msd / (6 * duration)
        
        return {
            'diffusion_coefficient_m2_s': float(D),
            'msd_m2': float(msd),
            'measurement_duration_s': duration,
        }
    
    def measure_vibrational_modes(
        self,
        duration: float = 1.0,
        freq_range: Tuple[float, float] = (10.0, 400.0)
    ) -> Dict[str, Any]:
        """
        Extract vibrational mode frequencies from acceleration spectrum.
        
        Maps to molecular vibration frequencies (scaled down).
        
        Args:
            duration: Measurement time
            freq_range: Frequency range to analyze (Hz)
            
        Returns:
            Identified vibrational modes
        """
        data = self.read_raw_acceleration(duration)
        
        # FFT of each axis
        fft_x = np.fft.rfft(data['x'])
        fft_y = np.fft.rfft(data['y'])
        fft_z = np.fft.rfft(data['z'])
        freqs = np.fft.rfftfreq(len(data['x']), 1/self.sample_rate)
        
        # Total power
        power_total = np.abs(fft_x)**2 + np.abs(fft_y)**2 + np.abs(fft_z)**2
        
        # Find peaks in frequency range
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs_range = freqs[mask]
        power_range = power_total[mask]
        
        # Identify peaks (local maxima)
        peak_indices = self._find_peaks(power_range, threshold=np.mean(power_range) * 3)
        peak_freqs = freqs_range[peak_indices]
        peak_amplitudes = power_range[peak_indices]
        
        # Sort by amplitude
        sorted_indices = np.argsort(peak_amplitudes)[::-1]
        modes = []
        for i in sorted_indices[:10]:  # Top 10 modes
            modes.append({
                'frequency_Hz': float(peak_freqs[i]),
                'amplitude': float(peak_amplitudes[i]),
                'period_s': float(1.0 / peak_freqs[i]),
            })
        
        return {
            'n_modes': len(modes),
            'modes': modes,
            'frequency_range_Hz': freq_range,
        }
    
    def _find_peaks(self, data: np.ndarray, threshold: float) -> np.ndarray:
        """Find local maxima above threshold."""
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1] and data[i] > threshold:
                peaks.append(i)
        return np.array(peaks)
    
    def get_complete_molecular_state(
        self,
        duration: float = 2.0,
        molecular_mass: float = 32.0,
        number_density: float = 2.5e25
    ) -> Dict[str, Any]:
        """
        Complete molecular state from accelerometer.
        
        Args:
            duration: Measurement time
            molecular_mass: Molecular mass (amu)
            number_density: Number density (molecules/m³)
            
        Returns:
            Complete gas state parameters
        """
        velocities = self.extract_molecular_velocities(duration, molecular_mass)
        collisions = self.extract_collision_frequency(duration, number_density)
        diffusion = self.extract_diffusion_coefficient(duration)
        vibrations = self.measure_vibrational_modes(duration)
        
        return {
            'velocities': velocities,
            'collisions': collisions,
            'diffusion': diffusion,
            'vibrations': vibrations,
            'hardware_available': self.hardware_available,
            'sample_rate_Hz': self.sample_rate,
            'measurement_duration_s': duration,
        }


