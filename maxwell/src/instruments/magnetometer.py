"""
Magnetometer sensor for magnetic field and paramagnetic O₂ measurements.

Maps 3-axis magnetic field data to:
- O₂ paramagnetic coupling (triplet state sensitivity)
- Zeeman splitting (spin state populations)
- Ion trajectories (Lorentz force)
- Magnetic field gradients (trap potentials)
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

try:
    import sensors  # Generic sensor library
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False


class MagnetometerSensor:
    """
    3-axis magnetometer for O₂ paramagnetic effects and ion dynamics.
    
    Typical range: ±4900 μT (49 Gauss)
    Resolution: ~0.15 μT
    Bandwidth: DC to 400 Hz
    """
    
    def __init__(self, sample_rate: float = 100.0):
        """
        Initialize magnetometer.
        
        Args:
            sample_rate: Sampling frequency in Hz
        """
        self.sample_rate = sample_rate
        self.hardware_available = HARDWARE_AVAILABLE
        self._calibrate()
        
    def _calibrate(self):
        """Calibrate magnetometer using Earth's magnetic field."""
        if self.hardware_available:
            # Read multiple samples
            samples = []
            for _ in range(100):
                try:
                    Bx = sensors.magnetometer.x
                    By = sensors.magnetometer.y
                    Bz = sensors.magnetometer.z
                    samples.append([Bx, By, Bz])
                except:
                    pass
            
            if samples:
                samples = np.array(samples)
                self.earth_field_magnitude = np.mean(np.linalg.norm(samples, axis=1))
            else:
                self.earth_field_magnitude = 50.0  # μT typical
        else:
            self.earth_field_magnitude = 50.0  # μT
    
    def read_magnetic_field(self, duration: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Read 3-axis magnetic field.
        
        Args:
            duration: Measurement duration in seconds
            
        Returns:
            Dictionary with 'Bx', 'By', 'Bz', 't' arrays in μT
        """
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples)
        
        if self.hardware_available:
            # Read from actual hardware
            Bx = np.array([sensors.magnetometer.x for _ in range(n_samples)])
            By = np.array([sensors.magnetometer.y for _ in range(n_samples)])
            Bz = np.array([sensors.magnetometer.z for _ in range(n_samples)])
        else:
            # Simulate Earth's field + noise
            # Earth's field ~ 50 μT with ±12 μT noise
            Bx = np.random.normal(30.0, 1.0, n_samples)  # μT
            By = np.random.normal(20.0, 1.0, n_samples)
            Bz = np.random.normal(35.0, 1.0, n_samples)
        
        return {
            'Bx': Bx,
            'By': By,
            'Bz': Bz,
            'B_magnitude': np.sqrt(Bx**2 + By**2 + Bz**2),
            't': t,
        }
    
    def calculate_o2_zeeman_splitting(
        self,
        duration: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate Zeeman splitting for O₂ triplet state.
        
        O₂ has S=1 (triplet), μ = 2 Bohr magnetons
        ΔE = g * μ_B * m_s * B
        
        Args:
            duration: Measurement time
            
        Returns:
            Zeeman splitting energies and populations
        """
        field_data = self.read_magnetic_field(duration)
        B_magnitude = np.mean(field_data['B_magnitude']) * 1e-6  # Convert to Tesla
        
        # Bohr magneton
        mu_B = 9.274e-24  # J/T
        g_factor = 2.0  # g-factor for electron spin
        
        # Energy splitting for m_s = -1, 0, +1
        # ΔE = g * μ_B * m_s * B
        E_minus1 = -g_factor * mu_B * B_magnitude  # m_s = -1
        E_0 = 0.0  # m_s = 0
        E_plus1 = g_factor * mu_B * B_magnitude  # m_s = +1
        
        # Boltzmann populations at T=300K
        k_B = 1.380649e-23  # J/K
        T = 300.0  # K
        
        Z = np.exp(-E_minus1/(k_B*T)) + np.exp(-E_0/(k_B*T)) + np.exp(-E_plus1/(k_B*T))
        pop_minus1 = np.exp(-E_minus1/(k_B*T)) / Z
        pop_0 = np.exp(-E_0/(k_B*T)) / Z
        pop_plus1 = np.exp(-E_plus1/(k_B*T)) / Z
        
        # Paramagnetic susceptibility
        # χ = (N * g² * μ_B² * S(S+1)) / (3 * k_B * T)
        S = 1.0  # Spin quantum number
        N = 2.5e25  # Number density
        chi = (N * g_factor**2 * mu_B**2 * S * (S+1)) / (3 * k_B * T)
        
        return {
            'magnetic_field_T': float(B_magnitude),
            'magnetic_field_uT': float(B_magnitude * 1e6),
            'zeeman_splitting_J': float(E_plus1 - E_minus1),
            'zeeman_splitting_meV': float((E_plus1 - E_minus1) * 6.242e21),  # meV
            'population_ms_minus1': float(pop_minus1),
            'population_ms_0': float(pop_0),
            'population_ms_plus1': float(pop_plus1),
            'paramagnetic_susceptibility': float(chi),
            'temperature_K': T,
        }
    
    def simulate_ion_trajectory(
        self,
        ion_mass: float = 16.0,  # amu (O+ ion)
        ion_charge: int = 1,     # elementary charges
        initial_velocity: np.ndarray = None,
        duration: float = 1.0
    ) -> Dict[str, Any]:
        """
        Simulate ion trajectory in magnetic field.
        
        Lorentz force: F = q(v × B)
        
        Args:
            ion_mass: Ion mass in amu
            ion_charge: Ion charge in elementary charge units
            initial_velocity: Initial velocity [vx, vy, vz] in m/s
            duration: Simulation time
            
        Returns:
            Ion trajectory information
        """
        if initial_velocity is None:
            # Thermal velocity at 300K
            k_B = 1.380649e-23
            m = ion_mass * 1.66054e-27  # kg
            v_thermal = np.sqrt(3 * k_B * 300 / m)
            initial_velocity = np.array([v_thermal, 0, 0])
        
        # Read magnetic field
        field_data = self.read_magnetic_field(duration)
        B = np.array([
            np.mean(field_data['Bx']),
            np.mean(field_data['By']),
            np.mean(field_data['Bz'])
        ]) * 1e-6  # Convert to Tesla
        
        # Ion properties
        q = ion_charge * 1.602e-19  # Coulombs
        m = ion_mass * 1.66054e-27  # kg
        
        # Cyclotron frequency: ω = qB/m
        B_magnitude = np.linalg.norm(B)
        omega_cyclotron = q * B_magnitude / m
        
        # Cyclotron radius: r = mv⊥ / (qB)
        v = initial_velocity
        v_magnitude = np.linalg.norm(v)
        v_parallel = np.dot(v, B) / B_magnitude  # Velocity along B
        v_perp = np.sqrt(v_magnitude**2 - v_parallel**2)  # Perpendicular velocity
        r_cyclotron = m * v_perp / (q * B_magnitude)
        
        # Pitch angle
        pitch_angle = np.arctan2(v_perp, v_parallel) * 180 / np.pi
        
        return {
            'ion_mass_amu': float(ion_mass),
            'ion_charge': int(ion_charge),
            'magnetic_field_T': float(B_magnitude),
            'cyclotron_frequency_Hz': float(omega_cyclotron / (2*np.pi)),
            'cyclotron_period_s': float(2*np.pi / omega_cyclotron),
            'cyclotron_radius_m': float(r_cyclotron),
            'cyclotron_radius_mm': float(r_cyclotron * 1e3),
            'pitch_angle_deg': float(pitch_angle),
            'parallel_velocity_m_s': float(v_parallel),
            'perpendicular_velocity_m_s': float(v_perp),
        }
    
    def measure_field_gradient(
        self,
        spatial_separation: float = 0.01  # meters between sensors
    ) -> Dict[str, Any]:
        """
        Estimate magnetic field gradient.
        
        Gradient needed for magnetic trapping of paramagnetic molecules.
        
        Args:
            spatial_separation: Assumed distance between measurements
            
        Returns:
            Field gradient estimates
        """
        # Take two measurements (simulate slight movement)
        field1 = self.read_magnetic_field(0.1)
        # Simulate movement
        import time
        time.sleep(0.1)
        field2 = self.read_magnetic_field(0.1)
        
        # Calculate gradient (rough estimate)
        dBx = np.mean(field2['Bx']) - np.mean(field1['Bx'])
        dBy = np.mean(field2['By']) - np.mean(field1['By'])
        dBz = np.mean(field2['Bz']) - np.mean(field1['Bz'])
        
        grad_B = np.array([dBx, dBy, dBz]) / spatial_separation  # μT/m
        
        # Trapping force on O2 molecule
        # F = μ * ∇B, where μ = 2 Bohr magnetons
        mu_B = 9.274e-24  # J/T
        mu = 2 * mu_B
        force_magnitude = mu * np.linalg.norm(grad_B) * 1e-6  # N
        
        return {
            'gradient_Bx_uT_m': float(dBx / spatial_separation),
            'gradient_By_uT_m': float(dBy / spatial_separation),
            'gradient_Bz_uT_m': float(dBz / spatial_separation),
            'gradient_magnitude_uT_m': float(np.linalg.norm(grad_B)),
            'trapping_force_N': float(force_magnitude),
            'spatial_separation_m': spatial_separation,
        }
    
    def measure_phase_coherence_from_field(
        self,
        duration: float = 10.0
    ) -> Dict[str, Any]:
        """
        Measure phase coherence from magnetic field stability.
        
        Stable field → coherent spin precession
        
        Args:
            duration: Measurement time
            
        Returns:
            Phase coherence metrics
        """
        field_data = self.read_magnetic_field(duration)
        B_magnitude = field_data['B_magnitude']
        
        # Larmor frequency: ω = γB (γ = gyromagnetic ratio)
        # For electron: γ = g * μ_B / ℏ
        g = 2.0
        mu_B = 9.274e-24  # J/T
        hbar = 1.055e-34  # J·s
        gamma = g * mu_B / hbar  # rad/(s·T)
        
        # Convert B to Tesla
        B_T = B_magnitude * 1e-6
        
        # Larmor frequencies
        omega_larmor = gamma * B_T
        f_larmor = omega_larmor / (2 * np.pi)
        
        # Phase accumulation (integral of frequency)
        dt = 1.0 / self.sample_rate
        phases = np.cumsum(2 * np.pi * f_larmor * dt)
        
        # Phase jitter
        phase_jitter = np.std(np.diff(phases))
        
        # Coherence = 1 / (1 + jitter)
        coherence = 1.0 / (1.0 + phase_jitter)
        
        return {
            'mean_larmor_frequency_Hz': float(np.mean(f_larmor)),
            'larmor_frequency_std_Hz': float(np.std(f_larmor)),
            'phase_jitter_rad': float(phase_jitter),
            'phase_coherence': float(coherence),
            'measurement_duration_s': duration,
        }
    
    def get_complete_magnetic_state(self) -> Dict[str, Any]:
        """
        Complete magnetic state from magnetometer.
        
        Returns:
            All magnetic measurements
        """
        field_data = self.read_magnetic_field(2.0)
        zeeman = self.calculate_o2_zeeman_splitting()
        ion_traj = self.simulate_ion_trajectory()
        gradient = self.measure_field_gradient()
        coherence = self.measure_phase_coherence_from_field()
        
        return {
            'field_data': {
                'Bx_mean_uT': float(np.mean(field_data['Bx'])),
                'By_mean_uT': float(np.mean(field_data['By'])),
                'Bz_mean_uT': float(np.mean(field_data['Bz'])),
                'B_magnitude_mean_uT': float(np.mean(field_data['B_magnitude'])),
            },
            'o2_zeeman_splitting': zeeman,
            'ion_trajectory': ion_traj,
            'field_gradient': gradient,
            'phase_coherence': coherence,
            'hardware_available': self.hardware_available,
        }


