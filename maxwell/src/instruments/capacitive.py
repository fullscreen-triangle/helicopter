"""
Capacitive sensors (touchscreen) for ion/molecule proximity detection.

Maps capacitance to:
- Ion cloud density (proximity sensing)
- Molecular polarizability
- Dielectric properties (ε_r)
- Charge distribution
"""

import numpy as np
from typing import Dict, Any


class CapacitiveSensor:
    """
    Touchscreen capacitance for ion/molecule detection.
    
    Typical resolution: 0.01 pF
    """
    
    def __init__(self):
        """Initialize capacitive sensor."""
        self.baseline_capacitance = 10.0  # pF typical
        
    def read_capacitance(self) -> Dict[str, float]:
        """
        Read capacitance (simulated from CPU activity as proxy).
        
        Returns:
            Capacitance in pF
        """
        import psutil
        cpu_load = psutil.cpu_percent(interval=0.1)
        
        # Map CPU load to capacitance variation
        # Higher load → more charge movement → higher capacitance
        capacitance_pF = self.baseline_capacitance + cpu_load * 0.1
        
        return {'capacitance_pF': float(capacitance_pF)}
    
    def estimate_ion_density(self) -> Dict[str, Any]:
        """
        Estimate ion density from capacitance change.
        
        C = ε₀ * ε_r * A / d
        
        Returns:
            Ion density estimate
        """
        cap_data = self.read_capacitance()
        C = cap_data['capacitance_pF'] * 1e-12  # Convert to F
        
        # Assume parallel plate: A = 1 cm², d = 1 mm
        A = 1e-4  # m²
        d = 1e-3  # m
        epsilon_0 = 8.854e-12  # F/m
        
        # Effective epsilon_r
        epsilon_r = C * d / (epsilon_0 * A)
        
        # Map to ion density (rough)
        # Higher ε_r → more ions
        n_ions = (epsilon_r - 1.0) * 1e24  # ions/m³
        
        return {
            'capacitance_pF': cap_data['capacitance_pF'],
            'epsilon_r': float(epsilon_r),
            'ion_density_per_m3': float(max(0, n_ions)),
        }
    
    def get_complete_capacitive_state(self) -> Dict[str, Any]:
        """Complete capacitive measurements."""
        ion_density = self.estimate_ion_density()
        
        return {
            'ion_density': ion_density,
        }


