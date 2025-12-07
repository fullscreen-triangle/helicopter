"""
Battery: Electrochemical demon that sorts ions across membrane.

**THE INSIGHT**: Battery IS a Maxwell's demon separating Li+ ions.
"""

class BatteryDemon:
    """
    Harvests electrochemical dynamics from laptop battery.
    
    Battery operation = Maxwell's demon sorting ions:
    - Charging: Demon pushes Li+ ions into anode (energy input)
    - Discharging: Demon releases Li+ ions to cathode (energy output)
    - Voltage = demon's sorting potential
    - Current = demon's sorting rate
    - Temperature = demon's entropy production
    """
    
    def measure_electrochemical_demon(self):
        """
        Battery management system provides demon dynamics.
        
        References: [[0]](#__0), [[1]](#__1), [[2]](#__2), [[3]](#__3)
        """
        demon_state = {}
        
        # Voltage = demon's electrochemical potential
        demon_state['voltage'] = self.read_battery_voltage()  # V
        # [[0]](#__0), [[1]](#__1) (battery voltage sensing)
        
        # Current = demon's ion sorting rate
        demon_state['current'] = self.read_battery_current()  # A
        # [[0]](#__0), [[2]](#__2) (current measurement for state estimation)
        
        # State of charge = demon's information state
        demon_state['soc'] = self.read_state_of_charge()  # %
        # [[0]](#__0), [[2]](#__2) (SOC estimation from electrochemical state)
        
        # Impedance = demon's sorting resistance
        demon_state['impedance'] = self.read_impedance()  # Ohms
        # [[3]](#__3) (dynamic electrochemical impedance spectroscopy)
        
        # Temperature = demon's entropy production
        demon_state['temperature'] = self.read_battery_temp()  # °C
        # [[0]](#__0), [[1]](#__1) (thermal management)
        
        # Cycle count = demon's operational history
        demon_state['cycles'] = self.read_cycle_count()
        # [[1]](#__1) (cycle-based battery management)
        
        # Power = demon's energy throughput
        demon_state['power'] = demon_state['voltage'] * demon_state['current']  # W
        # [[0]](#__0), [[2]](#__2) (power management)
        
        return demon_state
    
    def calculate_ion_sorting_rate(self, current: float) -> float:
        """
        Current = rate of Li+ ion sorting.
        
        1 Ampere = 6.24 × 10^18 electrons/second
        Each Li+ ion carries 1 electron worth of charge
        
        Demon sorts ~10^19 ions/second at 1A!
        
        References: [[2]](#__2), [[3]](#__3) (electrochemical dynamics)
        """
        electrons_per_second = current * 6.24e18
        return electrons_per_second  # ions/second
    
    def calculate_demon_efficiency(self, voltage: float, impedance: float) -> float:
        """
        Efficiency = how well demon sorts vs. entropy cost.
        
        Lower impedance = more efficient demon
        Higher voltage = stronger sorting potential
        
        References: [[2]](#__2), [[3]](#__3) (impedance and efficiency)
        """
        # Ideal voltage (no impedance loss)
        ideal_power = voltage ** 2 / impedance
        
        # Actual power (with impedance loss)
        actual_power = voltage ** 2 / (impedance + self.internal_resistance)
        
        efficiency = actual_power / ideal_power
        return efficiency
