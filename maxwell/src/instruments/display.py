"""
Optical sensors (camera/screen) for spectral information and display dynamics.

**HARDWARE DEMON**: The display IS a Maxwell's demon that sorts electrons → photons.
We harvest its dynamics as the thermodynamic reference for vision BMD.
"""

import numpy as np
import time
from typing import Dict, Tuple, List
import cv2

class DisplayDemon:
    """
    Harvests physical display dynamics as Maxwell's demon reference.
    
    Display hardware already implements Maxwell's demon:
    - Sorts electrons into RGB photons
    - Uses pixel values as information
    - Dissipates energy as heat + light
    - Operates far from thermodynamic equilibrium
    
    We measure these dynamics to ground vision BMD in hardware reality.
    """
    
    def __init__(self, display_device):
        self.display = display_device
        self.hardware_demon = {}
        
    def measure_pixel_response_times(self) -> Dict[Tuple[int, int], float]:
        """
        Measure pixel transition times for all gray levels.
        
        Hardware demon's sorting speed: how fast can display transition pixels?
        Typical range: 1-16ms depending on display technology.
        
        References: ,  (screen tearing/vsync papers)
        """
        response_times = {}
        
        # Test all gray level transitions (0-255)
        for start_gray in range(0, 256, 16):  # Sample every 16 levels
            for end_gray in range(0, 256, 16):
                if start_gray == end_gray:
                    continue
                
                # Measure transition time
                transition_time = self._measure_single_transition(
                    start_gray, 
                    end_gray
                )
                response_times[(start_gray, end_gray)] = transition_time
        
        self.hardware_demon['pixel_response_times'] = response_times
        return response_times
    
    def _measure_single_transition(self, start: int, end: int) -> float:
        """
        Measure single pixel transition time using photodiode or camera.
        
        Hardware demon's sorting operation: electron → photon conversion time.
        """
        # Set display to start gray level
        self.display.set_gray_level(start)
        time.sleep(0.1)  # Settle
        
        # Trigger transition and measure
        start_time = time.perf_counter()
        self.display.set_gray_level(end)
        
        # Wait for 90% of transition (standard measurement)
        target_brightness = start + 0.9 * (end - start)
        while self.display.measure_brightness() < target_brightness:
            pass
        
        end_time = time.perf_counter()
        transition_time = (end_time - start_time) * 1000  # Convert to ms
        
        return transition_time
    
    def measure_rgb_decay_curves(self) -> Dict[str, np.ndarray]:
        """
        Measure phosphor/LED decay curves for each RGB channel.
        
        Hardware demon's channel-specific behavior: different decay rates.
        Red typically slower (~5ms), Blue faster (~2ms).
        
        References:  (prev search on phosphor decay)
        """
        rgb_decay = {}
        
        for channel in ['R', 'G', 'B']:
            # Set channel to full brightness
            self.display.set_channel(channel, 255)
            time.sleep(0.1)  # Settle
            
            # Turn off and measure decay
            self.display.set_channel(channel, 0)
            
            # Sample at 1000 Hz for 20ms
            decay_curve = []
            start_time = time.perf_counter()
            while (time.perf_counter() - start_time) < 0.02:  # 20ms
                brightness = self.display.measure_channel_brightness(channel)
                decay_curve.append(brightness)
                time.sleep(0.001)  # 1ms sampling
            
            rgb_decay[channel] = np.array(decay_curve)
        
        self.hardware_demon['rgb_decay_curves'] = rgb_decay
        return rgb_decay
    
    def measure_refresh_synchronization(self) -> Dict[str, float]:
        """
        Measure display refresh rate and VSync timing.
        
        Hardware demon's clock: 60Hz, 120Hz, or 240Hz rhythm.
        VSync = demon's synchronization signal.
        
        References: , , ,  (vsync papers)
        """
        sync_data = {}
        
        # Measure refresh rate by detecting frame updates
        frame_times = []
        for _ in range(100):  # Measure 100 frames
            start = time.perf_counter()
            self.display.wait_for_vsync()  # Wait for vertical sync
            end = time.perf_counter()
            frame_times.append(end - start)
        
        # Calculate refresh rate
        avg_frame_time = np.mean(frame_times)
        refresh_rate = 1.0 / avg_frame_time
        
        sync_data['refresh_rate'] = refresh_rate  # Hz
        sync_data['frame_time'] = avg_frame_time * 1000  # ms
        sync_data['vsync_jitter'] = np.std(frame_times) * 1000  # ms
        
        self.hardware_demon['refresh_synchronization'] = sync_data
        return sync_data
    
    def measure_energy_dissipation(self) -> Dict[str, float]:
        """
        Measure display power consumption and heat dissipation.
        
        Hardware demon's thermodynamic cost: energy → heat + light.
        This is the demon's entropy production.
        """
        energy_data = {}
        
        # Measure power consumption at different brightness levels
        for brightness in [0, 64, 128, 192, 255]:
            self.display.set_brightness(brightness)
            time.sleep(1.0)  # Settle
            
            power = self.display.measure_power_consumption()  # Watts
            energy_data[f'power_{brightness}'] = power
        
        # Measure heat dissipation (if thermal sensor available)
        if hasattr(self.display, 'measure_temperature'):
            temp = self.display.measure_temperature()
            energy_data['temperature'] = temp
        
        self.hardware_demon['energy_dissipation'] = energy_data
        return energy_data
    
    def create_hardware_reference(self) -> Dict:
        """
        Complete hardware demon characterization.
        
        This becomes the thermodynamic reference for vision BMD.
        All software demon operations are grounded in this hardware reality.
        """
        print("Measuring hardware demon dynamics...")
        
        # Measure all hardware demon properties
        self.measure_pixel_response_times()
        self.measure_rgb_decay_curves()
        self.measure_refresh_synchronization()
        self.measure_energy_dissipation()
        
        print(f"Hardware demon characterized:")
        print(f"  Refresh rate: {self.hardware_demon['refresh_synchronization']['refresh_rate']:.1f} Hz")
        print(f"  Avg pixel response: {np.mean(list(self.hardware_demon['pixel_response_times'].values())):.2f} ms")
        print(f"  Red decay time: {self._calculate_decay_time(self.hardware_demon['rgb_decay_curves']['R']):.2f} ms")
        print(f"  Power consumption: {self.hardware_demon['energy_dissipation']['power_255']:.2f} W")
        
        return self.hardware_demon
    
    def _calculate_decay_time(self, decay_curve: np.ndarray) -> float:
        """Calculate 1/e decay time from curve."""
        max_val = decay_curve[0]
        target = max_val / np.e
        
        # Find where curve crosses target
        idx = np.where(decay_curve < target)[0]
        if len(idx) > 0:
            return idx[0]  # ms (sampled at 1ms)
        return len(decay_curve)


class VisionDemonFromHardware:
    """
    Software demon that learns from hardware demon.
    
    Uses display dynamics as reference for image understanding.
    Hardware demon → Software demon → Semantic understanding.
    """
    
    def __init__(self, hardware_demon: Dict):
        self.hardware_ref = hardware_demon
        self.software_demon = self._create_from_hardware()
    
    def _create_from_hardware(self) -> Dict:
        """
        Software demon inherits hardware demon's properties.
        
        Temporal reference: refresh rate
        Color reference: RGB decay curves
        Transition reference: pixel response times
        Energy reference: power dissipation
        """
        demon = {}
        
        # Temporal reference from hardware clock
        demon['temporal_freq'] = self.hardware_ref['refresh_synchronization']['refresh_rate']
        demon['temporal_period'] = self.hardware_ref['refresh_synchronization']['frame_time']
        
        # Color reference from RGB dynamics
        demon['red_decay'] = self._calculate_decay_time(
            self.hardware_ref['rgb_decay_curves']['R']
        )
        demon['green_decay'] = self._calculate_decay_time(
            self.hardware_ref['rgb_decay_curves']['G']
        )
        demon['blue_decay'] = self._calculate_decay_time(
            self.hardware_ref['rgb_decay_curves']['B']
        )
        
        # Transition reference from pixel response
        demon['avg_response'] = np.mean(
            list(self.hardware_ref['pixel_response_times'].values())
        )
        
        # Energy reference from power consumption
        demon['energy_cost'] = self.hardware_ref['energy_dissipation']['power_255']
        
        return demon
    
    def sort_image_regions(self, image: np.ndarray) -> Dict:
        """
        Software demon sorts image regions using hardware reference.
        
        Like enzyme sorting molecules by binding dynamics,
        vision demon sorts pixels by display dynamics.
        """
        results = {}
        
        # Split image into regions
        regions = self._split_into_regions(image)
        
        for region_id, region in enumerate(regions):
            # Compare region dynamics to hardware dynamics
            temporal_match = self._compare_temporal(region)
            color_match = self._compare_color(region)
            transition_match = self._compare_transitions(region)
            
            # Demon classifies based on hardware similarity
            classification = self._classify(
                temporal_match,
                color_match,
                transition_match
            )
            
            results[region_id] = {
                'classification': classification,
                'temporal_score': temporal_match,
                'color_score': color_match,
                'transition_score': transition_match
            }
        
        return results
    
    def _compare_temporal(self, region: np.ndarray) -> float:
        """
        Compare region temporal dynamics to hardware refresh rate.
        
        Fast changes (> refresh rate) = aliasing
        Slow changes (< refresh rate) = resolvable
        """
        # Calculate region's dominant frequency
        region_freq = self._calculate_dominant_frequency(region)
        
        # Compare to hardware refresh rate (Nyquist limit)
        nyquist = self.software_demon['temporal_freq'] / 2
        
        if region_freq < nyquist:
            return 1.0  # Resolvable
        else:
            return nyquist / region_freq  # Aliased
    
    def _compare_color(self, region: np.ndarray) -> float:
        """
        Compare region color dynamics to hardware RGB decay.
        
        Red-dominant = slow dynamics (like red phosphor)
        Blue-dominant = fast dynamics (like blue phosphor)
        """
        # Extract RGB channels
        r_channel = region[:, :, 0]
        g_channel = region[:, :, 1]
        b_channel = region[:, :, 2]
        
        # Calculate channel dominance
        r_dominance = np.mean(r_channel) / 255
        g_dominance = np.mean(g_channel) / 255
        b_dominance = np.mean(b_channel) / 255
        
        # Weight by hardware decay times
        color_score = (
            r_dominance * self.software_demon['red_decay'] +
            g_dominance * self.software_demon['green_decay'] +
            b_dominance * self.software_demon['blue_decay']
        ) / (self.software_demon['red_decay'] + 
             self.software_demon['green_decay'] + 
             self.software_demon['blue_decay'])
        
        return color_score
    
    def _compare_transitions(self, region: np.ndarray) -> float:
        """
        Compare region transitions to hardware pixel response.
        
        Sharp edges = fast transitions (like fast pixels)
        Smooth gradients = slow transitions (like slow pixels)
        """
        # Calculate edge strength (transition sharpness)
        edges = cv2.Canny(region, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Compare to hardware response time
        # Sharp edges need fast pixels
        transition_score = edge_density * (
            1.0 / self.software_demon['avg_response']
        )
        
        return transition_score
    
    def _classify(self, temporal: float, color: float, transition: float) -> str:
        """
        Demon classifies region based on hardware similarity.
        
        Like enzyme classifying substrate by binding dynamics.
        """
        # Weighted classification
        score = 0.33 * temporal + 0.33 * color + 0.33 * transition
        
        if score > 0.7:
            return 'high_quality'  # Matches hardware well
        elif score > 0.4:
            return 'medium_quality'  # Partial hardware match
        else:
            return 'low_quality'  # Poor hardware match
    
    def _split_into_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """Split image into regions for demon sorting."""
        # Simple grid splitting (can be more sophisticated)
        h, w = image.shape[:2]
        region_size = 64
        
        regions = []
        for y in range(0, h, region_size):
            for x in range(0, w, region_size):
                region = image[y:y+region_size, x:x+region_size]
                if region.shape[0] == region_size and region.shape[1] == region_size:
                    regions.append(region)
        
        return regions
    
    def _calculate_dominant_frequency(self, region: np.ndarray) -> float:
        """Calculate dominant temporal frequency in region."""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # FFT to find dominant frequency
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Find peak frequency (excluding DC component)
        h, w = magnitude.shape
        magnitude[h//2, w//2] = 0  # Remove DC
        
        peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        peak_freq = np.sqrt((peak_idx[0] - h//2)**2 + (peak_idx[1] - w//2)**2)
        
        return peak_freq
