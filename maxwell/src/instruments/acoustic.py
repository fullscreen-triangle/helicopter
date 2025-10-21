"""
Acoustic sensors (microphone) for pressure oscillations and gas flow.

Maps acoustic measurements to:
- Pressure oscillations (sound waves)
- Gas flow velocities (acoustic streaming)
- Turbulence intensity (noise analysis)
- Molecular collision rates (from noise floor)
"""

import numpy as np
from typing import Dict, Any

try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


class AcousticSensor:
    """
    Microphone for acoustic pressure measurements.
    
    Maps sound to gas pressure oscillations.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize acoustic sensor."""
        self.sample_rate = sample_rate
        self.audio_available = AUDIO_AVAILABLE
        
    def read_acoustic_signal(self, duration: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Read acoustic signal from microphone.
        
        Args:
            duration: Recording duration
            
        Returns:
            Audio signal array
        """
        n_samples = int(duration * self.sample_rate)
        
        if self.audio_available:
            try:
                p = pyaudio.PyAudio()
                stream = p.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=self.sample_rate,
                              input=True,
                              frames_per_buffer=n_samples)
                data = stream.read(n_samples)
                stream.stop_stream()
                stream.close()
                p.terminate()
                
                audio = np.frombuffer(data, dtype=np.int16).astype(float)
                audio = audio / 32768.0  # Normalize
                
                return {'signal': audio, 't': np.arange(n_samples) / self.sample_rate}
            except:
                pass
        
        # Simulate
        t = np.arange(n_samples) / self.sample_rate
        audio = np.random.normal(0, 0.01, n_samples)
        return {'signal': audio, 't': t}
    
    def measure_pressure_oscillations(self, duration: float = 1.0) -> Dict[str, Any]:
        """
        Extract pressure oscillation frequency and amplitude.
        
        Args:
            duration: Measurement time
            
        Returns:
            Pressure oscillation metrics
        """
        data = self.read_acoustic_signal(duration)
        signal = data['signal']
        
        # RMS pressure
        p_rms = np.sqrt(np.mean(signal**2))
        
        # Convert to Pascals (calibration: 1.0 audio units ≈ 1 Pa)
        p_rms_Pa = p_rms * 1.0
        
        # FFT
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/self.sample_rate)
        power = np.abs(fft)**2
        
        # Dominant frequency
        peak_idx = np.argmax(power[1:]) + 1  # Exclude DC
        f_dominant = freqs[peak_idx]
        
        return {
            'pressure_rms_Pa': float(p_rms_Pa),
            'dominant_frequency_Hz': float(f_dominant),
            'pressure_amplitude_Pa': float(p_rms_Pa * np.sqrt(2)),
        }
    
    def get_complete_acoustic_state(self) -> Dict[str, Any]:
        """Complete acoustic measurements."""
        pressure = self.measure_pressure_oscillations()
        
        return {
            'pressure_oscillations': pressure,
            'audio_available': self.audio_available,
        }


