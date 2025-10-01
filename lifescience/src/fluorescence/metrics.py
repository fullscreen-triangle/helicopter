"""
Fluorescence Microscopy Metrics - Data structures and measurement classes
"""

from typing import Dict
from dataclasses import dataclass
from enum import Enum


class FluorescenceChannel(Enum):
    """Common fluorescence channels"""
    DAPI = "dapi"  # DNA/nucleus
    GFP = "gfp"    # Green fluorescent protein
    RFP = "rfp"    # Red fluorescent protein
    FITC = "fitc"  # Fluorescein


@dataclass
class FluorescenceMetrics:
    """Fluorescence analysis metrics"""
    mean_intensity: float
    max_intensity: float
    integrated_intensity: float
    area: float
    signal_to_noise: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'mean_intensity': self.mean_intensity,
            'max_intensity': self.max_intensity, 
            'integrated_intensity': self.integrated_intensity,
            'area': self.area,
            'signal_to_noise': self.signal_to_noise
        }
