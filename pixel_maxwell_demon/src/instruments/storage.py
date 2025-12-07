"""
Storage I/O for information bandwidth and categorical state persistence.

Maps disk operations to:
- Information transfer rates
- State persistence (write cycles)
- 120 Hz hard drive reference (mechanical oscillator)
"""

import numpy as np
from typing import Dict, Any
import psutil


class StorageSensor:
    """
    Disk I/O for information bandwidth measurements.
    """
    
    def __init__(self):
        """Initialize storage sensor."""
        pass
        
    def measure_io_bandwidth(self) -> Dict[str, Any]:
        """
        Measure disk I/O bandwidth.
        
        Returns:
            I/O throughput metrics
        """
        # Get disk I/O stats
        disk_io = psutil.disk_io_counters()
        
        read_bytes = disk_io.read_bytes if disk_io else 0
        write_bytes = disk_io.write_bytes if disk_io else 0
        
        # Estimate bandwidth (bytes/s)
        bandwidth = (read_bytes + write_bytes) / 3600  # Rough estimate
        
        return {
            'read_bytes': int(read_bytes),
            'write_bytes': int(write_bytes),
            'bandwidth_MB_s': float(bandwidth / 1e6),
        }
    
    def get_complete_storage_state(self) -> Dict[str, Any]:
        """Complete storage measurements."""
        io = self.measure_io_bandwidth()
        
        return {
            'io_bandwidth': io,
        }

