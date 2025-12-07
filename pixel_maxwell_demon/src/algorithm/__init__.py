"""
HCCC Algorithm implementation.

Main algorithm for hardware-constrained categorical completion:
- HCCCAlgorithm: Main algorithm class
- RegionSelector: Region selection strategies
- HierarchicalIntegration: BMD integration operations
- ConvergenceMonitor: Convergence tracking
"""

from .hccc import HCCCAlgorithm
from .selection import RegionSelector
from .integration import HierarchicalIntegration
from .convergence import ConvergenceMonitor

__all__ = [
    'HCCCAlgorithm',
    'RegionSelector',
    'HierarchicalIntegration',
    'ConvergenceMonitor',
]
