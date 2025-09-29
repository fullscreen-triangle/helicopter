# Video Processing and Analysis Module
"""
Video processing and analysis functionality for life science applications.
Specialized for analyzing time-lapse microscopy, live cell imaging, and 
dynamic biological processes.

Key Features:
- VideoAnalyzer: Core video processing engine with biological context
- CellTracker: Individual cell tracking across time with nearest-neighbor assignment
- CellTrack: Data structure for temporal cell properties (position, area, intensity)

Applications:
- Live cell imaging analysis with individual cell tracking
- Cell migration pattern quantification
- Cell division timing and dynamics
- Calcium imaging temporal analysis
- Drug response kinetics in live cells
- Fluorescence recovery after photobleaching (FRAP)
"""

# Import core classes from submodules
from .tracking import VideoType, CellTrack, CellTracker
from .analyzer import VideoAnalyzer

# Export main classes
__all__ = ['VideoAnalyzer', 'VideoType', 'CellTrack', 'CellTracker']