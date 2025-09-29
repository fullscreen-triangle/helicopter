# S-Entropy Framework for Life Science Applications
"""
Self-contained entropy framework implementing S-entropy coordinate transformation
specifically designed for biological and medical image analysis.

Key Features:
- SEntropyTransformer: Core transformation engine for biological images
- SEntropyCoordinates: 4D coordinate system for biological structures
- BiologicalContext: Context enumeration for different biological scales

Applications:
- Cell phenotype classification through entropy coordinates
- Tissue morphology analysis using semantic dimensions
- Disease progression tracking via entropy trajectories
- Developmental biology staging using coordinate evolution
"""

# Import core classes from submodules
from .coordinates import SEntropyCoordinates, BiologicalContext
from .transformer import SEntropyTransformer

# Export main classes
__all__ = ['SEntropyTransformer', 'SEntropyCoordinates', 'BiologicalContext']