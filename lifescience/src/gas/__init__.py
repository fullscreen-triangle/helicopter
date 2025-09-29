# Gas Molecule Model Classes and Functionality for Life Science Applications
"""
This module implements the gas molecular dynamics framework specifically adapted for
biological and life science image analysis. Information elements are modeled as 
thermodynamic gas molecules that seek equilibrium states through Hamilton's equations,
enabling principled analysis of biological structures and processes.

Key Features:
- InformationGasMolecule: Individual information entities with thermodynamic properties
- GasMolecularSystem: Collection of molecules with interaction potentials
- BiologicalGasAnalyzer: Life science-specific analysis wrapper

Applications:
- Protein structure analysis through thermodynamic equilibrium
- Cellular process modeling (mitosis, apoptosis, migration)
- Metabolic pathway visualization as molecular interactions
- Drug-target binding dynamics through equilibrium states
"""

# Import core classes from submodules
from .molecules import (
    InformationGasMolecule,
    GasMolecularSystem,
    MoleculeType,
    BiologicalProperties
)

from .analyzer import BiologicalGasAnalyzer

# Export main classes
__all__ = [
    'InformationGasMolecule',
    'GasMolecularSystem', 
    'BiologicalGasAnalyzer',
    'MoleculeType',
    'BiologicalProperties'
]