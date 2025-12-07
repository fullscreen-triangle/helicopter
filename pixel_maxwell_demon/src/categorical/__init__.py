"""
Categorical completion operations for BMD-based vision.

This package implements:
- Ambiguity calculation: A(β, R)
- Categorical completion: β_{i+1} = Generate(β_i, R)
- Richness metrics: R(β)
- Constraint networks: Phase-lock graph management
"""

from .ambiguity import AmbiguityCalculator
from .completion import CategoricalCompletion
from .richness import CategoricalRichnessCalculator
from .constrains import ConstraintNetwork

__all__ = [
    'AmbiguityCalculator',
    'CategoricalCompletion',
    'CategoricalRichnessCalculator',
    'ConstraintNetwork',
]
