"""
Ambiguous Compression Processing Module
=====================================

This module contains classes and functions for processing ambiguous bits
through compression analysis and batch processing.
"""

from .batch_ambiguity_processor import (
    BatchAmbiguityProcessor,
    BatchCompressionAnalysis,
    AmbiguousBit
)

__all__ = [
    'BatchAmbiguityProcessor',
    'BatchCompressionAnalysis', 
    'AmbiguousBit'
]
