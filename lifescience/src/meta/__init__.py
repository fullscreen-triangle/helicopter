# Meta-Information Extraction and Ambiguous Compression Module
"""
Meta-information extraction and ambiguous compression implementation.
Focuses on analyzing structural patterns for problem space compression
and identifying compressible structures in biological data.

Key Features:
- MetaInformationExtractor: Core analysis engine for structural pattern identification
- MetaInformation: Comprehensive structural characterization with confidence metrics
- InformationType: Classification system for biological information patterns

Applications:
- Large-scale biological dataset compression (genomics, proteomics, imaging)
- Pattern recognition in cellular organization
- Compression-guided region-of-interest detection
- Multi-omics data integration through structural similarity
- Database optimization for biological repositories
"""

# Import core classes from submodules
from .structures import InformationType, MetaInformation
from .extractor import MetaInformationExtractor

# Export main classes
__all__ = ['MetaInformationExtractor', 'InformationType', 'MetaInformation']