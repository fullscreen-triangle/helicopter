"""
Integration Module: Bridging Pixel Maxwell Demons with HCCC Framework
=====================================================================

This module provides the integration layer between:
1. Pixel Maxwell Demon implementation (maxwell/src/maxwell/)
2. HCCC algorithm framework (maxwell/src/)

Key components:
- DualMembraneBMDState: Bridges pixel demon dual states with HCCC BMD states
- DualMembraneRegion: Image regions with pixel demon grids
- DualMembraneNetworkBMD: Hierarchical network with dual structure
- PixelDemonHardwareStream: Hardware stream from pixel demon measurements

Author: Kundai Sachikonye & AI Collaborator
Date: 2024
"""

from .dual_bmd_state import DualMembraneBMDState, pixel_demon_to_bmd_state
from .dual_region import DualMembraneRegion, create_dual_regions_from_image
from .dual_network_bmd import DualMembraneNetworkBMD
from .pixel_hardware_stream import PixelDemonHardwareStream
from .dual_ambiguity import DualMembraneAmbiguityCalculator
from .dual_hccc_algorithm import DualMembraneHCCCAlgorithm, DualHCCCResult
from .depth_extraction import DepthExtractor
from .validate_framework import FrameworkValidator, validate_framework

__all__ = [
    'DualMembraneBMDState',
    'pixel_demon_to_bmd_state',
    'DualMembraneRegion',
    'create_dual_regions_from_image',
    'DualMembraneNetworkBMD',
    'PixelDemonHardwareStream',
    'DualMembraneAmbiguityCalculator',
    'DualMembraneHCCCAlgorithm',
    'DualHCCCResult',
    'DepthExtractor',
    'FrameworkValidator',
    'validate_framework',
]

