"""
Helicopter: Reverse Pakati for Visual Knowledge Extraction

A revolutionary framework that extracts structured knowledge from images through
differential analysis, comparing actual images against domain expectations to
identify meaningful deviations.
"""

__version__ = "0.1.0"
__author__ = "Helicopter Team"
__email__ = "support@helicopter-ai.com"

# Core Reverse Helicopter components
from .core.reverse_helicopter import ReverseHelicopter, DeviationToken, ExpectationBaseline

# Pakati integration
from .models.pakati_integration import PakatiGenerator, RegionalControl

# Analysis components
from .core.expectation_analyzer import ExpectationAnalyzer
from .utils.deviation_analysis import DeviationExtractor
from .utils.image_processing import ImageProcessor

# Training and domain adaptation
from .training.domain_trainer import DomainTrainer
from .training.differential_trainer import DifferentialTrainer

# Web interface components
from .api.web_interface import HelicopterWebInterface
from .api.rest_api import HelicopterAPI

# Integration with ecosystem
from .integrations.purpose_integration import PurposeIntegration
from .integrations.combine_harvester_integration import CombineHarvesterIntegration
from .integrations.moriarty_integration import MoriartyIntegration

__all__ = [
    # Core components
    "ReverseHelicopter",
    "DeviationToken", 
    "ExpectationBaseline",
    
    # Pakati integration
    "PakatiGenerator",
    "RegionalControl",
    
    # Analysis
    "ExpectationAnalyzer",
    "DeviationExtractor",
    "ImageProcessor",
    
    # Training
    "DomainTrainer",
    "DifferentialTrainer",
    
    # Web/API
    "HelicopterWebInterface",
    "HelicopterAPI",
    
    # Ecosystem integrations
    "PurposeIntegration",
    "CombineHarvesterIntegration", 
    "MoriartyIntegration"
] 