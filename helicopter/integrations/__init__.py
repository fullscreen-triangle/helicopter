"""
Helicopter Integrations Module

This module provides integrations with external AI services and models,
particularly Hugging Face models for enhanced computer vision capabilities.

The integrations support the core Helicopter principle: reconstruction ability
demonstrates understanding. External models are used to validate and enhance
autonomous reconstruction insights.
"""

from .huggingface_api import (
    HuggingFaceModelAPI,
    ModelSelector,
    TaskRouter,
    ModelCapabilities,
    IntelligentModelOrchestrator
)

from .model_validation import (
    ModelValidationEngine,
    CrossModelValidator,
    ConsensusBuilder
)

__all__ = [
    'HuggingFaceModelAPI',
    'ModelSelector', 
    'TaskRouter',
    'ModelCapabilities',
    'IntelligentModelOrchestrator',
    'ModelValidationEngine',
    'CrossModelValidator',
    'ConsensusBuilder'
]

__version__ = "0.1.0" 