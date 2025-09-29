"""
Proof-Validated Compression Processing Module
==========================================

This module contains classes and functions for proof-validated compression
processing using formal theorem provers like Lean and Coq.
"""

from .proof_compression_processor import (
    ProofValidatedCompressionProcessor,
    FormalSystem,
    ProofValidationResult,
    ValidatedAmbiguousBit,
    ProofBasedMetaInformation,
    FormalProof,
    CompressionStep
)

__all__ = [
    'ProofValidatedCompressionProcessor',
    'FormalSystem',
    'ProofValidationResult',
    'ValidatedAmbiguousBit',
    'ProofBasedMetaInformation',
    'FormalProof',
    'CompressionStep'
]
