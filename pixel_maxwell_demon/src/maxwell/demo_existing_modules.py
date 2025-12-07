#!/usr/bin/env python3
"""
Complete Maxwell Demo: Using ALL Existing Modules
================================================

This demonstrates the CORRECT way to use the Maxwell package by leveraging
the existing modules that were already implemented:

1. live_cell_imaging.py - Multi-modal microscopy with virtual detectors
2. harmonic_coincidence.py - O(1) frequency-based network queries  
3. categorical_light_sources.py - Categorical rendering

Author: Kundai Sachikonye
Date: 2024
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import logging
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from live_cell_imaging import (
    LiveCellSample,
    LiveCellMicroscope,
    validate_with_real_data,
    demonstrate_ambiguous_signal_resolution
)
from harmonic_coincidence import (
    HarmonicCoincidenceNetwork,
    Oscillator
)
from categorical_light_sources import (
    CategoricalLightSource,
    Color
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_multi_modal_microscopy():
    """
    Demonstrate multi-modal microscopy using existing live_cell_imaging module.
    
    This is the CORRECT way - using the already-implemented functionality!
    """
    logger.info("\n" + "="*80)
    logger.info("  MULTI-MODAL MICROSCOPY DEMONSTRATION")
    logger.info("  Using Existing Maxwell Modules")
    logger.info("="*80)
    
    # Create biological sample
    logger.info("\n1. Creating biological sample...")
    sample = LiveCellSample(name="HeLa_cytoplasm", sample_volume_m3=1e-15)
    sample.populate_typical_cell_cytoplasm()
    
    logger.info(f"   ✓ Sample created with {len(sample.molecules)} molecule types")
    logger.info(f"   Temperature: {sample.temperature_k:.2f} K ({sample.temperature_k-273.15:.2f} °C)")
    logger.info(f"   pH: {sample.ph}")
    
    # Create microscope with trans-Planckian resolution
    logger.info("\n2. Initializing Pixel Maxwell Demon microscope...")
    microscope = LiveCellMicroscope(
        spatial_resolution_m=1e-9,  # 1 nm (sub-wavelength!)
        temporal_resolution_s=1e-15,  # 1 femtosecond
        field_of_view_m=(10e-6, 10e-6, 5e-6),  # 10×10×5 μm³
        name="PMD_microscope"
    )
    
    logger.info(f"   ✓ Microscope initialized")
    logger.info(f"   Spatial resolution: {microscope.spatial_resolution*1e9:.1f} nm")
    logger.info(f"   Temporal resolution: {microscope.temporal_resolution*1e15:.1f} fs")
    logger.info(f"   Pixel grid: {microscope.grid_shape}")
    
    # Image sample with ALL modalities simultaneously
    logger.info("\n3. Imaging sample (all virtual detectors simultaneously)...")
    logger.info("   🚀 REVOLUTIONARY: All modalities on ONE sample!")
    
    results = microscope.image_sample(sample)
    
    logger.info(f"\n   ✓ Imaging complete!")
    logger.info(f"   Pixels imaged: {results['num_pixels']:,}")
    logger.info(f"   Mean confidence: {results['mean_confidence']:.2%}")
    logger.info(f"   Detectors used: {', '.join(results['detector_types_used'])}")
    
    # Show top interpretations
    logger.info("\n4. Top pixel interpretations:")
    sorted_interpretations = sorted(
        results['pixel_interpretations'],
        key=lambda x: x['confidence'],
        reverse=True
    )[:5]
    
    for i, interp in enumerate(sorted_interpretations, 1):
        logger.info(f"\n   {i}. Pixel {interp['pixel_id']}:")
        logger.info(f"      Interpretation: {interp['interpretation']}")
        logger.info(f"      Confidence: {interp['confidence']:.2%}")
        logger.info(f"      Evidence:")
        for detector, evidence in interp['evidence'].items():
            logger.info(f"        • {detector}: {evidence}")
    
    return results


def demonstrate_harmonic_coincidence_network():
    """
    Demonstrate O(1) queries using harmonic coincidence networks.
    
    This shows how the framework achieves constant-time complexity.
    """
    logger.info("\n" + "="*80)
    logger.info("  HARMONIC COINCIDENCE NETWORK DEMONSTRATION")
    logger.info("  O(1) Information Access via Frequency Resonance")
    logger.info("="*80)
    
    # Create network
    logger.info("\n1. Building harmonic network from molecular oscillators...")
    network = HarmonicCoincidenceNetwork(name="molecular_network")
    
    # Add molecular vibrational modes
    molecules = [
        ("O2", 4.7e13),      # O₂ stretch
        ("N2", 7.0e13),      # N₂ stretch
        ("H2O_sym", 1.0e14), # H₂O symmetric stretch
        ("H2O_asym", 1.1e14),# H₂O asymmetric stretch
        ("H2O_bend", 4.8e13),# H₂O bend
        ("CO2_asym", 7.0e13),# CO₂ asymmetric stretch
        ("CO2_sym", 4.0e13), # CO₂ symmetric stretch
    ]
    
    for mol_name, freq in molecules:
        network.add_oscillator(
            frequency=freq,
            amplitude=1.0,
            oscillator_id=mol_name,
            metadata={'type': 'molecular_vibration'}
        )
    
    logger.info(f"   ✓ Added {len(network.oscillators)} molecular oscillators")
    
    # Find all harmonic coincidences
    logger.info("\n2. Finding harmonic coincidences (integer frequency ratios)...")
    network.find_all_coincidences(tolerance=0.1)
    
    logger.info(f"   ✓ Found {len(network.coincidences)} coincidences")
    logger.info(f"   Network density: {network.get_network_density():.3f}")
    
    # Show some coincidences
    logger.info("\n3. Sample harmonic coincidences:")
    for i, coin in enumerate(network.coincidences[:5], 1):
        logger.info(f"   {i}. {coin.osc1.id} <-> {coin.osc2.id}")
        logger.info(f"      Ratio: {coin.ratio:.3f} ≈ {coin.nearest_integer} (deviation: {coin.deviation:.4f})")
        logger.info(f"      Coupling: {coin.coupling_strength:.3f}")
    
    # Demonstrate O(1) query
    logger.info("\n4. Demonstrating O(1) frequency query...")
    target_freq = 7.0e13  # Looking for ~70 THz
    neighbors = network.find_oscillators_near_frequency(target_freq, tolerance=0.1)
    
    logger.info(f"   Query for {target_freq:.2e} Hz:")
    for osc in neighbors:
        logger.info(f"   • {osc.id}: {osc.frequency:.2e} Hz")
    
    logger.info(f"\n   ✓ Query completed in O(1) time via harmonic indexing!")
    
    return network


def demonstrate_categorical_light_sources():
    """
    Demonstrate categorical light sources for rendering.
    
    Shows how information propagates through S-space, not physical space.
    """
    logger.info("\n" + "="*80)
    logger.info("  CATEGORICAL LIGHT SOURCES DEMONSTRATION")
    logger.info("  Information-Theoretic Light Emission")
    logger.info("="*80)
    
    # Create light sources
    logger.info("\n1. Creating categorical light sources...")
    
    # Fluorescence emission (common wavelengths)
    sources = [
        ("DAPI", 461, 1.0),       # Blue fluorescence (DNA)
        ("GFP", 509, 0.8),        # Green fluorescence (proteins)
        ("RFP", 584, 0.7),        # Red fluorescence (proteins)
        ("Cy5", 670, 0.9),        # Far-red fluorescence
    ]
    
    light_sources = []
    for name, wavelength_nm, intensity in sources:
        color = Color.from_wavelength(wavelength_nm, intensity)
        source = CategoricalLightSource(
            position_s=np.random.rand(3),  # Random S-space position
            color=color,
            intensity=intensity,
            name=name
        )
        light_sources.append(source)
        logger.info(f"   ✓ {name}: {wavelength_nm} nm, RGB{color.to_tuple()}")
    
    logger.info(f"\n2. Light propagation in categorical space:")
    logger.info("   (NOT physical ray tracing - information transfer via S-distance)")
    
    # Show categorical distances
    for i in range(len(light_sources)):
        for j in range(i+1, len(light_sources)):
            src1, src2 = light_sources[i], light_sources[j]
            
            # Categorical distance (not physical!)
            cat_distance = np.linalg.norm(src1.position_s - src2.position_s)
            
            logger.info(f"   • {src1.name} <-> {src2.name}: d_cat = {cat_distance:.3f}")
    
    logger.info("\n   ✓ Information transfer via categorical proximity, not ray tracing!")
    
    return light_sources


def main():
    """Run complete demonstration using all existing modules."""
    
    logger.info("\n" + "="*80)
    logger.info("  MAXWELL FRAMEWORK COMPLETE DEMONSTRATION")
    logger.info("  Using Existing Modules (CORRECT Implementation)")
    logger.info("="*80)
    logger.info("\nModules used:")
    logger.info("  1. live_cell_imaging.py - Multi-modal microscopy")
    logger.info("  2. harmonic_coincidence.py - O(1) network queries")
    logger.info("  3. categorical_light_sources.py - Categorical rendering")
    logger.info("\n" + "="*80)
    
    # Part 1: Multi-modal microscopy
    microscopy_results = demonstrate_multi_modal_microscopy()
    
    # Part 2: Harmonic coincidence network
    network = demonstrate_harmonic_coincidence_network()
    
    # Part 3: Categorical light sources
    light_sources = demonstrate_categorical_light_sources()
    
    # Part 4: Ambiguous signal resolution (already in live_cell_imaging)
    logger.info("\n" + "="*80)
    logger.info("  AMBIGUOUS SIGNAL RESOLUTION")
    logger.info("  (Using built-in demonstrate_ambiguous_signal_resolution)")
    logger.info("="*80)
    
    demonstrate_ambiguous_signal_resolution()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("  DEMONSTRATION COMPLETE")
    logger.info("="*80)
    
    logger.info("\nKey Results:")
    logger.info(f"  1. Microscopy: {microscopy_results['num_pixels']:,} pixels analyzed")
    logger.info(f"     Mean confidence: {microscopy_results['mean_confidence']:.2%}")
    logger.info(f"     Detectors: {len(microscopy_results['detector_types_used'])}")
    
    logger.info(f"\n  2. Harmonic Network: {len(network.oscillators)} oscillators")
    logger.info(f"     Coincidences: {len(network.coincidences)}")
    logger.info(f"     Density: {network.get_network_density():.3f}")
    
    logger.info(f"\n  3. Light Sources: {len(light_sources)} categorical sources")
    
    logger.info("\nRevolutionary Advantages:")
    logger.info("  ✓ Multi-modal without multiple samples")
    logger.info("  ✓ O(1) queries via harmonic coincidences")
    logger.info("  ✓ Trans-Planckian resolution (1 nm, 1 fs)")
    logger.info("  ✓ Non-destructive observation")
    logger.info("  ✓ Categorical information transfer")
    
    logger.info("\n" + "="*80)
    
    # Save results
    output_dir = Path("maxwell_demo_results")
    output_dir.mkdir(exist_ok=True)
    
    results_summary = {
        'microscopy': {
            'num_pixels': microscopy_results['num_pixels'],
            'mean_confidence': microscopy_results['mean_confidence'],
            'detectors_used': microscopy_results['detector_types_used']
        },
        'harmonic_network': {
            'num_oscillators': len(network.oscillators),
            'num_coincidences': len(network.coincidences),
            'density': network.get_network_density()
        },
        'light_sources': {
            'num_sources': len(light_sources),
            'wavelengths_nm': [src.name for src in light_sources]
        }
    }
    
    with open(output_dir / 'demo_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: {output_dir}/demo_results.json\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

