#!/usr/bin/env python3
"""
Test that all imports work correctly.

Run this to verify the module is properly installed.
"""

import sys
from pathlib import Path

# Add src to path (for development)
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("="*80)
print("  TESTING PIXEL MAXWELL DEMON IMPORTS")
print("="*80)

tests_passed = 0
tests_failed = 0

# Test 1: Core Maxwell module
print("\n[1/8] Testing maxwell.pixel_maxwell_demon...")
try:
    from maxwell.pixel_maxwell_demon import PixelMaxwellDemon, SEntropyCoordinates
    print("  ✓ pixel_maxwell_demon imports OK")
    tests_passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    tests_failed += 1

# Test 2: Dual membrane
print("\n[2/8] Testing maxwell.dual_membrane_pixel_demon...")
try:
    from maxwell.dual_membrane_pixel_demon import DualMembranePixelDemon, DualState
    print("  ✓ dual_membrane_pixel_demon imports OK")
    tests_passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    tests_failed += 1

# Test 3: Simple pixel grid
print("\n[3/8] Testing maxwell.simple_pixel_grid...")
try:
    from maxwell.simple_pixel_grid import PixelDemonGrid
    print("  ✓ simple_pixel_grid imports OK")
    tests_passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    tests_failed += 1

# Test 4: Categorical light sources
print("\n[4/8] Testing maxwell.categorical_light_sources...")
try:
    from maxwell.categorical_light_sources import Color
    print("  ✓ categorical_light_sources imports OK")
    tests_passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    tests_failed += 1

# Test 5: Live cell imaging
print("\n[5/8] Testing maxwell.live_cell_imaging...")
try:
    from maxwell.live_cell_imaging import LiveCellMicroscope
    print("  ✓ live_cell_imaging imports OK")
    tests_passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    tests_failed += 1

# Test 6: Virtual detectors
print("\n[6/8] Testing maxwell.virtual_detectors...")
try:
    from maxwell.virtual_detectors import VirtualDetector
    print("  ✓ virtual_detectors imports OK")
    tests_passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    tests_failed += 1

# Test 7: Integration module
print("\n[7/8] Testing maxwell.integration...")
try:
    from maxwell.integration import DualMembraneBMDState, DualMembraneHCCCAlgorithm
    print("  ✓ integration module imports OK")
    tests_passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    tests_failed += 1

# Test 8: Standard library imports
print("\n[8/8] Testing standard dependencies...")
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    print("  ✓ numpy, matplotlib, opencv imports OK")
    tests_passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    tests_failed += 1

# Summary
print("\n" + "="*80)
print("  RESULTS")
print("="*80)
print(f"\nTests passed: {tests_passed}/{tests_passed + tests_failed}")
print(f"Tests failed: {tests_failed}/{tests_passed + tests_failed}")

if tests_failed == 0:
    print("\n✓ ALL TESTS PASSED!")
    print("\nYou're ready to run the analysis scripts:")
    print("  python demo_virtual_imaging.py ../maxwell/public/1585.jpg")
    print("  python visualize_npy_results.py --search-dir ../maxwell")
    print("  python validate_life_sciences_multi_modal.py --max-images 5")
else:
    print("\n✗ SOME TESTS FAILED")
    print("\nTroubleshooting:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Install package: pip install -e .")
    print("  3. Check Python version: python --version (need ≥ 3.8)")

print("="*80 + "\n")

sys.exit(0 if tests_failed == 0 else 1)

