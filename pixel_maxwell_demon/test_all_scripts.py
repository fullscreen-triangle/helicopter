#!/usr/bin/env python3
"""
Test All Analysis Scripts
========================

Tests all three main analysis scripts with a single command.

Usage:
    python test_all_scripts.py
"""

import subprocess
import sys
from pathlib import Path

print("="*80)
print("  TESTING ALL ANALYSIS SCRIPTS")
print("="*80)

# Determine base directory
base_dir = Path(__file__).parent
public_dir = base_dir.parent / 'maxwell' / 'public'

# Check if public directory exists
if not public_dir.exists():
    print(f"\n✗ Public directory not found: {public_dir}")
    print(f"\nPlease ensure maxwell/public/ contains test images.")
    sys.exit(1)

# Find a test image
test_images = list(public_dir.glob('*.jpg')) + list(public_dir.glob('*.png'))
if not test_images:
    print(f"\n✗ No test images found in: {public_dir}")
    sys.exit(1)

test_image = test_images[0]
print(f"\n✓ Found test image: {test_image.name}")

# Test 1: Virtual Imaging Demo
print(f"\n{'='*80}")
print("TEST 1: Virtual Imaging Demo")
print(f"{'='*80}")
print(f"Running: python demo_virtual_imaging.py {test_image}")

try:
    result = subprocess.run(
        [sys.executable, "demo_virtual_imaging.py", str(test_image), "--output-dir", "test_virtual_results"],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=120
    )
    
    if result.returncode == 0:
        print("✓ Virtual imaging demo PASSED")
        test1_passed = True
    else:
        print("✗ Virtual imaging demo FAILED")
        print(f"Error: {result.stderr[-500:]}")  # Last 500 chars
        test1_passed = False
except subprocess.TimeoutExpired:
    print("✗ Virtual imaging demo TIMEOUT")
    test1_passed = False
except Exception as e:
    print(f"✗ Virtual imaging demo ERROR: {e}")
    test1_passed = False

# Test 2: Life Sciences Validation
print(f"\n{'='*80}")
print("TEST 2: Life Sciences Validation")
print(f"{'='*80}")
print(f"Running: python validate_life_sciences_multi_modal.py --max-images 1")

try:
    result = subprocess.run(
        [sys.executable, "validate_life_sciences_multi_modal.py", "--max-images", "1", "--output-dir", "test_validation_results"],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode == 0:
        print("✓ Life sciences validation PASSED")
        test2_passed = True
    else:
        print("✗ Life sciences validation FAILED")
        print(f"Error: {result.stderr[-500:]}")
        test2_passed = False
except subprocess.TimeoutExpired:
    print("✗ Life sciences validation TIMEOUT")
    test2_passed = False
except Exception as e:
    print(f"✗ Life sciences validation ERROR: {e}")
    test2_passed = False

# Test 3: NPY Visualization
print(f"\n{'='*80}")
print("TEST 3: NPY Visualization")
print(f"{'='*80}")
print(f"Running: python visualize_npy_results.py --search-dir test_virtual_results")

try:
    result = subprocess.run(
        [sys.executable, "visualize_npy_results.py", "--search-dir", "test_virtual_results", "--output-dir", "test_viz_results"],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode == 0:
        print("✓ NPY visualization PASSED")
        test3_passed = True
    else:
        print("✗ NPY visualization FAILED")
        print(f"Error: {result.stderr[-500:]}")
        test3_passed = False
except subprocess.TimeoutExpired:
    print("✗ NPY visualization TIMEOUT")
    test3_passed = False
except Exception as e:
    print(f"✗ NPY visualization ERROR: {e}")
    test3_passed = False

# Summary
print(f"\n{'='*80}")
print("  TEST SUMMARY")
print(f"{'='*80}")

tests = [
    ("Virtual Imaging Demo", test1_passed),
    ("Life Sciences Validation", test2_passed),
    ("NPY Visualization", test3_passed)
]

passed = sum(1 for _, p in tests if p)
total = len(tests)

for name, passed_test in tests:
    status = "✓ PASS" if passed_test else "✗ FAIL"
    print(f"  {name}: {status}")

print(f"\nTotal: {passed}/{total} tests passed")

if passed == total:
    print(f"\n🎉 ALL TESTS PASSED!")
    print(f"\nYou can now run the analysis scripts:")
    print(f"  python demo_virtual_imaging.py ../maxwell/public/1585.jpg")
    print(f"  python validate_life_sciences_multi_modal.py --max-images 3")
    print(f"  python visualize_npy_results.py --search-dir .")
else:
    print(f"\n⚠ SOME TESTS FAILED")
    print(f"\nPlease check the error messages above.")
    print(f"\nCommon issues:")
    print(f"  1. Package not installed: run 'python install_and_test.py'")
    print(f"  2. Missing dependencies: run 'pip install -r requirements.txt'")
    print(f"  3. Image directory wrong: check '../maxwell/public/' exists")

print("="*80 + "\n")

sys.exit(0 if passed == total else 1)

