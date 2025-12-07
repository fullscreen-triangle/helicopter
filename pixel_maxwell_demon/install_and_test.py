#!/usr/bin/env python3
"""
Install and Test Pixel Maxwell Demon Package
===========================================

This script:
1. Reinstalls the package
2. Tests all imports
3. Creates quick-start scripts

Run this after making changes to the source code.
"""

import subprocess
import sys
from pathlib import Path

print("="*80)
print("  PIXEL MAXWELL DEMON: INSTALL AND TEST")
print("="*80)

# Step 1: Reinstall package
print("\n[1/3] Reinstalling package...")
try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--force-reinstall", "--no-deps"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("  ✓ Package reinstalled successfully")
    else:
        print(f"  ✗ Installation failed: {result.stderr}")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Installation error: {e}")
    sys.exit(1)

# Step 2: Test imports
print("\n[2/3] Testing imports...")
sys.path.insert(0, str(Path(__file__).parent / 'src'))

tests_passed = 0
tests_failed = 0

# Test core imports
try:
    from maxwell.pixel_maxwell_demon import PixelMaxwellDemon, SEntropyCoordinates
    from maxwell.simple_pixel_grid import PixelDemonGrid
    from maxwell.dual_membrane_pixel_demon import DualMembranePixelDemon, DualState
    print("  ✓ Core modules OK")
    tests_passed += 1
except Exception as e:
    print(f"  ✗ Core modules FAILED: {e}")
    tests_failed += 1

# Test auxiliary modules
try:
    from maxwell.categorical_light_sources import Color
    from maxwell.live_cell_imaging import LiveCellMicroscope
    from maxwell.virtual_detectors import VirtualDetector
    print("  ✓ Auxiliary modules OK")
    tests_passed += 1
except Exception as e:
    print(f"  ✗ Auxiliary modules FAILED: {e}")
    tests_failed += 1

# Test standard dependencies
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    print("  ✓ Standard dependencies OK")
    tests_passed += 1
except Exception as e:
    print(f"  ✗ Standard dependencies FAILED: {e}")
    tests_failed += 1

# Step 3: Create quick-start scripts
print("\n[3/3] Creating quick-start scripts...")

quick_scripts = {
    "run_demo.py": """#!/usr/bin/env python3
# Quick demo runner
import sys
sys.exit(__import__('demo_virtual_imaging').main())
""",
    "run_visualize.py": """#!/usr/bin/env python3
# Quick visualization runner
import sys
sys.exit(__import__('visualize_npy_results').main())
""",
    "run_validate.py": """#!/usr/bin/env python3
# Quick validation runner
import sys
sys.exit(__import__('validate_life_sciences_multi_modal').main())
"""
}

for script_name, script_content in quick_scripts.items():
    script_path = Path(__file__).parent / script_name
    script_path.write_text(script_content)
    print(f"  ✓ Created {script_name}")

# Summary
print("\n" + "="*80)
print("  RESULTS")
print("="*80)
print(f"\nTests passed: {tests_passed}/{tests_passed + tests_failed}")
print(f"Tests failed: {tests_failed}/{tests_passed + tests_failed}")

if tests_failed == 0:
    print("\n✓ INSTALLATION SUCCESSFUL!")
    print("\nQuick Start:")
    print("  python demo_virtual_imaging.py ../maxwell/public/1585.jpg")
    print("  python visualize_npy_results.py --search-dir ../maxwell")
    print("  python validate_life_sciences_multi_modal.py --max-images 3")
    print("\nOr use the quick-run scripts:")
    print("  python run_demo.py ../maxwell/public/1585.jpg")
    print("  python run_visualize.py --search-dir ../maxwell")
else:
    print("\n✗ SOME TESTS FAILED")
    print("\nPlease check the error messages above.")

print("="*80 + "\n")

sys.exit(0 if tests_failed == 0 else 1)

