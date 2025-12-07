#!/usr/bin/env python3
"""
Command-line interface for Pixel Maxwell Demon analysis tools.
"""

import sys
import argparse
from pathlib import Path


def visualize_npy():
    """CLI entry point for visualize_npy_results."""
    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from visualize_npy_results import main
    sys.exit(main())


def validate_life_sciences():
    """CLI entry point for life sciences validation."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from validate_life_sciences_multi_modal import main
    sys.exit(main())


def run_demo():
    """CLI entry point for virtual imaging demo."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from demo_virtual_imaging import main
    sys.exit(main())


if __name__ == '__main__':
    print("Use: pmd-visualize, pmd-validate, or pmd-demo")

