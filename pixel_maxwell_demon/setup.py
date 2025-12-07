#!/usr/bin/env python3
"""
Pixel Maxwell Demon: Dual-Membrane Computer Vision Framework
=============================================================

A revolutionary computer vision framework that generates 3D information from 2D images
using pixel Maxwell demons and dual-membrane information processing.

Author: Kundai Sachikonye
Date: 2024
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="pixel-maxwell-demon",
    version="0.1.0",
    author="Kundai Sachikonye",
    description="Dual-Membrane Computer Vision Framework with Pixel Maxwell Demons",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pixel-maxwell-demon",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "opencv-python>=4.5.0",
        "scipy>=1.6.0",
        "scikit-image>=0.18.0",
        "pillow>=8.0.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.900",
        ],
    },
    entry_points={
        "console_scripts": [
            "pmd-visualize=maxwell.cli:visualize_npy",
            "pmd-validate=maxwell.cli:validate_life_sciences",
            "pmd-demo=maxwell.cli:run_demo",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="computer-vision, maxwell-demon, information-theory, image-processing, 3d-reconstruction",
)

