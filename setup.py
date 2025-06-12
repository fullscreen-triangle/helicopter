#!/usr/bin/env python3
"""
Setup script for Helicopter - Reverse Pakati for Visual Knowledge Extraction
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Version
version = "0.1.0"

setup(
    name="helicopter",
    version=version,
    author="Fullscreen Triangle",
    author_email="dev@fullscreen-triangle.com",
    description="Reverse Pakati for Visual Knowledge Extraction - Convert images to domain-specific LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fullscreen-triangle/helicopter",
    project_urls={
        "Bug Tracker": "https://github.com/fullscreen-triangle/helicopter/issues",
        "Documentation": "https://helicopter.readthedocs.io/",
        "Source Code": "https://github.com/fullscreen-triangle/helicopter",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "diffusers>=0.21.0",
        "accelerate>=0.20.0",
        
        # Computer vision
        "opencv-python>=4.8.0",
        "pillow>=9.5.0",
        "scikit-image>=0.20.0",
        "albumentations>=1.3.0",
        
        # Scientific computing
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        
        # Neural networks and ML
        "timm>=0.9.0",
        "sentence-transformers>=2.2.0",
        "clip-by-openai>=1.0",
        
        # Utilities
        "tqdm>=4.65.0",
        "rich>=13.0.0",
        "typer>=0.9.0",
        "pydantic>=2.0.0",
        "omegaconf>=2.3.0",
        
        # Visualization
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        
        # Data handling
        "h5py>=3.9.0",
        "datasets>=2.14.0",
        "jsonlines>=3.1.0",
        
        # Web API
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "python-multipart>=0.0.6",
        
        # Logging and monitoring
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-xdist>=3.3.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
            "bandit>=1.7.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.24.0",
            "myst-parser>=2.0.0",
            "nbsphinx>=0.9.0",
        ],
        "experimental": [
            # NeRF and 3D processing
            "torch-ngp>=0.1.0",
            "nerfstudio>=0.3.0",
            
            # Advanced ML
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
            "optuna>=3.3.0",
            
            # Causal inference
            "causal-learn>=0.1.0",
            "dowhy>=0.11.0",
            
            # Graph neural networks
            "torch-geometric>=2.3.0",
            "networkx>=3.1.0",
        ],
        "full": [
            # All extras combined
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-xdist>=3.3.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
            "bandit>=1.7.0",
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.24.0",
            "myst-parser>=2.0.0",
            "nbsphinx>=0.9.0",
            "torch-ngp>=0.1.0",
            "nerfstudio>=0.3.0",
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
            "optuna>=3.3.0",
            "causal-learn>=0.1.0",
            "dowhy>=0.11.0",
            "torch-geometric>=2.3.0",
            "networkx>=3.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "helicopter=helicopter.cli.main:app",
            "helicopter-server=helicopter.api.server:main",
            "helicopter-train=helicopter.training.cli:app",
        ],
    },
    include_package_data=True,
    package_data={
        "helicopter": [
            "configs/*.yaml",
            "configs/*.json",
            "assets/*",
            "templates/*",
        ]
    },
    zip_safe=False,
    keywords=[
        "computer-vision",
        "machine-learning",
        "artificial-intelligence",
        "diffusion-models",
        "vision-language-models",
        "domain-specific-llms",
        "visual-tokenization",
        "reverse-diffusion",
        "metacognitive-ai",
    ],
)
