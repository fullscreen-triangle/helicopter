"""
Setup script for maxwell package.

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README_IMPLEMENTATION.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')
else:
    long_description = "Hardware-Constrained Categorical Completion Algorithm"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = [
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'scikit-image>=0.18.0',
        'opencv-python>=4.5.0',
        'matplotlib>=3.3.0',
        'networkx>=2.6.0'
    ]

setup(
    name='maxwell',
    version='1.0.0',
    description='Hardware-Constrained Categorical Completion for Image Understanding',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Kundai Farai Sachikonye',
    author_email='research@s-entropy.org',
    url='https://github.com/yourusername/maxwell',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.12.0',
            'black>=21.0',
            'mypy>=0.910',
            'flake8>=3.9.0',
        ],
        'hardware': [
            'pyserial>=3.5',
            'sounddevice>=0.4.0',
            'screen-brightness-control>=0.20.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'maxwell-demo=scripts.run_demo:main',
            'maxwell-benchmark=scripts.run_benchmark:main',
            'maxwell-process=scripts.process_image:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='computer-vision maxwell-demon bmd categorical-completion s-entropy',
    project_urls={
        'Documentation': 'https://github.com/yourusername/maxwell/blob/main/README_IMPLEMENTATION.md',
        'Source': 'https://github.com/yourusername/maxwell',
        'Tracker': 'https://github.com/yourusername/maxwell/issues',
    },
)
