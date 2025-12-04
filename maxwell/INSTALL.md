# Installation Guide

Complete installation instructions for the Maxwell package.

## System Requirements

### Operating System

- Linux (Ubuntu 20.04+, Debian 10+, etc.)
- macOS (10.15+)
- Windows 10/11 (with WSL recommended)

### Python Version

- Python 3.8 or higher
- Python 3.9 recommended
- Python 3.10, 3.11 supported

### Hardware

- Minimum: 4GB RAM, 2 CPU cores
- Recommended: 8GB+ RAM, 4+ CPU cores
- GPU: Optional (for future GPU acceleration)

## Installation Methods

### Method 1: Basic Installation (Recommended)

```bash
# Clone repository (if from git)
git clone https://github.com/yourusername/maxwell.git
cd maxwell

# Install package
pip install -e .
```

This installs:

- Core dependencies (NumPy, SciPy, scikit-image, OpenCV, etc.)
- Maxwell package in editable mode
- Command-line scripts

### Method 2: Development Installation

For contributors and developers:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or using Make
make install-dev
```

This adds:

- pytest, pytest-cov (testing)
- black (code formatting)
- mypy (type checking)
- flake8 (linting)

### Method 3: Complete Installation

With all optional dependencies:

```bash
# Install everything
pip install -e ".[dev,hardware]"

# Or using Make
make install-all
```

This adds:

- Development dependencies
- Hardware sensor libraries (pyserial, sounddevice, etc.)

### Method 4: Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install
pip install -e .
```

### Method 5: Conda Environment

```bash
# Create conda environment
conda create -n maxwell python=3.9
conda activate maxwell

# Install dependencies
conda install numpy scipy scikit-image opencv matplotlib networkx pandas

# Install maxwell
pip install -e .
```

## Dependency Installation

### Core Dependencies

```bash
pip install numpy>=1.20.0 \
            scipy>=1.7.0 \
            scikit-image>=0.18.0 \
            scikit-learn>=1.0.0 \
            opencv-python>=4.5.0 \
            matplotlib>=3.3.0 \
            networkx>=2.6.0 \
            pandas>=1.3.0 \
            pillow>=8.0.0
```

### Optional Hardware Dependencies

```bash
# For real hardware sensor support
pip install pyserial>=3.5 \
            sounddevice>=0.4.0 \
            screen-brightness-control>=0.20.0
```

### Development Dependencies

```bash
# For development and testing
pip install pytest>=6.0.0 \
            pytest-cov>=2.12.0 \
            black>=21.0 \
            mypy>=0.910 \
            flake8>=3.9.0
```

## Platform-Specific Instructions

### Ubuntu/Debian Linux

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip python3-venv \
                     libopencv-dev python3-opencv \
                     build-essential

# Install Maxwell
pip install -e .
```

### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.9

# Install OpenCV
brew install opencv

# Install Maxwell
pip3 install -e .
```

### Windows

#### Option 1: WSL (Recommended)

```bash
# Install WSL Ubuntu
wsl --install

# Follow Ubuntu instructions above
```

#### Option 2: Native Windows

```powershell
# Install Python from python.org
# Install Visual C++ Build Tools

# In PowerShell
pip install -e .
```

## Verification

After installation, verify it works:

```bash
# Check installation
python -c "import maxwell; print(maxwell.__version__)"

# Run quick test
python -m pytest tests/ -v

# Run demo (may take a few minutes)
python demo_hccc_vision.py
```

Expected output:

```
maxwell version: 1.0.0
Tests: PASSED
Demo: Processing complete!
```

## Troubleshooting

### Import Error: "No module named 'maxwell'"

Solution:

```bash
# Ensure you're in the maxwell directory
cd /path/to/maxwell

# Install in editable mode
pip install -e .
```

### OpenCV Import Error

Solution:

```bash
# Try headless version
pip uninstall opencv-python
pip install opencv-python-headless
```

### Permission Errors

Solution:

```bash
# Use user installation
pip install --user -e .

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e .
```

### NumPy/SciPy Build Errors

Solution:

```bash
# Install pre-built wheels
pip install --upgrade pip
pip install numpy scipy --only-binary :all:

# Then install maxwell
pip install -e .
```

### Hardware Sensor Errors

If you don't need hardware sensors:

```bash
# Install without hardware extras
pip install -e .
```

If you need hardware sensors but getting errors:

```bash
# Install each dependency individually
pip install pyserial
pip install sounddevice
# etc.
```

## Configuration

After installation, configure the package:

```bash
# Copy example config
cp config.yaml config.local.yaml

# Edit configuration
nano config.local.yaml
```

Or use environment variables:

```bash
export MAXWELL_ALGORITHM_LAMBDA_STREAM=0.7
export MAXWELL_SEGMENTATION_N_SEGMENTS=100
```

## Testing Installation

### Quick Test

```bash
python -c "
from maxwell import HCCCAlgorithm, HardwareBMDStream
from maxwell.validation import BenchmarkSuite

print('Imports successful!')

benchmark = BenchmarkSuite()
image = benchmark.generate_synthetic_image('geometric', size=(64, 64))
print(f'Generated test image: {image.shape}')

hardware_stream = HardwareBMDStream()
hccc = HCCCAlgorithm(hardware_stream=hardware_stream, max_iterations=2)
print('Algorithm initialized!')

results = hccc.process_image(image, segmentation_params={'n_segments': 5})
print(f'Processing complete: {results[\"convergence_step\"]} iterations')

print('✓ Installation verified!')
"
```

### Full Test Suite

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Uninstallation

```bash
# Uninstall package
pip uninstall maxwell

# Remove virtual environment (if used)
rm -rf venv/

# Remove cache files
make clean
```

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Review [QUICK_START.md](QUICK_START.md) for usage examples
3. Check [GitHub Issues](https://github.com/yourusername/maxwell/issues)
4. Email: <research@s-entropy.org>

## Next Steps

After installation:

1. Read [QUICK_START.md](QUICK_START.md) for basic usage
2. Run `python demo_hccc_vision.py` for a complete demo
3. Review [README_IMPLEMENTATION.md](README_IMPLEMENTATION.md) for architecture details
4. Try processing your own images with `python -m scripts.process_image`

---

Happy computing with Maxwell! 🔬
