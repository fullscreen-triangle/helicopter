#!/usr/bin/env python3
"""
Setup script for Helicopter Life Science Modules

This script helps you set up the environment and test your installation.
"""

import sys
import subprocess
import importlib.util
from pathlib import Path


def check_python_version():
    """Check if Python version is adequate"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 7):
        print(f"❌ Python 3.7+ required, found {sys.version}")
        return False
    
    print(f"✅ Python {sys.version} - OK")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'numpy',
        'scipy', 
        'cv2',           # opencv-python
        'skimage',       # scikit-image
        'sklearn',       # scikit-learn
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                importlib.import_module('cv2')
                print(f"  ✅ opencv-python")
            elif package == 'skimage':
                importlib.import_module('skimage')
                print(f"  ✅ scikit-image")
            elif package == 'sklearn':
                importlib.import_module('sklearn')
                print(f"  ✅ scikit-learn")
            else:
                importlib.import_module(package)
                print(f"  ✅ {package}")
        except ImportError:
            if package == 'cv2':
                missing_packages.append('opencv-python')
                print(f"  ❌ opencv-python")
            elif package == 'skimage':
                missing_packages.append('scikit-image')
                print(f"  ❌ scikit-image")
            elif package == 'sklearn':
                missing_packages.append('scikit-learn')
                print(f"  ❌ scikit-learn")
            else:
                missing_packages.append(package)
                print(f"  ❌ {package}")
    
    return missing_packages


def install_dependencies(missing_packages):
    """Install missing dependencies"""
    if not missing_packages:
        return True
    
    print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
    print("Installing missing dependencies...")
    
    try:
        # Use pip to install missing packages
        cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Please install manually using:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def check_data_directory():
    """Check if data directory exists and has files"""
    print("\n📂 Checking data directory...")
    
    data_dir = Path(__file__).parent / "public"
    
    if not data_dir.exists():
        print(f"⚠️  Data directory not found: {data_dir}")
        print("Creating data directory...")
        data_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Data directory created")
        print("📝 Place your microscopy images and videos in lifescience/public/")
        return False
    
    # Count files
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mpg', '.mpeg'}
    archive_extensions = {'.zip', '.tar', '.gz'}
    
    image_count = sum(1 for f in data_dir.iterdir() if f.suffix.lower() in image_extensions)
    video_count = sum(1 for f in data_dir.iterdir() if f.suffix.lower() in video_extensions)
    archive_count = sum(1 for f in data_dir.iterdir() if f.suffix.lower() in archive_extensions)
    
    print(f"📁 Data directory: {data_dir}")
    print(f"   Images: {image_count}")
    print(f"   Videos: {video_count}")
    print(f"   Archives: {archive_count}")
    
    if image_count + video_count + archive_count == 0:
        print("⚠️  No data files found")
        print("📝 Place your microscopy data in lifescience/public/")
        return False
    
    print("✅ Data files found")
    return True


def test_imports():
    """Test if lifescience modules can be imported"""
    print("\n🔍 Testing module imports...")
    
    modules_to_test = [
        ('src.gas', 'Gas molecular dynamics'),
        ('src.entropy', 'S-entropy framework'),
        ('src.fluorescence', 'Fluorescence microscopy'),
        ('src.electron', 'Electron microscopy'),
        ('src.video', 'Video analysis'),
        ('src.meta', 'Meta-information extraction')
    ]
    
    failed_imports = []
    
    for module_name, description in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"  ✅ {description}")
        except ImportError as e:
            print(f"  ❌ {description}: {e}")
            failed_imports.append((module_name, description))
    
    return len(failed_imports) == 0


def create_results_directory():
    """Create results directory"""
    print("\n📁 Setting up results directory...")
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    print(f"✅ Results directory: {results_dir}")
    return results_dir


def show_next_steps():
    """Show next steps for the user"""
    print("\n🎯 Next Steps:")
    print("=" * 50)
    
    print("\n1. 📝 Configure your data paths:")
    print("   • Edit lifescience/config.py")
    print("   • Update MICROSCOPY_IMAGES, MICROSCOPY_VIDEOS paths")
    print("   • Adjust analysis parameters if needed")
    
    print("\n2. 🧪 Test your setup:")
    print("   python demo_quick_test.py")
    
    print("\n3. 🚀 Run analysis:")
    print("   python demo_all_modules.py        # Complete analysis")
    print("   python demo_fluorescence.py       # Fluorescence only")
    print("   python demo_video.py              # Video analysis only")
    
    print("\n4. 📊 Check results:")
    print("   • Results saved in lifescience/results/")
    print("   • Visualizations as PNG files")
    print("   • Analysis data printed to console")
    
    print("\n💡 Tips:")
    print("   • Start with demo_quick_test.py to verify everything works")
    print("   • Modify config.py to match your specific data types")
    print("   • Check individual demo scripts for focused analysis")


def main():
    """Main setup function"""
    print("🚁 Helicopter Life Science Framework - Setup")
    print("=" * 60)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Check dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        if input(f"\nInstall missing packages? (y/N): ").lower().strip() == 'y':
            if not install_dependencies(missing_packages):
                success = False
        else:
            print("⚠️  Some dependencies are missing. Manual installation required.")
            success = False
    
    # Test imports
    if success and not test_imports():
        print("❌ Module import test failed!")
        success = False
    
    # Check data directory
    has_data = check_data_directory()
    
    # Create results directory
    create_results_directory()
    
    # Summary
    print("\n" + "=" * 60)
    
    if success:
        print("🎉 Setup Complete!")
        if has_data:
            print("✅ Your Helicopter Life Science framework is ready to use!")
        else:
            print("⚠️  Setup successful, but add your data files to continue.")
    else:
        print("❌ Setup incomplete. Please resolve the issues above.")
    
    show_next_steps()


if __name__ == "__main__":
    main()
