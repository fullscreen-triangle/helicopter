#!/usr/bin/env python3
"""
Quick Test Script for Helicopter Life Science Framework

This script quickly tests if all modules are working correctly with your data.
Perfect for debugging and initial testing.

Usage:
    python demo_quick_test.py
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add the lifescience package to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_valid_files


def test_imports():
    """Test if all modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        from src.gas import BiologicalGasAnalyzer
        print("  ✅ Gas molecular dynamics")
    except ImportError as e:
        print(f"  ❌ Gas molecular dynamics: {e}")
        return False
    
    try:
        from src.entropy import SEntropyTransformer
        print("  ✅ S-entropy framework")
    except ImportError as e:
        print(f"  ❌ S-entropy framework: {e}")
        return False
    
    try:
        from src.flourescence import FluorescenceAnalyzer
        print("  ✅ Fluorescence microscopy")
    except ImportError as e:
        print(f"  ❌ Fluorescence microscopy: {e}")
        return False
    
    try:
        from src.electron import ElectronMicroscopyAnalyzer
        print("  ✅ Electron microscopy")
    except ImportError as e:
        print(f"  ❌ Electron microscopy: {e}")
        return False
    
    try:
        from src.video import VideoAnalyzer
        print("  ✅ Video analysis")
    except ImportError as e:
        print(f"  ❌ Video analysis: {e}")
        return False
    
    try:
        from src.meta import MetaInformationExtractor
        print("  ✅ Meta-information extraction")
    except ImportError as e:
        print(f"  ❌ Meta-information extraction: {e}")
        return False
    
    return True


def test_data_loading():
    """Test if data files can be loaded"""
    print("\n📂 Testing data loading...")
    
    valid_images, valid_videos, valid_archives = get_valid_files()
    
    print(f"  Found {len(valid_images)} images")
    print(f"  Found {len(valid_videos)} videos") 
    print(f"  Found {len(valid_archives)} archives")
    
    if not valid_images and not valid_videos:
        print("  ❌ No valid data files found!")
        print("  Check your paths in config.py")
        return False
    
    # Test loading one image
    if valid_images:
        image_name, image_path = next(iter(valid_images.items()))
        try:
            image = cv2.imread(str(image_path))
            if image is not None:
                print(f"  ✅ Successfully loaded test image: {image_name} ({image.shape})")
            else:
                print(f"  ❌ Could not load test image: {image_name}")
                return False
        except Exception as e:
            print(f"  ❌ Error loading test image: {e}")
            return False
    
    # Test loading one video
    if valid_videos:
        video_name, video_path = next(iter(valid_videos.items()))
        try:
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  ✅ Successfully loaded test video: {video_name} ({frame.shape})")
            else:
                print(f"  ❌ Could not load test video: {video_name}")
                return False
            cap.release()
        except Exception as e:
            print(f"  ❌ Error loading test video: {e}")
            return False
    
    return True


def test_basic_analysis():
    """Test basic analysis on one image"""
    print("\n🧪 Testing basic analysis...")
    
    valid_images, _, _ = get_valid_files()
    if not valid_images:
        print("  ⏭️  Skipping - no images available")
        return True
    
    # Get first available image
    image_name, image_path = next(iter(valid_images.items()))
    image = cv2.imread(str(image_path))
    
    if image is None:
        print("  ❌ Could not load test image")
        return False
    
    try:
        # Test S-entropy analysis (simplest)
        from src.entropy import SEntropyTransformer
        transformer = SEntropyTransformer()
        coordinates = transformer.transform(image)
        print(f"  ✅ S-entropy analysis successful")
        print(f"     Coordinates: [{coordinates.structural:.3f}, {coordinates.functional:.3f}, {coordinates.morphological:.3f}, {coordinates.temporal:.3f}]")
        
    except Exception as e:
        print(f"  ❌ S-entropy analysis failed: {e}")
        return False
    
    try:
        # Test meta-information extraction
        from src.meta import MetaInformationExtractor
        extractor = MetaInformationExtractor()
        meta_info = extractor.extract_meta_information(image)
        print(f"  ✅ Meta-information analysis successful")
        print(f"     Type: {meta_info.information_type.value}, Compression: {meta_info.compression_potential:.3f}")
        
    except Exception as e:
        print(f"  ❌ Meta-information analysis failed: {e}")
        return False
    
    return True


def main():
    """Run quick test"""
    print("🚁 Helicopter Life Science Framework - Quick Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed!")
        print("Make sure you're running from the lifescience directory")
        return
    
    # Test data loading
    if not test_data_loading():
        print("\n❌ Data loading test failed!")
        print("Check your file paths in config.py")
        return
    
    # Test basic analysis
    if not test_basic_analysis():
        print("\n❌ Basic analysis test failed!")
        return
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Your Helicopter Life Science framework is ready to use!")
    print("\nNext steps:")
    print("  • Run 'python demo_all_modules.py' for complete analysis")
    print("  • Modify config.py to customize your analysis")
    print("  • Check individual demo scripts for focused analysis")


if __name__ == "__main__":
    main()
