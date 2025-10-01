#!/usr/bin/env python3
"""
Fluorescence Microscopy Demo

Focused demo for fluorescence microscopy analysis.
Configure your fluorescence images in config.py and run this script.

Usage:
    python demo_fluorescence.py
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add the lifescience package to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_valid_files, ensure_output_dir, SAVE_FIGURES, SHOW_FIGURES
from src.fluorescence import FluorescenceAnalyzer, FluorescenceChannel


def analyze_fluorescence_image(image_path, channel=FluorescenceChannel.GFP, output_dir=None):
    """Analyze a single fluorescence image with comprehensive metrics and JSON output"""
    print(f"🔬 Analyzing fluorescence image: {image_path.name}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None, None
    
    print(f"📁 Image loaded: {image.shape}")
    
    # Initialize analyzer with realistic pixel size
    analyzer = FluorescenceAnalyzer(pixel_size_um=0.065)  # Typical microscopy pixel size
    
    # Run comprehensive analysis with time series
    results = analyzer.analyze_image(image, channel, enable_time_series=True, num_time_points=50)
    
    # Save JSON results if output directory provided
    if output_dir:
        saved_files = analyzer.save_comprehensive_results(results, output_dir, f"{image_path.stem}_{channel.value}")
        print(f"💾 JSON results saved: {list(saved_files.values())}")
    
    # Print comprehensive results
    print(f"\n📊 Comprehensive Analysis Results ({channel.value}):")
    print(f"   Structures detected: {results['num_structures']}")
    print(f"   Processing time: {results.get('processing_time', 0):.2f}s")
    
    # Show segmentation performance
    if 'segmentation_dice' in results:
        print(f"   Segmentation Dice: {results['segmentation_dice']:.3f}")
        print(f"   Segmentation IoU: {results['segmentation_iou']:.3f}")
        print(f"   Pixel accuracy: {results['pixel_accuracy']:.3f}")
    
    if results['structures']:
        # Show details for first few structures with enhanced metrics
        for i, structure in enumerate(results['structures'][:3]):
            print(f"\n   Structure {structure['id']}:")
            print(f"     Mean intensity: {structure['mean_intensity']:.1f} AU")
            print(f"     Max intensity: {structure['max_intensity']:.1f} AU")
            print(f"     Area: {structure['area_pixels']:.0f} pixels ({structure['area_um2']:.2f} μm²)")
            print(f"     Signal-to-noise: {structure['signal_to_noise']:.2f}")
            print(f"     Contrast: {structure['contrast']:.2f}")
            print(f"     Eccentricity: {structure['eccentricity']:.3f}")
            print(f"     Solidity: {structure['solidity']:.3f}")
    
    # Enhanced summary statistics
    summary = results.get('summary', {})
    print(f"\n📈 Enhanced Summary Statistics:")
    print(f"   Total intensity: {summary.get('total_intensity', 0):.0f} AU")
    print(f"   Mean area: {summary.get('mean_area', 0):.1f} pixels")
    print(f"   Median area: {summary.get('median_area', 0):.1f} pixels")
    print(f"   Mean SNR: {summary.get('mean_snr', 0):.2f}")
    print(f"   Median SNR: {summary.get('median_snr', 0):.2f}")
    
    # Intensity distribution
    intensity_dist = summary.get('intensity_distribution', {})
    if intensity_dist:
        print(f"   Intensity range: {intensity_dist.get('min', 0):.1f} - {intensity_dist.get('max', 0):.1f} AU")
        print(f"   Intensity std: {intensity_dist.get('std', 0):.1f} AU")
    
    # Time series results
    comprehensive_metrics = results.get('comprehensive_metrics')
    if comprehensive_metrics and hasattr(comprehensive_metrics, 'time_series_data') and comprehensive_metrics.time_series_data:
        time_series = comprehensive_metrics.time_series_data
        print(f"\n⏱️  Time Series Analysis:")
        print(f"   Data points: {len(time_series.get('fluorescence_intensity', []))}")
        if time_series['fluorescence_intensity']:
            intensities = time_series['fluorescence_intensity']
            print(f"   Mean intensity over time: {np.mean(intensities):.1f} AU")
            print(f"   Photobleaching detected: {intensities[0] > intensities[-1]}")
            print(f"   SNR over time: {np.mean(time_series['signal_to_noise']):.2f}")
    
    return analyzer, results


def analyze_multi_channel(images_dict):
    """Analyze multiple channels for colocalization"""
    print(f"\n🌈 Multi-Channel Analysis ({len(images_dict)} channels)")
    
    # Load images
    loaded_images = {}
    for channel_name, image_path in images_dict.items():
        image = cv2.imread(str(image_path))
        if image is not None:
            loaded_images[channel_name] = image
            print(f"   ✅ {channel_name}: {image_path.name}")
        else:
            print(f"   ❌ {channel_name}: Failed to load {image_path.name}")
    
    if len(loaded_images) < 2:
        print("❌ Need at least 2 valid images for multi-channel analysis")
        return None
    
    # Map to fluorescence channels
    channel_map = {
        'dapi': FluorescenceChannel.DAPI,
        'gfp': FluorescenceChannel.GFP, 
        'rfp': FluorescenceChannel.RFP,
        'fitc': FluorescenceChannel.FITC
    }
    
    # Convert to channel enum mapping
    channel_images = {}
    for name, image in loaded_images.items():
        # Try to guess channel from name or use GFP as default
        channel_key = name.lower()
        if channel_key in channel_map:
            channel_images[channel_map[channel_key]] = image
        else:
            channel_images[FluorescenceChannel.GFP] = image
    
    # Run multi-channel analysis
    analyzer = FluorescenceAnalyzer()
    results = analyzer.analyze_multi_channel(channel_images)
    
    # Print results
    print(f"\n📊 Multi-Channel Results:")
    print(f"   Total channels: {results['summary']['total_channels']}")
    print(f"   Total structures: {results['summary']['total_structures']}")
    
    # Colocalization results
    colocalization = results.get('colocalization', {})
    if colocalization:
        print(f"\n🔗 Colocalization Analysis:")
        for pair, metrics in colocalization.items():
            if 'error' not in metrics:
                print(f"   {pair}:")
                print(f"     Pearson correlation: {metrics['pearson_correlation']:.3f}")
                print(f"     Manders M1: {metrics['manders_m1']:.3f}")
                print(f"     Manders M2: {metrics['manders_m2']:.3f}")
            else:
                print(f"   {pair}: {metrics['error']}")
    
    return analyzer, results


def main():
    """Main fluorescence demo"""
    print("🚁 Helicopter Life Science - Fluorescence Microscopy Demo")
    print("=" * 60)
    
    # Setup
    output_dir = ensure_output_dir()
    valid_images, _, _ = get_valid_files()
    
    if not valid_images:
        print("❌ No valid images found!")
        print("Please check your paths in config.py")
        return
    
    print(f"📂 Found {len(valid_images)} images")
    print(f"📁 Output directory: {output_dir}")
    
    # Analyze individual images
    analyzers = []
    all_results = []
    
    # Process first 3 images as different channels
    channels = [FluorescenceChannel.GFP, FluorescenceChannel.DAPI, FluorescenceChannel.RFP]
    
    for i, (image_name, image_path) in enumerate(list(valid_images.items())[:3]):
        print(f"\n" + "-" * 40)
        
        channel = channels[i % len(channels)]
        
        # Run comprehensive analysis with JSON output
        analyzer, results = analyze_fluorescence_image(image_path, channel, output_dir)
        
        if analyzer and results:
            # Load original image for visualization
            original_image = cv2.imread(str(image_path))
            
            # Create comprehensive visualizations
            if SAVE_FIGURES or SHOW_FIGURES:
                fig = analyzer.visualize_results(results, original_image)
                
                if SAVE_FIGURES:
                    save_path = output_dir / f"fluorescence_comprehensive_{image_name}_{channel.value}.png"
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"💾 Comprehensive visualization saved: {save_path}")
                
                if SHOW_FIGURES:
                    plt.show()
                else:
                    plt.close(fig)
            
            analyzers.append(analyzer)
            all_results.append(results)
    
    # Multi-channel analysis example
    if len(valid_images) >= 2:
        print(f"\n" + "=" * 60)
        
        # Take first 2 images as multi-channel example
        multi_channel_images = {
            'gfp': list(valid_images.values())[0],
            'dapi': list(valid_images.values())[1]
        }
        
        analyzer, results = analyze_multi_channel(multi_channel_images)
        
        if analyzer and results:
            # Visualize multi-channel results
            if SAVE_FIGURES or SHOW_FIGURES:
                fig = analyzer.visualize_results(results)
                
                if SAVE_FIGURES:
                    save_path = output_dir / "fluorescence_multi_channel.png"
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"💾 Saved multi-channel analysis: {save_path}")
                
                if SHOW_FIGURES:
                    plt.show()
                else:
                    plt.close(fig)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("🎉 Fluorescence Analysis Complete!")
    
    total_structures = sum(len(r['structures']) for r in all_results)
    print(f"✅ Analyzed {len(all_results)} images")
    print(f"🔬 Detected {total_structures} fluorescent structures")
    
    if SAVE_FIGURES:
        print(f"💾 Results saved to: {output_dir}")
    
    print("\n🎯 Tips for fluorescence analysis:")
    print("   • Adjust channel types in config.py for your specific fluorophores")
    print("   • Multi-channel analysis reveals protein colocalization")
    print("   • High SNR indicates good image quality")
    print("   • Try different background subtraction methods for optimal results")


if __name__ == "__main__":
    main()
