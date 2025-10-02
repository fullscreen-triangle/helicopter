#!/usr/bin/env python3
"""
Helicopter Life Science - Complete Demo Script

This script demonstrates all 6 modules of the Helicopter Life Science framework.
Simply configure paths in config.py and run this script.

Usage:
    python demo_all_modules.py

The script will:
1. Load images/videos from configured paths
2. Run analysis with all enabled modules  
3. Generate visualizations and results
4. Save outputs to the results directory
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add the lifescience package to path
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration
from config import (
    get_valid_files, ensure_output_dir, 
    RUN_MODULES, ANALYSIS_PARAMS, 
    SAVE_FIGURES, SHOW_FIGURES
)

# Import ALL Helicopter framework modules
try:
    # Core Helicopter Framework modules
    from src.gas import BiologicalGasAnalyzer, MoleculeType, BiologicalProperties
    from src.entropy import SEntropyTransformer, BiologicalContext
    from src.meta import MetaInformationExtractor, InformationType
    
    # Life Science specialization modules
    from src.fluorescence import FluorescenceAnalyzer, FluorescenceChannel
    from src.electron import ElectronMicroscopyAnalyzer, EMType, UltrastructureType
    from src.video import VideoAnalyzer, VideoType
    
    print("✅ ALL Helicopter framework modules imported successfully!")
    print("   • Gas Molecular Dynamics: ✅")
    print("   • S-Entropy Coordinates: ✅") 
    print("   • Meta-Information: ✅")
    print("   • Fluorescence Analysis: ✅")
    print("   • Electron Microscopy: ✅")
    print("   • Video Analysis: ✅")
    
except ImportError as e:
    print(f"❌ Error importing Helicopter modules: {e}")
    print("Make sure you're running from the lifescience directory")
    sys.exit(1)


def load_image(image_path):
    """Load and return image"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"⚠️  Warning: Could not load image {image_path}")
            return None
        print(f"📁 Loaded image: {image_path.name} ({image.shape})")
        return image
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return None


def load_video_frames(video_path, max_frames=50):
    """Load frames from video file"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_count = 0
        
        while cap.read()[0] and frame_count < max_frames:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                frame_count += 1
            else:
                break
        
        cap.release()
        print(f"📹 Loaded {len(frames)} frames from {video_path.name}")
        return frames
    except Exception as e:
        print(f"❌ Error loading video {video_path}: {e}")
        return []


def run_gas_molecular_analysis(image, image_name, output_dir):
    """Run gas molecular dynamics analysis - CORE HELICOPTER MODULE"""
    if not RUN_MODULES['gas_molecular']:
        print("⏭️  Gas molecular analysis disabled")
        return None
        
    print("\n🧪 Running Gas Molecular Dynamics Analysis (CORE HELICOPTER)...")
    
    try:
        start_time = time.time()
        analyzer = BiologicalGasAnalyzer()
        
        # Run protein structure analysis
        results = analyzer.analyze_protein_structure(
            image, 
            structure_type=ANALYSIS_PARAMS['gas_molecular']['structure_type']
        )
        
        processing_time = time.time() - start_time
        
        # Visualize results
        if SAVE_FIGURES or SHOW_FIGURES:
            fig = analyzer.visualize_system()
            
            if SAVE_FIGURES:
                save_path = output_dir / f"gas_molecular_{image_name}.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Saved gas molecular analysis: {save_path}")
            
            if SHOW_FIGURES:
                plt.show()
            else:
                plt.close(fig)
        
        print(f"🔬 Gas Analysis Complete ({processing_time:.2f}s):")
        print(f"   🧬 Molecular residues: {results['num_residues']}")
        print(f"   ⚖️ Equilibrium reached: {results['equilibrium_reached']}")
        print(f"   🔗 Binding sites: {len(results['binding_sites'])}")
        print(f"   📊 Folding quality: {results['folding_quality']['quality']:.3f}")
        print(f"   🌡️ System temperature: {results['system_temperature']:.2f}K")
        print(f"   ⚡ Total energy: {results['total_energy']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Gas molecular analysis failed: {e}")
        return None


def run_entropy_analysis(image, image_name, output_dir):
    """Run S-entropy coordinate analysis - CORE HELICOPTER MODULE"""
    if not RUN_MODULES['entropy_analysis']:
        print("⏭️  S-entropy analysis disabled")
        return None
        
    print("\n🎯 Running S-Entropy Coordinate Transformation (CORE HELICOPTER)...")
    
    try:
        start_time = time.time()
        
        context_map = {
            'cellular': BiologicalContext.CELLULAR,
            'tissue': BiologicalContext.TISSUE, 
            'molecular': BiologicalContext.MOLECULAR
        }
        context = context_map[ANALYSIS_PARAMS['entropy']['biological_context']]
        
        transformer = SEntropyTransformer(biological_context=context)
        coordinates = transformer.transform(image)
        
        processing_time = time.time() - start_time
        
        # Visualize results
        if SAVE_FIGURES or SHOW_FIGURES:
            fig = transformer.visualize_coordinates(coordinates)
            
            if SAVE_FIGURES:
                save_path = output_dir / f"entropy_{image_name}.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Saved entropy analysis: {save_path}")
            
            if SHOW_FIGURES:
                plt.show()
            else:
                plt.close(fig)
        
        print(f"🎯 S-Entropy Analysis Complete ({processing_time:.2f}s):")
        print(f"   📐 4D Coordinates:")
        print(f"     • Structural: {coordinates.structural:.3f}")
        print(f"     • Functional: {coordinates.functional:.3f}")
        print(f"     • Morphological: {coordinates.morphological:.3f}")
        print(f"     • Temporal: {coordinates.temporal:.3f}")
        print(f"   🎯 Transformation confidence: {coordinates.confidence:.3f}")
        print(f"   🧬 Biological context: {coordinates.biological_context.value}")
        print(f"   🔄 Processing mode: {coordinates.processing_mode}")
        
        return coordinates
        
    except Exception as e:
        print(f"❌ S-entropy analysis failed: {e}")
        return None


def run_fluorescence_analysis(image, image_name, output_dir):
    """Run fluorescence microscopy analysis"""
    if not RUN_MODULES['fluorescence']:
        return None
        
    print("\n🔬 Running Fluorescence Analysis...")
    
    try:
        channel_map = {
            'dapi': FluorescenceChannel.DAPI,
            'gfp': FluorescenceChannel.GFP,
            'rfp': FluorescenceChannel.RFP,
            'fitc': FluorescenceChannel.FITC
        }
        channel = channel_map[ANALYSIS_PARAMS['fluorescence']['channel']]
        
        analyzer = FluorescenceAnalyzer()
        results = analyzer.analyze_image(image, channel)
        
        # Visualize results
        if SAVE_FIGURES or SHOW_FIGURES:
            fig = analyzer.visualize_results(results)
            
            if SAVE_FIGURES:
                save_path = output_dir / f"fluorescence_{image_name}.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Saved fluorescence analysis: {save_path}")
            
            if SHOW_FIGURES:
                plt.show()
            else:
                plt.close(fig)
        
        print(f"🌟 Fluorescence Analysis ({channel.value}):")
        print(f"   Structures detected: {results['num_structures']}")
        print(f"   Total intensity: {results['summary']['total_intensity']:.0f}")
        print(f"   Mean SNR: {results['summary']['mean_snr']:.2f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Fluorescence analysis failed: {e}")
        return None


def run_electron_microscopy_analysis(image, image_name, output_dir):
    """Run electron microscopy analysis"""
    if not RUN_MODULES['electron_microscopy']:
        return None
        
    print("\n⚡ Running Electron Microscopy Analysis...")
    
    try:
        em_type_map = {
            'sem': EMType.SEM,
            'tem': EMType.TEM,
            'cryo_em': EMType.CRYO_EM
        }
        em_type = em_type_map[ANALYSIS_PARAMS['electron_microscopy']['em_type']]
        
        structure_map = {
            'mitochondria': UltrastructureType.MITOCHONDRIA,
            'nucleus': UltrastructureType.NUCLEUS,
            'vesicles': UltrastructureType.VESICLES,
            'membrane': UltrastructureType.MEMBRANE
        }
        target_structures = [structure_map[s] for s in ANALYSIS_PARAMS['electron_microscopy']['target_structures']]
        
        analyzer = ElectronMicroscopyAnalyzer(em_type=em_type)
        results = analyzer.analyze_image(image, target_structures=target_structures)
        
        # Visualize results
        if SAVE_FIGURES or SHOW_FIGURES:
            fig = analyzer.visualize_results(results, image)
            
            if SAVE_FIGURES:
                save_path = output_dir / f"electron_{image_name}.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Saved electron microscopy analysis: {save_path}")
            
            if SHOW_FIGURES:
                plt.show()
            else:
                plt.close(fig)
        
        print(f"🔬 Electron Microscopy ({em_type.value}):")
        print(f"   Structures detected: {results['num_structures']}")
        print(f"   Mean confidence: {results['summary']['mean_confidence']:.3f}")
        if results['summary']['type_distribution']:
            print(f"   Structure types: {results['summary']['type_distribution']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Electron microscopy analysis failed: {e}")
        return None


def run_video_analysis(frames, video_name, output_dir):
    """Run video analysis on frames"""
    if not RUN_MODULES['video_analysis'] or not frames:
        return None
        
    print(f"\n🎬 Running Video Analysis on {len(frames)} frames...")
    
    try:
        video_type_map = {
            'live_cell': VideoType.LIVE_CELL,
            'time_lapse': VideoType.TIME_LAPSE,
            'calcium_imaging': VideoType.CALCIUM_IMAGING,
            'cell_migration': VideoType.CELL_MIGRATION
        }
        video_type = video_type_map[ANALYSIS_PARAMS['video']['video_type']]
        
        analyzer = VideoAnalyzer(video_type=video_type)
        
        # Generate timestamps
        frame_interval = ANALYSIS_PARAMS['video']['frame_interval']
        timestamps = [i * frame_interval for i in range(len(frames))]
        
        results = analyzer.analyze_video(frames, timestamps)
        
        # Visualize results
        if SAVE_FIGURES or SHOW_FIGURES:
            fig = analyzer.visualize_results(results, frames[0])
            
            if SAVE_FIGURES:
                save_path = output_dir / f"video_{video_name}.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Saved video analysis: {save_path}")
            
            if SHOW_FIGURES:
                plt.show()
            else:
                plt.close(fig)
        
        print(f"🎥 Video Analysis ({video_type.value}):")
        print(f"   Frames analyzed: {results['num_frames']}")
        print(f"   Cell tracks: {results['summary']['num_tracks']}")
        print(f"   Mean activity: {results['summary']['mean_activity']:.3f}")
        print(f"   Total displacement: {results['summary']['total_displacement']:.1f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Video analysis failed: {e}")
        return None


def run_meta_information_analysis(image, image_name, output_dir):
    """Run meta-information extraction - CORE HELICOPTER MODULE"""
    if not RUN_MODULES['meta_information']:
        print("⏭️  Meta-information analysis disabled")
        return None
        
    print("\n🔍 Running Meta-Information Extraction (CORE HELICOPTER)...")
    
    try:
        start_time = time.time()
        extractor = MetaInformationExtractor()
        
        # Extract meta-information
        meta_info = extractor.extract_meta_information(image)
        
        # Analyze compression ratios if requested
        compression_ratios = None
        if ANALYSIS_PARAMS['meta']['compression_analysis']:
            compression_ratios = extractor.analyze_compression_ratio(image)
        
        processing_time = time.time() - start_time
        
        # Visualize results
        if SAVE_FIGURES or SHOW_FIGURES:
            fig = extractor.visualize_meta_information(image, meta_info)
            
            if SAVE_FIGURES:
                save_path = output_dir / f"meta_{image_name}.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Saved meta-information analysis: {save_path}")
            
            if SHOW_FIGURES:
                plt.show()
            else:
                plt.close(fig)
        
        print(f"📊 Meta-Information Analysis Complete ({processing_time:.2f}s):")
        print(f"   📋 Information type: {meta_info.information_type.value}")
        print(f"   🗜️ Compression potential: {meta_info.compression_potential:.3f}")
        print(f"   🔬 Semantic density: {meta_info.semantic_density:.3f}")
        print(f"   🏗️ Structural complexity: {meta_info.structural_complexity:.3f}")
        print(f"   🎯 Analysis confidence: {meta_info.confidence:.3f}")
        
        if compression_ratios:
            print(f"   📈 Compression Analysis:")
            print(f"     • Lossless: {compression_ratios['lossless_ratio']:.1f}x")
            print(f"     • Lossy: {compression_ratios['lossy_ratio']:.1f}x")
            print(f"     • Semantic: {compression_ratios['semantic_ratio']:.1f}x")
            print(f"     • Pattern Recognition: {compression_ratios.get('pattern_ratio', 1.0):.1f}x")
        
        return meta_info, compression_ratios
        
    except Exception as e:
        print(f"❌ Meta-information analysis failed: {e}")
        return None


def main():
    """Main demo function"""
    print("🚁 Helicopter Life Science Framework - Complete Demo")
    print("=" * 60)
    
    # Setup
    output_dir = ensure_output_dir()
    valid_images, valid_videos, valid_archives = get_valid_files()
    
    print(f"\n📂 Data Summary:")
    print(f"   Images: {len(valid_images)}")
    print(f"   Videos: {len(valid_videos)}")
    print(f"   Archives: {len(valid_archives)}")
    print(f"   Output directory: {output_dir}")
    
    if not valid_images and not valid_videos:
        print("❌ No valid image or video files found!")
        print("Please check your paths in config.py")
        return
    
    # Process images
    image_results = {}
    for image_name, image_path in list(valid_images.items())[:3]:  # Process first 3 images
        print(f"\n" + "="*60)
        print(f"🖼️  Processing Image: {image_name}")
        print(f"   Path: {image_path}")
        
        image = load_image(image_path)
        if image is None:
            continue
        
        start_time = time.time()
        
        # Run all analyses
        results = {
            'gas_molecular': run_gas_molecular_analysis(image, image_name, output_dir),
            'entropy': run_entropy_analysis(image, image_name, output_dir),
            'fluorescence': run_fluorescence_analysis(image, image_name, output_dir),
            'electron_microscopy': run_electron_microscopy_analysis(image, image_name, output_dir),
            'meta_information': run_meta_information_analysis(image, image_name, output_dir)
        }
        
        processing_time = time.time() - start_time
        print(f"\n⏱️  Processing time: {processing_time:.2f} seconds")
        
        image_results[image_name] = results
    
    # Process videos
    video_results = {}
    for video_name, video_path in list(valid_videos.items())[:2]:  # Process first 2 videos
        print(f"\n" + "="*60)
        print(f"🎬 Processing Video: {video_name}")
        print(f"   Path: {video_path}")
        
        frames = load_video_frames(video_path)
        if not frames:
            continue
        
        start_time = time.time()
        
        # Run video analysis
        results = run_video_analysis(frames, video_name, output_dir)
        
        processing_time = time.time() - start_time
        print(f"\n⏱️  Processing time: {processing_time:.2f} seconds")
        
        video_results[video_name] = results
    
    # Summary
    print(f"\n" + "="*60)
    print("📊 ANALYSIS COMPLETE!")
    print("="*60)
    
    total_processed = len(image_results) + len(video_results)
    print(f"✅ Successfully processed {total_processed} files")
    
    if SAVE_FIGURES:
        print(f"💾 Results saved to: {output_dir}")
        print("   Check the directory for visualization outputs")
    
    print("\n🎯 Next steps:")
    print("   • Modify config.py to analyze your specific data")
    print("   • Adjust analysis parameters for your use case")
    print("   • Run individual module demos for focused analysis")
    print("   • Check the results directory for outputs")


if __name__ == "__main__":
    main()
