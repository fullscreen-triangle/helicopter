#!/usr/bin/env python3
"""
Video Analysis Demo

Focused demo for time-lapse microscopy and live cell imaging.
Configure your video files in config.py and run this script.

Usage:
    python demo_video.py
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add the lifescience package to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_valid_files, ensure_output_dir, SAVE_FIGURES, SHOW_FIGURES
from src.video import VideoAnalyzer, VideoType


def load_video_frames(video_path, max_frames=100, skip_frames=1):
    """Load frames from video with optional frame skipping"""
    print(f"🎬 Loading video: {video_path.name}")
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"   📹 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
        
        frames = []
        frame_indices = []
        frame_count = 0
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames if requested
            if frame_count % skip_frames == 0:
                frames.append(frame)
                frame_indices.append(frame_count)
            
            frame_count += 1
        
        cap.release()
        
        print(f"   ✅ Loaded {len(frames)} frames (every {skip_frames} frame{'s' if skip_frames > 1 else ''})")
        
        # Generate timestamps
        timestamps = [idx / fps for idx in frame_indices] if fps > 0 else list(range(len(frames)))
        
        return frames, timestamps, {
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'loaded_frames': len(frames)
        }
        
    except Exception as e:
        print(f"❌ Error loading video {video_path}: {e}")
        return [], [], {}


def analyze_video(video_path, video_type=VideoType.LIVE_CELL, max_frames=50, output_dir=None):
    """Analyze a single video file with comprehensive metrics and JSON output"""
    print(f"\n🎥 Analyzing video: {video_path.name}")
    
    # Load video frames
    frames, timestamps, video_info = load_video_frames(video_path, max_frames=max_frames)
    
    if not frames:
        print("❌ No frames loaded")
        return None, None, None
    
    # Initialize video analyzer with FPS from video info
    fps = video_info.get('fps', 30.0)
    analyzer = VideoAnalyzer(video_type=video_type, fps=fps)
    
    # Run comprehensive analysis
    print(f"🔍 Running comprehensive {video_type.value} analysis...")
    results = analyzer.analyze_video(frames, timestamps)
    
    # Save JSON results if output directory provided
    if output_dir:
        saved_files = analyzer.save_comprehensive_results(results, output_dir, f"{video_path.stem}_{video_type.value}")
        print(f"💾 JSON results saved: {list(saved_files.values())}")
    
    # Print comprehensive results
    print(f"\n📊 Comprehensive Video Analysis Results:")
    print(f"   Video type: {results['video_type']}")
    print(f"   Frames processed: {results['num_frames']}")
    print(f"   Duration: {timestamps[-1] - timestamps[0]:.1f} seconds")
    print(f"   Processing time: {results.get('processing_time', 0):.2f}s")
    
    # Enhanced tracking results
    tracking_metrics = results.get('tracking_metrics', {})
    tracks = results['tracks']
    print(f"\n🔬 Enhanced Cell Tracking:")
    print(f"   Cell tracks detected: {len(tracks)}")
    print(f"   Tracking accuracy: {tracking_metrics.get('accuracy', 0):.1%}")
    print(f"   Track completeness: {tracking_metrics.get('completeness', 0):.1%}")
    print(f"   False positive rate: {tracking_metrics.get('false_positive_rate', 0):.1%}")
    print(f"   False negative rate: {tracking_metrics.get('false_negative_rate', 0):.1%}")
    
    if tracks:
        # Show details for first few tracks with enhanced metrics
        for i, track in enumerate(tracks[:3]):
            displacement = track.get_total_displacement()
            track_duration = track.timestamps[-1] - track.timestamps[0]
            print(f"\n   Track {track.track_id}:")
            print(f"     Frames tracked: {len(track.positions)}")
            print(f"     Total displacement: {displacement:.1f} pixels")
            print(f"     Track duration: {track_duration:.1f} seconds")
            if track_duration > 0:
                avg_velocity = displacement / track_duration
                print(f"     Average velocity: {avg_velocity:.2f} pixels/second")
    
    # Enhanced motion analysis
    motion_activity = results['motion_activity']
    peak_frames = results.get('peak_activity_frames', [])
    if motion_activity:
        print(f"\n📈 Enhanced Motion Analysis:")
        print(f"   Mean activity level: {np.mean(motion_activity):.3f}")
        print(f"   Max activity level: {np.max(motion_activity):.3f}")
        print(f"   Activity variance: {np.std(motion_activity):.3f}")
        print(f"   Peak activity frames: {len(peak_frames)}")
    
    # Behavioral analysis
    behavior_analysis = results.get('behavior_analysis', {})
    if behavior_analysis and 'distribution' in behavior_analysis:
        behaviors = behavior_analysis['distribution']
        print(f"\n🧬 Behavioral Analysis:")
        total_cells = sum(behaviors.values())
        if total_cells > 0:
            for behavior, count in behaviors.items():
                percentage = (count / total_cells) * 100
                print(f"   {behavior.capitalize()}: {count} cells ({percentage:.1f}%)")
    
    # Velocity analysis
    velocity_metrics = results.get('velocity_metrics', {})
    if velocity_metrics:
        print(f"\n🏃 Velocity Analysis:")
        print(f"   Mean velocity: {velocity_metrics.get('mean_velocity', 0):.2f} pixels/second")
        displacement_metrics = velocity_metrics.get('displacement_metrics', {})
        if displacement_metrics:
            print(f"   Mean displacement: {displacement_metrics.get('mean_displacement', 0):.1f} pixels")
            print(f"   Max displacement: {displacement_metrics.get('max_displacement', 0):.1f} pixels")
            print(f"   Displacement variance: {displacement_metrics.get('displacement_variance', 0):.1f}")
    
    # Summary statistics
    summary = results['summary']
    print(f"\n📊 Enhanced Summary:")
    print(f"   Total displacement: {summary['total_displacement']:.1f} pixels")
    print(f"   Mean motion activity: {summary['mean_activity']:.3f}")
    
    return analyzer, results, video_info


def analyze_cell_behavior(tracks):
    """Analyze cell behavior patterns from tracks"""
    if not tracks:
        return {}
    
    print(f"\n🧬 Cell Behavior Analysis:")
    
    behaviors = {
        'stationary': 0,    # Very low displacement
        'migrating': 0,     # High displacement, directional
        'oscillating': 0,   # Medium displacement, non-directional
        'dividing': 0       # Area changes (simplified detection)
    }
    
    for track in tracks:
        displacement = track.get_total_displacement()
        track_length = len(track.positions)
        
        if track_length < 5:
            continue
            
        # Calculate track path length
        path_length = 0
        for i in range(1, len(track.positions)):
            pos_current = np.array(track.positions[i])
            pos_previous = np.array(track.positions[i-1])
            path_length += np.linalg.norm(pos_current - pos_previous)
        
        # Directionality (efficiency of movement)
        directionality = displacement / path_length if path_length > 0 else 0
        
        # Classify behavior
        if displacement < 10:
            behaviors['stationary'] += 1
        elif directionality > 0.6 and displacement > 50:
            behaviors['migrating'] += 1
        elif directionality < 0.3 and displacement > 20:
            behaviors['oscillating'] += 1
        else:
            # Check for area changes (division indicator)
            if hasattr(track, 'areas') and len(track.areas) > 5:
                area_changes = np.diff(track.areas)
                if np.any(area_changes > np.std(track.areas)):
                    behaviors['dividing'] += 1
            else:
                behaviors['migrating'] += 1  # Default to migrating if no area data
    
    # Print behavior analysis
    total_analyzed = sum(behaviors.values())
    if total_analyzed > 0:
        print(f"   Cells analyzed: {total_analyzed}")
        for behavior, count in behaviors.items():
            percentage = (count / total_analyzed) * 100
            print(f"   {behavior.capitalize()}: {count} ({percentage:.1f}%)")
    
    return behaviors


def main():
    """Main video demo"""
    print("🚁 Helicopter Life Science - Video Analysis Demo")
    print("=" * 60)
    
    # Setup
    output_dir = ensure_output_dir()
    _, valid_videos, _ = get_valid_files()
    
    if not valid_videos:
        print("❌ No valid videos found!")
        print("Please check your video paths in config.py")
        return
    
    print(f"🎬 Found {len(valid_videos)} videos")
    print(f"📁 Output directory: {output_dir}")
    
    # Video types to try
    video_types = [VideoType.LIVE_CELL, VideoType.TIME_LAPSE, VideoType.CELL_MIGRATION]
    
    # Analyze videos
    all_results = []
    all_tracks = []
    
    for i, (video_name, video_path) in enumerate(valid_videos.items()):
        print(f"\n" + "=" * 60)
        
        # Use different video type for each video
        video_type = video_types[i % len(video_types)]
        
        # Run comprehensive analysis with JSON output
        analyzer, results, video_info = analyze_video(video_path, video_type, max_frames=30, output_dir=output_dir)
        
        if analyzer and results:
            # Collect tracks for behavior analysis
            all_tracks.extend(results['tracks'])
            all_results.append(results)
            
            # Create comprehensive visualizations
            if SAVE_FIGURES or SHOW_FIGURES:
                # Use first frame as representative frame
                frames, _, _ = load_video_frames(video_path, max_frames=1)
                if frames:
                    fig = analyzer.visualize_results(results, frames[0])
                    
                    if SAVE_FIGURES:
                        save_path = output_dir / f"video_comprehensive_{video_name}_{video_type.value}.png"
                        fig.savefig(save_path, dpi=300, bbox_inches='tight')
                        print(f"💾 Comprehensive visualization saved: {save_path}")
                    
                    if SHOW_FIGURES:
                        plt.show()
                    else:
                        plt.close(fig)
    
    # Analyze cell behavior across all videos
    if all_tracks:
        print(f"\n" + "=" * 60)
        behaviors = analyze_cell_behavior(all_tracks)
    
    # Overall summary
    print(f"\n" + "=" * 60)
    print("🎉 Video Analysis Complete!")
    
    total_tracks = len(all_tracks)
    total_frames = sum(r['num_frames'] for r in all_results)
    
    print(f"✅ Processed {len(all_results)} videos")
    print(f"🎬 Total frames analyzed: {total_frames}")
    print(f"🔬 Total cell tracks: {total_tracks}")
    
    if all_results:
        avg_activity = np.mean([r['summary']['mean_activity'] for r in all_results])
        total_displacement = sum(r['summary']['total_displacement'] for r in all_results)
        print(f"📈 Average motion activity: {avg_activity:.3f}")
        print(f"🏃 Total cell displacement: {total_displacement:.1f} pixels")
    
    if SAVE_FIGURES:
        print(f"💾 Results saved to: {output_dir}")
    
    print("\n🎯 Tips for video analysis:")
    print("   • Use appropriate video_type for your experiment")
    print("   • Higher frame rates improve tracking accuracy")
    print("   • Cell_migration type focuses on directional movement")
    print("   • Time_lapse type is optimized for developmental processes")
    print("   • Live_cell type provides general cell tracking")


if __name__ == "__main__":
    main()
