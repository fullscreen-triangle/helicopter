"""
Video Analysis - Core video analysis engine
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

from .tracking import CellTracker, CellTrack, VideoType

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Core video analysis engine for biological applications"""
    
    def __init__(self, video_type: VideoType = VideoType.LIVE_CELL):
        self.video_type = video_type
        self.cell_tracker = CellTracker()
        self.analysis_results = []
        
    def analyze_video(self, video_frames: List[np.ndarray],
                     timestamps: Optional[List[float]] = None) -> Dict[str, Any]:
        """Analyze biological video data"""
        
        if timestamps is None:
            timestamps = list(range(len(video_frames)))
        
        logger.info(f"Analyzing {len(video_frames)} frames")
        
        # Simple frame differencing analysis
        motion_activity = []
        
        for i in range(1, len(video_frames)):
            # Convert frames to grayscale
            if len(video_frames[i].shape) == 3:
                frame1 = cv2.cvtColor(video_frames[i-1], cv2.COLOR_RGB2GRAY)
                frame2 = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2GRAY)
            else:
                frame1 = video_frames[i-1].astype(np.float32)
                frame2 = video_frames[i].astype(np.float32)
            
            # Calculate frame difference
            diff = np.abs(frame2 - frame1)
            activity = np.mean(diff)
            motion_activity.append(activity)
        
        # Simple cell detection and tracking
        tracks = self.cell_tracker.track_cells(video_frames, timestamps)
        
        results = {
            'video_type': self.video_type.value,
            'num_frames': len(video_frames),
            'motion_activity': motion_activity,
            'tracks': tracks,
            'summary': {
                'num_tracks': len(tracks),
                'mean_activity': np.mean(motion_activity) if motion_activity else 0,
                'total_displacement': sum(track.get_total_displacement() for track in tracks)
            }
        }
        
        self.analysis_results.append(results)
        return results
    
    def visualize_results(self, results: Dict[str, Any],
                         representative_frame: np.ndarray) -> plt.Figure:
        """Visualize video analysis results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Video Analysis: {results['video_type']}")
        
        # Representative frame with tracks
        axes[0, 0].imshow(representative_frame, cmap='gray')
        
        tracks = results['tracks']
        for i, track in enumerate(tracks[:5]):  # Show first 5 tracks
            positions = np.array(track.positions)
            axes[0, 0].plot(positions[:, 0], positions[:, 1], linewidth=2, label=f'Track {i}')
            axes[0, 0].scatter(positions[0, 0], positions[0, 1], s=50, marker='o')  # Start
            axes[0, 0].scatter(positions[-1, 0], positions[-1, 1], s=50, marker='s')  # End
        
        axes[0, 0].set_title(f'Cell Tracks ({len(tracks)} total)')
        axes[0, 0].axis('off')
        
        # Motion activity over time
        motion_activity = results['motion_activity']
        if motion_activity:
            axes[0, 1].plot(motion_activity)
            axes[0, 1].set_title('Motion Activity Over Time')
            axes[0, 1].set_xlabel('Frame')
            axes[0, 1].set_ylabel('Activity Level')
        
        # Track lengths
        if tracks:
            track_lengths = [len(track.positions) for track in tracks]
            axes[1, 0].hist(track_lengths, bins=10)
            axes[1, 0].set_title('Track Length Distribution')
            axes[1, 0].set_xlabel('Track Length (frames)')
            axes[1, 0].set_ylabel('Count')
        
        # Summary statistics
        summary = results['summary']
        summary_text = f"Tracks: {summary['num_tracks']}\n"
        summary_text += f"Frames: {results['num_frames']}\n"
        summary_text += f"Mean Activity: {summary['mean_activity']:.2f}\n"
        summary_text += f"Total Displacement: {summary['total_displacement']:.1f}"
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, va='center')
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
