# Video Processing and Analysis Module
"""
Video processing and analysis functionality for life science applications.
Specialized for analyzing time-lapse microscopy, live cell imaging, and 
dynamic biological processes.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VideoType(Enum):
    """Types of biological video data"""
    LIVE_CELL = "live_cell"
    TIME_LAPSE = "time_lapse"
    CALCIUM_IMAGING = "calcium_imaging"
    CELL_MIGRATION = "cell_migration"


@dataclass
class CellTrack:
    """Individual cell tracking data"""
    track_id: int
    positions: List[Tuple[int, int]]
    timestamps: List[float]
    areas: List[float]
    
    def get_total_displacement(self) -> float:
        """Calculate total displacement"""
        if len(self.positions) < 2:
            return 0.0
        
        start_pos = np.array(self.positions[0])
        end_pos = np.array(self.positions[-1])
        return np.linalg.norm(end_pos - start_pos)


class VideoAnalyzer:
    """Core video analysis engine for biological applications"""
    
    def __init__(self, video_type: VideoType = VideoType.LIVE_CELL):
        self.video_type = video_type
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
        
        # Simple cell detection and tracking (simplified)
        tracks = self._simple_cell_tracking(video_frames, timestamps)
        
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
    
    def _simple_cell_tracking(self, frames: List[np.ndarray], 
                             timestamps: List[float]) -> List[CellTrack]:
        """Simplified cell tracking"""
        
        tracks = []
        
        # Very basic tracking - detect brightest regions
        for frame_idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame.astype(np.float32)
            
            # Find bright spots (simplified cell detection)
            threshold = np.percentile(gray, 90)
            binary = (gray > threshold).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if frame_idx == 0:
                # Initialize tracks
                for i, contour in enumerate(contours[:5]):  # Limit to 5 tracks
                    if cv2.contourArea(contour) > 100:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            track = CellTrack(
                                track_id=i,
                                positions=[(cx, cy)],
                                timestamps=[timestamp],
                                areas=[cv2.contourArea(contour)]
                            )
                            tracks.append(track)
            else:
                # Simple nearest neighbor tracking
                for track in tracks:
                    if not contours:
                        continue
                        
                    last_pos = np.array(track.positions[-1])
                    best_distance = float('inf')
                    best_contour = None
                    
                    for contour in contours:
                        if cv2.contourArea(contour) > 50:
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                
                                distance = np.linalg.norm(np.array([cx, cy]) - last_pos)
                                if distance < best_distance and distance < 50:  # Max movement
                                    best_distance = distance
                                    best_contour = contour
                    
                    if best_contour is not None:
                        M = cv2.moments(best_contour)
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        track.positions.append((cx, cy))
                        track.timestamps.append(timestamp)
                        track.areas.append(cv2.contourArea(best_contour))
        
        return tracks
    
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


# Export main classes
__all__ = ['VideoAnalyzer', 'VideoType', 'CellTrack']