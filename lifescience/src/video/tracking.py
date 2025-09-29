"""
Video Tracking - Cell tracking and data structures
"""

import numpy as np
import cv2
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


class CellTracker:
    """Track individual cells across video frames"""
    
    def __init__(self, max_distance: float = 50.0):
        self.max_distance = max_distance
        self.tracks = {}
        self.next_track_id = 0
        
    def track_cells(self, video_frames: List[np.ndarray],
                   timestamps: List[float]) -> List[CellTrack]:
        """Track cells across video frames"""
        
        logger.info(f"Tracking cells across {len(video_frames)} frames")
        
        # Very basic tracking - detect brightest regions
        for frame_idx, (frame, timestamp) in enumerate(zip(video_frames, timestamps)):
            # Convert to grayscale
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
                            self.tracks[i] = track
                            self.next_track_id = i + 1
            else:
                # Simple nearest neighbor tracking
                for track in self.tracks.values():
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
                                if distance < best_distance and distance < self.max_distance:
                                    best_distance = distance
                                    best_contour = contour
                    
                    if best_contour is not None:
                        M = cv2.moments(best_contour)
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        track.positions.append((cx, cy))
                        track.timestamps.append(timestamp)
                        track.areas.append(cv2.contourArea(best_contour))
        
        # Return list of tracks
        active_tracks = [track for track in self.tracks.values() if len(track.positions) > 3]
        logger.info(f"Generated {len(active_tracks)} cell tracks")
        return active_tracks
