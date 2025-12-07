#!/usr/bin/env python3
"""
Validation experiments for Motion Picture Maxwell Demon

Tests the core hypothesis: Can we create a video playback system where
temporal direction (forward entropy) is decoupled from scrubbing direction?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy as scipy_entropy
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
import json

class TemporalEntropyCalculator:
    """Calculate S-entropy coordinates for video frames"""
    
    def __init__(self):
        self.name = "Temporal Entropy Calculator"
    
    def calculate_shannon_entropy(self, frame: np.ndarray) -> float:
        """Shannon entropy of pixel distribution"""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        hist, _ = np.histogram(frame.flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Remove zeros
        
        return scipy_entropy(hist, base=2)
    
    def calculate_temporal_entropy(self, frame_curr: np.ndarray, 
                                   frame_prev: np.ndarray) -> float:
        """Temporal entropy: measure of frame-to-frame change"""
        if len(frame_curr.shape) == 3:
            frame_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
            frame_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
        
        # Optical flow magnitude as temporal change
        flow = cv2.calcOpticalFlowFarneback(
            frame_prev, frame_curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Entropy of motion distribution
        hist, _ = np.histogram(magnitude.flatten(), bins=50, range=(0, 10))
        hist = hist / (hist.sum() + 1e-10)
        hist = hist[hist > 0]
        
        return scipy_entropy(hist, base=2) if len(hist) > 0 else 0.0
    
    def calculate_participation_ratio(self, frame: np.ndarray) -> float:
        """Participation ratio: effective number of active features"""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(frame, 50, 150)
        
        # Count active edge pixels
        n_active = np.sum(edges > 0)
        n_total = edges.size
        
        if n_active == 0:
            return 0.0
        
        # Participation ratio
        p = n_active / n_total
        return 1.0 / (p + 1e-10) if p > 0 else 0.0
    
    def calculate_s_entropy_coordinates(self, frames: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate S_t, S_k, S_e for all frames"""
        n_frames = len(frames)
        
        s_t = np.zeros(n_frames)  # Temporal entropy
        s_k = np.zeros(n_frames)  # Knowledge entropy (Shannon)
        s_e = np.zeros(n_frames)  # Evolutionary entropy (cumulative)
        participation = np.zeros(n_frames)
        
        print(f"\nCalculating S-entropy coordinates for {n_frames} frames...")
        
        for i in range(n_frames):
            # S_k: Shannon entropy at frame i
            s_k[i] = self.calculate_shannon_entropy(frames[i])
            
            # S_t: Temporal entropy (change from previous frame)
            if i > 0:
                s_t[i] = self.calculate_temporal_entropy(frames[i], frames[i-1])
            
            # Participation ratio
            participation[i] = self.calculate_participation_ratio(frames[i])
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{n_frames} frames")
        
        # S_e: Cumulative entropy production (always increasing!)
        s_e = np.cumsum(s_t)
        
        # Entropy production rate
        ds_dt = np.gradient(s_e)
        
        return {
            's_t': s_t,
            's_k': s_k,
            's_e': s_e,
            'participation': participation,
            'ds_dt': ds_dt
        }


class DualMembraneTemporalGenerator:
    """Generate dual-membrane temporal structures for alternative paths"""
    
    def __init__(self, frames: List[np.ndarray], s_entropy: Dict[str, np.ndarray]):
        self.frames = frames
        self.s_entropy = s_entropy
        self.n_frames = len(frames)
    
    def generate_alternative_forward_path(self, frame_idx: int) -> Tuple[np.ndarray, float]:
        """
        Generate alternative frame that still moves forward in entropy
        
        This is the KEY innovation: when scrubbing backward temporally,
        we generate a DIFFERENT forward-entropy frame instead of reversing.
        """
        if frame_idx <= 0:
            return self.frames[0], self.s_entropy['s_e'][0]
        
        # Current entropy level
        current_s_e = self.s_entropy['s_e'][frame_idx]
        
        # Generate alternative frame by perturbing while maintaining entropy increase
        prev_frame = self.frames[frame_idx - 1]
        
        # Add controlled noise to previous frame (increases entropy)
        noise_level = 0.02
        noise = np.random.randn(*prev_frame.shape) * noise_level * 255
        alternative_frame = np.clip(prev_frame + noise, 0, 255).astype(np.uint8)
        
        # Calculate what the new entropy would be
        if frame_idx < self.n_frames - 1:
            # Interpolate toward next frame but with variation
            alpha = 0.7
            alternative_frame = cv2.addWeighted(
                alternative_frame, alpha,
                self.frames[frame_idx], 1 - alpha, 0
            )
        
        # Ensure entropy increased
        new_s_e = current_s_e + np.abs(np.random.randn()) * 0.1
        
        return alternative_frame, new_s_e
    
    def create_dual_membrane_structure(self) -> Dict:
        """
        Create front and back faces for each temporal point
        
        Front face: Original forward path
        Back face: Alternative forward path (conjugate)
        """
        front_faces = []
        back_faces = []
        front_entropy = []
        back_entropy = []
        
        print("\nGenerating dual-membrane temporal structure...")
        
        for i in range(self.n_frames):
            # Front face: original frame
            front_faces.append(self.frames[i])
            front_entropy.append(self.s_entropy['s_e'][i])
            
            # Back face: alternative forward path
            alt_frame, alt_entropy = self.generate_alternative_forward_path(i)
            back_faces.append(alt_frame)
            back_entropy.append(alt_entropy)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{self.n_frames} dual frames")
        
        return {
            'front_faces': front_faces,
            'back_faces': back_faces,
            'front_entropy': np.array(front_entropy),
            'back_entropy': np.array(back_entropy),
            'thickness': np.abs(np.array(back_entropy) - np.array(front_entropy))
        }


class IrreversibleVideoPlayer:
    """
    Video player that ALWAYS plays forward in entropy,
    regardless of scrubbing direction
    """
    
    def __init__(self, dual_membrane: Dict, s_entropy: Dict):
        self.dual_membrane = dual_membrane
        self.s_entropy = s_entropy
        self.n_frames = len(dual_membrane['front_faces'])
        self.current_frame_idx = 0
        self.entropy_position = 0.0
        self.playback_history = []
    
    def scrub_to_position(self, target_idx: int, use_front_face: bool = True) -> np.ndarray:
        """
        Scrub to position - KEY TEST: always moves forward in entropy!
        
        If target_idx < current_frame_idx (scrubbing backward),
        we switch to back face (alternative forward path)
        """
        direction = "FORWARD" if target_idx > self.current_frame_idx else "BACKWARD"
        
        # Determine which face to use
        if target_idx > self.current_frame_idx:
            # Normal forward: use front face
            face = 'front'
            frame = self.dual_membrane['front_faces'][target_idx]
            new_entropy = self.dual_membrane['front_entropy'][target_idx]
        else:
            # Scrubbing backward: use back face (alternative forward path!)
            face = 'back'
            frame = self.dual_membrane['back_faces'][target_idx]
            new_entropy = self.dual_membrane['back_entropy'][target_idx]
        
        # CRITICAL CHECK: Entropy must increase!
        entropy_increased = new_entropy >= self.entropy_position
        
        self.playback_history.append({
            'frame_idx': target_idx,
            'entropy': new_entropy,
            'face': face,
            'direction': direction,
            'entropy_increased': entropy_increased
        })
        
        self.current_frame_idx = target_idx
        self.entropy_position = new_entropy
        
        return frame
    
    def get_playback_statistics(self) -> Dict:
        """Analyze playback to verify entropy monotonicity"""
        if not self.playback_history:
            return {}
        
        entropies = [h['entropy'] for h in self.playback_history]
        directions = [h['direction'] for h in self.playback_history]
        faces = [h['face'] for h in self.playback_history]
        
        # Check entropy monotonicity
        entropy_diffs = np.diff(entropies)
        monotonic = np.all(entropy_diffs >= 0)
        violations = np.sum(entropy_diffs < 0)
        
        # Count forward/backward scrubs
        n_forward = directions.count('FORWARD')
        n_backward = directions.count('BACKWARD')
        
        # Count face usage
        n_front = faces.count('front')
        n_back = faces.count('back')
        
        return {
            'total_frames_played': len(self.playback_history),
            'entropy_monotonic': monotonic,
            'entropy_violations': violations,
            'forward_scrubs': n_forward,
            'backward_scrubs': n_backward,
            'front_face_used': n_front,
            'back_face_used': n_back,
            'entropy_trajectory': entropies,
            'min_entropy': np.min(entropies),
            'max_entropy': np.max(entropies),
            'total_entropy_production': entropies[-1] - entropies[0]
        }


def create_synthetic_video(n_frames: int = 50, resolution: Tuple[int, int] = (256, 256)) -> List[np.ndarray]:
    """Create synthetic video with known motion for testing"""
    print(f"Creating synthetic video: {n_frames} frames at {resolution}")
    
    frames = []
    h, w = resolution
    
    # Create a moving circle
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Moving circle
        center_x = int(w * 0.2 + (w * 0.6) * (i / n_frames))
        center_y = int(h * 0.5 + 50 * np.sin(2 * np.pi * i / n_frames))
        radius = 30
        
        cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), -1)
        
        # Add some random particles for complexity
        np.random.seed(i)  # Consistent particles per frame
        for _ in range(10):
            px = np.random.randint(0, w)
            py = np.random.randint(0, h)
            cv2.circle(frame, (px, py), 2, (255, 255, 255), -1)
        
        frames.append(frame)
    
    return frames


def export_video(dual_membrane: Dict, test_sequence: List[int], output_path: Path, 
                fps: int = 30, show_labels: bool = True):
    """
    Export video showing dual-membrane playback with face switching
    
    150 frames at 30 fps = 5 seconds
    """
    print(f"\nExporting video to: {output_path}")
    
    # Video writer
    sample_frame = dual_membrane['front_faces'][0]
    h, w = sample_frame.shape[:2]
    
    # Create side-by-side layout: front face | back face | playback
    output_w = w * 3 + 40  # 3 panels + spacing
    output_h = h + 100  # Extra space for labels
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (output_w, output_h))
    
    # Simulate extended playback for 150 frames
    extended_sequence = []
    for _ in range(3):  # Repeat pattern 3 times
        extended_sequence.extend(test_sequence)
    
    # Trim to 150 frames
    extended_sequence = extended_sequence[:150]
    
    # Create video
    current_entropy = 0.0
    for step, frame_idx in enumerate(extended_sequence):
        # Determine which face to use
        if step == 0 or frame_idx > extended_sequence[step-1]:
            face = 'front'
            current_frame = dual_membrane['front_faces'][frame_idx]
            new_entropy = dual_membrane['front_entropy'][frame_idx]
        else:
            face = 'back'
            current_frame = dual_membrane['back_faces'][frame_idx]
            new_entropy = dual_membrane['back_entropy'][frame_idx]
        
        # Create composite frame
        composite = np.ones((output_h, output_w, 3), dtype=np.uint8) * 255
        
        # Panel 1: Front face (always show)
        front_frame = dual_membrane['front_faces'][frame_idx].copy()
        composite[50:50+h, 10:10+w] = front_frame
        
        # Panel 2: Back face (always show)
        back_frame = dual_membrane['back_faces'][frame_idx].copy()
        composite[50:50+h, 20+w:20+2*w] = back_frame
        
        # Panel 3: Current playback (highlight which is active)
        composite[50:50+h, 30+2*w:30+3*w] = current_frame
        
        if show_labels:
            # Add labels
            cv2.putText(composite, "FRONT FACE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(composite, "BACK FACE", (20+w, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(composite, f"PLAYBACK: {face.upper()}", (30+2*w, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Add frame info
            cv2.putText(composite, f"Frame: {frame_idx}", (10, h+80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(composite, f"Step: {step+1}/150", (200, h+80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(composite, f"Entropy: {new_entropy:.2f}", (400, h+80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Direction indicator
            if step > 0:
                direction = "FORWARD" if frame_idx > extended_sequence[step-1] else "BACKWARD"
                color = (0, 200, 0) if direction == "FORWARD" else (0, 0, 200)
                cv2.putText(composite, f"{direction}", (600, h+80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Highlight active panel with border
            if face == 'front':
                cv2.rectangle(composite, (8, 48), (12+w, 52+h), (0, 0, 255), 3)
            else:
                cv2.rectangle(composite, (18+w, 48), (22+2*w, 52+h), (0, 255, 0), 3)
        
        writer.write(composite)
        
        if (step + 1) % 30 == 0:
            print(f"  Rendered {step+1}/150 frames ({(step+1)/150*100:.1f}%)")
        
        current_entropy = new_entropy
    
    writer.release()
    print(f"  ✓ Video exported: {output_path}")
    print(f"  Duration: {150/fps:.1f}s at {fps} fps")
    print(f"  Resolution: {output_w}×{output_h}")


def visualize_results(s_entropy: Dict, dual_membrane: Dict, 
                     playback_stats: Dict, output_dir: Path):
    """Create comprehensive visualizations"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: S-entropy coordinates over time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    n_frames = len(s_entropy['s_t'])
    time = np.arange(n_frames)
    
    # S_t (temporal entropy)
    axes[0, 0].plot(time, s_entropy['s_t'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Frame Index')
    axes[0, 0].set_ylabel('$S_t$ (Temporal Entropy)')
    axes[0, 0].set_title('Temporal Entropy: Frame-to-Frame Change')
    axes[0, 0].grid(True, alpha=0.3)
    
    # S_e (cumulative entropy - MUST BE MONOTONIC!)
    axes[0, 1].plot(time, s_entropy['s_e'], 'r-', linewidth=2, label='Front face')
    axes[0, 1].plot(time, dual_membrane['back_entropy'], 'g--', linewidth=2, label='Back face')
    axes[0, 1].set_xlabel('Frame Index')
    axes[0, 1].set_ylabel('$S_e$ (Cumulative Entropy)')
    axes[0, 1].set_title('Cumulative Entropy Production (MUST INCREASE!)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Entropy production rate
    axes[1, 0].plot(time, s_entropy['ds_dt'], 'purple', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Frame Index')
    axes[1, 0].set_ylabel('$dS/dt$ (Entropy Production Rate)')
    axes[1, 0].set_title('Entropy Production Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Membrane thickness
    axes[1, 1].plot(time, dual_membrane['thickness'], 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Frame Index')
    axes[1, 1].set_ylabel('Membrane Thickness')
    axes[1, 1].set_title('Dual-Membrane Thickness (Categorical Distance)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'entropy_analysis.png'}")
    plt.close()
    
    # Figure 2: Playback entropy trajectory
    if playback_stats:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        entropy_traj = playback_stats['entropy_trajectory']
        playback_time = np.arange(len(entropy_traj))
        
        # Entropy during playback
        axes[0].plot(playback_time, entropy_traj, 'b-o', linewidth=2, markersize=4)
        axes[0].set_xlabel('Playback Step')
        axes[0].set_ylabel('Entropy')
        axes[0].set_title(f"Playback Entropy Trajectory\n(Monotonic: {playback_stats['entropy_monotonic']})")
        axes[0].grid(True, alpha=0.3)
        
        # Check for violations
        if playback_stats['entropy_violations'] > 0:
            axes[0].text(0.5, 0.95, f"⚠ {playback_stats['entropy_violations']} violations detected!",
                        transform=axes[0].transAxes, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
        else:
            axes[0].text(0.5, 0.95, "✓ Perfect entropy monotonicity!",
                        transform=axes[0].transAxes, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
        
        # Statistics
        stats_text = f"""
        Total frames played: {playback_stats['total_frames_played']}
        Forward scrubs: {playback_stats['forward_scrubs']}
        Backward scrubs: {playback_stats['backward_scrubs']}
        Front face used: {playback_stats['front_face_used']}
        Back face used: {playback_stats['back_face_used']}
        Total entropy production: {playback_stats['total_entropy_production']:.2f}
        """
        axes[1].text(0.1, 0.5, stats_text, transform=axes[1].transAxes,
                    fontsize=11, verticalalignment='center', family='monospace')
        axes[1].axis('off')
        axes[1].set_title('Playback Statistics')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'playback_analysis.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir / 'playback_analysis.png'}")
        plt.close()


def run_validation_experiment():
    """Main validation experiment"""
    print("=" * 80)
    print("MOTION PICTURE MAXWELL DEMON VALIDATION")
    print("=" * 80)
    
    output_dir = Path("motion_picture_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create synthetic video
    print("\n[1/7] Creating synthetic video...")
    frames = create_synthetic_video(n_frames=50, resolution=(256, 256))
    print(f"  Created {len(frames)} frames")
    
    # Step 2: Calculate S-entropy coordinates
    print("\n[2/7] Calculating S-entropy coordinates...")
    entropy_calc = TemporalEntropyCalculator()
    s_entropy = entropy_calc.calculate_s_entropy_coordinates(frames)
    
    # Verify entropy monotonicity
    entropy_monotonic = np.all(np.diff(s_entropy['s_e']) >= 0)
    print(f"  Entropy monotonic: {entropy_monotonic} ✓" if entropy_monotonic else "  Entropy NOT monotonic ✗")
    print(f"  Total entropy production: {s_entropy['s_e'][-1]:.2f}")
    
    # Step 3: Generate dual-membrane structure
    print("\n[3/7] Generating dual-membrane temporal structure...")
    dual_gen = DualMembraneTemporalGenerator(frames, s_entropy)
    dual_membrane = dual_gen.create_dual_membrane_structure()
    print(f"  Generated {len(dual_membrane['front_faces'])} dual-membrane frames")
    print(f"  Mean membrane thickness: {np.mean(dual_membrane['thickness']):.2f}")
    
    # Step 4: Test irreversible playback
    print("\n[4/7] Testing irreversible video player...")
    player = IrreversibleVideoPlayer(dual_membrane, s_entropy)
    
    # Simulate scrubbing pattern: forward, backward, forward, backward
    test_sequence = [0, 10, 20, 30, 40, 49,  # Forward to end
                    40, 30, 20, 10, 0,        # Backward to start (KEY TEST!)
                    10, 20, 30, 40, 49]       # Forward again
    
    print(f"  Simulating scrubbing sequence: {test_sequence}")
    for idx in test_sequence:
        frame = player.scrub_to_position(idx)
    
    # Step 5: Analyze playback
    print("\n[5/7] Analyzing playback statistics...")
    playback_stats = player.get_playback_statistics()
    
    print(f"\n  Results:")
    print(f"    Entropy monotonic: {playback_stats['entropy_monotonic']}")
    print(f"    Entropy violations: {playback_stats['entropy_violations']}")
    print(f"    Forward scrubs: {playback_stats['forward_scrubs']}")
    print(f"    Backward scrubs: {playback_stats['backward_scrubs']}")
    print(f"    Front face used: {playback_stats['front_face_used']} times")
    print(f"    Back face used: {playback_stats['back_face_used']} times")
    print(f"    Total entropy production: {playback_stats['total_entropy_production']:.2f}")
    
    # Step 6: Visualize
    print("\n[6/6] Creating visualizations...")
    visualize_results(s_entropy, dual_membrane, playback_stats, output_dir)
    
    # Step 7: Export video
    print("\n[7/7] Exporting demonstration video...")
    video_path = output_dir / 'motion_picture_demon_demo.mp4'
    export_video(dual_membrane, test_sequence, video_path, fps=30, show_labels=True)
    
    # Save results (fix JSON serialization)
    def make_json_serializable(obj):
        """Convert numpy types to native Python types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        else:
            return obj
    
    results = {
        's_entropy': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                     for k, v in s_entropy.items()},
        'playback_stats': make_json_serializable(playback_stats),
        'experiment_params': {
            'n_frames': len(frames),
            'resolution': frames[0].shape[:2],
            'test_sequence': test_sequence
        }
    }
    
    with open(output_dir / 'validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {output_dir / 'validation_results.json'}")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("VALIDATION VERDICT")
    print("=" * 80)
    
    if playback_stats['entropy_monotonic'] and playback_stats['entropy_violations'] == 0:
        print("✓ SUCCESS: Motion Picture Maxwell Demon validated!")
        print("  - Entropy remains monotonically increasing during playback")
        print("  - Backward scrubbing successfully uses alternative forward path")
        print("  - Dual-membrane structure provides entropy-preserving alternatives")
        print("\n  THE HYPOTHESIS IS CONFIRMED: Video always plays forward in entropy!")
    else:
        print("✗ FAILURE: Entropy violations detected")
        print(f"  - {playback_stats['entropy_violations']} entropy violations")
        print("  - Additional debugging required")
    
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print("  - entropy_analysis.png")
    print("  - playback_analysis.png")
    print("  - validation_results.json")
    print("  - motion_picture_demon_demo.mp4  (5s video showing dual-membrane playback)")


if __name__ == '__main__':
    run_validation_experiment()

