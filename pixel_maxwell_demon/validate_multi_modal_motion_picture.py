#!/usr/bin/env python3
"""
Multi-Modal Motion Picture Maxwell Demon

Combines virtual imaging with motion picture Maxwell demon:
- Generate videos at different wavelengths/resolutions from a single capture
- Apply entropy-preserving playback to each virtual video
- Demonstrate that the concept works across imaging modalities

This is the UNIFIED FRAMEWORK: 
  Spatial: Virtual imaging (wavelength/resolution changes)
  Temporal: Motion picture demon (entropy-preserving playback)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
import json
from scipy.ndimage import gaussian_filter, zoom
from scipy.stats import entropy as scipy_entropy


class VirtualImagingGenerator:
    """Generate virtual videos at different wavelengths and resolutions"""
    
    def __init__(self, reference_frames: List[np.ndarray], reference_wavelength: float = 550.0):
        self.reference_frames = reference_frames
        self.reference_wavelength = reference_wavelength
        self.n_frames = len(reference_frames)
    
    def wavelength_shift(self, frames: List[np.ndarray], target_wavelength: float) -> List[np.ndarray]:
        """
        Shift wavelength by modulating color channels
        
        Simulates molecular response at different wavelengths
        """
        ratio = target_wavelength / self.reference_wavelength
        shifted_frames = []
        
        for frame in frames:
            if len(frame.shape) == 2:
                # Grayscale: modulate intensity
                shifted = frame * (1.0 + 0.3 * (ratio - 1.0))
            else:
                # Color: shift channels
                shifted = frame.copy().astype(np.float32)
                if ratio < 1.0:  # Blue shift
                    shifted[:, :, 0] *= (1.0 + 0.5 * (1.0 - ratio))  # Boost blue
                    shifted[:, :, 2] *= (0.5 + 0.5 * ratio)  # Reduce red
                else:  # Red shift
                    shifted[:, :, 2] *= (0.5 + 0.5 * ratio)  # Boost red
                    shifted[:, :, 0] *= (1.5 - 0.5 * ratio)  # Reduce blue
                shifted = shifted.astype(np.uint8)
            
            shifted = np.clip(shifted, 0, 255).astype(np.uint8)
            shifted_frames.append(shifted)
        
        return shifted_frames
    
    def resolution_change(self, frames: List[np.ndarray], scale_factor: float) -> List[np.ndarray]:
        """
        Change resolution by scaling
        
        Simulates different sensor pixel sizes or optical magnification
        """
        scaled_frames = []
        
        for frame in frames:
            if scale_factor != 1.0:
                # Zoom (interpolation)
                if len(frame.shape) == 3:
                    scaled = zoom(frame, (scale_factor, scale_factor, 1), order=1)
                else:
                    scaled = zoom(frame, scale_factor, order=1)
            else:
                scaled = frame.copy()
            
            scaled_frames.append(scaled.astype(np.uint8))
        
        return scaled_frames
    
    def generate_virtual_videos(self) -> Dict[str, List[np.ndarray]]:
        """
        Generate multiple virtual videos from reference
        """
        virtual_videos = {
            f'{int(self.reference_wavelength)}nm_original': self.reference_frames,
            '650nm_red': self.wavelength_shift(self.reference_frames, 650.0),
            '450nm_blue': self.wavelength_shift(self.reference_frames, 450.0),
            'high_res_2x': self.resolution_change(self.reference_frames, 2.0),
            'low_res_0.5x': self.resolution_change(self.reference_frames, 0.5)
        }
        
        return virtual_videos


class MotionPictureDemon:
    """Motion picture Maxwell demon for single video"""
    
    def __init__(self, frames: List[np.ndarray], name: str):
        self.frames = frames
        self.name = name
        self.n_frames = len(frames)
        self.s_entropy = self._calculate_entropy()
        self.dual_membrane = self._generate_dual_membrane()
    
    def _calculate_entropy(self) -> Dict[str, np.ndarray]:
        """Calculate S-entropy coordinates"""
        s_t = np.zeros(self.n_frames)
        s_k = np.zeros(self.n_frames)
        
        for i in range(self.n_frames):
            frame = self.frames[i]
            if len(frame.shape) == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame
            
            # Shannon entropy
            hist, _ = np.histogram(frame_gray.flatten(), bins=256, range=(0, 256))
            hist = hist / (hist.sum() + 1e-10)
            hist = hist[hist > 0]
            s_k[i] = scipy_entropy(hist, base=2) if len(hist) > 0 else 0.0
            
            # Temporal entropy (frame difference)
            if i > 0:
                prev_frame = self.frames[i-1]
                if len(prev_frame.shape) == 3:
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                else:
                    prev_gray = prev_frame
                
                diff = np.abs(frame_gray.astype(float) - prev_gray.astype(float))
                s_t[i] = np.mean(diff) / 255.0
        
        # Cumulative entropy
        s_e = np.cumsum(s_t)
        
        return {'s_t': s_t, 's_k': s_k, 's_e': s_e}
    
    def _generate_dual_membrane(self) -> Dict:
        """Generate dual-membrane temporal structure"""
        front_faces = self.frames
        back_faces = []
        front_entropy = self.s_entropy['s_e']
        back_entropy = []
        
        for i in range(self.n_frames):
            if i == 0:
                back_faces.append(self.frames[0].copy())
                back_entropy.append(front_entropy[0])
            else:
                # Alternative forward path: perturb previous frame
                prev_frame = self.frames[i-1]
                noise = np.random.randn(*prev_frame.shape) * 5
                alt_frame = np.clip(prev_frame.astype(float) + noise, 0, 255).astype(np.uint8)
                
                # Blend with current for continuity
                alpha = 0.7
                alt_frame = cv2.addWeighted(alt_frame, alpha, self.frames[i], 1-alpha, 0)
                
                back_faces.append(alt_frame)
                back_entropy.append(front_entropy[i] + np.abs(np.random.randn()) * 0.05)
        
        return {
            'front_faces': front_faces,
            'back_faces': back_faces,
            'front_entropy': front_entropy,
            'back_entropy': np.array(back_entropy),
            'thickness': np.abs(np.array(back_entropy) - front_entropy)
        }
    
    def simulate_playback(self, scrub_sequence: List[int]) -> Dict:
        """Simulate entropy-preserving playback"""
        playback_frames = []
        playback_entropy = []
        faces_used = []
        
        for step, frame_idx in enumerate(scrub_sequence):
            if step == 0 or frame_idx > scrub_sequence[step-1]:
                # Forward: use front face
                face = 'front'
                frame = self.dual_membrane['front_faces'][frame_idx]
                entropy = self.dual_membrane['front_entropy'][frame_idx]
            else:
                # Backward: use back face
                face = 'back'
                frame = self.dual_membrane['back_faces'][frame_idx]
                entropy = self.dual_membrane['back_entropy'][frame_idx]
            
            playback_frames.append(frame)
            playback_entropy.append(entropy)
            faces_used.append(face)
        
        # Check monotonicity
        entropy_diffs = np.diff(playback_entropy)
        monotonic = np.all(entropy_diffs >= -1e-6)  # Small tolerance
        violations = np.sum(entropy_diffs < -1e-6)
        
        return {
            'frames': playback_frames,
            'entropy': np.array(playback_entropy),
            'faces': faces_used,
            'monotonic': monotonic,
            'violations': int(violations)
        }


def create_synthetic_video(n_frames: int = 60, resolution: Tuple[int, int] = (256, 256)) -> List[np.ndarray]:
    """Create synthetic video with complex motion"""
    print(f"Creating synthetic reference video: {n_frames} frames at {resolution}")
    
    frames = []
    h, w = resolution
    
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Moving circle 1
        x1 = int(w * 0.2 + (w * 0.6) * (i / n_frames))
        y1 = int(h * 0.5 + 50 * np.sin(2 * np.pi * i / n_frames))
        cv2.circle(frame, (x1, y1), 25, (0, 255, 0), -1)
        
        # Moving circle 2 (counter-rotating)
        x2 = int(w * 0.8 - (w * 0.6) * (i / n_frames))
        y2 = int(h * 0.5 - 40 * np.cos(2 * np.pi * i / n_frames))
        cv2.circle(frame, (x2, y2), 20, (255, 0, 0), -1)
        
        # Background particles
        np.random.seed(i)
        for _ in range(15):
            px = np.random.randint(0, w)
            py = np.random.randint(0, h)
            cv2.circle(frame, (px, py), 2, (255, 255, 255), -1)
        
        frames.append(frame)
    
    return frames


def export_multi_modal_video(virtual_videos: Dict[str, List[np.ndarray]], 
                             motion_demons: Dict[str, MotionPictureDemon],
                             scrub_sequence: List[int],
                             output_path: Path, fps: int = 30):
    """
    Export video showing multiple modalities with entropy-preserving playback
    
    Layout: Grid of videos (2×3 = 6 panels)
    Each panel shows one modality with face switching
    """
    print(f"\nExporting multi-modal video to: {output_path}")
    
    # Get all modality keys (sorted)
    modalities = sorted(list(virtual_videos.keys()))
    
    # Get sample dimensions
    sample_frames = {}
    for mod in modalities:
        sample_frames[mod] = virtual_videos[mod][0]
    
    # Normalize all to same size (use max dimensions)
    max_h = max(f.shape[0] for f in sample_frames.values())
    max_w = max(f.shape[1] for f in sample_frames.values())
    target_size = (max_w, max_h)
    
    # Grid layout: 2 rows × 3 cols
    panel_w, panel_h = target_size
    grid_w = panel_w * 3 + 40
    grid_h = panel_h * 2 + 100
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (grid_w, grid_h))
    
    # Extended sequence for 180 frames (6 seconds at 30fps)
    extended_sequence = []
    for _ in range(3):
        extended_sequence.extend(scrub_sequence)
    extended_sequence = extended_sequence[:180]
    
    # Simulate playback for each modality
    print("  Simulating playback for each modality...")
    playback_results = {}
    for mod_name in modalities:
        demon = motion_demons[mod_name]
        playback_results[mod_name] = demon.simulate_playback(extended_sequence)
    
    # Render video
    print("  Rendering multi-modal video...")
    for step in range(len(extended_sequence)):
        canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240
        
        # Render each modality in grid
        for idx, mod_name in enumerate(modalities[:6]):  # Max 6 panels
            row = idx // 3
            col = idx % 3
            
            x_offset = 10 + col * (panel_w + 10)
            y_offset = 50 + row * (panel_h + 30)
            
            # Get current frame
            frame = playback_results[mod_name]['frames'][step]
            
            # Resize if needed
            if frame.shape[0] != panel_h or frame.shape[1] != panel_w:
                frame = cv2.resize(frame, target_size)
            
            # Place frame
            canvas[y_offset:y_offset+panel_h, x_offset:x_offset+panel_w] = frame
            
            # Add label
            label = mod_name.replace('_', ' ').upper()
            cv2.putText(canvas, label, (x_offset, y_offset-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Face indicator
            face = playback_results[mod_name]['faces'][step]
            color = (0, 0, 200) if face == 'front' else (0, 200, 0)
            cv2.circle(canvas, (x_offset + panel_w - 15, y_offset + 15), 8, color, -1)
        
        # Global info
        cv2.putText(canvas, f"MULTI-MODAL MOTION PICTURE DEMON", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(canvas, f"Frame: {step+1}/180", (grid_w - 200, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Legend
        cv2.circle(canvas, (grid_w - 200, grid_h - 30), 8, (0, 0, 200), -1)
        cv2.putText(canvas, "Front Face", (grid_w - 185, grid_h - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.circle(canvas, (grid_w - 100, grid_h - 30), 8, (0, 200, 0), -1)
        cv2.putText(canvas, "Back Face", (grid_w - 85, grid_h - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        writer.write(canvas)
        
        if (step + 1) % 30 == 0:
            print(f"    Rendered {step+1}/180 frames")
    
    writer.release()
    print(f"  ✓ Multi-modal video exported: {output_path}")
    print(f"  Duration: 6.0s at {fps} fps")


def visualize_multi_modal_entropy(motion_demons: Dict[str, MotionPictureDemon],
                                  scrub_sequence: List[int], output_dir: Path):
    """Visualize entropy across modalities"""
    print("  Creating multi-modal entropy visualization...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Extended sequence
    extended_sequence = []
    for _ in range(3):
        extended_sequence.extend(scrub_sequence)
    extended_sequence = extended_sequence[:180]
    
    # Plot 1: Entropy evolution during playback
    ax1 = axes[0]
    for mod_name, demon in motion_demons.items():
        playback = demon.simulate_playback(extended_sequence)
        ax1.plot(playback['entropy'], linewidth=2, alpha=0.7, label=mod_name)
    
    ax1.set_xlabel('Playback Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Entropy', fontsize=12, fontweight='bold')
    ax1.set_title('Multi-Modal Entropy Evolution During Playback\n(All Must Be Monotonic!)',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Monotonicity verification
    ax2 = axes[1]
    modalities = list(motion_demons.keys())
    monotonic_counts = []
    violation_counts = []
    
    for mod_name in modalities:
        demon = motion_demons[mod_name]
        playback = demon.simulate_playback(extended_sequence)
        monotonic_counts.append(1 if playback['monotonic'] else 0)
        violation_counts.append(playback['violations'])
    
    x = np.arange(len(modalities))
    width = 0.35
    
    ax2.bar(x - width/2, monotonic_counts, width, label='Monotonic (1=Yes)', 
           color='green', alpha=0.7, edgecolor='black', linewidth=2)
    ax2.bar(x + width/2, violation_counts, width, label='Violations', 
           color='red', alpha=0.7, edgecolor='black', linewidth=2)
    
    ax2.set_xlabel('Modality', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Entropy Monotonicity Verification Across Modalities',
                 fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('_', '\n') for m in modalities], fontsize=8, rotation=0)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multi_modal_entropy_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {output_dir / 'multi_modal_entropy_analysis.png'}")


def run_multi_modal_experiment():
    """Main experiment combining virtual imaging + motion picture demon"""
    print("=" * 80)
    print("MULTI-MODAL MOTION PICTURE MAXWELL DEMON")
    print("Unified Framework: Virtual Imaging × Entropy-Preserving Playback")
    print("=" * 80)
    
    output_dir = Path("multi_modal_motion_picture")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create reference video
    print("\n[1/6] Creating reference video...")
    reference_frames = create_synthetic_video(n_frames=60, resolution=(256, 256))
    print(f"  Created {len(reference_frames)} frames")
    
    # Step 2: Generate virtual videos
    print("\n[2/6] Generating virtual videos at different modalities...")
    virtual_gen = VirtualImagingGenerator(reference_frames, reference_wavelength=550.0)
    virtual_videos = virtual_gen.generate_virtual_videos()
    print(f"  Generated {len(virtual_videos)} virtual videos:")
    for name, frames in virtual_videos.items():
        print(f"    • {name}: {len(frames)} frames, {frames[0].shape}")
    
    # Step 3: Create motion picture demon for each modality
    print("\n[3/6] Creating motion picture demons for each modality...")
    motion_demons = {}
    for mod_name, frames in virtual_videos.items():
        demon = MotionPictureDemon(frames, mod_name)
        motion_demons[mod_name] = demon
        print(f"    • {mod_name}: S_e = {demon.s_entropy['s_e'][-1]:.2f}")
    
    # Step 4: Test playback
    print("\n[4/6] Testing entropy-preserving playback...")
    scrub_sequence = [0, 10, 20, 30, 40, 50, 59,  # Forward
                     50, 40, 30, 20, 10, 0,        # Backward
                     10, 20, 30, 40, 50, 59]       # Forward again
    
    results_summary = {}
    for mod_name, demon in motion_demons.items():
        playback = demon.simulate_playback(scrub_sequence)
        results_summary[mod_name] = {
            'monotonic': bool(playback['monotonic']),
            'violations': playback['violations'],
            'total_entropy': float(playback['entropy'][-1])
        }
    
    print("\n  Playback Results:")
    for mod_name, results in results_summary.items():
        status = "✓" if results['monotonic'] else "✗"
        print(f"    {status} {mod_name}: violations={results['violations']}, "
              f"total_entropy={results['total_entropy']:.2f}")
    
    # Step 5: Export multi-modal video
    print("\n[5/6] Exporting multi-modal demonstration video...")
    video_path = output_dir / 'multi_modal_motion_picture_demo.mp4'
    export_multi_modal_video(virtual_videos, motion_demons, scrub_sequence, 
                            video_path, fps=30)
    
    # Step 6: Visualize entropy analysis
    print("\n[6/6] Creating entropy visualizations...")
    visualize_multi_modal_entropy(motion_demons, scrub_sequence, output_dir)
    
    # Save results
    with open(output_dir / 'multi_modal_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"  Saved: {output_dir / 'multi_modal_results.json'}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("MULTI-MODAL VALIDATION COMPLETE")
    print("=" * 80)
    
    all_monotonic = all(r['monotonic'] for r in results_summary.values())
    total_violations = sum(r['violations'] for r in results_summary.values())
    
    if all_monotonic and total_violations == 0:
        print("✓ SUCCESS: Motion picture demon validated across ALL modalities!")
        print(f"  • {len(virtual_videos)} different imaging modalities tested")
        print(f"  • All maintain entropy monotonicity during playback")
        print(f"  • Zero violations across all modalities")
        print("\n  CONCLUSION: Entropy-preserving playback is UNIVERSAL")
        print("              Works for any wavelength, resolution, or modality!")
    else:
        print(f"⚠ PARTIAL SUCCESS: {total_violations} total violations across modalities")
        print("  Some modalities require refinement")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {output_dir}/")
    print("  - multi_modal_motion_picture_demo.mp4  (6s grid video)")
    print("  - multi_modal_entropy_analysis.png")
    print("  - multi_modal_results.json")
    print("=" * 80)


if __name__ == '__main__':
    run_multi_modal_experiment()

