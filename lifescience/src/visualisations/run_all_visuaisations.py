# run_all_visualisations.py
"""
Master Script for Publication-Quality Figure Generation
Generates all comprehensive analysis figures from your data
"""

from pathlib import Path
import sys

# Add current directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(Path(__file__).parent))

# Import all plotting functions
from plot_fluorescence_overview import plot_fluorescence_overview
from plot_fluorescence_detailed import plot_fluorescence_detailed
from plot_video_tracking_overview import plot_video_tracking_overview
from plot_video_tracking_detailed import plot_video_tracking_detailed
from plot_comparative_summary import plot_comparative_summary


def main():
    """Generate all publication-quality figures"""

    # Define paths
    results_dir = project_root / "results"
    output_dir = project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PUBLICATION-QUALITY FIGURE GENERATION")
    print("=" * 70)

    # ========================================================================
    # FLUORESCENCE MICROSCOPY FIGURES
    # ========================================================================
    print("\n📊 Generating Fluorescence Microscopy Figures...")

    fluor_files = [
        results_dir / "1585_dapi_comprehensive.json",
        results_dir / "1585_gfp_comprehensive.json",
        results_dir / "10954_rfp_comprehensive.json"
    ]

    # Check if files exist
    fluor_exist = all(f.exists() for f in fluor_files)

    if fluor_exist:
        # Figure 1: Multi-Channel Overview
        print("  → Figure 1: Multi-Channel Fluorescence Overview")
        try:
            plot_fluorescence_overview(
                fluor_files,
                output_dir / "figure1_fluorescence_overview"
            )
            print("    ✓ Success!")
        except Exception as e:
            print(f"    ✗ Error: {e}")

        # Figure 2: Channel-Specific Detailed Analysis
        print("  → Figure 2: Channel-Specific Detailed Analysis")
        try:
            plot_fluorescence_detailed(
                fluor_files,
                output_dir / "figure2_fluorescence_detailed"
            )
            print("    ✓ Success!")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    else:
        print("  ⚠ Fluorescence files not found, skipping...")

    # ========================================================================
    # VIDEO TRACKING FIGURES
    # ========================================================================
    print("\n🎬 Generating Video Tracking Figures...")

    video_files = [
        results_dir / "7199_web_live_cell_comprehensive.json",
        results_dir / "astrosoma-g2s2_VOL_time_lapse_comprehensive.json",
        results_dir / "astrosoma-g3s10_vol_cell_migration_comprehensive.json"
    ]

    # Check if files exist
    video_exist = all(f.exists() for f in video_files)

    if video_exist:
        # Figure 3: Video Tracking Overview
        print("  → Figure 3: Video Tracking Overview")
        try:
            plot_video_tracking_overview(
                video_files,
                output_dir / "figure3_video_tracking_overview"
            )
            print("    ✓ Success!")
        except Exception as e:
            print(f"    ✗ Error: {e}")

        # Figure 4: Video Tracking Detailed
        print("  → Figure 4: Video Tracking Detailed Analysis")
        try:
            plot_video_tracking_detailed(
                video_files,
                output_dir / "figure4_video_tracking_detailed"
            )
            print("    ✓ Success!")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    else:
        print("  ⚠ Video files not found, skipping...")

    # ========================================================================
    # COMPARATIVE ANALYSIS
    # ========================================================================
    print("\n📈 Generating Comparative Analysis...")

    # Figure 5: Video Comparative Summary
    if video_exist:
        print("  → Figure 5: Video Comparative Summary")
        try:
            plot_comparative_summary(
                video_files,
                output_dir / "figure5_video_comparative_summary"
            )
            print("    ✓ Success!")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    # Figure 6: Fluorescence Comparative Summary
    if fluor_exist:
        print("  → Figure 6: Fluorescence Comparative Summary")
        try:
            plot_comparative_summary(
                fluor_files,
                output_dir / "figure6_fluorescence_comparative_summary"
            )
            print("    ✓ Success!")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    # ========================================================================
    # COMPLETION SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ FIGURE GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Output directory: {output_dir}")
    print("\n📊 Generated figures:")

    # List generated files
    generated_files = sorted(output_dir.glob("figure*.png"))
    if generated_files:
        for i, file in enumerate(generated_files, 1):
            print(f"  {i}. {file.name}")
    else:
        print("  No figures were generated. Check error messages above.")

    print("\n🎉 Ready for publication!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
