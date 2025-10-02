# generate_all_figures.py
"""
Master script to generate all publication-quality figures
"""

from plot_fluorescence_panels import plot_fluorescence_comprehensive
from plot_video_analysis_panels import plot_video_analysis_comprehensive
from plot_comparative_summary import plot_comparative_summary
from plot_statistical_analysis import plot_statistical_analysis
from plot_timeseries_detailed import plot_timeseries_detailed
from plot_video_tracking_detailed import plot_video_tracking_detailed

def main():
    """Generate all figures for publication"""
    
    # Define file paths
    fluorescence_files = [
        '1585_dapi_comprehensive.json',
        '1585_gfp_comprehensive.json',
        '10954_rfp_comprehensive.json'
    ]
    
    video_files = [
        '7199_web_live_cell_comprehensive.json',
        'astrosoma-g2s2_VOL_time_lapse_comprehensive.json',
        'astrosoma-g3s10_vol_cell_migration_comprehensive.json'
    ]
    
    print("="*60)
    print("Generating Publication-Quality Figures")
    print("="*60)
    
    # Figure 1: Fluorescence Analysis
    print("\n[1/6] Generating Fluorescence Analysis Figure...")
    plot_fluorescence_comprehensive(fluorescence_files, "figure1_fluorescence_analysis")
    
    # Figure 2: Video Analysis
    print("\n[2/6] Generating Video Analysis Figure...")
    plot_video_analysis_comprehensive(video_files, "figure2_video_analysis")
    
    # Figure 3: Comparative Summary
    print("\n[3/6] Generating Comparative Summary Figure...")
    plot_comparative_summary(fluorescence_files, video_files, "figure3_comparative_summary")
    
    # Figure 4: Statistical Analysis
    print("\n[4/6] Generating Statistical Analysis Figure...")
    plot_statistical_analysis(fluorescence_files, "figure4_statistical_analysis")
    
    # Figure 5: Time-Series Detailed
    print("\n[5/6] Generating Time-Series Detailed Figure...")
    plot_timeseries_detailed(fluorescence_files, "figure5_timeseries_detailed")
    
    # Figure 6: Video Tracking Detailed
    print("\n[6/6] Generating Video Tracking Detailed Figure...")
    plot_video_tracking_detailed(video_files, "figure6_video_tracking_detailed")
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("Output formats: PNG, PDF, SVG")
    print("="*60)

if __name__ == "__main__":
    main()
