"""
Configuration file for Helicopter Life Science demos.

Simply change the paths below to point to your data files and run the demos.
"""

import os
from pathlib import Path

# Base directory for all data
DATA_DIR = Path(__file__).parent / "public"

# ====================================
# CONFIGURE YOUR DATA PATHS HERE
# ====================================

# Single microscopy images for static analysis
MICROSCOPY_IMAGES = {
    'fluorescence_cell': DATA_DIR / "1585.jpg",
    'high_res_tissue': DATA_DIR / "1585.tif", 
    'cellular_structure': DATA_DIR / "10954.jpg",
    'electron_microscopy': DATA_DIR / "36068.jpg",
    'protein_structure': DATA_DIR / "36075.jpg",
    'tissue_sample': DATA_DIR / "9701.jpg",
    'cell_culture': DATA_DIR / "2_512s.jpg",
    'organelle_detail': DATA_DIR / "23_512v.jpg",
    'membrane_structure': DATA_DIR / "50_512v.jpg",
    'cellular_division': DATA_DIR / "6729_512v.jpg",
    'mitochondria': DATA_DIR / "6733_512r.jpg",
    'nucleus_detail': DATA_DIR / "7298_512r.jpg"
}

# Video files for temporal analysis
MICROSCOPY_VIDEOS = {
    'live_cell_imaging': DATA_DIR / "7199_web.mp4",
    'time_lapse_development': DATA_DIR / "astrosoma-g2s2_VOL.mpg",
    'cellular_dynamics': DATA_DIR / "astrosoma-g3s10_vol.mpg"
}

# Archive files (may contain image sequences)
MICROSCOPY_ARCHIVES = {
    'dataset_1': DATA_DIR / "10105.zip",
    'dataset_2': DATA_DIR / "55784.zip"
}

# ====================================
# ANALYSIS CONFIGURATION
# ====================================

# Which modules to run (set to False to skip)
RUN_MODULES = {
    'gas_molecular': True,
    'entropy_analysis': True, 
    'fluorescence': True,
    'electron_microscopy': True,
    'video_analysis': True,
    'meta_information': True
}

# Output directory for results
OUTPUT_DIR = Path(__file__).parent / "results"

# Visualization settings
SAVE_FIGURES = True
SHOW_FIGURES = False  # Set to True to display plots interactively

# Analysis parameters
ANALYSIS_PARAMS = {
    'gas_molecular': {
        'evolution_steps': 1000,
        'structure_type': 'folded'  # or 'unfolded'
    },
    'entropy': {
        'biological_context': 'cellular'  # 'cellular', 'tissue', 'molecular'
    },
    'fluorescence': {
        'channel': 'gfp',  # 'dapi', 'gfp', 'rfp', 'fitc'
        'background_subtraction': True
    },
    'electron_microscopy': {
        'em_type': 'tem',  # 'sem', 'tem', 'cryo_em'
        'target_structures': ['mitochondria', 'vesicles', 'nucleus']
    },
    'video': {
        'video_type': 'live_cell',  # 'live_cell', 'time_lapse', 'calcium_imaging'
        'frame_interval': 1.0
    },
    'meta': {
        'compression_analysis': True
    }
}

def get_valid_files():
    """Get list of files that actually exist"""
    valid_images = {}
    valid_videos = {}
    valid_archives = {}
    
    for name, path in MICROSCOPY_IMAGES.items():
        if path.exists():
            valid_images[name] = path
    
    for name, path in MICROSCOPY_VIDEOS.items():
        if path.exists():
            valid_videos[name] = path
            
    for name, path in MICROSCOPY_ARCHIVES.items():
        if path.exists():
            valid_archives[name] = path
    
    return valid_images, valid_videos, valid_archives

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR
