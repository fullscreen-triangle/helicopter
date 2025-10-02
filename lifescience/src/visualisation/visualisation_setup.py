# visualization_setup.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path

# Publication-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.titlesize'] = 11

# Color palettes
CHANNEL_COLORS = {
    'dapi': '#4A90E2',  # Blue
    'gfp': '#7ED321',   # Green
    'rfp': '#E74C3C',   # Red
}

def load_json(filepath):
    """Load JSON data from file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_figure(fig, filename, tight=True):
    """Save figure in multiple formats"""
    if tight:
        fig.tight_layout()
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{filename}.pdf", bbox_inches='tight')
    fig.savefig(f"{filename}.svg", bbox_inches='tight')
    print(f"Saved: {filename}")
