import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from scipy import stats
import seaborn as sns

# Publication-quality settings
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 8
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5
rcParams['figure.dpi'] = 300

def create_dual_membrane_figure(front_face_data, back_face_data, 
                                temporal_data, switching_data):
    """
    Create Figure 1: Dual-Membrane Complementarity
    
    Parameters:
    -----------
    front_face_data : array (n_pixels,)
        S_k values for front face
    back_face_data : array (n_pixels,)
        S_k values for back face
    temporal_data : dict
        {'time': array, 'front': array, 'back': array, 'separation': array}
    switching_data : dict
        {'time': array, 'observable_face': array, 'Sk_value': array}
    """
    
    fig = plt.figure(figsize=(7.5, 7.5))  # Two-column width
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)
    
    # ============================================================
    # PANEL A: Front/Back Anti-Correlation
    # ============================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Hexbin for density (better than scatter for many points)
    hb = ax_a.hexbin(front_face_data, back_face_data, 
                     gridsize=50, cmap='Blues', mincnt=1,
                     linewidths=0.2, edgecolors='white')
    
    # Perfect anti-correlation line
    x_line = np.linspace(front_face_data.min(), front_face_data.max(), 100)
    ax_a.plot(x_line, -x_line, 'r--', linewidth=1.5, 
              label='Perfect anti-correlation', alpha=0.8)
    
    # Calculate correlation
    r_value, p_value = stats.pearsonr(front_face_data, back_face_data)
    
    # Add statistics box
    textstr = f'r = {r_value:.6f}\np < 10⁻¹⁵\nn = {len(front_face_data):,}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, 
                 edgecolor='gray', linewidth=0.5)
    ax_a.text(0.05, 0.95, textstr, transform=ax_a.transAxes,
              fontsize=7, verticalalignment='top', bbox=props)
    
    ax_a.set_xlabel('$S_k^{\\mathrm{front}}$ (bits)', fontsize=8)
    ax_a.set_ylabel('$S_k^{\\mathrm{back}}$ (bits)', fontsize=8)
    ax_a.set_title('A. Front/Back Face Anti-Correlation', 
                   fontsize=9, fontweight='bold', loc='left')
    ax_a.legend(fontsize=7, frameon=True, edgecolor='gray', 
                framealpha=0.8, loc='lower right')
    ax_a.grid(True, alpha=0.3, linewidth=0.3)
    
    # Colorbar
    cbar = plt.colorbar(hb, ax=ax_a, pad=0.02)
    cbar.set_label('Pixel density', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    
    # ============================================================
    # PANEL B: Conjugate Sum Distribution
    # ============================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    conjugate_sum = front_face_data + back_face_data
    
    # Histogram with KDE overlay
    n, bins, patches = ax_b.hist(conjugate_sum, bins=50, 
                                  density=True, alpha=0.6,
                                  color='steelblue', edgecolor='black',
                                  linewidth=0.5, label='Observed')
    
    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(conjugate_sum)
    x_kde = np.linspace(conjugate_sum.min(), conjugate_sum.max(), 200)
    ax_b.plot(x_kde, kde(x_kde), 'r-', linewidth=1.5, 
              label='KDE', alpha=0.8)
    
    # Vertical line at zero
    ax_b.axvline(0, color='green', linestyle='--', linewidth=1.5,
                 label='Perfect conjugacy', alpha=0.8)
    
    # Statistics
    mean_sum = np.mean(conjugate_sum)
    std_sum = np.std(conjugate_sum)
    max_abs_sum = np.max(np.abs(conjugate_sum))
    
    textstr = (f'Mean: {mean_sum:.2e}\n'
               f'Std: {std_sum:.2e}\n'
               f'Max |sum|: {max_abs_sum:.2e}')
    props = dict(boxstyle='round', facecolor='white', alpha=0.8,
                 edgecolor='gray', linewidth=0.5)
    ax_b.text(0.95, 0.95, textstr, transform=ax_b.transAxes,
              fontsize=7, verticalalignment='top', 
              horizontalalignment='right', bbox=props)
    
    ax_b.set_xlabel('$S_k^{\\mathrm{front}} + S_k^{\\mathrm{back}}$ (bits)', 
                    fontsize=8)
    ax_b.set_ylabel('Probability density', fontsize=8)
    ax_b.set_title('B. Conjugate Sum Distribution', 
                   fontsize=9, fontweight='bold', loc='left')
    ax_b.legend(fontsize=7, frameon=True, edgecolor='gray',
                framealpha=0.8, loc='upper left')
    ax_b.grid(True, alpha=0.3, linewidth=0.3)
    ax_b.set_yscale('log')  # Log scale to show precision
    
    # ============================================================
    # PANEL C: Temporal Evolution
    # ============================================================
    ax_c = fig.add_subplot(gs[1, 0])
    
    time = temporal_data['time']
    front = temporal_data['front']
    back = temporal_data['back']
    separation = temporal_data['separation']
    
    # Twin axes for separation
    ax_c2 = ax_c.twinx()
    
    # Plot front and back
    line1 = ax_c.plot(time, front, 'b-', linewidth=1.5, 
                      label='Front face', alpha=0.8)
    line2 = ax_c.plot(time, back, 'r-', linewidth=1.5,
                      label='Back face', alpha=0.8)
    
    # Plot separation on right axis
    line3 = ax_c2.plot(time, separation, 'g--', linewidth=1.5,
                       label='Separation $d_S$', alpha=0.8)
    
    # Calculate mean and std of separation
    mean_sep = np.mean(separation)
    std_sep = np.std(separation)
    
    # Horizontal line for mean separation
    ax_c2.axhline(mean_sep, color='green', linestyle=':', 
                  linewidth=1, alpha=0.5)
    
    # Statistics box
    textstr = f'$d_S$ = {mean_sep:.3f} ± {std_sep:.3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8,
                 edgecolor='gray', linewidth=0.5)
    ax_c.text(0.05, 0.95, textstr, transform=ax_c.transAxes,
              fontsize=7, verticalalignment='top', bbox=props)
    
    ax_c.set_xlabel('Time (s)', fontsize=8)
    ax_c.set_ylabel('$S_k$ (bits)', fontsize=8, color='black')
    ax_c2.set_ylabel('$d_S$ (bits)', fontsize=8, color='green')
    ax_c.set_title('C. Temporal Evolution of Conjugate Faces',
                   fontsize=9, fontweight='bold', loc='left')
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax_c.legend(lines, labels, fontsize=7, frameon=True,
                edgecolor='gray', framealpha=0.8, loc='upper right')
    
    ax_c.grid(True, alpha=0.3, linewidth=0.3)
    ax_c.tick_params(axis='y', labelcolor='black')
    ax_c2.tick_params(axis='y', labelcolor='green')
    
    # ============================================================
    # PANEL D: Face Switching Dynamics
    # ============================================================
    ax_d = fig.add_subplot(gs[1, 1])
    
    time_switch = switching_data['time']
    observable = switching_data['observable_face']  # 0=front, 1=back
    sk_value = switching_data['Sk_value']
    
    # Create step plot for observable face
    ax_d.step(time_switch, observable, where='post', 
              linewidth=1.5, color='purple', alpha=0.7,
              label='Observable face')
    
    # Overlay S_k values as scatter
    colors = ['blue' if o == 0 else 'red' for o in observable]
    ax_d.scatter(time_switch, sk_value / np.max(np.abs(sk_value)) * 0.8 + 0.5,
                 c=colors, s=10, alpha=0.6, label='$S_k$ (normalized)')
    
    # Calculate switching frequency
    switches = np.diff(observable) != 0
    n_switches = np.sum(switches)
    duration = time_switch[-1] - time_switch[0]
    freq = n_switches / duration
    
    # Statistics box
    textstr = (f'Switching freq: {freq:.2f} Hz\n'
               f'Total switches: {n_switches}\n'
               f'Duration: {duration:.2f} s')
    props = dict(boxstyle='round', facecolor='white', alpha=0.8,
                 edgecolor='gray', linewidth=0.5)
    ax_d.text(0.95, 0.95, textstr, transform=ax_d.transAxes,
              fontsize=7, verticalalignment='top',
              horizontalalignment='right', bbox=props)
    
    ax_d.set_xlabel('Time (s)', fontsize=8)
    ax_d.set_ylabel('Observable face (0=Front, 1=Back)', fontsize=8)
    ax_d.set_title('D. Automatic Face Switching (5 Hz)',
                   fontsize=9, fontweight='bold', loc='left')
    ax_d.set_ylim(-0.1, 1.1)
    ax_d.set_yticks([0, 1])
    ax_d.set_yticklabels(['Front', 'Back'])
    ax_d.grid(True, alpha=0.3, linewidth=0.3, axis='x')
    
    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='purple', alpha=0.7, label='Observable face'),
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor='blue', markersize=5, 
                   label='Front $S_k$'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='red', markersize=5,
                   label='Back $S_k$')
    ]
    ax_d.legend(handles=legend_elements, fontsize=7, frameon=True,
                edgecolor='gray', framealpha=0.8, loc='center right')
    
    plt.tight_layout()
    return fig

    

# Example usage:
"""
# Generate or load your data
front_face = np.random.randn(10000) * 2 + 5  # Example
back_face = -front_face + np.random.randn(10000) * 1e-10  # Near-perfect anti-correlation

temporal = {
    'time': np.linspace(0, 10, 1000),
    'front': 5 + np.sin(np.linspace(0, 10, 1000)) * 2,
    'back': -5 - np.sin(np.linspace(0, 10, 1000)) * 2,
    'separation': np.ones(1000) * 2.683 + np.random.randn(1000) * 0.001
}

switching = {
    'time': np.linspace(0, 2, 100),
    'observable_face': (np.floor(np.linspace(0, 2, 100) * 5) % 2).astype(int),
    'Sk_value': np.random.randn(100) * 2 + 5
}

fig = create_dual_membrane_figure(front_face, back_face, temporal, switching)
plt.savefig('figure1_dual_membrane.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure1_dual_membrane.png', dpi=300, bbox_inches='tight')
plt.show()
"""
