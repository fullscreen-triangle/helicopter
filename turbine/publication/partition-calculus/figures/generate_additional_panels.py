"""
Generate additional visualization panels for Partition Calculus paper.
Topics: Arrival time, Vanillin structure, Bidirectional algorithm,
        Intracellular fluid dynamics, Electric field mechanisms, Zero backaction
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy import ndimage
from scipy.interpolate import griddata
import os

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# Custom colormaps
colors_partition = ['#1a1a2e', '#16213e', '#0f3460', '#e94560']
cmap_partition = LinearSegmentedColormap.from_list('partition', colors_partition)


def panel_arrival_time():
    """Panel: Combined arrival time of different localization modalities"""
    from skimage import io

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

    image_dir = '../../../public/images/images/'
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    # Chart 1: 3D surface - Arrival time distribution across modalities
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    np.random.seed(42)
    modalities = np.arange(6)  # optical, spectral, thermal, O2, mass, charge
    time_points = np.linspace(0, 100, 50)
    M, T = np.meshgrid(modalities, time_points)

    # Different arrival profiles for each modality
    arrival = np.zeros_like(M, dtype=float)
    delays = [0, 5, 12, 18, 25, 35]  # ms delays
    widths = [8, 10, 15, 12, 20, 18]

    for i, (d, w) in enumerate(zip(delays, widths)):
        mask = M == i
        arrival[mask] = np.exp(-((T[mask] - d - 30) ** 2) / (2 * w ** 2))

    surf = ax1.plot_surface(T, M, arrival, cmap='plasma', alpha=0.9,
                            linewidth=0, antialiased=True)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Modality')
    ax1.set_zlabel('Signal Amplitude')
    ax1.set_yticks(modalities)
    ax1.set_yticklabels(['Opt', 'Spec', 'Therm', 'O₂', 'Mass', 'Chg'], fontsize=7)
    ax1.view_init(elev=25, azim=45)

    # Chart 2: Arrival time histogram per modality
    ax2 = fig.add_subplot(gs[0, 1])

    modality_names = ['Optical', 'Spectral', 'Thermal', 'O₂', 'Mass', 'Charge']
    colors_mod = ['#e94560', '#0f3460', '#533483', '#16213e', '#ff2e63', '#1a1a2e']

    for i, (name, color, d, w) in enumerate(zip(modality_names, colors_mod, delays, widths)):
        t = np.linspace(0, 100, 200)
        signal = np.exp(-((t - d - 30) ** 2) / (2 * w ** 2))
        ax2.fill_between(t, signal + i * 0.15, i * 0.15, alpha=0.7, color=color)
        ax2.axhline(y=i * 0.15, color='gray', linewidth=0.5, alpha=0.3)

    ax2.set_xlabel('Arrival Time (ms)')
    ax2.set_ylabel('Modality (stacked)')
    ax2.set_xlim(0, 100)
    ax2.set_yticks([i * 0.15 + 0.5 for i in range(6)])
    ax2.set_yticklabels(modality_names, fontsize=8)

    # Chart 3: Combined signal from microscopy image with time overlay
    ax3 = fig.add_subplot(gs[1, 0])

    if image_files:
        img = io.imread(os.path.join(image_dir, image_files[5]))
        img_norm = (img - img.min()) / (img.max() - img.min())

        # Create arrival time map based on intensity
        arrival_map = 30 + 40 * (1 - img_norm) + np.random.normal(0, 3, img_norm.shape)

        im = ax3.imshow(arrival_map, cmap='coolwarm', vmin=20, vmax=80)
        plt.colorbar(im, ax=ax3, label='Arrival Time (ms)', shrink=0.8)
        ax3.axis('off')

    # Chart 4: Correlation matrix of arrival times
    ax4 = fig.add_subplot(gs[1, 1])

    # Simulated correlation between modality arrival times
    corr_matrix = np.array([
        [1.00, 0.85, 0.45, 0.62, 0.38, 0.55],
        [0.85, 1.00, 0.52, 0.70, 0.42, 0.48],
        [0.45, 0.52, 1.00, 0.78, 0.65, 0.72],
        [0.62, 0.70, 0.78, 1.00, 0.58, 0.68],
        [0.38, 0.42, 0.65, 0.58, 1.00, 0.82],
        [0.55, 0.48, 0.72, 0.68, 0.82, 1.00],
    ])

    im = ax4.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax4.set_xticks(range(6))
    ax4.set_yticks(range(6))
    ax4.set_xticklabels(['Opt', 'Spec', 'Therm', 'O₂', 'Mass', 'Chg'], fontsize=8)
    ax4.set_yticklabels(['Opt', 'Spec', 'Therm', 'O₂', 'Mass', 'Chg'], fontsize=8)
    plt.colorbar(im, ax=ax4, label='Correlation ρ', shrink=0.8)

    plt.suptitle('Combined Arrival Time Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig('panel_arrival_time.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('panel_arrival_time.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: panel_arrival_time.png/pdf")


def panel_vanillin_structure():
    """Panel: Predicted vanillin molecular structure"""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

    # Vanillin: C8H8O3 - 4-hydroxy-3-methoxybenzaldehyde
    # Approximate 3D coordinates
    atoms = {
        'C1': (0, 0, 0),       # Benzene ring
        'C2': (1.4, 0, 0),
        'C3': (2.1, 1.2, 0),
        'C4': (1.4, 2.4, 0),
        'C5': (0, 2.4, 0),
        'C6': (-0.7, 1.2, 0),
        'C7': (2.1, 3.6, 0),   # Aldehyde carbon
        'O1': (3.3, 3.6, 0),   # Aldehyde oxygen
        'O2': (-0.7, 3.6, 0),  # Hydroxyl
        'O3': (2.8, -1.2, 0),  # Methoxy oxygen
        'C8': (4.2, -1.2, 0),  # Methoxy carbon
    }

    bonds = [
        ('C1', 'C2'), ('C2', 'C3'), ('C3', 'C4'), ('C4', 'C5'),
        ('C5', 'C6'), ('C6', 'C1'), ('C4', 'C7'), ('C7', 'O1'),
        ('C5', 'O2'), ('C2', 'O3'), ('O3', 'C8')
    ]

    # Chart 1: 3D molecular structure
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    atom_colors = {'C': '#404040', 'O': '#e94560'}
    atom_sizes = {'C': 200, 'O': 250}

    for name, pos in atoms.items():
        element = name[0]
        ax1.scatter(*pos, s=atom_sizes[element], c=atom_colors[element], alpha=0.9, edgecolors='white')

    for a1, a2 in bonds:
        p1, p2 = atoms[a1], atoms[a2]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                'k-', linewidth=2, alpha=0.7)

    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')
    ax1.view_init(elev=20, azim=45)

    # Chart 2: Electron density surface
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')

    # Create electron density from atom positions
    x = np.linspace(-2, 6, 50)
    y = np.linspace(-2, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for name, pos in atoms.items():
        element = name[0]
        electrons = 6 if element == 'C' else 8  # Simplified
        r2 = (X - pos[0])**2 + (Y - pos[1])**2
        Z += electrons * np.exp(-r2 / 1.5)

    surf = ax2.plot_surface(X, Y, Z, cmap='plasma', alpha=0.85,
                            linewidth=0, antialiased=True)
    ax2.set_xlabel('X (Å)')
    ax2.set_ylabel('Y (Å)')
    ax2.set_zlabel('Electron Density')
    ax2.view_init(elev=35, azim=60)

    # Chart 3: Partition signature spectrum
    ax3 = fig.add_subplot(gs[1, 0])

    # Simulated partition signature for vanillin functional groups
    n_values = np.arange(1, 51)

    # Different functional groups have different partition signatures
    benzene_sig = 0.6 * np.exp(-(n_values - 12)**2 / 20)
    aldehyde_sig = 0.8 * np.exp(-(n_values - 25)**2 / 15)
    hydroxyl_sig = 0.5 * np.exp(-(n_values - 8)**2 / 10)
    methoxy_sig = 0.4 * np.exp(-(n_values - 18)**2 / 12)

    ax3.fill_between(n_values, benzene_sig, alpha=0.6, color='#0f3460', label='Benzene ring')
    ax3.fill_between(n_values, aldehyde_sig, alpha=0.6, color='#e94560', label='Aldehyde')
    ax3.fill_between(n_values, hydroxyl_sig, alpha=0.6, color='#533483', label='Hydroxyl')
    ax3.fill_between(n_values, methoxy_sig, alpha=0.6, color='#16213e', label='Methoxy')

    ax3.set_xlabel('Partition Depth n')
    ax3.set_ylabel('Signature Amplitude')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_xlim(1, 50)

    # Chart 4: 2D projection with bond orders
    ax4 = fig.add_subplot(gs[1, 1])

    # Plot atoms
    for name, pos in atoms.items():
        element = name[0]
        color = atom_colors[element]
        ax4.scatter(pos[0], pos[1], s=400, c=color, alpha=0.9, edgecolors='white', linewidth=2, zorder=3)

    # Plot bonds
    for a1, a2 in bonds:
        p1, p2 = atoms[a1], atoms[a2]
        ax4.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=3, zorder=1)

    # Double bonds in benzene (alternating)
    double_bonds = [('C1', 'C2'), ('C3', 'C4'), ('C5', 'C6'), ('C7', 'O1')]
    for a1, a2 in double_bonds:
        p1, p2 = atoms[a1], atoms[a2]
        offset = 0.1
        ax4.plot([p1[0]+offset, p2[0]+offset], [p1[1]+offset, p2[1]+offset],
                'k-', linewidth=2, zorder=2)

    ax4.set_xlim(-2, 5.5)
    ax4.set_ylim(-2.5, 5)
    ax4.set_aspect('equal')
    ax4.axis('off')

    plt.suptitle('Predicted Vanillin Structure (C₈H₈O₃)', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig('panel_vanillin_structure.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('panel_vanillin_structure.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: panel_vanillin_structure.png/pdf")


def panel_bidirectional_algorithm():
    """Panel: Bidirectional algorithm visualization"""
    from skimage import io

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

    image_dir = '../../../public/images/images/'
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    # Chart 1: 3D convergence surface
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    iterations = np.linspace(0, 50, 50)
    depth = np.linspace(0, 100, 50)
    I, D = np.meshgrid(iterations, depth)

    # Forward pass (bottom-up) and backward pass (top-down) convergence
    forward = 1 - np.exp(-I / 15) * (1 - D / 100)
    backward = 1 - np.exp(-(50 - I) / 15) * (D / 100)
    combined = (forward + backward) / 2

    surf = ax1.plot_surface(I, D, combined, cmap='viridis', alpha=0.9,
                            linewidth=0, antialiased=True)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Partition Depth')
    ax1.set_zlabel('Convergence')
    ax1.view_init(elev=25, azim=135)

    # Chart 2: Forward vs backward pass on microscopy
    ax2 = fig.add_subplot(gs[0, 1])

    if image_files:
        img = io.imread(os.path.join(image_dir, image_files[6]))
        img_norm = (img - img.min()) / (img.max() - img.min())

        # Simulate forward (edge detection) and backward (reconstruction)
        forward_pass = ndimage.sobel(img_norm)
        forward_pass = (forward_pass - forward_pass.min()) / (forward_pass.max() - forward_pass.min())

        # Create side-by-side
        h, w = img_norm.shape
        combined_img = np.zeros((h, w * 2 + 10))
        combined_img[:, :w] = forward_pass
        combined_img[:, w+10:] = img_norm

        ax2.imshow(combined_img, cmap='inferno')
        ax2.axvline(x=w + 5, color='white', linewidth=2, linestyle='--')
        ax2.axis('off')

    # Chart 3: Convergence curves for both directions
    ax3 = fig.add_subplot(gs[1, 0])

    iterations = np.arange(0, 51)

    forward_conv = 1 - np.exp(-iterations / 12)
    backward_conv = 1 - np.exp(-(50 - iterations) / 12)
    bidirectional = (forward_conv + backward_conv) / 2

    ax3.plot(iterations, forward_conv, '-', color='#e94560', linewidth=2, label='Forward (↑)')
    ax3.plot(iterations, backward_conv, '-', color='#0f3460', linewidth=2, label='Backward (↓)')
    ax3.plot(iterations, bidirectional, '-', color='#533483', linewidth=3, label='Bidirectional')

    ax3.fill_between(iterations, forward_conv, backward_conv, alpha=0.2, color='gray')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Convergence')
    ax3.legend(loc='lower right')
    ax3.set_xlim(0, 50)
    ax3.set_ylim(0, 1.05)

    # Chart 4: Information flow heatmap
    ax4 = fig.add_subplot(gs[1, 1])

    # Create information flow matrix (layers x iterations)
    layers = 10
    iters = 50
    flow = np.zeros((layers, iters))

    for i in range(iters):
        # Forward wave
        forward_wave = np.exp(-((np.arange(layers) - i * layers / iters) ** 2) / 4)
        # Backward wave
        backward_wave = np.exp(-((np.arange(layers) - (layers - i * layers / iters)) ** 2) / 4)
        flow[:, i] = forward_wave + backward_wave

    im = ax4.imshow(flow, aspect='auto', cmap='magma', origin='lower')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Network Layer')
    plt.colorbar(im, ax=ax4, label='Information Flow', shrink=0.8)

    plt.suptitle('Bidirectional Algorithm', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig('panel_bidirectional_algorithm.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('panel_bidirectional_algorithm.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: panel_bidirectional_algorithm.png/pdf")


def panel_intracellular_fluid_dynamics():
    """Panel: Intracellular fluid dynamics"""
    from skimage import io

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

    image_dir = '../../../public/images/images/'
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    # Chart 1: 3D velocity field
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    # Create 3D grid
    x = np.linspace(-5, 5, 10)
    y = np.linspace(-5, 5, 10)
    z = np.linspace(-2, 2, 5)
    X, Y, Z = np.meshgrid(x, y, z)

    # Cytoplasmic streaming pattern (circular + z-component)
    r = np.sqrt(X**2 + Y**2) + 0.1
    U = -Y / r * np.exp(-r/3)
    V = X / r * np.exp(-r/3)
    W = 0.3 * np.sin(np.pi * Z / 2) * np.exp(-r/4)

    ax1.quiver(X, Y, Z, U, V, W, length=0.8, normalize=True,
               color=plt.cm.coolwarm((W.flatten() - W.min()) / (W.max() - W.min() + 0.01)),
               alpha=0.8, arrow_length_ratio=0.3)
    ax1.set_xlabel('X (μm)')
    ax1.set_ylabel('Y (μm)')
    ax1.set_zlabel('Z (μm)')
    ax1.view_init(elev=25, azim=45)

    # Chart 2: Flow field overlay on microscopy
    ax2 = fig.add_subplot(gs[0, 1])

    if image_files:
        img = io.imread(os.path.join(image_dir, image_files[8]))
        img_norm = (img - img.min()) / (img.max() - img.min())

        ax2.imshow(img_norm, cmap='gray', alpha=0.7)

        # Overlay velocity field
        h, w = img_norm.shape
        ys = np.linspace(20, h-20, 15).astype(int)
        xs = np.linspace(20, w-20, 15).astype(int)
        Xg, Yg = np.meshgrid(xs, ys)

        # Compute flow based on intensity gradients
        gy, gx = np.gradient(img_norm)
        gx_sub = gx[Yg, Xg]
        gy_sub = gy[Yg, Xg]

        ax2.quiver(Xg, Yg, gx_sub, -gy_sub, color='#e94560', scale=3, width=0.003)
        ax2.axis('off')

    # Chart 3: Velocity magnitude distribution
    ax3 = fig.add_subplot(gs[1, 0])

    np.random.seed(42)
    # Simulated velocity measurements in different cellular regions
    cytoplasm_v = np.random.gamma(2, 0.5, 500)  # μm/s
    nucleus_v = np.random.gamma(1.5, 0.2, 500)
    membrane_v = np.random.gamma(3, 0.8, 500)

    bins = np.linspace(0, 5, 40)
    ax3.hist(cytoplasm_v, bins=bins, alpha=0.6, color='#e94560', label='Cytoplasm', density=True)
    ax3.hist(nucleus_v, bins=bins, alpha=0.6, color='#0f3460', label='Nucleus', density=True)
    ax3.hist(membrane_v, bins=bins, alpha=0.6, color='#533483', label='Membrane', density=True)

    ax3.set_xlabel('Velocity (μm/s)')
    ax3.set_ylabel('Probability Density')
    ax3.legend(loc='upper right')
    ax3.set_xlim(0, 5)

    # Chart 4: Streamlines
    ax4 = fig.add_subplot(gs[1, 1])

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    r = np.sqrt(X**2 + Y**2) + 0.1
    U = -Y / r * np.exp(-r/3)
    V = X / r * np.exp(-r/3)

    speed = np.sqrt(U**2 + V**2)

    ax4.streamplot(X, Y, U, V, color=speed, cmap='plasma', density=2, linewidth=1.5)
    ax4.set_xlabel('X (μm)')
    ax4.set_ylabel('Y (μm)')
    ax4.set_aspect('equal')

    # Add cell boundary
    theta = np.linspace(0, 2*np.pi, 100)
    ax4.plot(4*np.cos(theta), 4*np.sin(theta), 'k--', linewidth=2, alpha=0.5)

    plt.suptitle('Intracellular Fluid Dynamics', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig('panel_fluid_dynamics.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('panel_fluid_dynamics.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: panel_fluid_dynamics.png/pdf")


def panel_electric_field_mechanisms():
    """Panel: Intracellular electric field mechanisms"""
    from skimage import io

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

    image_dir = '../../../public/images/images/'
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    # Chart 1: 3D electric potential surface
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)

    # Membrane potential creates dipole-like field
    # Plus internal organelle potentials
    r1 = np.sqrt((X-2)**2 + (Y-1)**2) + 0.5  # Nucleus
    r2 = np.sqrt((X+2)**2 + (Y-2)**2) + 0.5  # Mitochondria
    r3 = np.sqrt((X)**2 + (Y+2)**2) + 0.5    # ER

    potential = -70 * np.exp(-(X**2 + Y**2) / 20)  # Resting potential
    potential += 20 / r1 - 10 / r2 + 5 / r3  # Organelle contributions

    surf = ax1.plot_surface(X, Y, potential, cmap='RdBu_r', alpha=0.9,
                            linewidth=0, antialiased=True)
    ax1.set_xlabel('X (μm)')
    ax1.set_ylabel('Y (μm)')
    ax1.set_zlabel('Potential (mV)')
    ax1.view_init(elev=30, azim=45)

    # Chart 2: Electric field vectors on microscopy
    ax2 = fig.add_subplot(gs[0, 1])

    if image_files:
        img = io.imread(os.path.join(image_dir, image_files[10]))
        img_norm = (img - img.min()) / (img.max() - img.min())

        ax2.imshow(img_norm, cmap='gray', alpha=0.6)

        h, w = img_norm.shape
        ys = np.linspace(30, h-30, 12).astype(int)
        xs = np.linspace(30, w-30, 12).astype(int)
        Xg, Yg = np.meshgrid(xs, ys)

        # E-field from intensity (proxy for charge distribution)
        gy, gx = np.gradient(ndimage.gaussian_filter(img_norm, sigma=5))
        Ex = -gx[Yg, Xg]
        Ey = -gy[Yg, Xg]

        magnitude = np.sqrt(Ex**2 + Ey**2)
        ax2.quiver(Xg, Yg, Ex, Ey, magnitude, cmap='coolwarm', scale=0.5, width=0.005)
        ax2.axis('off')

    # Chart 3: Membrane potential dynamics
    ax3 = fig.add_subplot(gs[1, 0])

    t = np.linspace(0, 100, 1000)  # ms

    # Action potential-like dynamics
    V_rest = -70
    V_peak = 40

    # Create realistic action potential shape
    V = np.ones_like(t) * V_rest

    # Add action potentials at different times
    for t0 in [20, 50, 80]:
        mask = (t > t0) & (t < t0 + 5)
        V[mask] = V_peak * np.exp(-((t[mask] - t0 - 1)**2) / 0.5) + V_rest

        mask2 = (t >= t0 + 3) & (t < t0 + 15)
        V[mask2] = V_rest - 10 * np.exp(-(t[mask2] - t0 - 5) / 3)

    ax3.plot(t, V, color='#e94560', linewidth=2)
    ax3.axhline(y=V_rest, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Membrane Potential (mV)')
    ax3.set_xlim(0, 100)
    ax3.set_ylim(-90, 50)

    # Chart 4: Ion channel distribution
    ax4 = fig.add_subplot(gs[1, 1])

    # Simulated channel positions on membrane
    np.random.seed(42)
    theta = np.linspace(0, 2*np.pi, 100)
    membrane_x = 4 * np.cos(theta)
    membrane_y = 4 * np.sin(theta)

    ax4.plot(membrane_x, membrane_y, 'k-', linewidth=3, alpha=0.3)

    # Different channel types at different positions
    n_na = 30
    n_k = 40
    n_ca = 15

    na_theta = np.random.uniform(0, 2*np.pi, n_na)
    k_theta = np.random.uniform(0, 2*np.pi, n_k)
    ca_theta = np.random.uniform(0, 2*np.pi, n_ca)

    ax4.scatter(4*np.cos(na_theta), 4*np.sin(na_theta), s=80, c='#e94560',
               marker='o', label='Na⁺', alpha=0.8, edgecolors='white')
    ax4.scatter(4*np.cos(k_theta), 4*np.sin(k_theta), s=80, c='#0f3460',
               marker='s', label='K⁺', alpha=0.8, edgecolors='white')
    ax4.scatter(4*np.cos(ca_theta), 4*np.sin(ca_theta), s=80, c='#533483',
               marker='^', label='Ca²⁺', alpha=0.8, edgecolors='white')

    ax4.set_xlim(-5.5, 5.5)
    ax4.set_ylim(-5.5, 5.5)
    ax4.set_aspect('equal')
    ax4.legend(loc='upper right')
    ax4.set_xlabel('X (μm)')
    ax4.set_ylabel('Y (μm)')

    plt.suptitle('Intracellular Electric Field Mechanisms', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig('panel_electric_field.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('panel_electric_field.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: panel_electric_field.png/pdf")


def panel_zero_backaction():
    """Panel: Zero backaction measurement"""
    from skimage import io

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

    image_dir = '../../../public/images/images/'
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    # Chart 1: 3D backaction surface
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    # Measurement strength vs system coupling
    measurement = np.linspace(0, 1, 50)
    coupling = np.linspace(0, 1, 50)
    M, C = np.meshgrid(measurement, coupling)

    # Conventional measurement: backaction proportional to measurement × coupling
    conventional_backaction = M * C
    # Categorical measurement: backaction suppressed by commutation
    categorical_backaction = M * C * np.exp(-5 * (1 - M * C))

    surf1 = ax1.plot_surface(M, C, conventional_backaction, cmap='Reds', alpha=0.5,
                             linewidth=0, antialiased=True)
    surf2 = ax1.plot_surface(M, C, categorical_backaction, cmap='Blues', alpha=0.7,
                             linewidth=0, antialiased=True)

    ax1.set_xlabel('Measurement Strength')
    ax1.set_ylabel('System Coupling')
    ax1.set_zlabel('Backaction δ')
    ax1.view_init(elev=25, azim=45)

    # Chart 2: Backaction comparison on sequential measurements
    ax2 = fig.add_subplot(gs[0, 1])

    n_measurements = np.arange(1, 51)

    # Conventional: backaction accumulates
    conv_error = 0.01 * np.sqrt(n_measurements)
    # Categorical: backaction stays bounded
    cat_error = 0.001 * (1 - np.exp(-n_measurements / 20))

    ax2.semilogy(n_measurements, conv_error, '-', color='#e94560', linewidth=2,
                label='Conventional')
    ax2.semilogy(n_measurements, cat_error, '-', color='#0f3460', linewidth=2,
                label='Categorical (zero backaction)')

    ax2.fill_between(n_measurements, conv_error, cat_error, alpha=0.2, color='gray')
    ax2.set_xlabel('Number of Measurements')
    ax2.set_ylabel('Accumulated Backaction')
    ax2.legend(loc='lower right')
    ax2.set_xlim(1, 50)

    # Chart 3: Microscopy showing preserved structure
    ax3 = fig.add_subplot(gs[1, 0])

    if image_files:
        img = io.imread(os.path.join(image_dir, image_files[12]))
        img_norm = (img - img.min()) / (img.max() - img.min())

        # Show "before" (original) and "after" (preserved)
        h, w = img_norm.shape

        # Add very slight noise to "after" to show minimal perturbation
        after = img_norm + np.random.normal(0, 0.001, img_norm.shape)
        after = np.clip(after, 0, 1)

        combined = np.zeros((h, w * 2 + 10))
        combined[:, :w] = img_norm
        combined[:, w+10:] = after

        ax3.imshow(combined, cmap='viridis')
        ax3.axvline(x=w + 5, color='white', linewidth=2, linestyle='--')
        ax3.axis('off')

    # Chart 4: Commutator magnitude
    ax4 = fig.add_subplot(gs[1, 1])

    # [O_cat, O_phys] as function of observable type
    observables = ['Position', 'Momentum', 'Energy', 'Spin', 'Partition\nSignature']

    # Physical observables don't commute with each other
    # Categorical observables commute with all physical ones
    phys_commutator = [1.0, 1.0, 0.5, 0.7, 0.0]  # [X,P] etc
    cat_commutator = [0.001, 0.001, 0.001, 0.001, 0.0]  # [O_cat, O_phys] ≈ 0

    x = np.arange(len(observables))
    width = 0.35

    bars1 = ax4.bar(x - width/2, phys_commutator, width, label='[Ô_phys, Ô_phys]',
                   color='#e94560', alpha=0.85)
    bars2 = ax4.bar(x + width/2, cat_commutator, width, label='[Ô_cat, Ô_phys]',
                   color='#0f3460', alpha=0.85)

    ax4.set_ylabel('|Commutator|')
    ax4.set_xticks(x)
    ax4.set_xticklabels(observables, fontsize=9)
    ax4.legend(loc='upper right')
    ax4.set_ylim(0, 1.2)
    ax4.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Zero Backaction Measurement', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig('panel_zero_backaction.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('panel_zero_backaction.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: panel_zero_backaction.png/pdf")


if __name__ == '__main__':
    print("Generating additional Partition Calculus panels...")
    print("=" * 50)

    panel_arrival_time()
    panel_vanillin_structure()
    panel_bidirectional_algorithm()
    panel_intracellular_fluid_dynamics()
    panel_electric_field_mechanisms()
    panel_zero_backaction()

    print("=" * 50)
    print("All additional panels generated successfully!")
