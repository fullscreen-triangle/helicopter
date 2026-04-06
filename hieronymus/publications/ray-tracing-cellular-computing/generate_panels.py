#!/usr/bin/env python
"""
Paper 5: Ray-Tracing as Cellular Computation -- Panel Generation
================================================================
Five panels (1x4 each) visualizing results from validation experiments.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

# ── paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

# ── color palette ──────────────────────────────────────────────────────
BLUE   = "#2196F3"
ORANGE = "#FF9800"
GREEN  = "#4CAF50"
PURPLE = "#9C27B0"
RED    = "#E53935"
COLORS = [BLUE, ORANGE, GREEN, PURPLE, RED]

COMP_NAMES = ["Background", "Cytoplasm", "Membrane", "Nucleus"]
COMP_COLORS = [BLUE, ORANGE, GREEN, PURPLE]

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
})


# =====================================================================
# PANEL 1: Triple Observation Consistency
# =====================================================================
def panel_1():
    print("Generating Panel 1: Triple Observation Consistency...")
    scatter = pd.read_csv(os.path.join(RES, "exp1_scatter_data.csv"))
    corr_df = pd.read_csv(os.path.join(RES, "exp1_triple_consistency.csv"))

    fig = plt.figure(figsize=(20, 4.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # subsample for plotting
    n_pts = min(2000, len(scatter))
    idx = np.random.choice(len(scatter), n_pts, replace=False)
    s = scatter.iloc[idx]

    # (A) 3D scatter: mu_a vs 1/(tau*dS) vs G*RT
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    for c in range(4):
        mask = s["compartment"] == c
        if mask.sum() > 0:
            ax.scatter(s.loc[mask, "mu_a"],
                       s.loc[mask, "inv_tau_dS"],
                       s.loc[mask, "G_RT"],
                       c=COMP_COLORS[c], alpha=0.4, s=6,
                       label=COMP_NAMES[c])
    ax.set_xlabel("mu_a", fontsize=8)
    ax.set_ylabel("1/(tau*dS)", fontsize=8)
    ax.set_zlabel("G*RT", fontsize=8)
    ax.set_title("(A) Triple Observation Space", fontsize=9)
    ax.legend(fontsize=6, loc="upper left", markerscale=2)
    ax.tick_params(labelsize=6)

    # (B) Scatter: mu_a vs 1/(tau*dS)
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(s["mu_a"], s["inv_tau_dS"], c=BLUE, alpha=0.3, s=4)
    # trend line
    z = np.polyfit(s["mu_a"], s["inv_tau_dS"], 1)
    x_line = np.linspace(s["mu_a"].min(), s["mu_a"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), color=RED, linewidth=2)
    from scipy import stats as sp_stats
    r_val, _ = sp_stats.pearsonr(s["mu_a"], s["inv_tau_dS"])
    ax.set_xlabel("mu_a (absorption)")
    ax.set_ylabel("1/(tau * d_S) (retention)")
    ax.set_title(f"(B) mu_a vs Retention (r={r_val:.3f})")
    ax.grid(True, alpha=0.3)

    # (C) Scatter: mu_a vs G*RT
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(s["mu_a"], s["G_RT"], c=GREEN, alpha=0.3, s=4)
    z = np.polyfit(s["mu_a"], s["G_RT"], 1)
    ax.plot(x_line, np.polyval(z, x_line), color=RED, linewidth=2)
    r_val, _ = sp_stats.pearsonr(s["mu_a"], s["G_RT"])
    ax.set_xlabel("mu_a (absorption)")
    ax.set_ylabel("G * RT (conductance)")
    ax.set_title(f"(C) mu_a vs Conductance (r={r_val:.3f})")
    ax.grid(True, alpha=0.3)

    # (D) Bar: mean correlation per volume (all 3 pairs)
    ax = fig.add_subplot(gs[0, 3])
    means = [corr_df["r_mu_inv"].mean(), corr_df["r_mu_g"].mean(), corr_df["r_inv_g"].mean()]
    stds = [corr_df["r_mu_inv"].std(), corr_df["r_mu_g"].std(), corr_df["r_inv_g"].std()]
    labels = ["mu_a vs\n1/tau*dS", "mu_a vs\nG*RT", "1/tau*dS\nvs G*RT"]
    bars = ax.bar(labels, means, yerr=stds, color=[BLUE, GREEN, PURPLE],
                  capsize=5, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0.95, color=RED, linestyle="--", linewidth=1, label="Threshold 0.95")
    ax.set_ylabel("Pearson r")
    ax.set_title("(D) Mean Correlation (30 volumes)")
    ax.set_ylim(0.8, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Panel 1: Triple Observation Consistency", fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(os.path.join(FIG, "panel1_triple_consistency.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  -> Saved panel1_triple_consistency.png")


# =====================================================================
# PANEL 2: Holographic Reconstruction
# =====================================================================
def panel_2():
    print("Generating Panel 2: Holographic Reconstruction...")
    slice_df = pd.read_csv(os.path.join(RES, "exp2_slice_errors.csv"))
    voxel_df = pd.read_csv(os.path.join(RES, "exp2_voxel_errors.csv"))
    error_surface = np.load(os.path.join(RES, "exp2_error_surface.npy"))
    with open(os.path.join(RES, "exp2_summary.json")) as f:
        summary = json.load(f)

    fig = plt.figure(figsize=(20, 4.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # (A) 3D surface: error vs x-position vs depth
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    NX, NZ = error_surface.shape
    X_grid = np.arange(NX)
    Z_grid = np.arange(NZ)
    X_mesh, Z_mesh = np.meshgrid(X_grid, Z_grid, indexing="ij")
    surf = ax.plot_surface(X_mesh, Z_mesh, error_surface,
                           cmap="viridis", alpha=0.8, linewidth=0)
    ax.set_xlabel("X position", fontsize=8)
    ax.set_ylabel("Depth slice", fontsize=8)
    ax.set_zlabel("Error %", fontsize=8)
    ax.set_title("(A) Error Surface", fontsize=9)
    ax.tick_params(labelsize=6)

    # (B) Line: error per depth slice
    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(slice_df["depth_slice"],
                    slice_df["mean"] - slice_df["std"],
                    slice_df["mean"] + slice_df["std"],
                    alpha=0.3, color=BLUE)
    ax.plot(slice_df["depth_slice"], slice_df["mean"], color=BLUE, linewidth=2)
    ax.axhline(y=5.0, color=RED, linestyle="--", linewidth=1, label="5% threshold")
    ax.set_xlabel("Depth Slice")
    ax.set_ylabel("Reconstruction Error (%)")
    ax.set_title("(B) Error per Depth Slice")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (C) Histogram: voxel error distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(voxel_df["voxel_error"].dropna(), bins=50, color=GREEN, alpha=0.7,
            edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Absolute Voxel Error")
    ax.set_ylabel("Count")
    ax.set_title("(C) Voxel Error Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    # (D) Bar: error per compartment
    ax = fig.add_subplot(gs[0, 3])
    comp_err = summary["compartment_errors"]
    comp_vals = [comp_err[str(i)] for i in range(4)]
    bars = ax.bar(COMP_NAMES, comp_vals, color=COMP_COLORS,
                  edgecolor="black", linewidth=0.5)
    ax.axhline(y=5.0, color=RED, linestyle="--", linewidth=1, label="5% threshold")
    ax.set_ylabel("Mean Error (%)")
    ax.set_title("(D) Error per Compartment")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

    fig.suptitle("Panel 2: Holographic Reconstruction Fidelity", fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(os.path.join(FIG, "panel2_holographic_reconstruction.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  -> Saved panel2_holographic_reconstruction.png")


# =====================================================================
# PANEL 3: Coherence Diagnostic
# =====================================================================
def panel_3():
    print("Generating Panel 3: Coherence Diagnostic...")
    coh_df = pd.read_csv(os.path.join(RES, "exp3_coherence.csv"))
    roc_df = pd.read_csv(os.path.join(RES, "exp3_roc.csv"))
    with open(os.path.join(RES, "exp3_summary.json")) as f:
        summary = json.load(f)

    fig = plt.figure(figsize=(20, 4.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    healthy = coh_df[coh_df["healthy"] == 1]
    diseased = coh_df[coh_df["healthy"] == 0]

    # (A) 3D scatter: V_cell vs eta_cell vs class
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax.scatter(healthy["V_cell"], healthy["eta_cell"],
               np.ones(len(healthy)), c=GREEN, alpha=0.7, s=30, label="Healthy")
    ax.scatter(diseased["V_cell"], diseased["eta_cell"],
               np.zeros(len(diseased)), c=RED, alpha=0.7, s=30, label="Diseased")
    ax.set_xlabel("V_cell", fontsize=8)
    ax.set_ylabel("eta_cell", fontsize=8)
    ax.set_zlabel("Class", fontsize=8)
    ax.set_title("(A) Coherence Space", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=6)

    # (B) ROC curve
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(roc_df["fpr"], roc_df["tpr"], color=BLUE, linewidth=2,
            label=f"AUC = {summary['auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.fill_between(roc_df["fpr"], roc_df["tpr"], alpha=0.15, color=BLUE)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("(B) ROC Curve")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # (C) Box plot: V_cell healthy vs diseased
    ax = fig.add_subplot(gs[0, 2])
    bp_data = [healthy["V_cell"].values, diseased["V_cell"].values]
    bp = ax.boxplot(bp_data, tick_labels=["Healthy", "Diseased"], patch_artist=True,
                    widths=0.5)
    bp["boxes"][0].set_facecolor(GREEN)
    bp["boxes"][1].set_facecolor(RED)
    for box in bp["boxes"]:
        box.set_alpha(0.6)
    ax.set_ylabel("V_cell (Interference Visibility)")
    ax.set_title("(C) Visibility by Class")
    ax.grid(True, alpha=0.3, axis="y")

    # (D) Scatter: V_cell vs eta_cell with identity
    ax = fig.add_subplot(gs[0, 3])
    ax.scatter(healthy["eta_cell"], healthy["V_cell"], c=GREEN, alpha=0.7,
               s=30, label="Healthy", edgecolors="black", linewidths=0.3)
    ax.scatter(diseased["eta_cell"], diseased["V_cell"], c=RED, alpha=0.7,
               s=30, label="Diseased", edgecolors="black", linewidths=0.3)
    lims = [0, 1]
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="Identity")
    r_val = summary["pearson_r"]
    ax.set_xlabel("eta_cell (Ground Truth)")
    ax.set_ylabel("V_cell (Ray-Traced)")
    ax.set_title(f"(D) V_cell vs eta_cell (r={r_val:.3f})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Panel 3: Coherence Diagnostic (Healthy vs Diseased)", fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(os.path.join(FIG, "panel3_coherence_diagnostic.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  -> Saved panel3_coherence_diagnostic.png")


# =====================================================================
# PANEL 4: Flow Field Recovery
# =====================================================================
def panel_4():
    print("Generating Panel 4: Flow Field Recovery...")
    scatter_df = pd.read_csv(os.path.join(RES, "exp4_scatter.csv"))
    v_surface = np.load(os.path.join(RES, "exp4_poiseuille_surface.npy"))
    with open(os.path.join(RES, "exp4_summary.json")) as f:
        summary = json.load(f)

    fig = plt.figure(figsize=(20, 4.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # (A) 3D surface: Poiseuille velocity field
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    N = v_surface.shape[0]
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    surf = ax.plot_surface(X, Y, v_surface, cmap="coolwarm", alpha=0.85, linewidth=0)
    ax.set_xlabel("X (norm)", fontsize=8)
    ax.set_ylabel("Y (norm)", fontsize=8)
    ax.set_zlabel("v_z (um/s)", fontsize=8)
    ax.set_title("(A) Poiseuille Flow Profile", fontsize=9)
    ax.tick_params(labelsize=6)

    # (B) Scatter: recovered vs ground-truth velocity
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(scatter_df["v_gt"], scatter_df["v_rec"], c=BLUE, alpha=0.3, s=6)
    lims = [0, scatter_df["v_gt"].max() * 1.1]
    ax.plot(lims, lims, color=RED, linestyle="--", linewidth=1.5, label="Identity")
    from scipy import stats as sp_stats
    r_val, _ = sp_stats.pearsonr(scatter_df["v_gt"], scatter_df["v_rec"])
    ax.set_xlabel("Ground Truth v (um/s)")
    ax.set_ylabel("Recovered v (um/s)")
    ax.set_title(f"(B) Recovery Accuracy (r={r_val:.3f})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (C) Histogram: velocity error distribution
    ax = fig.add_subplot(gs[0, 2])
    valid_err = scatter_df["error_pct"].dropna()
    ax.hist(valid_err, bins=40, color=ORANGE, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.axvline(x=10.0, color=RED, linestyle="--", linewidth=1.5, label="10% threshold")
    ax.set_xlabel("Velocity Error (%)")
    ax.set_ylabel("Count")
    ax.set_title("(C) Error Distribution")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # (D) Line: error vs radial position
    ax = fig.add_subplot(gs[0, 3])
    # bin by radius
    scatter_valid = scatter_df.dropna(subset=["error_pct"])
    bins = np.linspace(0, scatter_valid["radius"].max(), 15)
    scatter_valid = scatter_valid.copy()
    scatter_valid["r_bin"] = pd.cut(scatter_valid["radius"], bins=bins)
    grouped = scatter_valid.groupby("r_bin", observed=True)["error_pct"].agg(["mean", "std"]).reset_index()
    bin_centers = [(b.left + b.right) / 2 for b in grouped["r_bin"]]

    ax.fill_between(bin_centers,
                    grouped["mean"] - grouped["std"],
                    grouped["mean"] + grouped["std"],
                    alpha=0.2, color=PURPLE)
    ax.plot(bin_centers, grouped["mean"], color=PURPLE, linewidth=2, marker="o", markersize=4)
    ax.axhline(y=10.0, color=RED, linestyle="--", linewidth=1, label="10% threshold")
    ax.set_xlabel("Radial Position (normalized)")
    ax.set_ylabel("Mean Error (%)")
    ax.set_title("(D) Error vs Radius")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Panel 4: Flow Field Recovery", fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(os.path.join(FIG, "panel4_flow_recovery.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  -> Saved panel4_flow_recovery.png")


# =====================================================================
# PANEL 5: Throughput & Conservation
# =====================================================================
def panel_5():
    print("Generating Panel 5: Throughput & Conservation...")
    timing_df = pd.read_csv(os.path.join(RES, "exp5_timing.csv"))
    entropy_df = pd.read_csv(os.path.join(RES, "entropy_conservation.csv"))

    fig = plt.figure(figsize=(20, 4.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # (A) 3D bar: timing breakdown per size per stage
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    stages = ["t_encode", "t_ray", "t_interference", "t_readback"]
    stage_labels = ["Encode", "Ray March", "Interference", "Readback"]
    stage_colors = [BLUE, ORANGE, GREEN, PURPLE]
    sizes = timing_df["size"].values

    for si, stage in enumerate(stages):
        xs = np.arange(len(sizes))
        ys = np.ones(len(sizes)) * si
        zs = np.zeros(len(sizes))
        dx = 0.6
        dy = 0.6
        dz = timing_df[stage].values
        ax.bar3d(xs, ys, zs, dx, dy, dz, color=stage_colors[si], alpha=0.7,
                 label=stage_labels[si])

    ax.set_xticks(np.arange(len(sizes)))
    ax.set_xticklabels([f"{s}^3" for s in sizes], fontsize=6)
    ax.set_yticks(np.arange(len(stages)))
    ax.set_yticklabels(stage_labels, fontsize=6)
    ax.set_zlabel("Time (s)", fontsize=8)
    ax.set_title("(A) Timing Breakdown", fontsize=9)
    ax.tick_params(labelsize=6)

    # (B) Line: FPS vs volume size for 3 GPU tiers
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(timing_df["voxels"], timing_df["fps_intel_uhd"],
            color=BLUE, marker="o", linewidth=2, label="Intel UHD (8x)")
    ax.plot(timing_df["voxels"], timing_df["fps_amd_vega8"],
            color=ORANGE, marker="s", linewidth=2, label="AMD Vega 8 (22x)")
    ax.plot(timing_df["voxels"], timing_df["fps_apple_m1"],
            color=GREEN, marker="^", linewidth=2, label="Apple M1 (52x)")
    ax.plot(timing_df["voxels"], timing_df["fps_cpu"],
            color="gray", marker="x", linewidth=1.5, linestyle="--", label="CPU baseline")
    ax.set_xlabel("Volume Size (voxels)")
    ax.set_ylabel("Estimated FPS")
    ax.set_title("(B) GPU-Projected FPS")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # (C) 3D scatter: S_k vs S_t vs S_e on conservation manifold
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    ax.scatter(entropy_df["S_k"], entropy_df["S_t"], entropy_df["S_e"],
               c=PURPLE, alpha=0.7, s=25, edgecolors="black", linewidths=0.3)
    # draw conservation plane S_k + S_t + S_e = 1
    sk_line = np.linspace(0, 0.6, 20)
    st_line = np.linspace(0, 0.3, 20)
    SK, ST = np.meshgrid(sk_line, st_line)
    SE = 1.0 - SK - ST
    mask = SE >= 0
    SK_m, ST_m, SE_m = SK.copy(), ST.copy(), SE.copy()
    SK_m[~mask] = np.nan
    ax.plot_surface(SK_m, ST_m, SE_m, alpha=0.15, color="gray")
    ax.set_xlabel("S_k", fontsize=8)
    ax.set_ylabel("S_t", fontsize=8)
    ax.set_zlabel("S_e", fontsize=8)
    ax.set_title("(C) S-Entropy Manifold", fontsize=9)
    ax.tick_params(labelsize=6)

    # (D) Histogram: conservation deviation
    ax = fig.add_subplot(gs[0, 3])
    ax.hist(entropy_df["deviation"], bins=30, color=RED, alpha=0.7,
            edgecolor="black", linewidth=0.3)
    ax.axvline(x=0.01, color="black", linestyle="--", linewidth=1.5,
               label="0.01 threshold")
    ax.set_xlabel("Conservation Deviation |S_k+S_t+S_e - 1|")
    ax.set_ylabel("Count")
    ax.set_title("(D) Conservation Check")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Panel 5: Throughput & S-Entropy Conservation", fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(os.path.join(FIG, "panel5_throughput_conservation.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  -> Saved panel5_throughput_conservation.png")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    print("\n" + "#" * 72)
    print("# Paper 5: Ray-Tracing as Cellular Computation -- Panel Generation")
    print("#" * 72 + "\n")

    panel_1()
    panel_2()
    panel_3()
    panel_4()
    panel_5()

    print("\nAll panels saved to:", FIG)
    print("Done.")
