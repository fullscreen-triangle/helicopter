#!/usr/bin/env python3
"""
Generate 5 panel figures for Paper 2: Image Harmonic Matching Circuits.
Each panel: white background, 4 charts in a row, at least one 3D, no tables/text.
"""

import json, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

RNG = np.random.default_rng(42)

RESULTS = pathlib.Path(__file__).parent / "results"
FIGURES = pathlib.Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
with open(RESULTS / "harmonic_summary.json") as f:
    summary = json.load(f)
df_match = pd.read_csv(RESULTS / "harmonic_matching_accuracy.csv")
df_noise = pd.read_csv(RESULTS / "harmonic_noise_robustness.csv")
df_seg   = pd.read_csv(RESULTS / "harmonic_segmentation.csv")
df_net   = pd.read_csv(RESULTS / "harmonic_network_statistics.csv")
df_ent   = pd.read_csv(RESULTS / "harmonic_entropy_conservation.csv")
df_trip  = pd.read_csv(RESULTS / "harmonic_triple_equivalence.csv")

# ── Colours ──────────────────────────────────────────────────────────────────
C_SIFT = "#2196F3"
C_ORB  = "#FF9800"
C_OSC  = "#4CAF50"
C_ACC  = "#9C27B0"
C_RED  = "#E53935"
C_GRID = "#E0E0E0"

def style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor("white")
    ax.set_xlabel(xlabel, fontsize=8, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8, fontweight="bold")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, color=C_GRID)

def style_3d(ax, xlabel="", ylabel="", zlabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=7, fontweight="bold", labelpad=2)
    ax.set_ylabel(ylabel, fontsize=7, fontweight="bold", labelpad=2)
    ax.set_zlabel(zlabel, fontsize=7, fontweight="bold", labelpad=2)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(labelsize=6)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL 1: Matching Accuracy Comparison
# ═════════════════════════════════════════════════════════════════════════════

def panel_1():
    fig = plt.figure(figsize=(20, 4.5), facecolor="white")
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # (a) 3D scatter: SIFT acc vs ORB acc vs Oscillatory acc
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.scatter(df_match["sift_accuracy"], df_match["orb_accuracy"],
                df_match["oscillatory_accuracy"],
                c=df_match["oscillatory_accuracy"], cmap="RdYlGn",
                s=30, alpha=0.8, edgecolors="k", linewidths=0.3)
    style_3d(ax1, "SIFT", "ORB", "Oscillatory",
             "Matching Accuracy Space")
    ax1.view_init(elev=25, azim=135)

    # (b) Grouped bar: mean accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    sm = summary["matching"]
    methods = ["SIFT", "ORB", "Oscillatory"]
    means = [sm["sift_accuracy_mean"], sm["orb_accuracy_mean"],
             sm["osc_accuracy_mean"]]
    stds = [sm["sift_accuracy_std"], sm["orb_accuracy_std"],
            sm["osc_accuracy_std"]]
    ax2.bar(methods, means, yerr=stds, capsize=5,
            color=[C_SIFT, C_ORB, C_OSC], edgecolor="k", linewidth=0.5,
            alpha=0.85)
    ax2.set_ylim(0, 1.1)
    style_ax(ax2, "", "Accuracy", "Mean Matching Accuracy")

    # (c) Box plot: accuracy distributions
    ax3 = fig.add_subplot(gs[0, 2])
    bp = ax3.boxplot([df_match["sift_accuracy"], df_match["orb_accuracy"],
                      df_match["oscillatory_accuracy"]],
                     labels=["SIFT", "ORB", "Osc."], patch_artist=True,
                     widths=0.5, flierprops=dict(markersize=3))
    for patch, color in zip(bp["boxes"], [C_SIFT, C_ORB, C_OSC]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    style_ax(ax3, "", "Accuracy", "Accuracy Distribution")

    # (d) Bar: matching speed (ms, log scale)
    ax4 = fig.add_subplot(gs[0, 3])
    times = [sm["sift_time_mean_s"] * 1000, sm["orb_time_mean_s"] * 1000,
             sm["osc_time_mean_s"] * 1000]
    ax4.bar(methods, times, color=[C_SIFT, C_ORB, C_OSC],
            edgecolor="k", linewidth=0.5, alpha=0.85)
    ax4.set_yscale("log")
    style_ax(ax4, "", "Time (ms, log)", "Matching Speed")

    fig.savefig(FIGURES / "panel_1_matching_accuracy.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Panel 1 saved")


# ═════════════════════════════════════════════════════════════════════════════
# PANEL 2: Noise Robustness
# ═════════════════════════════════════════════════════════════════════════════

def panel_2():
    fig = plt.figure(figsize=(20, 4.5), facecolor="white")
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    sigmas = df_noise["noise_sigma"].values

    # (a) 3D surface: accuracy(method, sigma)
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    method_idx = np.array([0, 1, 2])
    M, S = np.meshgrid(method_idx, sigmas)
    Z = np.column_stack([df_noise["sift_accuracy_mean"],
                          df_noise["orb_accuracy_mean"],
                          df_noise["osc_accuracy_mean"]])
    ax1.plot_surface(M, S, Z, cmap="coolwarm", alpha=0.7, edgecolor="k",
                     linewidth=0.3)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["SIFT", "ORB", "Osc"], fontsize=6)
    style_3d(ax1, "", "Noise $\\sigma$", "Accuracy",
             "Accuracy vs Noise")
    ax1.view_init(elev=25, azim=135)

    # (b) Line plot: accuracy vs noise sigma
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.errorbar(sigmas, df_noise["sift_accuracy_mean"],
                 yerr=df_noise["sift_accuracy_std"], color=C_SIFT,
                 linewidth=2, marker="o", markersize=5, capsize=3,
                 label="SIFT")
    ax2.errorbar(sigmas, df_noise["orb_accuracy_mean"],
                 yerr=df_noise["orb_accuracy_std"], color=C_ORB,
                 linewidth=2, marker="s", markersize=5, capsize=3,
                 label="ORB")
    ax2.errorbar(sigmas, df_noise["osc_accuracy_mean"],
                 yerr=df_noise["osc_accuracy_std"], color=C_OSC,
                 linewidth=2, marker="^", markersize=5, capsize=3,
                 label="Oscillatory")
    ax2.legend(fontsize=7, framealpha=0.8)
    style_ax(ax2, "Noise $\\sigma$", "Accuracy", "Noise Robustness")

    # (c) Degradation: accuracy drop from sigma=0
    ax3 = fig.add_subplot(gs[0, 2])
    base_sift = df_noise["sift_accuracy_mean"].iloc[0]
    base_orb  = df_noise["orb_accuracy_mean"].iloc[0]
    base_osc  = df_noise["osc_accuracy_mean"].iloc[0]
    deg_sift = base_sift - df_noise["sift_accuracy_mean"]
    deg_orb  = base_orb  - df_noise["orb_accuracy_mean"]
    deg_osc  = base_osc  - df_noise["osc_accuracy_mean"]
    ax3.plot(sigmas, deg_sift, color=C_SIFT, linewidth=2, marker="o",
             markersize=5, label="SIFT")
    ax3.plot(sigmas, deg_orb, color=C_ORB, linewidth=2, marker="s",
             markersize=5, label="ORB")
    ax3.plot(sigmas, deg_osc, color=C_OSC, linewidth=2, marker="^",
             markersize=5, label="Oscillatory")
    ax3.legend(fontsize=7)
    style_ax(ax3, "Noise $\\sigma$", "$\\Delta$Accuracy",
             "Accuracy Degradation")

    # (d) Grouped bar at sigma=100 (worst case)
    ax4 = fig.add_subplot(gs[0, 3])
    worst = df_noise[df_noise["noise_sigma"] == 100].iloc[0]
    methods = ["SIFT", "ORB", "Osc."]
    vals = [worst["sift_accuracy_mean"], worst["orb_accuracy_mean"],
            worst["osc_accuracy_mean"]]
    errs = [worst["sift_accuracy_std"], worst["orb_accuracy_std"],
            worst["osc_accuracy_std"]]
    ax4.bar(methods, vals, yerr=errs, capsize=5,
            color=[C_SIFT, C_ORB, C_OSC], edgecolor="k", linewidth=0.5,
            alpha=0.85)
    ax4.set_ylim(0, 1.1)
    style_ax(ax4, "", "Accuracy", "Worst Case ($\\sigma$=100)")

    fig.savefig(FIGURES / "panel_2_noise_robustness.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Panel 2 saved")


# ═════════════════════════════════════════════════════════════════════════════
# PANEL 3: Segmentation & Standing-Wave Analysis
# ═════════════════════════════════════════════════════════════════════════════

def panel_3():
    fig = plt.figure(figsize=(20, 4.5), facecolor="white")
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # (a) 3D scatter: wave dice vs otsu dice vs n-thresh dice
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.scatter(df_seg["dice_standing_wave"], df_seg["dice_otsu"],
                df_seg["dice_n_threshold"],
                c=df_seg["dice_otsu"], cmap="viridis", s=30, alpha=0.8,
                edgecolors="k", linewidths=0.3)
    style_3d(ax1, "Wave", "Otsu", "N-thresh",
             "Segmentation Methods")
    ax1.view_init(elev=25, azim=135)

    # (b) Grouped bar: mean Dice
    ax2 = fig.add_subplot(gs[0, 1])
    ss = summary["segmentation"]
    methods = ["Standing\nWave", "Otsu", "N-Threshold"]
    means = [ss["dice_wave_mean"], ss["dice_otsu_mean"], ss["dice_n_thresh_mean"]]
    stds = [ss["dice_wave_std"], ss["dice_otsu_std"], ss["dice_n_thresh_std"]]
    ax2.bar(methods, means, yerr=stds, capsize=5,
            color=[C_OSC, C_ORB, C_SIFT], edgecolor="k", linewidth=0.5,
            alpha=0.85)
    ax2.set_ylim(0, 1.0)
    style_ax(ax2, "", "Dice", "Segmentation Performance")

    # (c) Histogram: Dice distributions
    ax3 = fig.add_subplot(gs[0, 2])
    bins = np.linspace(0.3, 0.85, 30)
    ax3.hist(df_seg["dice_standing_wave"], bins=bins, alpha=0.6, color=C_OSC,
             label="Wave", edgecolor="white", linewidth=0.5)
    ax3.hist(df_seg["dice_otsu"], bins=bins, alpha=0.6, color=C_ORB,
             label="Otsu", edgecolor="white", linewidth=0.5)
    ax3.hist(df_seg["dice_n_threshold"], bins=bins, alpha=0.6, color=C_SIFT,
             label="N-thresh", edgecolor="white", linewidth=0.5)
    ax3.legend(fontsize=7)
    style_ax(ax3, "Dice", "Count", "Dice Distribution")

    # (d) Scatter: boundary pixels vs Dice (standing wave)
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.scatter(df_seg["n_boundary_pixels"], df_seg["dice_standing_wave"],
                c=C_OSC, s=30, alpha=0.7, edgecolors="k", linewidths=0.3)
    z = np.polyfit(df_seg["n_boundary_pixels"], df_seg["dice_standing_wave"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_seg["n_boundary_pixels"].min(),
                          df_seg["n_boundary_pixels"].max(), 50)
    ax4.plot(x_line, p(x_line), "--", color=C_RED, linewidth=1.5, alpha=0.7)
    style_ax(ax4, "Boundary Pixels", "Dice (Wave)",
             "Boundaries vs Quality")

    fig.savefig(FIGURES / "panel_3_segmentation.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Panel 3 saved")


# ═════════════════════════════════════════════════════════════════════════════
# PANEL 4: Harmonic Network Structure
# ═════════════════════════════════════════════════════════════════════════════

def panel_4():
    fig = plt.figure(figsize=(20, 4.5), facecolor="white")
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # (a) 3D bar: nodes, edges, loops per image
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    n_imgs = len(df_net)
    x_pos = np.arange(n_imgs)
    # edges (scaled down for visual)
    ax1.bar(x_pos, df_net["n_edges"] / 1000, zs=0, zdir="y",
            color=C_SIFT, alpha=0.7, width=0.6)
    ax1.bar(x_pos, df_net["n_independent_loops"] / 1000, zs=1, zdir="y",
            color=C_OSC, alpha=0.7, width=0.6)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Edges\n(x1000)", "Loops\n(x1000)"], fontsize=6)
    style_3d(ax1, "Image", "", "Count (x1000)",
             "Network Structure")
    ax1.view_init(elev=25, azim=135)

    # (b) Scatter: edges vs loops (perfect linear relationship expected)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(df_net["n_edges"], df_net["n_independent_loops"],
                c=C_ACC, s=40, alpha=0.8, edgecolors="k", linewidths=0.3)
    # perfect fit line
    z = np.polyfit(df_net["n_edges"], df_net["n_independent_loops"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_net["n_edges"].min(), df_net["n_edges"].max(), 50)
    ax2.plot(x_line, p(x_line), "--", color=C_RED, linewidth=1.5)
    from scipy.stats import pearsonr
    r, _ = pearsonr(df_net["n_edges"], df_net["n_independent_loops"])
    ax2.text(0.05, 0.95, f"r={r:.4f}", transform=ax2.transAxes,
             fontsize=8, va="top", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="gray", alpha=0.8))
    style_ax(ax2, "Edges", "Ind. Loops", "Edges vs Loops")

    # (c) Histogram: coupling strength distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df_net["mean_coupling_strength"], bins=15, color=C_OSC,
             edgecolor="white", linewidth=0.5, alpha=0.85)
    ax3.axvline(df_net["mean_coupling_strength"].mean(), color=C_RED,
                linestyle="--", linewidth=1.5)
    style_ax(ax3, "Coupling Strength", "Count",
             "Coupling Distribution")

    # (d) Histogram: harmonic deviation
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(df_net["mean_harmonic_deviation"], bins=15, color=C_SIFT,
             edgecolor="white", linewidth=0.5, alpha=0.85)
    ax4.axvline(0.05, color=C_RED, linestyle="--", linewidth=1.5,
                label="$\\delta_{max}=0.05$")
    ax4.legend(fontsize=7)
    style_ax(ax4, "Harmonic Deviation", "Count",
             "Deviation Distribution")

    fig.savefig(FIGURES / "panel_4_harmonic_network.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Panel 4 saved")


# ═════════════════════════════════════════════════════════════════════════════
# PANEL 5: S-Entropy Conservation & Triple Equivalence
# ═════════════════════════════════════════════════════════════════════════════

def panel_5():
    fig = plt.figure(figsize=(20, 4.5), facecolor="white")
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # (a) 3D scatter: S_k, S_t, S_e with conservation surface
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    S_k = df_ent["mean_S_k"].values
    S_t = df_ent["mean_S_t"].values
    S_e = df_ent["mean_S_e"].values
    ax1.scatter(S_k, S_t, S_e, c=C_OSC, s=40, alpha=0.8,
                edgecolors="k", linewidths=0.3, zorder=5)
    sk_grid = np.linspace(0, 1, 30)
    st_grid = np.linspace(0, 1, 30)
    SK, ST = np.meshgrid(sk_grid, st_grid)
    SE = 1.0 - SK - ST
    SE[SE < 0] = np.nan
    SE[SE > 1] = np.nan
    ax1.plot_surface(SK, ST, SE, alpha=0.12, color=C_ACC, zorder=1)
    style_3d(ax1, "$S_k$", "$S_t$", "$S_e$",
             "Conservation Manifold")
    ax1.view_init(elev=20, azim=130)

    # (b) Stacked bar: entropy partitioning per image
    ax2 = fig.add_subplot(gs[0, 1])
    n_show = min(25, len(df_ent))
    idx = np.arange(n_show)
    ax2.bar(idx, S_k[:n_show], color="#1565C0", label="$S_k$",
            edgecolor="white", linewidth=0.3)
    ax2.bar(idx, S_t[:n_show], bottom=S_k[:n_show], color="#F57C00",
            label="$S_t$", edgecolor="white", linewidth=0.3)
    ax2.bar(idx, S_e[:n_show], bottom=S_k[:n_show] + S_t[:n_show],
            color="#2E7D32", label="$S_e$", edgecolor="white", linewidth=0.3)
    ax2.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=7, loc="upper right")
    style_ax(ax2, "Image", "Entropy", "Entropy Partitioning")

    # (c) Triple equivalence: scatter S_osc vs S_cat vs S_part
    ax3 = fig.add_subplot(gs[0, 2])
    # all three are identical by construction, so plot S_osc vs S_cat
    ax3.scatter(df_trip["S_oscillatory"], df_trip["S_categorical"],
                c=C_OSC, s=20, alpha=0.6, edgecolors="k", linewidths=0.2)
    lims = [df_trip["S_oscillatory"].min(), df_trip["S_oscillatory"].max()]
    ax3.plot(lims, lims, "--", color=C_RED, linewidth=1.5, label="y=x")
    ax3.legend(fontsize=7)
    ax3.set_aspect("equal")
    style_ax(ax3, "$S_{osc}$", "$S_{cat}$",
             "Triple Equivalence")

    # (d) Histogram: max deviation from conservation
    ax4 = fig.add_subplot(gs[0, 3])
    deviations = df_ent["max_deviation"].values
    ax4.hist(deviations, bins=30, color=C_ACC, edgecolor="white",
             linewidth=0.5, alpha=0.85)
    ax4.axvline(0, color=C_RED, linestyle="--", linewidth=1.5)
    mean_dev = np.mean(deviations)
    ax4.text(0.95, 0.95, f"Mean: {mean_dev:.1e}", transform=ax4.transAxes,
             fontsize=8, va="top", ha="right", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="gray", alpha=0.8))
    style_ax(ax4, "Max |$S_{total} - 1$|", "Count",
             "Conservation Deviation")

    fig.savefig(FIGURES / "panel_5_entropy_triple.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Panel 5 saved")


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating Paper 2 panels...")
    panel_1()
    panel_2()
    panel_3()
    panel_4()
    panel_5()
    print(f"\nAll panels saved to: {FIGURES}")
