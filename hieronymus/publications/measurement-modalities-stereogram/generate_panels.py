#!/usr/bin/env python3
"""
Generate 5 panel figures for Paper 1: Measurement-Modality Stereogram.
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
with open(RESULTS / "stereogram_summary.json") as f:
    summary = json.load(f)
df = pd.read_csv(RESULTS / "stereogram_per_image_results.csv")

# ── Colour palette ───────────────────────────────────────────────────────────
C_VIS  = "#2196F3"   # blue  - visible
C_INV  = "#FF9800"   # orange - invisible
C_DUAL = "#4CAF50"   # green - dual
C_ACC  = "#9C27B0"   # purple - accent
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
# PANEL 1: Segmentation Performance & Dice Analysis
# ═════════════════════════════════════════════════════════════════════════════

def panel_1():
    fig = plt.figure(figsize=(20, 4.5), facecolor="white")
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # (a) 3D scatter: Dice_vis vs Dice_inv vs Dice_dual
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.scatter(df["dice_visible"], df["dice_invisible"], df["dice_dual"],
                c=df["consistency_rate"], cmap="viridis", s=30, alpha=0.8,
                edgecolors="k", linewidths=0.3)
    style_3d(ax1, "Dice (vis)", "Dice (inv)", "Dice (dual)",
             "Dice Coefficients")
    ax1.view_init(elev=25, azim=135)

    # (b) Grouped bar: Dice comparison (vis, inv, dual)
    ax2 = fig.add_subplot(gs[0, 1])
    methods = ["Visible", "Invisible", "Dual"]
    means = [df["dice_visible"].mean(), df["dice_invisible"].mean(),
             df["dice_dual"].mean()]
    stds = [df["dice_visible"].std(), df["dice_invisible"].std(),
            df["dice_dual"].std()]
    bars = ax2.bar(methods, means, yerr=stds, capsize=4,
                   color=[C_VIS, C_INV, C_DUAL], edgecolor="k", linewidth=0.5,
                   alpha=0.85)
    ax2.set_ylim(0.70, 0.85)
    style_ax(ax2, "", "Dice Coefficient", "Segmentation Performance")

    # (c) Histogram: Dice distribution for all three methods
    ax3 = fig.add_subplot(gs[0, 2])
    bins = np.linspace(0.72, 0.84, 25)
    ax3.hist(df["dice_visible"], bins=bins, alpha=0.6, color=C_VIS,
             label="Visible", edgecolor="white", linewidth=0.5)
    ax3.hist(df["dice_invisible"], bins=bins, alpha=0.6, color=C_INV,
             label="Invisible", edgecolor="white", linewidth=0.5)
    ax3.hist(df["dice_dual"], bins=bins, alpha=0.6, color=C_DUAL,
             label="Dual", edgecolor="white", linewidth=0.5)
    ax3.legend(fontsize=7, framealpha=0.8)
    style_ax(ax3, "Dice", "Count", "Dice Distribution (n=50)")

    # (d) Box plot: per-image Dice
    ax4 = fig.add_subplot(gs[0, 3])
    bp = ax4.boxplot([df["dice_visible"], df["dice_invisible"], df["dice_dual"]],
                     labels=["Vis", "Inv", "Dual"], patch_artist=True,
                     widths=0.5, showfliers=True, flierprops=dict(markersize=3))
    for patch, color in zip(bp["boxes"], [C_VIS, C_INV, C_DUAL]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    style_ax(ax4, "", "Dice", "Dice Box Plot")

    fig.savefig(FIGURES / "panel_1_segmentation_performance.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Panel 1 saved")


# ═════════════════════════════════════════════════════════════════════════════
# PANEL 2: S-Entropy Conservation
# ═════════════════════════════════════════════════════════════════════════════

def panel_2():
    fig = plt.figure(figsize=(20, 4.5), facecolor="white")
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # (a) 3D scatter: S_k, S_t, S_e coordinates with conservation surface
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    S_k = df["conservation_mean_S_k"].values
    S_t = df["conservation_mean_S_t"].values
    S_e = df["conservation_mean_S_e"].values
    ax1.scatter(S_k, S_t, S_e, c=C_DUAL, s=40, alpha=0.8,
                edgecolors="k", linewidths=0.3, zorder=5)
    # conservation surface S_k + S_t + S_e = 1
    sk_grid = np.linspace(0, 1, 30)
    st_grid = np.linspace(0, 1, 30)
    SK, ST = np.meshgrid(sk_grid, st_grid)
    SE = 1.0 - SK - ST
    mask = (SE >= 0) & (SE <= 1)
    SE[~mask] = np.nan
    ax1.plot_surface(SK, ST, SE, alpha=0.15, color=C_ACC, zorder=1)
    style_3d(ax1, "$S_k$", "$S_t$", "$S_e$",
             "S-Entropy Conservation")
    ax1.view_init(elev=20, azim=130)

    # (b) Stacked bar: S_k, S_t, S_e per image (first 20)
    ax2 = fig.add_subplot(gs[0, 1])
    n_show = 20
    idx = np.arange(n_show)
    ax2.bar(idx, S_k[:n_show], color="#1565C0", label="$S_k$",
            edgecolor="white", linewidth=0.3)
    ax2.bar(idx, S_t[:n_show], bottom=S_k[:n_show], color="#F57C00",
            label="$S_t$", edgecolor="white", linewidth=0.3)
    ax2.bar(idx, S_e[:n_show], bottom=S_k[:n_show] + S_t[:n_show],
            color="#2E7D32", label="$S_e$", edgecolor="white", linewidth=0.3)
    ax2.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=7, loc="upper right", framealpha=0.8)
    style_ax(ax2, "Image ID", "Entropy", "Entropy Partitioning")

    # (c) Histogram: conservation deviation
    ax3 = fig.add_subplot(gs[0, 2])
    total = S_k + S_t + S_e
    deviation = np.abs(total - 1.0)
    ax3.hist(deviation, bins=30, color=C_ACC, edgecolor="white",
             linewidth=0.5, alpha=0.85)
    ax3.axvline(0, color="red", linestyle="--", linewidth=1)
    ax3.set_xlim(-0.001, 0.005)
    style_ax(ax3, "|$S_k + S_t + S_e - 1$|", "Count",
             "Conservation Deviation")

    # (d) Pie chart: mean entropy partitioning
    ax4 = fig.add_subplot(gs[0, 3])
    sizes = [summary["conservation_mean_S_k"]["mean"],
             summary["conservation_mean_S_t"]["mean"],
             summary["conservation_mean_S_e"]["mean"]]
    labels_pie = ["$S_k$\n{:.1%}".format(sizes[0]),
                  "$S_t$\n{:.1%}".format(sizes[1]),
                  "$S_e$\n{:.1%}".format(sizes[2])]
    wedges, texts = ax4.pie(sizes, labels=labels_pie,
                            colors=["#1565C0", "#F57C00", "#2E7D32"],
                            startangle=90, wedgeprops=dict(edgecolor="white",
                            linewidth=2))
    for t in texts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
    ax4.set_title("Mean Entropy Partition", fontsize=9, fontweight="bold", pad=8)

    fig.savefig(FIGURES / "panel_2_entropy_conservation.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Panel 2 saved")


# ═════════════════════════════════════════════════════════════════════════════
# PANEL 3: Information Theory (MI, H, Gain)
# ═════════════════════════════════════════════════════════════════════════════

def panel_3():
    fig = plt.figure(figsize=(20, 4.5), facecolor="white")
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # (a) 3D surface: MI as function of H_vis and H_inv
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.scatter(df["H_visible"], df["H_invisible"], df["mutual_information"],
                c=df["mutual_information"], cmap="plasma", s=35, alpha=0.8,
                edgecolors="k", linewidths=0.3)
    style_3d(ax1, "H(vis)", "H(inv)", "MI",
             "Mutual Information")
    ax1.view_init(elev=25, azim=45)

    # (b) Grouped bar: H_vis, H_inv, MI, I_dual
    ax2 = fig.add_subplot(gs[0, 1])
    labels = ["H(vis)", "H(inv)", "MI", "I(dual)"]
    vals = [df["H_visible"].mean(), df["H_invisible"].mean(),
            df["mutual_information"].mean(), df["I_dual_total"].mean()]
    errs = [df["H_visible"].std(), df["H_invisible"].std(),
            df["mutual_information"].std(), df["I_dual_total"].std()]
    colors = [C_VIS, C_INV, C_ACC, C_DUAL]
    ax2.bar(labels, vals, yerr=errs, capsize=4, color=colors,
            edgecolor="k", linewidth=0.5, alpha=0.85)
    style_ax(ax2, "", "Bits", "Information Content")

    # (c) Scatter: MI vs Dice
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(df["mutual_information"], df["dice_dual"],
                c=C_DUAL, s=30, alpha=0.7, edgecolors="k", linewidths=0.3)
    # trend line
    z = np.polyfit(df["mutual_information"], df["dice_dual"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["mutual_information"].min(),
                          df["mutual_information"].max(), 50)
    ax3.plot(x_line, p(x_line), "--", color="red", linewidth=1.5, alpha=0.7)
    from scipy.stats import pearsonr
    r, pval = pearsonr(df["mutual_information"], df["dice_dual"])
    ax3.text(0.05, 0.95, f"r={r:.3f}", transform=ax3.transAxes,
             fontsize=8, va="top", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="gray", alpha=0.8))
    style_ax(ax3, "MI (bits)", "Dice (dual)", "MI vs Segmentation")

    # (d) Violin plot: information gain distribution
    ax4 = fig.add_subplot(gs[0, 3])
    parts = ax4.violinplot([df["information_gain_bits"].values],
                           positions=[1], showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(C_DUAL)
        pc.set_alpha(0.6)
    ax4.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax4.set_xticks([1])
    ax4.set_xticklabels(["Dual - Best Single"])
    style_ax(ax4, "", "Info Gain (bits)", "Information Gain")

    fig.savefig(FIGURES / "panel_3_information_theory.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Panel 3 saved")


# ═════════════════════════════════════════════════════════════════════════════
# PANEL 4: Consistency & Categorical Distance
# ═════════════════════════════════════════════════════════════════════════════

def panel_4():
    fig = plt.figure(figsize=(20, 4.5), facecolor="white")
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # (a) 3D bar: consistency rate vs categorical distance vs Dice
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    x = df["consistency_rate"].values
    y = df["mean_categorical_distance"].values
    z = df["dice_dual"].values
    ax1.scatter(x, y, z, c=z, cmap="RdYlGn", s=35, alpha=0.8,
                edgecolors="k", linewidths=0.3)
    style_3d(ax1, "Cons. Rate", "$d_{cat}$", "Dice",
             "Consistency-Distance-Dice")
    ax1.view_init(elev=20, azim=45)

    # (b) Scatter: consistency rate vs Dice
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(df["consistency_rate"], df["dice_dual"], c=C_DUAL, s=30,
                alpha=0.7, edgecolors="k", linewidths=0.3)
    z = np.polyfit(df["consistency_rate"], df["dice_dual"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["consistency_rate"].min(),
                          df["consistency_rate"].max(), 50)
    ax2.plot(x_line, p(x_line), "--", color="red", linewidth=1.5, alpha=0.7)
    style_ax(ax2, "Consistency Rate", "Dice", "Consistency vs Dice")

    # (c) Histogram: categorical distance distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df["mean_categorical_distance"], bins=20, color=C_ACC,
             edgecolor="white", linewidth=0.5, alpha=0.85)
    ax3.axvline(df["mean_categorical_distance"].mean(), color="red",
                linestyle="--", linewidth=1.5, label="Mean")
    ax3.legend(fontsize=7)
    style_ax(ax3, "Mean $d_{cat}$", "Count",
             "Categorical Distance Distribution")

    # (d) Scatter: consistency rate vs mean d_cat (coloured by Dice)
    ax4 = fig.add_subplot(gs[0, 3])
    sc = ax4.scatter(df["consistency_rate"], df["mean_categorical_distance"],
                     c=df["dice_dual"], cmap="RdYlGn", s=35, alpha=0.8,
                     edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax4, label="Dice", shrink=0.8)
    style_ax(ax4, "Consistency Rate", "Mean $d_{cat}$",
             "Consistency vs Distance")

    fig.savefig(FIGURES / "panel_4_consistency_analysis.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Panel 4 saved")


# ═════════════════════════════════════════════════════════════════════════════
# PANEL 5: Ternary States & Resolution
# ═════════════════════════════════════════════════════════════════════════════

def panel_5():
    fig = plt.figure(figsize=(20, 4.5), facecolor="white")
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # (a) 3D surface: ternary state populations as function of O2 concentration
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    o2_range = np.linspace(5e-6, 40e-6, 50)
    c_norm = o2_range / 40e-6
    p2 = np.clip(0.15 * c_norm, 0.01, 0.98)
    p0 = np.clip(0.35 * c_norm, 0.01, 0.98)
    p1 = np.clip(1.0 - p0 - p2, 0.01, 0.98)
    total = p0 + p1 + p2
    p0 /= total; p1 /= total; p2 /= total
    # plot as 3 lines in 3D (concentration, state index, probability)
    for state, prob, color, label in [(0, p0, "#E53935", "Absorption"),
                                       (1, p1, "#1E88E5", "Ground"),
                                       (2, p2, "#43A047", "Emission")]:
        ax1.plot(o2_range * 1e6, [state] * len(o2_range), prob,
                 color=color, linewidth=2.5, label=label)
        ax1.scatter(o2_range[::5] * 1e6, [state] * len(o2_range[::5]),
                    prob[::5], c=color, s=15, alpha=0.8)
    ax1.legend(fontsize=6, loc="upper left")
    style_3d(ax1, "[O$_2$] ($\\mu$M)", "State", "Probability",
             "Ternary State Populations")
    ax1.set_yticks([0, 1, 2])
    ax1.view_init(elev=20, azim=135)

    # (b) Line plot: ternary populations vs O2 concentration (2D)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(o2_range * 1e6, p0, color="#E53935", linewidth=2, label="$p_0$ (Abs)")
    ax2.plot(o2_range * 1e6, p1, color="#1E88E5", linewidth=2, label="$p_1$ (Gnd)")
    ax2.plot(o2_range * 1e6, p2, color="#43A047", linewidth=2, label="$p_2$ (Em)")
    ax2.fill_between(o2_range * 1e6, p0, alpha=0.15, color="#E53935")
    ax2.fill_between(o2_range * 1e6, p2, alpha=0.15, color="#43A047")
    ax2.legend(fontsize=7)
    style_ax(ax2, "[O$_2$] ($\\mu$M)", "Population", "Ternary Populations")

    # (c) Bar: information content
    ax3 = fig.add_subplot(gs[0, 2])
    ti = summary["ternary_info"]
    categories = ["Per Cycle\n(bits)", "Data Rate\n(Gbps)"]
    values = [ti["bits_per_cycle"] / 1e9, ti["data_rate_Gbps"]]
    bars = ax3.bar(categories, values, color=[C_VIS, C_DUAL],
                   edgecolor="k", linewidth=0.5, alpha=0.85)
    ax3.set_yscale("log")
    style_ax(ax3, "", "Value (log scale)", "Ternary Information")

    # (d) Scatter: REF vs consistency rate
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.scatter(df["consistency_rate"], df["resolution_enhancement_factor"],
                c=df["dice_dual"], cmap="viridis", s=35, alpha=0.8,
                edgecolors="k", linewidths=0.3)
    ax4.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.5,
                label="REF=1")
    ax4.legend(fontsize=7)
    cb = plt.colorbar(ax4.collections[0], ax=ax4, label="Dice", shrink=0.8)
    style_ax(ax4, "Consistency Rate", "REF",
             "Resolution Enhancement")

    fig.savefig(FIGURES / "panel_5_ternary_resolution.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Panel 5 saved")


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating Paper 1 panels...")
    panel_1()
    panel_2()
    panel_3()
    panel_4()
    panel_5()
    print(f"\nAll panels saved to: {FIGURES}")
