"""
_style.py — Shared chart styling for Group 10: Wind Curtailment & Data Centres

Import this at the top of each analysis script to get consistent,
publication-quality charts across the project.

Usage:
    from _style import apply_style, COLORS, save_fig
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ── COLOUR PALETTE ────────────────────────────────────────────────────────────
# Cohesive palette inspired by academic energy papers.

COLORS = {
    # Primary
    "blue":        "#1B4F72",
    "blue_light":  "#5DADE2",
    "blue_pale":   "#AED6F1",
    # Accents
    "green":       "#1E8449",
    "green_light": "#82E0AA",
    "amber":       "#D4AC0D",
    "red":         "#C0392B",
    "purple":      "#7D3C98",
    "grey":        "#5D6D7E",
    "grey_light":  "#D5D8DC",
    # Scenario palette (ordered for bar/line charts)
    "scen1":       "#1B4F72",
    "scen2":       "#2E86C1",
    "scen3":       "#D4AC0D",
    "scen4":       "#C0392B",
    "scen5":       "#7D3C98",
    # Heatmap
    "night":       "#1B2631",
    "shoulder":    "#5DADE2",
    "day":         "#F4D03F",
}

SCENARIO_COLORS = [COLORS["scen1"], COLORS["scen2"], COLORS["scen3"],
                   COLORS["scen4"], COLORS["scen5"]]


def apply_style():
    """Apply consistent matplotlib rcParams. Call once at script start."""
    plt.rcParams.update({
        # Figure
        "figure.facecolor":   "white",
        "figure.dpi":         150,
        "figure.figsize":     (12, 5),
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
        # Axes
        "axes.facecolor":     "white",
        "axes.edgecolor":     "#CCCCCC",
        "axes.linewidth":     0.8,
        "axes.grid":          True,
        "axes.grid.which":    "major",
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.titlepad":      12,
        "axes.labelsize":     11,
        "axes.labelpad":      6,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        # Grid
        "grid.color":         "#E8E8E8",
        "grid.linewidth":     0.5,
        "grid.alpha":         0.7,
        # Ticks
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        # Font
        "font.size":          10,
        "font.family":        "sans-serif",
        # Legend
        "legend.fontsize":    9,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "#CCCCCC",
        # Lines
        "lines.linewidth":    1.8,
        "lines.markersize":   5,
    })


def save_fig(fig, path, close=True):
    """Save figure and optionally close it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    if close:
        plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")


def hour_labels(ax, step=1):
    """Set clean hour-of-day x-axis labels."""
    ticks = range(0, 24, step)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([f"{h:02d}:00" for h in ticks], rotation=45, ha="right")
