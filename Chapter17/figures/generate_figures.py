"""
Chapter 17: conceptual diagrams.

Saves, in the same folder as this script:
    fig17_1_network.png
    fig17_5_hetero_gnn.png

The data-derived figures (17.2, 17.3, 17.4, 17.6, 17.7) are produced by run.py.

Run: python generate_figures.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

np.random.seed(0)

OUT = os.path.dirname(os.path.abspath(__file__))

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, FILLS, COL_WIDTH)

apply()
BG = "white"
G0 = "#111111"
G1 = "#333333"
G2 = "#555555"
G3 = "#777777"
G4 = "#999999"
G5 = "#BBBBBB"
G6 = "#DDDDDD"


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))
    print(f"  saved {name}")


def rbox(ax, x, y, w, h, lines, fill=None, fc=None, fs=8, gap=None,
         bold_first=True):
    """Square corners, dark type on a light fill; the caller's grey is mapped
    onto the light palette and `fc` is ignored."""
    from matplotlib.patches import Rectangle
    light = {G0: G4, G1: G5, G2: G5, G3: G6}.get(fill, fill or G6)
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=light,
                           edgecolor=G3, linewidth=0.8, zorder=3))
    for i, line in enumerate(lines):
        ax.text(x, y, "\n".join(lines), ha='center', va='center',
                fontsize=fs, color=INK, fontweight='bold', fontfamily=FONT,
                linespacing=1.5, zorder=4)
        break


def arr(ax, x1, y1, x2, y2, color=G3, lw=1.1, ls='-'):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, mutation_scale=9), zorder=2)


# =============================================================================
# PART 1: Figure 17.1, the emulated network of CIDDS-001
# =============================================================================

print("Figure 17.1 …")

# The original was 13 units across in a 5.5-inch figure, which left each
# subnet box an inch for "192.168.100.0/24". Taller, with fewer units across.
fig, ax = figure(aspect=0.66)
ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(1.9, 8.0)

ax.add_patch(FancyBboxPatch((0.2, 2.1), 6.3, 5.3,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor="#F6F6F6", edgecolor=G4,
                            linewidth=0.8, linestyle='--', zorder=1))
ax.text(3.35, 7.1, "OpenStack environment (internal traffic)", ha='center',
        fontsize=8, fontweight='bold', color=INK, fontfamily=FONT)

rbox(ax, 3.35, 4.7, 2.4, 0.7, ["Internal network"], G2, fs=8)
subnets = [
    (1.5, 6.2, ["Server", "192.168.100.0/24"], G0),
    (5.2, 6.2, ["Management", "192.168.200.0/24"], G1),
    (5.2, 3.1, ["Office", "192.168.210.0/24"], G1),
    (1.5, 3.1, ["Developer", "192.168.220.0/24"], G0),
]
for x, y, lines, fill in subnets:
    rbox(ax, x, y, 2.6, 0.85, lines, fill, fs=8)
    ax.plot([x, 3.35], [y, 4.7], color=G3, linewidth=0.9, zorder=2)
ax.text(1.5, 2.45, "the developer subnet holds the two attacking clients",
        ha='left', va='center', fontsize=8, color=INK_SOFT, fontstyle='italic',
        fontfamily=FONT)

ax.add_patch(plt.Circle((7.6, 4.7), 0.62, facecolor=G6, edgecolor=G3,
                        linewidth=0.8, zorder=2))
ax.text(7.6, 4.7, "Internet", ha='center', va='center', fontsize=8,
        fontweight='bold', color=INK, fontfamily=FONT, zorder=3)
ax.plot([4.55, 6.98], [4.7, 4.7], color=G3, linewidth=0.9, zorder=1)

rbox(ax, 8.75, 6.6, 2.3, 1.15,
     ["External server", "web server,", "file sync"], G3, fs=8)
ax.plot([8.05, 8.55], [5.2, 6.05], color=G3, linewidth=0.9, zorder=1)
rbox(ax, 8.75, 2.9, 2.3, 1.0, ["External", "attackers"], G6, fs=8)
arr(ax, 8.75, 3.4, 8.75, 6.03, color=G2, ls='--')

ax.set_title("CIDDS-001: emulated small business network",
             fontsize=10.5, fontweight='bold', color=INK, fontfamily=FONT)
save(fig, "fig17_1_network.png")


# =============================================================================
# PART 2: Figure 17.5, the heterogeneous GNN
# =============================================================================

print("Figure 17.5 …")

# The three HeteroConv layers ran across 14 units, which at 5.5 inches gave
# each one 0.9 inches for "SAGEConv (received_by)". One layer is drawn in
# full and the repetition is stated, rather than drawn three times.
fig, ax = figure(aspect=0.86)
ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 8.6)

Y_FLOW, Y_HOST = 6.3, 3.4

rbox(ax, 1.5, Y_FLOW, 2.6, 0.95, ["Flow nodes", "flow features"], G3, fs=8)
rbox(ax, 1.5, Y_HOST, 2.6, 0.95, ["Host nodes", "16 address bits"], G1, fs=8)

ax.add_patch(plt.Rectangle((3.4, 2.3), 3.9, 5.1, fill=False, edgecolor=G4,
                           linestyle='--', linewidth=0.8, zorder=1))
ax.text(5.35, 7.7, "HeteroConv layer + LeakyReLU", ha='center', fontsize=8,
        color=INK_SOFT, fontstyle='italic', fontfamily=FONT)
rbox(ax, 5.35, Y_FLOW, 3.5, 1.15,
     ["Update flows", "SAGEConv (sends)", "+ SAGEConv (receives)"], G2, fs=8)
rbox(ax, 5.35, Y_HOST, 3.5, 1.15,
     ["Update hosts", "SAGEConv (sent_by)", "+ SAGEConv (received_by)"], G2, fs=8)
ax.annotate("", xy=(4.75, Y_FLOW - 0.6), xytext=(4.75, Y_HOST + 0.6),
            arrowprops=dict(arrowstyle="-|>", color=G3, lw=0.9,
                            mutation_scale=8), zorder=2)
ax.annotate("", xy=(5.95, Y_HOST + 0.6), xytext=(5.95, Y_FLOW - 0.6),
            arrowprops=dict(arrowstyle="-|>", color=G3, lw=0.9,
                            mutation_scale=8), zorder=2)
ax.text(5.35, 2.0, "this layer is repeated three times", ha='center',
        va='center', fontsize=8, color=INK_SOFT, fontstyle='italic',
        fontfamily=FONT)

arr(ax, 2.85, Y_FLOW, 3.55, Y_FLOW)
arr(ax, 2.85, Y_HOST, 3.55, Y_HOST)
arr(ax, 7.15, Y_FLOW, 7.85, Y_FLOW)
rbox(ax, 8.9, Y_FLOW, 1.9, 0.95, ["Linear", "64 \u2192 5"], G0, fs=8)
ax.text(8.9, 4.9, "5 class scores\nper flow", ha='center', va='center',
        fontsize=8, fontweight='bold', color=INK, fontfamily=FONT)
arr(ax, 8.9, Y_FLOW - 0.48, 8.9, 5.25)
ax.text(8.9, Y_HOST, "host embeddings\nare not used by\nthe classifier",
        ha='center', va='center', fontsize=8, color=INK_SOFT,
        fontstyle='italic', fontfamily=FONT)

ax.set_title("Heterogeneous GNN: four SAGEConv relations\nsummed per node type",
             fontsize=10.5, fontweight='bold', color=INK, fontfamily=FONT)
save(fig, "fig17_5_hetero_gnn.png")

print(f"\nAll figures saved to {OUT}")