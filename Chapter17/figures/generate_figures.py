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

FONT = "DejaVu Sans"
BG = "white"
G0 = "#111111"
G1 = "#333333"
G2 = "#555555"
G3 = "#777777"
G4 = "#999999"
G5 = "#BBBBBB"
G6 = "#DDDDDD"


def save(fig, name, dpi=200):
    fig.savefig(os.path.join(OUT, name), dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"  saved {name}")


def rbox(ax, x, y, w, h, lines, fill, fc='white', fs=10, gap=0.24,
         bold_first=True, edge=G3, ls='-'):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=fill, edgecolor=edge, linewidth=1.2,
                                linestyle=ls, zorder=3))
    n = len(lines)
    for i, line in enumerate(lines):
        ax.text(x, y + (n - 1) / 2 * gap - i * gap, line, ha='center',
                va='center', fontsize=fs if (i == 0 or not bold_first) else fs - 1,
                color=fc, fontweight='bold' if (i == 0 or not bold_first) else 'normal',
                fontfamily=FONT, zorder=4)


def arr(ax, x1, y1, x2, y2, color=G2, lw=1.6, ls='-'):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, mutation_scale=14), zorder=2)


# =============================================================================
# PART 1: Figure 17.1, the emulated network of CIDDS-001
# =============================================================================

print("Figure 17.1 …")

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 13); ax.set_ylim(0, 7)

ax.add_patch(FancyBboxPatch((0.3, 0.5), 7.6, 5.6,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor="#F4F4F4", edgecolor=G4,
                            linewidth=1.2, linestyle='--', zorder=1))
ax.text(4.1, 5.8, "OpenStack environment (internal traffic)", ha='center',
        fontsize=11, fontweight='bold', color=G1, fontfamily=FONT)

rbox(ax, 4.1, 3.3, 2.0, 0.8, ["Internal network"], G2)

subnets = [
    (1.8, 4.7, ["Server", "192.168.100.0/24"], G0),
    (6.4, 4.7, ["Management", "192.168.200.0/24"], G1),
    (6.4, 1.6, ["Office", "192.168.210.0/24"], G1),
    (1.8, 1.6, ["Developer", "192.168.220.0/24",
                "incl. the 2 attacking clients"], G0),
]
for x, y, lines, fill in subnets:
    rbox(ax, x, y, 2.5, 1.15 if len(lines) == 3 else 0.95, lines, fill)
    ax.plot([x, 4.1], [y, 3.3], color=G3, linewidth=1.6, zorder=2)

circle = plt.Circle((9.4, 3.3), 0.85, facecolor=G6, edgecolor=G3,
                    linewidth=1.2, zorder=2)
ax.add_patch(circle)
ax.text(9.4, 3.3, "Internet", ha='center', va='center', fontsize=11,
        fontweight='bold', color=G0, fontfamily=FONT, zorder=3)
ax.plot([5.1, 8.55], [3.3, 3.3], color=G3, linewidth=1.6, zorder=1)

rbox(ax, 11.6, 4.8, 2.3, 1.15, ["External server", "web server",
                                "file synchronization"], G3)
ax.plot([10.1, 10.45], [3.8, 4.4], color=G3, linewidth=1.6, zorder=1)

rbox(ax, 11.6, 1.6, 2.3, 0.95, ["External attackers", "attack the external server"],
     G6, fc=G0)
arr(ax, 11.6, 2.1, 11.6, 4.2, color=G0, lw=1.8, ls='--')

ax.set_title("CIDDS-001: emulated small business network",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=8)
fig.tight_layout()
save(fig, "fig17_1_network.png")


# =============================================================================
# PART 2: Figure 17.5, the heterogeneous GNN
# =============================================================================

print("Figure 17.5 …")

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 14); ax.set_ylim(0, 6)

Y_FLOW, Y_HOST = 4.1, 1.9

rbox(ax, 1.0, Y_FLOW, 1.7, 0.9, ["Flow nodes", "flow features"], G3)
rbox(ax, 1.0, Y_HOST, 1.7, 0.9, ["Host nodes", "16 address bits"], G1)

xs = [3.6, 6.4, 9.2]
for l, x in enumerate(xs):
    ax.add_patch(plt.Rectangle((x - 1.25, 1.05), 2.5, 4.0, fill=False,
                               edgecolor=G4, linestyle='--', linewidth=1.2,
                               zorder=1))
    ax.text(x, 5.3, f"HeteroConv {l + 1} + LeakyReLU", ha='center',
            fontsize=10, color=G1, fontstyle='italic', fontfamily=FONT)
    rbox(ax, x, Y_FLOW, 2.2, 1.05,
         ["Update flows", "SAGEConv (sends)", "+ SAGEConv (receives)"], G2, fs=10)
    rbox(ax, x, Y_HOST, 2.2, 1.05,
         ["Update hosts", "SAGEConv (sent_by)", "+ SAGEConv (received_by)"], G2, fs=10)
    ax.annotate("", xy=(x - 0.35, Y_FLOW - 0.55), xytext=(x - 0.35, Y_HOST + 0.55),
                arrowprops=dict(arrowstyle="-|>", color=G4, lw=1.1,
                                mutation_scale=10), zorder=2)
    ax.annotate("", xy=(x + 0.35, Y_HOST + 0.55), xytext=(x + 0.35, Y_FLOW - 0.55),
                arrowprops=dict(arrowstyle="-|>", color=G4, lw=1.1,
                                mutation_scale=10), zorder=2)

arr(ax, 1.85, Y_FLOW, 2.45, Y_FLOW)
arr(ax, 1.85, Y_HOST, 2.45, Y_HOST)
for a, b in zip(xs[:-1], xs[1:]):
    arr(ax, a + 1.1, Y_FLOW, b - 1.1, Y_FLOW)
    arr(ax, a + 1.1, Y_HOST, b - 1.1, Y_HOST)

rbox(ax, 11.6, Y_FLOW, 1.6, 0.9, ["Linear", "64 → 5"], G0)
arr(ax, 10.3, Y_FLOW, 10.8, Y_FLOW, color=G0)
arr(ax, 12.4, Y_FLOW, 12.85, Y_FLOW, color=G0)
ax.text(13.45, Y_FLOW, "5 class\nscores\nper flow", ha='center', va='center',
        fontsize=10, fontweight='bold', color=G0, fontfamily=FONT)
ax.text(11.6, Y_HOST, "Host embeddings are\nnot used by the classifier",
        ha='center', va='center', fontsize=9.5, color=G3, fontstyle='italic',
        fontfamily=FONT)

ax.set_title("Heterogeneous GNN: three HeteroConv layers, each with four SAGEConv "
             "relations summed per node type",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig17_5_hetero_gnn.png")

print(f"\nAll figures saved to {OUT}")