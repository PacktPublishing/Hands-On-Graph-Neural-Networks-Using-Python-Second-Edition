"""
Chapter 16 - Conceptual diagrams.

Three schematics that illustrate ideas rather than data: the road sensor
network (16.1), the A3T-GCN architecture (16.7) and the STAEformer
architecture (16.11). Every other figure in the chapter is produced by
run.py from a real run, so that no printed number comes from synthetic data.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import os

np.random.seed(0)

OUT  = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

import sys

sys.path.insert(0, os.path.join(OUT, '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, FILLS, G1, G2, G3, G4, G5, G6)
from matplotlib.patches import Rectangle

plt = apply()
G0 = "#111111"


def save(fig, name, dpi=None):
    save_fig(fig, f"{OUT}/{name}")
    print(f"  saved {name}")


def rbox(ax, x, y, w, h, label, fill=G6, fc=None, fs=8, r=0.04):
    """Square corners, dark type on a light fill; `fc` is ignored."""
    ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, facecolor=fill,
                           edgecolor=G3, linewidth=0.8, zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=INK,
            fontweight='bold', fontfamily=FONT, linespacing=1.5, zorder=4)


def arr(ax, x1, y1, x2, y2, color=G3, lw=1.1):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=9), zorder=2)


# ── Synthetic PeMS-M data ─────────────────────────────────────────────────────
N_STATIONS = 228
N_STEPS    = 2016   # ~7 days at 5-min intervals (representative subset)

t = np.arange(N_STEPS)
# Daily seasonality (288 steps/day) + weekly pattern + noise
base_speed = 55.0
speeds_syn = np.zeros((N_STEPS, N_STATIONS))
for s in range(N_STATIONS):
    phase    = np.random.uniform(0, 0.2)
    amp_day  = 12 + np.random.uniform(-3, 3)
    amp_week = 5  + np.random.uniform(-2, 2)
    speeds_syn[:, s] = (
        base_speed + amp_day * np.sin(2*np.pi*(t/288 - 0.25 + phase))
        + amp_week * np.sin(2*np.pi*t/2016)
        + np.random.normal(0, 3, N_STEPS)
    )
speeds_syn = np.clip(speeds_syn, 5, 85)

# Distance matrix (228×228), exponential decay
coords = np.random.uniform(0, 100, (N_STATIONS, 2))
dist_mat = np.zeros((N_STATIONS, N_STATIONS))
for i in range(N_STATIONS):
    for j in range(N_STATIONS):
        dist_mat[i,j] = np.linalg.norm(coords[i]-coords[j])

# Weighted adjacency matrix (same formula as chapter)
sigma2, epsilon = 0.1, 0.5
d = dist_mat / 10000.
d2 = d * d
w_mask = np.ones((N_STATIONS, N_STATIONS)) - np.eye(N_STATIONS)
adj = np.exp(-d2/sigma2) * (np.exp(-d2/sigma2) >= epsilon) * w_mask

mean_s = speeds_syn.mean(axis=1)
std_s  = speeds_syn.std(axis=1)


# ── Fig 16.1: Road sensor network schematic ───────────────────────────────────
print("Figure 16.1 …")

fig, ax = figure(aspect=0.60)
ax.set_facecolor("#F4F4F4"); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
# Draw stylised roads
roads = [
    [(0.5,3.0),(9.5,3.0)],   # horizontal main
    [(3.0,0.5),(3.0,5.5)],   # vertical 1
    [(7.0,0.5),(7.0,5.5)],   # vertical 2
    [(0.5,1.5),(5.0,1.5)],   # branch 1
    [(5.0,1.5),(9.5,4.5)],   # diagonal
    [(1.0,0.5),(2.0,3.0)],   # short branch
]
for road in roads:
    xs = [p[0] for p in road]; ys = [p[1] for p in road]
    ax.plot(xs, ys, color=G4, linewidth=3.2, solid_capstyle='round', zorder=1)
    ax.plot(xs, ys, color='white', linewidth=0.9,
            linestyle='--', solid_capstyle='round', zorder=2)

# Sensor nodes
sensor_pos = [(1.2,3.0),(2.0,3.0),(3.5,3.0),(5.0,3.0),(6.2,3.0),
              (7.8,3.0),(9.0,3.0),(3.0,1.2),(3.0,2.0),(3.0,4.0),
              (3.0,5.0),(7.0,1.2),(7.0,2.0),(7.0,4.0),(2.0,1.5),
              (3.5,1.5),(6.0,2.5),(7.5,3.8),(1.3,1.5),(1.6,2.2)]
for x, y in sensor_pos:
    ax.add_patch(plt.Circle((x, y), 0.20, color='white', zorder=3))
    ax.add_patch(plt.Circle((x, y), 0.13, facecolor=G2, edgecolor=G1,
                            linewidth=0.5, zorder=4))

# Labels
ax.text(4.8, 5.5, "N ↑", ha='center', fontsize=9, fontweight='bold',
        color=INK, fontfamily=FONT)
ax.text(0.3, 0.3, "District 7\nCalifornia", ha='left', fontsize=8,
        color=INK, fontfamily=FONT, fontstyle='italic')

ax.legend(
    [mpatches.Patch(facecolor=G2, edgecolor=G1),
     (Line2D([0], [0], color=G4, lw=3.2),
      Line2D([0], [0], color='white', lw=0.9, ls='--'))],
    ['Sensor station', 'Highway corridor'],
    handler_map={tuple: HandlerTuple(ndivide=1)},
    loc='lower right', fontsize=8, framealpha=0.95, edgecolor=G5)
save(fig, "fig16_1_sensor_network.png")


# ── Fig 16.7: A3T-GCN architecture diagram ──────────────────────────
print("Figure 16.7 \u2026")

# The original ran 14.5 inches across. The three time steps keep their columns
# but the attention chain moves under them instead of beside them.
fig, ax = figure(aspect=0.86)
ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

XS = [0.20, 0.50, 0.80]
for i, x in enumerate(XS):
    rbox(ax, x, 0.95, 0.22, 0.075, f"Graph t={i}", fill=G6, fs=8)
    arr(ax, x, 0.912, x, 0.862)
    rbox(ax, x, 0.805, 0.22, 0.095, f"GCN + GRU\nt={i}", fill=FILLS[1], fs=8)
    arr(ax, x, 0.757, x, 0.707)
    rbox(ax, x, 0.665, 0.22, 0.075, f"h{i} hidden", fill=FILLS[1], fs=8)
    if i:
        arr(ax, XS[i-1] + 0.11, 0.805, x - 0.11, 0.805)
    # The hidden states are what attention scores, so they converge there.
    arr(ax, x, 0.627, 0.5, 0.545)

# The tail of the model runs down the page: three boxes side by side would be
# 1.6 inches each, and "Attention scoring" does not fit in that.
rbox(ax, 0.5, 0.485, 0.34, 0.085, "Attention scoring", fill=FILLS[2], fs=8)
arr(ax, 0.5, 0.442, 0.5, 0.392)
rbox(ax, 0.5, 0.335, 0.30, 0.085, "Context vector", fill=FILLS[1], fs=8)
arr(ax, 0.5, 0.292, 0.5, 0.242)
rbox(ax, 0.5, 0.185, 0.30, 0.085, "Linear layer", fill=FILLS[1], fs=8)
arr(ax, 0.5, 0.142, 0.5, 0.095)
ax.text(0.5, 0.055, "forecast \u0177", ha='center', va='center', fontsize=8.5,
        fontweight='bold', color=INK, fontfamily=FONT)
save(fig, "fig16_7_a3tgcn.png")


# ── Fig 16.11: STAEformer architecture ────────────────────────────
print("Figure 16.11 \u2026")

# Six stages across needed 13 inches. The pipeline runs down the page, with
# the two branches side by side where it forks.
fig, ax = figure(aspect=0.88)
ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

rbox(ax, 0.5, 0.93, 0.34, 0.09, "Input (B, N, T)", fill=G6, fs=8)
arr(ax, 0.42, 0.885, 0.27, 0.815)
arr(ax, 0.58, 0.885, 0.73, 0.815)
rbox(ax, 0.27, 0.73, 0.40, 0.13,
     "Temporal input projection\n(T \u2192 d_model)", fill=FILLS[1], fs=8)
rbox(ax, 0.73, 0.73, 0.40, 0.13,
     "Adaptive node embeddings\n(learned per node)", fill=FILLS[1], fs=8)
arr(ax, 0.27, 0.665, 0.44, 0.595)
arr(ax, 0.73, 0.665, 0.56, 0.595)
rbox(ax, 0.5, 0.545, 0.10, 0.08, "+", fill=FILLS[2], fs=10)
arr(ax, 0.5, 0.505, 0.5, 0.445)
rbox(ax, 0.5, 0.375, 0.52, 0.125,
     "Transformer encoder\n(self-attention over N nodes)", fill=FILLS[2], fs=8)
arr(ax, 0.5, 0.312, 0.5, 0.252)
rbox(ax, 0.5, 0.195, 0.34, 0.095, "Output projection", fill=FILLS[1], fs=8)
arr(ax, 0.5, 0.147, 0.5, 0.095)
ax.text(0.5, 0.055, "forecast \u0177", ha='center', va='center', fontsize=8.5,
        fontweight='bold', color=INK, fontfamily=FONT)
save(fig, "fig16_11_staeformer.png")


print(f"\nAll figures saved to {OUT}")