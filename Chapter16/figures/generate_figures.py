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

FONT = "DejaVu Sans"
BG   = "white"
G0 = "#111111"; G1 = "#333333"; G2 = "#555555"
G3 = "#777777"; G4 = "#999999"; G5 = "#BBBBBB"; G6 = "#DDDDDD"

def save(fig, name, dpi=200):
    fig.savefig(f"{OUT}/{name}", dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"  saved {name}")

def rbox(ax, x, y, w, h, label, fill=G2, fc='white', fs=9, r=0.04):
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle=f"round,pad=0.02,rounding_size={r}",
                          facecolor=fill, edgecolor=G4,
                          linewidth=1.1, zorder=3)
    ax.add_patch(rect)
    for li, line in enumerate(label.split('\n')):
        off = 0.11*(li - label.count('\n')/2)
        ax.text(x, y-off, line, ha='center', va='center',
                fontsize=fs, color=fc, fontweight='bold',
                fontfamily=FONT, zorder=4)

def arr(ax, x1, y1, x2, y2, color=G2, lw=1.4):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12), zorder=2)


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

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor("#F0F0F0"); ax.axis('off')
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
    ax.plot(xs, ys, color=G3, linewidth=5, solid_capstyle='round', zorder=1)
    ax.plot(xs, ys, color='white', linewidth=1.5,
            linestyle='--', solid_capstyle='round', zorder=2)

# Sensor nodes
sensor_pos = [(1.2,3.0),(2.0,3.0),(3.5,3.0),(5.0,3.0),(6.2,3.0),
              (7.8,3.0),(9.0,3.0),(3.0,1.2),(3.0,2.0),(3.0,4.0),
              (3.0,5.0),(7.0,1.2),(7.0,2.0),(7.0,4.0),(2.0,1.5),
              (3.5,1.5),(6.0,2.5),(7.5,3.8),(1.3,1.5),(1.6,2.2)]
for x, y in sensor_pos:
    circle = plt.Circle((x,y), 0.18, color=G0, zorder=4)
    ax.add_patch(circle)
    circle2 = plt.Circle((x,y), 0.26, color='white', zorder=3)
    ax.add_patch(circle2)

# Labels
ax.text(4.8, 5.5, "N ↑", ha='center', fontsize=11, fontweight='bold',
        color=G2, fontfamily=FONT)
ax.text(0.3, 0.3, "District 7\nCalifornia", ha='left', fontsize=8,
        color=G2, fontfamily=FONT, fontstyle='italic')

ax.legend(
    [mpatches.Patch(color=G0),
     (Line2D([0],[0], color=G3, lw=5), Line2D([0],[0], color='white', lw=1.5, ls='--'))],
    ['Sensor station (loop detector)', 'Highway corridor'],
    handler_map={tuple: HandlerTuple(ndivide=1)},
    loc='lower right', fontsize=9, framealpha=0.95, edgecolor=G5)
fig.tight_layout()
save(fig, "fig16_1_sensor_network.png", dpi=300)


# ── Fig 16.7: A3T-GCN architecture diagram ───────────────────────────────────
print("Figure 16.7 …")

fig, ax = plt.subplots(figsize=(14.5, 5.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 14.5); ax.set_ylim(0, 5.5)

# Time steps, spread across the width so the two rows share the canvas
for i, x in enumerate([1.3, 3.3, 5.3]):
    t_lbl = f"t={i}"
    rbox(ax, x, 4.6, 1.5, 0.7, f"Graph\n{t_lbl}", fill=G1, fc='white', fs=9)
    arr(ax, x, 4.25, x, 3.6, color=G1)
    rbox(ax, x, 3.1, 1.5, 0.8, f"GCN+GRU\n{t_lbl}", fill=G2, fc='white', fs=8)
    arr(ax, x, 2.7, x, 2.15, color=G2)
    rbox(ax, x, 1.7, 1.5, 0.7, f"h{i}\nhidden", fill=G3, fc='white', fs=8)
    if i > 0:
        arr(ax, x-1.4, 3.1, x-0.8, 3.1, color=G2)

# Attention row
for x in (1.3, 3.3, 5.3):
    arr(ax, x, 1.35, 6.6, 0.95, color=G3)
rbox(ax, 7.6, 0.8, 1.9, 0.9, "Attention\nscoring", fill=G0, fc='white', fs=9)
arr(ax, 8.55, 0.8, 9.15, 0.8, color=G0)
rbox(ax, 10.0, 0.8, 1.5, 0.9, "Context\nvector", fill=G2, fc='white', fs=9)
arr(ax, 10.75, 0.8, 11.35, 0.8, color=G2)
rbox(ax, 12.2, 0.8, 1.6, 0.9, "Linear\n-> forecast", fill=G4, fc='white', fs=9)
arr(ax, 13.0, 0.8, 13.5, 0.8, color=G4)
ax.text(13.65, 0.8, "y_hat", ha='left', va='center', fontsize=12,
        fontweight='bold', color=G0, fontfamily=FONT)
fig.tight_layout()
save(fig, "fig16_7_a3tgcn.png")


# ── Fig 16.11: STAEformer architecture ────────────────────────────────────────
print("Figure 16.11 …")

fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 13); ax.set_ylim(0, 5.5)

# Input: (B, N, T)
rbox(ax, 1.0, 3.0, 1.6, 0.9, "Input\n(B, N, T)", fill=G1, fc='white', fs=9)
arr(ax, 1.8, 3.0, 2.6, 3.0, color=G1)

# Two parallel branches: temporal projection and adaptive node embedding
rbox(ax, 3.6, 4.0, 2.0, 0.9,
     "Temporal input\nprojection\n(T -> d_model)", fill=G2, fc='white', fs=8)
rbox(ax, 3.6, 2.0, 2.0, 0.9,
     "Adaptive node\nembeddings\n(learned per node)", fill=G3, fc='white', fs=8)
arr(ax, 2.7, 3.2, 2.7, 4.0, color=G2)
arr(ax, 2.7, 2.8, 2.7, 2.0, color=G3)

# Sum
rbox(ax, 6.0, 3.0, 0.7, 0.7, "+", fill=G0, fc='white', fs=14)
arr(ax, 4.7, 3.9, 5.7, 3.2, color=G2)
arr(ax, 4.7, 2.1, 5.7, 2.8, color=G3)

# Transformer encoder
rbox(ax, 8.0, 3.0, 2.2, 1.4,
     "Transformer encoder\n(self-attention\nover N nodes)",
     fill=G0, fc='white', fs=9)
arr(ax, 6.4, 3.0, 6.9, 3.0, color=G0)

# Output projection
rbox(ax, 10.5, 3.0, 1.6, 0.9, "Output\nprojection", fill=G2, fc='white', fs=9)
arr(ax, 9.1, 3.0, 9.7, 3.0, color=G0)

# Prediction
rbox(ax, 12.4, 3.0, 1.0, 0.9, "y_hat", fill=G4, fc='white', fs=10)
arr(ax, 11.3, 3.0, 11.9, 3.0, color=G2)
fig.tight_layout()
save(fig, "fig16_11_staeformer.png")


print(f"\nAll figures saved to {OUT}")