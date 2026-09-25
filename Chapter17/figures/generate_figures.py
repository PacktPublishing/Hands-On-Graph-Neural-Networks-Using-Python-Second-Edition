"""
Chapter 17 – Detecting Anomalies Using Heterogeneous GNNs
generate_figures17_anomaly.py

Generates the two conceptual diagrams of the chapter: the CIDDS-001
network layout (17.1) and the HeteroGNN architecture (17.5). Every other
figure is data-derived and is produced by run.py from a real run, so that
nothing printed in the chapter comes from synthetic values.

Usage:
    python generate_figures.py
    -> saves fig17_1_network.png and fig17_5_hetero_gnn.png here
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

np.random.seed(0)

OUT  = os.path.dirname(os.path.abspath(__file__))
FONT = "DejaVu Sans"; BG = "white"
G0="#111111"; G1="#333333"; G2="#555555"
G3="#777777"; G4="#999999"; G5="#BBBBBB"; G6="#DDDDDD"

def save(fig, name, dpi=200):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"  saved {path}")

def rbox(ax, x, y, w, h, label, fill=G2, fc='white', fs=9, r=0.04):
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle=f"round,pad=0.02,rounding_size={r}",
                          facecolor=fill, edgecolor=G4,
                          linewidth=1.1, zorder=3)
    ax.add_patch(rect)
    for li, line in enumerate(label.split('\n')):
        off = 0.11 * (li - label.count('\n') / 2)
        ax.text(x, y-off, line, ha='center', va='center', fontsize=fs,
                color=fc, fontweight='bold', fontfamily=FONT, zorder=4)

def arr(ax, x1, y1, x2, y2, color=G2, lw=1.4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12), zorder=2)


# ── Fig 17.1: Virtual network diagram ────────────────────────────────────────
print("Figure 17.1 …")
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor("#F4F4F4"); ax.axis('off')
ax.set_xlim(0, 12); ax.set_ylim(0, 7)

subnets = [
    (1.4, 5.5, 2.0, 1.4, "Developer\n192.168.100.x", G1),
    (4.5, 5.5, 2.0, 1.4, "Office\n192.168.200.x",    G2),
    (7.6, 5.5, 2.0, 1.4, "Management\n192.168.210.x", G3),
    (4.5, 2.4, 2.0, 1.4, "Server\n192.168.220.x",    G0),
]
for x, y, w, h, lbl, fill in subnets:
    rbox(ax, x, y, w, h, lbl, fill=fill, fc='white', fs=9)

rbox(ax, 4.5, 4.2, 1.4, 0.7, "Firewall", fill=G1, fc='white', fs=9)

circle = plt.Circle((9.5, 4.2), 0.9, color=G5, fill=True, zorder=2)
ax.add_patch(circle)
ax.text(9.5, 4.2, "Internet", ha='center', va='center',
        fontsize=9, fontweight='bold', color=G0, fontfamily=FONT, zorder=3)

rbox(ax, 11.0, 4.2, 1.6, 1.0, "External\nserver\n(FTP + Web)", fill=G4, fc='white', fs=8)
ax.text(9.5, 1.8, "⚠ Attackers", ha='center', fontsize=10,
        color=G0, fontweight='bold', fontfamily=FONT)
ax.annotate("", xy=(9.5, 3.2), xytext=(9.5, 2.1),
            arrowprops=dict(arrowstyle="-|>", color=G0, lw=2.0,
                            mutation_scale=14, linestyle='dashed'))

connections = [
    (1.4, 4.83, 3.8, 4.33), (4.5, 4.83, 4.5, 4.55),
    (7.6, 4.83, 5.2, 4.33), (4.5, 3.65, 4.5, 3.17),
    (5.2, 4.2, 10.4, 4.2),
]
for x1, y1, x2, y2 in connections:
    ax.plot([x1, x2], [y1, y2], color=G2, linewidth=1.8, zorder=1)

ax.set_title("CIDDS-001 — virtual network environment\n"
             "Connections collected from local and external servers",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=8)
fig.tight_layout()
save(fig, "fig17_1_network.png")


# ── Fig 17.5: HeteroGNN architecture ─────────────────────────────────────────
print("Figure 17.5 …")
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 13); ax.set_ylim(0, 6)

rbox(ax, 1.0, 4.5, 1.6, 0.8, "Host nodes\n(IP features)",   fill=G1, fc='white', fs=9)
rbox(ax, 1.0, 2.5, 1.6, 0.8, "Flow nodes\n(traffic feat.)", fill=G3, fc='white', fs=9)

for l, x in enumerate([3.5, 5.8, 8.1]):
    rbox(ax, x, 4.5, 1.8, 0.85, f"SAGEConv x2\nhost->flow\n(sends / receives)",
         fill=G2, fc='white', fs=7.5)
    rbox(ax, x, 2.5, 1.8, 0.85, f"SAGEConv x2\nflow->host\n(sent_by / received_by)",
         fill=G2, fc='white', fs=7.5)
    rect = plt.Rectangle((x-1.1, 1.85), 2.2, 3.4,
                          fill=False, edgecolor=G4,
                          linestyle='--', linewidth=1.2, zorder=1)
    ax.add_patch(rect)
    ax.text(x, 5.55, f"HeteroConv {l+1}", ha='center',
            fontsize=8, color=G3, fontstyle='italic', fontfamily=FONT)
    if l > 0:
        arr(ax, x-1.1, 4.5, x-0.85, 4.5, color=G2)
        arr(ax, x-1.1, 2.5, x-0.85, 2.5, color=G2)

arr(ax, 1.8, 4.5, 2.65, 4.5, color=G1)
arr(ax, 1.8, 2.5, 2.65, 2.5, color=G3)

arr(ax, 9.0, 4.5, 9.5, 3.8, color=G2)
arr(ax, 9.0, 2.5, 9.5, 3.2, color=G2)
rbox(ax, 10.0, 3.5, 1.4, 1.4, "LeakyReLU\n+\nLinear(5)", fill=G0, fc='white', fs=9)
arr(ax, 10.7, 3.5, 11.5, 3.5, color=G0, lw=1.6)
ax.text(11.7, 3.5, "5 classes\n(output)",
        ha='left', va='center', fontsize=9,
        fontweight='bold', color=G0, fontfamily=FONT)

ax.set_title("Heterogeneous GNN architecture: 3 HeteroConv layers, 4 relations each",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig17_5_hetero_gnn.png")


print("\nDone: fig17_1 and fig17_5 saved.")