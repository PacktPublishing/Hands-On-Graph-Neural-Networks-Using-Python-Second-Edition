"""
Chapter 12: conceptual diagrams.

Saves, in the same folder as this script:
    fig12_1_hops.png
    fig12_2_matrix_factorization.png
    fig12_3_seal_framework.png

Figure 12.4, the VGAE and SEAL results table, is produced by run.py.

Run: python generate_figures.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx

SEED = 0
np.random.seed(SEED)

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


def rbox(ax, x, y, w, h, lines, fill, fc='white', fs=11, line_gap=0.30):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=fill, edgecolor=G3,
                                linewidth=1.2, zorder=3))
    n = len(lines)
    for i, line in enumerate(lines):
        ax.text(x, y + (n - 1) / 2 * line_gap - i * line_gap, line,
                ha='center', va='center', fontsize=fs, color=fc,
                fontweight='bold', fontfamily=FONT, zorder=4)


def arr(ax, x1, y1, x2, y2, color=G2, lw=2.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=16), zorder=2)


# =============================================================================
# PART 1: Figure 12.1, 1-hop, 2-hop, and 3-hop neighbors of node 0
# =============================================================================

print("Figure 12.1 …")

edges = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (1, 5), (2, 6), (3, 7), (3, 8),
    (4, 9), (4, 14), (5, 10), (6, 11), (7, 12), (8, 13),
]
G = nx.Graph()
G.add_edges_from(edges)

hops = nx.single_source_shortest_path_length(G, 0)
SHADE = {0: G0, 1: G2, 2: G4, 3: G6}
SIZE = {0: 1000, 1: 700, 2: 600, 3: 500}
LABEL = {0: 'white', 1: 'white', 2: 'white', 3: G0}

pos = {
    0: (0.0, 0.0),
    1: (-1.8, -1.0), 2: (0.0, -1.0), 3: (1.8, -1.0),
    4: (-2.6, -2.1), 5: (-1.0, -2.1), 6: (0.0, -2.1), 7: (1.2, -2.1), 8: (2.6, -2.1),
    9: (-3.2, -3.2), 14: (-2.2, -3.2), 10: (-1.0, -3.2), 11: (0.0, -3.2),
    12: (1.2, -3.2), 13: (2.6, -3.2),
}

fig, ax = plt.subplots(figsize=(10, 6.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')

edge_color = [G1 if min(hops[u], hops[v]) == 0 else
              G3 if min(hops[u], hops[v]) == 1 else G5 for u, v in G.edges()]
edge_width = [2.2 if min(hops[u], hops[v]) == 0 else
              1.6 if min(hops[u], hops[v]) == 1 else 1.0 for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=edge_width)
nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_color=[SHADE[hops[n]] for n in G.nodes()],
                       node_size=[SIZE[hops[n]] for n in G.nodes()],
                       edgecolors=G3, linewidths=1.0)
for n, (x, y) in pos.items():
    ax.text(x, y, str(n), ha='center', va='center', fontsize=10,
            fontweight='bold', color=LABEL[hops[n]], fontfamily=FONT, zorder=5)

handles = [
    mpatches.Patch(facecolor=G0, edgecolor=G3, label='Target node (0)'),
    mpatches.Patch(facecolor=G2, edgecolor=G3, label='1-hop neighbors'),
    mpatches.Patch(facecolor=G4, edgecolor=G3, label='2-hop neighbors'),
    mpatches.Patch(facecolor=G6, edgecolor=G3, label='3-hop neighbors'),
]
ax.legend(handles=handles, loc='upper right', fontsize=10,
          framealpha=0.95, edgecolor=G5)
ax.set_xlim(-3.8, 3.8); ax.set_ylim(-3.7, 0.5)
ax.set_title("1-hop, 2-hop, and 3-hop neighbors of node 0",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig12_1_hops.png")


# =============================================================================
# PART 2: Figure 12.2, matrix factorization A ≈ Z × Zᵀ
# =============================================================================

print("Figure 12.2 …")

rng = np.random.default_rng(SEED)
n, d = 5, 3
upper = np.triu(rng.random((n, n)) < 0.45, k=1)
A = (upper | upper.T).astype(int)
Z = 0.15 + 0.7 * rng.random((n, d))

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 11); ax.set_ylim(0, 5.2)

CELL = 0.5


def draw_matrix(ax, x0, y0, values, label, dims):
    rows, cols = values.shape
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(plt.Rectangle((x0 + c * CELL, y0 + (rows - 1 - r) * CELL),
                                       CELL, CELL, facecolor=str(values[r, c]),
                                       edgecolor='white', linewidth=0.8))
    ax.add_patch(plt.Rectangle((x0, y0), cols * CELL, rows * CELL,
                               facecolor='none', edgecolor=G1, linewidth=1.5))
    ax.text(x0 + cols * CELL / 2, y0 + rows * CELL + 0.2, label,
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color=G0, fontfamily=FONT)
    ax.text(x0 + cols * CELL / 2, y0 - 0.25, dims, ha='center', va='top',
            fontsize=10, color=G3, fontstyle='italic', fontfamily=FONT)


mid = 2.6
draw_matrix(ax, 0.4, mid - n * CELL / 2, np.where(A == 1, 0.2, 0.87), "A", "n × n")
ax.text(3.35, mid, "≈", ha='center', va='center', fontsize=30,
        color=G0, fontfamily=FONT)
draw_matrix(ax, 4.0, mid - n * CELL / 2, Z, "Z", "n × d")
ax.text(6.0, mid, "×", ha='center', va='center', fontsize=26,
        color=G0, fontfamily=FONT)
draw_matrix(ax, 6.6, mid - d * CELL / 2, Z.T, "Zᵀ", "d × n")

ax.text(5.5, 0.15,
        "n = number of nodes,  d = embedding dimension (d ≪ n)",
        ha='center', fontsize=10, color=G3, fontstyle='italic', fontfamily=FONT)
ax.set_title("Matrix factorization: A ≈ Z × Zᵀ",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig12_2_matrix_factorization.png")


# =============================================================================
# PART 3: Figure 12.3, the SEAL framework
# =============================================================================

print("Figure 12.3 …")

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 14); ax.set_ylim(0, 5)

Gs = nx.Graph()
Gs.add_edges_from([(0, 2), (2, 3), (3, 1), (2, 4), (4, 3), (0, 5), (5, 4)])
pos_s = {0: (0.45, 3.3), 1: (1.75, 3.3), 2: (0.75, 2.4),
         3: (1.45, 2.4), 4: (1.1, 1.6), 5: (0.3, 1.6)}
nx.draw_networkx_edges(Gs, pos_s, ax=ax, edge_color=G4, width=1.2)
ax.plot([pos_s[0][0], pos_s[1][0]], [pos_s[0][1], pos_s[1][1]],
        linestyle='--', color=G0, linewidth=2.0, zorder=2)
nx.draw_networkx_nodes(Gs, pos_s, ax=ax,
                       node_color=[G0 if v in (0, 1) else G5 for v in Gs.nodes()],
                       node_size=260, edgecolors=G3, linewidths=0.8)
ax.text(1.1, 4.15, "Input graph", ha='center', fontsize=11,
        fontweight='bold', color=G0, fontfamily=FONT)
ax.text(1.1, 0.95, "dashed line = candidate link\nbetween the two dark nodes",
        ha='center', va='top', fontsize=9, color=G3, fontstyle='italic',
        fontfamily=FONT)

boxes = [
    (4.0, ["Step 1", "Enclosing subgraph", "extraction"], G1,
     "k-hop neighborhood around\neach candidate link"),
    (7.3, ["Step 2", "Node information", "matrix construction"], G2,
     "DRNL labels + optional\nembeddings + node features"),
    (10.6, ["Step 3", "GNN training", "(DGCNN)"], G3,
     "Sort pooling, 1D convolutions,\ndense layers"),
]
for x, lines, fill, sub in boxes:
    rbox(ax, x, 2.6, 2.6, 1.7, lines, fill)
    ax.text(x, 1.45, sub, ha='center', va='top', fontsize=9.5, color=G3,
            fontstyle='italic', fontfamily=FONT)

arr(ax, 2.15, 2.6, 2.65, 2.6)
arr(ax, 5.35, 2.6, 5.95, 2.6)
arr(ax, 8.65, 2.6, 9.25, 2.6)
arr(ax, 11.95, 2.6, 12.55, 2.6)
ax.text(13.3, 2.6, "Link\nprobability", ha='center', va='center',
        fontsize=11, color=G0, fontweight='bold', fontfamily=FONT)

ax.set_title("The SEAL framework: three steps from a candidate link to a link probability",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig12_3_seal_framework.png")

print(f"\nAll figures saved to {OUT}")