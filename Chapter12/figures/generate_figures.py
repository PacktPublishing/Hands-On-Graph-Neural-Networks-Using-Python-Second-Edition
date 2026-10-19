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
import sys

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, FILLS, G1, G2, G3, G4, G5, G6)

plt = apply()
G0 = "#111111"
SEED = 0
np.random.seed(SEED)
OUT = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))
    print(f"  saved {name}")


def rbox(ax, x, y, w, h, lines, fill, fc=None, fs=8, line_gap=None):
    """Square corners, dark type on a light fill; `fc` and `line_gap` are
    accepted and ignored, matplotlib spaces the lines itself."""
    ax.add_patch(Rectangle((x - w/2, y - h/2), w, h, facecolor=fill,
                           edgecolor=G3, linewidth=0.8, zorder=3))
    ax.text(x, y, "\n".join(lines), ha='center', va='center', fontsize=fs,
            color=INK, fontweight='bold', fontfamily=FONT, linespacing=1.5,
            zorder=4)


def arr(ax, x1, y1, x2, y2, color=G3, lw=1.1):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=9), zorder=2)


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
# Light fills with dark numbers: white digits on a mid grey close up in
# print at this size.
SHADE = {0: G4, 1: G5, 2: G6, 3: '#F0F0F0'}
SIZE = {0: 330, 1: 250, 2: 220, 3: 190}
LABEL = {0: INK, 1: INK, 2: INK, 3: INK}

pos = {
    0: (0.0, 0.0),
    1: (-1.8, -1.0), 2: (0.0, -1.0), 3: (1.8, -1.0),
    4: (-2.6, -2.1), 5: (-1.0, -2.1), 6: (0.0, -2.1), 7: (1.2, -2.1), 8: (2.6, -2.1),
    9: (-3.2, -3.2), 14: (-2.2, -3.2), 10: (-1.0, -3.2), 11: (0.0, -3.2),
    12: (1.2, -3.2), 13: (2.6, -3.2),
}

fig, ax = figure(aspect=0.64)
ax.set_facecolor(BG); ax.axis('off')

edge_color = [G1 if min(hops[u], hops[v]) == 0 else
              G3 if min(hops[u], hops[v]) == 1 else G5 for u, v in G.edges()]
edge_width = [1.5 if min(hops[u], hops[v]) == 0 else
              1.1 if min(hops[u], hops[v]) == 1 else 0.7 for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=edge_width)
nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_color=[SHADE[hops[n]] for n in G.nodes()],
                       node_size=[SIZE[hops[n]] for n in G.nodes()],
                       edgecolors=G3, linewidths=0.7)
for n, (x, y) in pos.items():
    ax.text(x, y, str(n), ha='center', va='center', fontsize=7.5,
            fontweight='bold', color=LABEL[hops[n]], fontfamily=FONT, zorder=5)

handles = [
    mpatches.Patch(facecolor=SHADE[0], edgecolor=G3, label='Target node (0)'),
    mpatches.Patch(facecolor=SHADE[1], edgecolor=G3, label='1 hop'),
    mpatches.Patch(facecolor=SHADE[2], edgecolor=G3, label='2 hops'),
    mpatches.Patch(facecolor=SHADE[3], edgecolor=G3, label='3 hops'),
]
ax.legend(handles=handles, loc='upper right', fontsize=8,
          frameon=False, handlelength=1.2, handletextpad=0.4)
ax.set_xlim(-3.8, 3.8); ax.set_ylim(-3.7, 0.5)
ax.set_title("1-hop, 2-hop, and 3-hop neighbors of node 0",
             fontsize=10.5, fontweight='bold', color=INK, fontfamily=FONT)
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

fig, ax = figure(aspect=0.46)
ax.set_facecolor(BG); ax.axis('off')
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
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            color=INK, fontfamily=FONT)
    ax.text(x0 + cols * CELL / 2, y0 - 0.25, dims, ha='center', va='top',
            fontsize=8, color=INK_SOFT, fontstyle='italic', fontfamily=FONT)


mid = 2.6
draw_matrix(ax, 0.4, mid - n * CELL / 2, np.where(A == 1, 0.2, 0.87), "A", "n × n")
ax.text(3.35, mid, "≈", ha='center', va='center', fontsize=18,
        color=INK, fontfamily=FONT)
draw_matrix(ax, 4.0, mid - n * CELL / 2, Z, "Z", "n × d")
ax.text(6.0, mid, "×", ha='center', va='center', fontsize=16,
        color=INK, fontfamily=FONT)
draw_matrix(ax, 6.6, mid - d * CELL / 2, Z.T, "Zᵀ", "d × n")

ax.text(5.5, 0.15,
        "n = number of nodes,  d = embedding dimension (d ≪ n)",
        ha='center', fontsize=8, color=INK_SOFT, fontstyle='italic',
        fontfamily=FONT)
ax.set_title("Matrix factorization: A ≈ Z × Zᵀ",
             fontsize=10.5, fontweight='bold', color=INK, fontfamily=FONT)
save(fig, "fig12_2_matrix_factorization.png")


# =============================================================================
# PART 3: Figure 12.3, the SEAL framework
# =============================================================================

print("Figure 12.3 …")

# Five stages in a row needed 14 inches. They run down the page instead: the
# input graph at the top, the three steps under it, the answer at the bottom.
# Each step carries its detail as a third line inside the box; set underneath,
# the detail landed on the arrow to the next step.
fig, ax = figure(aspect=1.00)
ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

Gs = nx.Graph()
Gs.add_edges_from([(0, 2), (2, 3), (3, 1), (2, 4), (4, 3), (0, 5), (5, 4)])
# The panel is square, so these positions are not stretched by the aspect.
pos_s = {0: (0.38, 0.955), 1: (0.62, 0.955), 2: (0.44, 0.912),
         3: (0.57, 0.912), 4: (0.50, 0.872), 5: (0.36, 0.872)}
nx.draw_networkx_edges(Gs, pos_s, ax=ax, edge_color=G4, width=0.8)
ax.plot([pos_s[0][0], pos_s[1][0]], [pos_s[0][1], pos_s[1][1]],
        linestyle='--', color=INK, linewidth=1.4, zorder=2)
nx.draw_networkx_nodes(Gs, pos_s, ax=ax,
                       node_color=[G4 if v in (0, 1) else G6 for v in Gs.nodes()],
                       node_size=70, edgecolors=G3, linewidths=0.7)
ax.text(0.12, 0.93, "Input graph", ha='center', va='center', fontsize=9,
        fontweight='bold', color=INK, fontfamily=FONT)
ax.text(0.86, 0.93, "dashed line =\ncandidate link", ha='center', va='center',
        fontsize=8, color=INK_SOFT, fontstyle='italic', fontfamily=FONT,
        linespacing=1.4)

STEPS = [
    (0.655, "Step 1  Enclosing subgraph extraction\n"
            "k-hop neighborhood around each candidate link"),
    (0.435, "Step 2  Node information matrix\n"
            "DRNL labels + optional embeddings + node features"),
    (0.215, "Step 3  GNN training (DGCNN)\n"
            "sort pooling, 1D convolutions, dense layers"),
]
for y, text in STEPS:
    rbox(ax, 0.5, y, 0.84, 0.115, text.split("\n"), FILLS[1], fs=8)

arr(ax, 0.5, 0.835, 0.5, 0.720)
arr(ax, 0.5, 0.593, 0.5, 0.500)
arr(ax, 0.5, 0.373, 0.5, 0.280)
arr(ax, 0.5, 0.153, 0.5, 0.078)
rbox(ax, 0.5, 0.035, 0.40, 0.070, ["Link probability"], FILLS[2], fs=8.5)

ax.set_title("The SEAL framework: from a candidate link to a probability",
             fontsize=10.5, fontweight='bold', color=INK, fontfamily=FONT)
save(fig, "fig12_3_seal_framework.png")

print(f"\nAll figures saved to {OUT}")