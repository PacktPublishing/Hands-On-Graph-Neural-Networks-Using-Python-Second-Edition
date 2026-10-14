import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Grayscale palette shared with the other chapters: G0 darkest, G6 lightest.
G1 = '#333333'; G3 = '#777777'; G6 = '#DDDDDD'
NODE_FILL = G1
EDGE_LINE = G3
R = 0.30

fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=300)
ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis('off'); ax.set_aspect('equal')

# ---- LEFT graph (disconnected) ----
L = {1:(2.3,4.0), 2:(0.9,1.2), 3:(3.5,1.2), 4:(4.7,2.6), 5:(4.7,1.2)}
L_edges = [(1,2),(1,3),(2,3),(4,5)]   # 4-5 now SOLID

# ---- RIGHT graph (connected) ----
Rg = {1:(8.0,4.0), 2:(6.6,1.2), 3:(9.2,1.2), 4:(10.4,2.6)}
R_edges = [(1,2),(1,3),(2,3),(1,4)]

def draw(pos, edges):
    for a, b in edges:
        (x1,y1), (x2,y2) = pos[a], pos[b]
        ax.plot([x1,x2], [y1,y2], color=EDGE_LINE, lw=2.2, zorder=1,
                solid_capstyle='round')
    for n, (x,y) in pos.items():
        ax.add_patch(Circle((x,y), R, facecolor=NODE_FILL, edgecolor='none', zorder=2))
        ax.text(x, y, str(n), color='white', ha='center', va='center',
                fontsize=12, fontweight='bold', zorder=3)

draw(L, L_edges)
draw(Rg, R_edges)

# subtle vertical divider like the original
ax.plot([5.8,5.8], [0.3,4.7], color=G6, lw=1.0, zorder=0)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'fig2_4_connected_vs_disconnected.png'),
            dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')

# ===========================================================================
# Figures 2.5 and 2.6, redrawn from the manuscript.
#
# They were supplied at 144 dpi and in colour, and were rejected as blurred.
# Figure 2.4 above is left exactly as it is: it was accepted as supplied, so
# there is nothing to gain by redrawing it and a proof cycle to lose.
# ===========================================================================
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, G2 as BG2, G4, G5 as BG5, G6 as BG6)
from matplotlib.patches import Ellipse, Rectangle

plt = apply()
OUT = os.path.dirname(os.path.abspath(__file__))


def dot(ax, x, y, r, label, fill, aspect, fs=7.5):
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r / aspect, facecolor=fill,
                         edgecolor=INK, linewidth=0.7, zorder=3))
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=INK,
                fontweight='bold', fontfamily=FONT, zorder=4)


def link(ax, p, q, color=BG2, lw=0.8, arrow=False, scale=8, trim=None):
    """`trim` is (radius, aspect): stop the arrow on the node's outline, or
    its head is drawn underneath and the edge looks undirected."""
    if arrow:
        if trim:
            r, aspect = trim
            dx, dy = q[0] - p[0], q[1] - p[1]
            t = 1.0 / ((dx / r) ** 2 + (dy / (r / aspect)) ** 2) ** 0.5
            p = (p[0] + dx * t, p[1] + dy * t)
            q = (q[0] - dx * t, q[1] - dy * t)
        ax.annotate("", xy=q, xytext=p,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    mutation_scale=scale), zorder=2)
    else:
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, zorder=1)


# ── Figure 2.5 — common types of graphs ─────────────────────────────────────
print("Figure 2.5 …")

fig, axes = figure(aspect=0.92, nrows=2, ncols=2)
ASP = 0.92 / 2 * 2            # two rows and two columns cancel out
for ax in axes.ravel():
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

R = 0.055


def panel_title(ax, text):
    ax.text(0.5, 0.99, text, ha='center', va='top', fontsize=8.5,
            fontweight='bold', color=INK, fontfamily=FONT)


# rooted tree
ax = axes[0, 0]; panel_title(ax, "Rooted tree")
tree = {'r': (0.50, 0.80), 'a': (0.28, 0.52), 'b': (0.72, 0.52),
        'c': (0.14, 0.22), 'd': (0.42, 0.22), 'e': (0.58, 0.22),
        'f': (0.86, 0.22)}
for a, b in [('r', 'a'), ('r', 'b'), ('a', 'c'), ('a', 'd'),
             ('b', 'e'), ('b', 'f')]:
    link(ax, tree[a], tree[b])
for key, (x, y) in tree.items():
    dot(ax, x, y, R, "", BG6 if key != 'r' else BG5, ASP)
ax.text(0.59, 0.80, "root", ha='left', va='center', fontsize=7.5,
        color=INK_SOFT, fontfamily=FONT, fontstyle='italic')

# directed acyclic graph
ax = axes[0, 1]; panel_title(ax, "Directed acyclic graph")
dag = {1: (0.14, 0.52), 2: (0.42, 0.24), 3: (0.50, 0.62),
       4: (0.84, 0.80), 5: (0.84, 0.30)}
for a, b in [(1, 3), (1, 2), (3, 4), (3, 5)]:
    link(ax, dag[a], dag[b], arrow=True, trim=(R, ASP))
for key, (x, y) in dag.items():
    dot(ax, x, y, R, "", BG6, ASP)

# bipartite
ax = axes[1, 0]; panel_title(ax, "Bipartite graph")
left = {i: (0.26, 0.80 - i * 0.16) for i in range(5)}
right = {i: (0.74, 0.72 - i * 0.16) for i in range(4)}
for a, b in [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 0), (3, 2), (4, 3)]:
    link(ax, left[a], right[b], lw=0.6)
for x, y in left.values():
    dot(ax, x, y, R * 0.8, "", BG6, ASP)
for x, y in right.values():
    dot(ax, x, y, R * 0.8, "", BG5, ASP)

# complete
ax = axes[1, 1]; panel_title(ax, "Complete graph")
n = 6
pts = [(0.5 + 0.30 * np.cos(2 * np.pi * k / n + np.pi / 2),
        0.46 + 0.30 * np.sin(2 * np.pi * k / n + np.pi / 2)) for k in range(n)]
for i in range(n):
    for j in range(i + 1, n):
        link(ax, pts[i], pts[j], lw=0.5, color=G4)
for x, y in pts:
    dot(ax, x, y, R * 0.8, "", BG6, ASP)

save_fig(fig, os.path.join(OUT, "fig2_5_graph_types.png"))
print("  saved fig2_5_graph_types.png")


# ── Figure 2.6 — a graph and its adjacency matrix ───────────────────────────
print("Figure 2.6 …")

NAMES = list("ABCDEFG")
EDGES = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("C", "G")]
POS = {"A": (0.52, 0.62), "B": (0.30, 0.34), "C": (0.74, 0.76),
       "D": (0.08, 0.26), "E": (0.30, 0.08), "F": (0.90, 0.62),
       "G": (0.70, 0.96)}

fig, (ax_g, ax_m) = figure(aspect=0.44, ncols=2)
for ax in (ax_g, ax_m):
    ax.set_facecolor(BG); ax.axis('off')
ax_g.set_xlim(0, 1); ax_g.set_ylim(0, 1)
ASP2 = 0.44 * 2
for a, b in EDGES:
    link(ax_g, POS[a], POS[b])
for name, (x, y) in POS.items():
    dot(ax_g, x, y, 0.062, name, BG5, ASP2, fs=8)

A = np.zeros((7, 7), dtype=int)
idx = {n: i for i, n in enumerate(NAMES)}
for a, b in EDGES:
    A[idx[a], idx[b]] = A[idx[b], idx[a]] = 1
table = ax_m.table(cellText=A, rowLabels=NAMES, colLabels=NAMES,
                   cellLoc='center', bbox=[0.12, 0.04, 0.86, 0.88])
table.auto_set_font_size(False)
table.set_fontsize(8)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor(G4)
    cell.set_linewidth(0.5)
    cell.get_text().set_fontfamily(FONT)
    cell.get_text().set_color(INK)
    if r == 0 or c == -1:
        cell.set_facecolor(BG6)
        cell.get_text().set_fontweight('bold')
    else:
        cell.set_facecolor(BG5 if A[r - 1, c] else BG)
ax_m.set_title("Its adjacency matrix", fontsize=9, fontweight='bold',
               color=INK, fontfamily=FONT)
ax_g.set_title("A graph", fontsize=9, fontweight='bold', color=INK,
               fontfamily=FONT)

save_fig(fig, os.path.join(OUT, "fig2_6_adjacency_matrix.png"))
print("  saved fig2_6_adjacency_matrix.png")
