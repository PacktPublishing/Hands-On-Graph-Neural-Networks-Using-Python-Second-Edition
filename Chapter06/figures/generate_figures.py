"""
Chapter 6 - conceptual figures (Figures 6.1 and 6.2).
Figures 6.3 to 6.8 come from the data and are saved by run.py.

Usage:
    python figures/generate_figures.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, INK, INK_SOFT,
                       G3, G4, G5, G6)

plt = apply()

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

EDGE = G3
EDGE_BG = G6


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))
    print(f"  saved {name}")


def node(ax, xy, text, fill, size=330, ring=0.8, fontsize=9):
    ax.scatter([xy[0]], [xy[1]], s=size, facecolor=fill, edgecolors=INK,
               linewidths=ring, zorder=3)
    ax.text(xy[0], xy[1], text, ha='center', va='center', fontsize=fontsize,
            color=INK, fontweight='bold', fontfamily=FONT, zorder=4)


# =============================================================================
# Figure 6.1 - a graph whose nodes have different numbers of neighbors
# =============================================================================
print("Figure 6.1 ...")

pos = {'A': (0.0, 1.0), 'B': (-1.2, 0.0), 'C': (1.2, 0.0), 'D': (0.0, -1.0)}
edges = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('A', 'D')]
deg = {n: sum(n in e for e in edges) for n in pos}

fig, ax = figure(aspect=0.50)
ax.axis('off')
for a, b in edges:
    ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=EDGE,
            lw=1.2, zorder=1)
for n, xy in pos.items():
    node(ax, xy, n, G5)
offsets = {'A': (0.0, 0.26, 'center', 'bottom'),
           'B': (-0.24, 0.0, 'right', 'center'),
           'C': (0.24, 0.0, 'left', 'center'),
           'D': (0.0, -0.26, 'center', 'top')}
for n, (dx, dy, ha, va) in offsets.items():
    ax.text(pos[n][0] + dx, pos[n][1] + dy, f"degree = {deg[n]}", ha=ha,
            va=va, fontsize=8, color=INK_SOFT, fontfamily=FONT)
ax.set_xlim(-2.4, 2.4)
ax.set_ylim(-1.45, 1.45)
save(fig, "fig6_1_simple_graph.png")


# =============================================================================
# Figure 6.2 - row versus symmetric normalization of the same edge
# =============================================================================
print("Figure 6.2 ...")

d_i, d_j = 2, 6
fig, axes = figure(aspect=0.48, ncols=2)


def panel(ax, title, weight_text, lw, note):
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0.2, 8.2)
    ax.set_title(title, fontweight='bold', color=INK, pad=4, fontfamily=FONT)
    pi, pj = (2.3, 4.6), (7.5, 4.6)
    for k in range(d_i - 1):
        a = np.pi * 0.62
        p = (pi[0] + 1.5 * np.cos(a), pi[1] + 1.5 * np.sin(a))
        ax.plot([pi[0], p[0]], [pi[1], p[1]], color=EDGE_BG, lw=1.0, zorder=1)
        ax.scatter([p[0]], [p[1]], s=60, facecolor=G6, edgecolors=G4,
                   linewidths=0.6, zorder=2)
    for k in range(d_j - 1):
        a = np.pi * (-0.42 + 0.21 * k)
        p = (pj[0] + 1.6 * np.cos(a), pj[1] + 1.6 * np.sin(a))
        ax.plot([pj[0], p[0]], [pj[1], p[1]], color=EDGE_BG, lw=1.0, zorder=1)
        ax.scatter([p[0]], [p[1]], s=60, facecolor=G6, edgecolors=G4,
                   linewidths=0.6, zorder=2)
    ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color=INK, lw=lw, zorder=1,
            solid_capstyle='butt')
    ax.text(4.9, 6.4, weight_text, ha='center', va='center', fontsize=9.5,
            color=INK, fontfamily=FONT,
            bbox=dict(facecolor='white', edgecolor=G3,
                      boxstyle='round,pad=0.3', linewidth=0.7))
    node(ax, pi, 'i', G4, size=300, ring=1.4)
    node(ax, pj, 'j', G5, size=300)
    ax.text(pi[0], pi[1] - 0.75, f"receiver\n$d_i$ = {d_i}", ha='center',
            va='top', fontsize=8, color=INK, fontfamily=FONT)
    ax.text(pj[0], pj[1] - 1.95, f"sender\n$d_j$ = {d_j}", ha='center',
            va='top', fontsize=8, color=INK, fontfamily=FONT)
    ax.text(4.9, 0.6, note, ha='center', fontsize=8, color=INK_SOFT,
            fontstyle='italic', fontfamily=FONT)


panel(axes[0], "Row normalization",
      r"$\dfrac{1}{d_i} = \dfrac{1}{2} = 0.50$", 3.2,
      "scaled by the receiver's degree only")
panel(axes[1], "Symmetric normalization",
      r"$\dfrac{1}{\sqrt{d_i\,d_j}} = \dfrac{1}{\sqrt{12}} \approx 0.29$", 1.8,
      "scaled by the degrees of both nodes")
save(fig, "fig6_2_row_vs_symmetric_norm.png")

print("Figures 6.1-6.2 done.")