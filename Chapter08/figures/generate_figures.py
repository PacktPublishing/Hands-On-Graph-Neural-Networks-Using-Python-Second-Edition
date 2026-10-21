"""
Chapter 8 - Conceptual diagrams in grayscale, drawn at the width they are
printed.

Figure 8.1: 1-hop and 2-hop neighbors of a target node.
Figure 8.2: computation graph of the target node in a two-layer GNN.
Figure 8.3: computation graph with and without neighbor sampling.

The data-derived figures (8.4-8.6) are produced by run.py.
"""

import math
import os
import sys

import numpy as np
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, FILLS, G3, G5, G6)

plt = apply()
OUT = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))
    print(f"  saved {name}")


def canvas(aspect, **kwargs):
    fig, ax = figure(aspect=aspect, **kwargs)
    for one in np.atleast_1d(ax).ravel():
        one.set_facecolor(BG)
        one.axis('off')
        one.set_xlim(0, 1)
        one.set_ylim(0, 1)
    return fig, ax


def box(ax, x, y, w, h, label, fill=G6, fs=8, edge=G3):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fill,
                           edgecolor=edge, linewidth=0.8, zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=INK,
            fontweight='bold', fontfamily=FONT, linespacing=1.4, zorder=4)


def arr(ax, x1, y1, x2, y2, color=G3, lw=1.0, scale=8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=scale, shrinkA=0, shrinkB=0),
                zorder=2)


def note(ax, x, y, text, fs=8, ha='center'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs, color=INK_SOFT,
            fontfamily=FONT, fontstyle='italic')


# Figure 8.1 - 1-hop and 2-hop neighbors of node 0
print("Figure 8.1 ...")

ASPECT = 0.80
fig, ax = canvas(ASPECT)
ax.set_ylim(0, ASPECT)
ax.set_aspect('equal')

cx, cy = 0.42, 0.40
hop1 = {1: 90, 2: 210, 3: 330}
hop2 = {4: (1, 70), 5: (1, 110), 6: (2, 210), 7: (3, 310), 8: (3, 350)}
other = {9: 4, 10: 5, 11: 6, 12: 7}
R1, R2, R3 = 0.13, 0.25, 0.36
pos = {0: (cx, cy)}
for n, a in hop1.items():
    pos[n] = (cx + R1 * math.cos(math.radians(a)),
              cy + R1 * math.sin(math.radians(a)))
for n, (p, a) in hop2.items():
    pos[n] = (cx + R2 * math.cos(math.radians(a)),
              cy + R2 * math.sin(math.radians(a)))
angles2 = {n: a for n, (p, a) in hop2.items()}
for n, p in other.items():
    a = angles2[p]
    pos[n] = (cx + R3 * math.cos(math.radians(a)),
              cy + R3 * math.sin(math.radians(a)))

edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (3, 7), (3, 8),
         (4, 9), (5, 10), (6, 11), (7, 12)]
fill = {0: FILLS[2]}
fill.update({n: FILLS[1] for n in hop1})
fill.update({n: FILLS[0] for n in hop2})
fill.update({n: BG for n in other})
RAD = 0.032

for u, v in edges:
    (x1, y1), (x2, y2) = pos[u], pos[v]
    ax.plot([x1, x2], [y1, y2], color=G3 if 0 in (u, v) or u in hop1
            else G5, linewidth=1.0, zorder=1)
for n, (x, y) in pos.items():
    ax.add_patch(Circle((x, y), RAD, facecolor=fill[n], edgecolor=G3,
                        linewidth=0.8, zorder=3))
    ax.text(x, y, str(n), ha='center', va='center', fontsize=8, color=INK,
            fontweight='bold', fontfamily=FONT, zorder=4)

legend = [("Target node", FILLS[2]), ("1-hop neighbors", FILLS[1]),
          ("2-hop neighbors", FILLS[0]), ("Other nodes", BG)]
for i, (label, f) in enumerate(legend):
    y = 0.70 - i * 0.065
    ax.add_patch(Circle((0.80, y), 0.018, facecolor=f, edgecolor=G3,
                        linewidth=0.8))
    ax.text(0.83, y, label, ha='left', va='center', fontsize=8, color=INK,
            fontfamily=FONT)
save(fig, "fig8_1_hops.png")


# Figure 8.2 - computation graph of node 0 in a two-layer GNN
print("Figure 8.2 ...")

fig, ax = canvas(0.50)
W, H = 0.11, 0.11
Y2, Y1, Y0 = 0.82, 0.50, 0.18
box(ax, 0.60, Y2, W, H, "0", fill=FILLS[2])
layer1 = {1: 0.37, 2: 0.60, 3: 0.84}
layer0 = {4: (1, 0.30), 5: (1, 0.44), 6: (2, 0.60), 7: (3, 0.76),
          8: (3, 0.92)}
for n, x in layer1.items():
    box(ax, x, Y1, W, H, str(n), fill=FILLS[1])
    arr(ax, x, Y1 + H / 2, 0.60, Y2 - H / 2)
for n, (p, x) in layer0.items():
    box(ax, x, Y0, W, H, str(n), fill=FILLS[0])
    arr(ax, x, Y0 + H / 2, layer1[p], Y1 - H / 2)
note(ax, 0.02, Y2, "Layer 2\noutput", ha='left')
note(ax, 0.02, Y1, "Layer 1\n1-hop neighbors", ha='left')
note(ax, 0.02, Y0, "Layer 0\n2-hop neighbors", ha='left')
save(fig, "fig8_2_computation_graph.png")


# Figure 8.3 - computation graph without and with neighbor sampling
print("Figure 8.3 ...")

fig, axes = canvas(0.95, nrows=2)
panels = [
    ("Without sampling: all neighbors",
     {1: 0.17, 2: 0.39, 3: 0.61, 4: 0.83},
     {5: (1, 0.11), 6: (1, 0.23), 7: (2, 0.33), 8: (2, 0.45),
      9: (3, 0.55), 10: (3, 0.67), 11: (4, 0.77), 12: (4, 0.89)}),
    ("With neighbor sampling: at most 2 neighbors per hop",
     {1: 0.30, 3: 0.70},
     {5: (1, 0.22), 6: (1, 0.38), 9: (3, 0.62), 10: (3, 0.78)}),
]
W, H = 0.085, 0.16
Y2, Y1, Y0 = 0.72, 0.42, 0.12
for ax, (heading, l1, l0) in zip(axes, panels):
    ax.text(0.5, 0.97, heading, ha='center', va='top', fontsize=8.5,
            fontweight='bold', color=INK, fontfamily=FONT)
    box(ax, 0.5, Y2, W, H, "0", fill=FILLS[2])
    for n, x in l1.items():
        box(ax, x, Y1, W, H, str(n), fill=FILLS[1])
        arr(ax, x, Y1 + H / 2, 0.5, Y2 - H / 2)
    for n, (p, x) in l0.items():
        box(ax, x, Y0, W, H, str(n), fill=FILLS[0])
        arr(ax, x, Y0 + H / 2, l1[p], Y1 - H / 2)
save(fig, "fig8_3_neighbor_sampling.png")

print("Done.")