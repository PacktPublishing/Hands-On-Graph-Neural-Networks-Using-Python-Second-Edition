"""
Chapter 15 - Explaining Graph Neural Networks
Conceptual diagrams, grayscale.

fig15_1_taxonomy.png              four families of instance-level explanation
                                  techniques, with the two covered in the chapter
fig15_2_gnnexplainer_schema.png   full graph, soft edge mask, explanation subgraph
                                  on a toy graph (illustrative, not model output)

Figures 15.3 to 15.5 are derived from data and produced by run.py.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx

SEED = 0
OUT = os.path.dirname(os.path.abspath(__file__))

import sys

sys.path.insert(0, os.path.join(OUT, '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, FILLS, G1, G2, G3, G4, G5, G6)
from matplotlib.patches import Rectangle

plt = apply()
G0 = '#111111'


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))
    print(f'  saved {name}')


def rbox(ax, x, y, w, h, label, fill=G6, fc=None, fs=8, r=0.04):
    """Square corners, dark type on a light fill; `fc` is ignored."""
    ax.add_patch(Rectangle((x - w/2, y - h/2), w, h, facecolor=fill,
                           edgecolor=G3, linewidth=0.8, zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=INK,
            fontweight='bold', linespacing=1.5, multialignment='center',
            fontfamily=FONT, zorder=4)


def arr(ax, x1, y1, x2, y2, color=G3, lw=1.1):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                mutation_scale=9), zorder=2)


print('Figure 15.1 ...')

# Four families across needed 13 inches; at 5.5 each would get 1.4, less than
# "Perturbation-based" is wide. Two by two put the root's arrows through the
# upper boxes, so the families are stacked and fed from one spine.
fig, ax = figure(aspect=0.74)
ax.axis('off'); ax.set_facecolor(BG)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

rbox(ax, 0.5, 0.93, 0.50, 0.095, 'GNN explanation techniques',
     fill=FILLS[2], fs=9)

FAMILIES = [
    ('Gradient-based \u2014 e.g. Integrated Gradients', True),
    ('Perturbation-based \u2014 e.g. GNNExplainer', True),
    ('Decomposition \u2014 e.g. GNN-LRP', False),
    ('Surrogate \u2014 e.g. GraphLIME', False),
]
TOP, STEP, BH = 0.72, 0.205, 0.105
SPINE_X = 0.10
ax.plot([SPINE_X, SPINE_X], [0.885, TOP - 3 * STEP], color=G3, lw=1.0,
        zorder=1)
ax.plot([SPINE_X, 0.5], [0.885, 0.885], color=G3, lw=1.0, zorder=1)
for i, (label, in_chapter) in enumerate(FAMILIES):
    y = TOP - i * STEP
    rbox(ax, 0.56, y, 0.80, BH, label, fill=FILLS[1], fs=8)
    arr(ax, SPINE_X, y, 0.155, y)
    if in_chapter:
        ax.text(0.56, y - 0.085, '\u2713 covered in this chapter',
                ha='center', va='center', fontsize=8, color=INK_SOFT,
                fontfamily=FONT, fontstyle='italic')

save(fig, 'fig15_1_taxonomy.png')


print('Figure 15.2 ...')

G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 5), (4, 6),
                  (5, 6)])
pos = nx.spring_layout(G, seed=3)
target = 0
mask = {(0, 1): 0.9, (0, 2): 0.9, (1, 2): 0.85, (0, 3): 0.3,
        (2, 4): 0.2, (3, 5): 0.1, (4, 6): 0.1, (5, 6): 0.1}
kept = [e for e, m in mask.items() if m > 0.5]
kept_nodes = sorted({n for e in kept for n in e})
SHADES = [G6, G5, G4, G3, G2]


def shade(value):
    return SHADES[min(int(value * len(SHADES)), len(SHADES) - 1)]


def draw_nodes(ax, nodes):
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax,
                           node_color=[G4 if n == target else G6
                                       for n in nodes],
                           node_size=170, edgecolors=G3, linewidths=0.7)
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in nodes}, ax=ax,
                            font_size=8, font_color=INK,
                            font_weight='bold', font_family=FONT)


fig, axes = figure(aspect=0.44, ncols=3)

ax = axes[0]
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=G5, width=1.0)
draw_nodes(ax, list(G.nodes()))
ax.set_title('Full graph', fontsize=9, fontweight='bold', color=INK)

ax = axes[1]
edges = list(mask)
nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax,
                       edge_color=[shade(mask[e]) for e in edges],
                       width=[0.5 + 3.5 * mask[e] for e in edges])
draw_nodes(ax, list(G.nodes()))
ax.set_title('Edge mask', fontsize=9, fontweight='bold', color=INK)

ax = axes[2]
nx.draw_networkx_edges(G, pos, edgelist=kept, ax=ax, edge_color=G1, width=3.5)
draw_nodes(ax, kept_nodes)
ax.set_title('Explanation subgraph', fontsize=9, fontweight='bold', color=INK)

xs = [p[0] for p in pos.values()]
ys = [p[1] for p in pos.values()]
for ax in axes:
    ax.set_facecolor(BG)
    ax.axis('off')
    ax.set_xlim(min(xs) - 0.2, max(xs) + 0.2)
    ax.set_ylim(min(ys) - 0.2, max(ys) + 0.2)

save(fig, 'fig15_2_gnnexplainer_schema.png')

print(f'\nFigures saved to {OUT}')