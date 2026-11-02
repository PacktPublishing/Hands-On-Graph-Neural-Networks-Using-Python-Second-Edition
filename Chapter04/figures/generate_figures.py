"""
Chapter 4 - conceptual figures (Figures 4.1 to 4.3).
Figures 4.4 and 4.5 come from the data and are saved by run.py.

Usage:
    python figures/generate_figures.py
"""

import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, INK, INK_SOFT,
                       G3, G4, G5, G6)

plt = apply()

np.random.seed(0)

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# Roles, mapped onto the three fills that carry dark type. The chapter ranks
# its candidates by the distance d(t,x), so they take an ordered ramp: the
# nearer the previous node, the darker the fill. White type on those fills is
# not an option at 8.5 pt - the press closes up the counters of reversed type.
FOCUS = G4       # the node the walk is standing on
MIDWAY = G5      # d(t,x) = 1
DISTANT = G6     # d(t,x) = 2, and the node the walk came from (d(t,x) = 0)
EDGE = G5        # edges that only carry the shape of the graph
EDGE_HL = INK    # edges the panel is about


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))
    print(f"  saved {name}")


def nodes(ax, pos, fills, size=260, ring=0.7):
    """Draw the markers with ax.scatter, which keeps them round.

    networkx would do the same, but a circle added in data coordinates turns
    into an ellipse here: these panels have their axes switched off, so the two
    data ranges are stretched to the box by different factors.
    """
    for name, (x, y) in pos.items():
        ax.scatter([x], [y], s=size, facecolor=fills[name], edgecolors=INK,
                   linewidths=ring[name] if isinstance(ring, dict) else ring,
                   zorder=3)


def label(ax, pos, texts, fontsize=8.5, weight='bold'):
    for name, (x, y) in pos.items():
        ax.text(x, y, texts[name], ha='center', va='center',
                fontsize=fontsize, color=INK, fontweight=weight,
                fontfamily=FONT, zorder=4)


def arrow(ax, start, end, color=INK_SOFT, lw=0.9, scale=9):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=scale, shrinkA=7, shrinkB=7),
                zorder=2)


def tag(ax, x, y, text, edgecolor=G3):
    """A probability or an alpha, boxed so the edge below does not cross it."""
    ax.text(x, y, text, ha='center', va='center', fontsize=8, color=INK,
            fontweight='bold', fontfamily=FONT, zorder=5,
            bbox=dict(facecolor='white', edgecolor=edgecolor,
                      boxstyle='round,pad=0.18', linewidth=0.7))


# =============================================================================
# Figure 4.1 - BFS versus DFS neighborhood
# =============================================================================
print("Figure 4.1 ...")

edges41 = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'F'),
           ('F', 'G')]
G41 = nx.Graph()
G41.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
G41.add_edges_from(edges41)
pos41 = {'A': (0, 0), 'B': (-0.8, -0.9), 'C': (0.8, -0.9), 'D': (-1.4, -1.8),
         'E': (-0.2, -1.8), 'F': (1.4, -1.8), 'G': (1.8, -2.7)}

# Two panels at column width are 2.75in each. The legend is shared instead of
# repeated in both, which buys back the room the taller panels need.
fig, axes = figure(aspect=0.56, ncols=2)

for ax, title, hilite in zip(
        axes,
        ["BFS neighborhood\nN(A) = {B, C}",
         "DFS neighborhood\nN(A) = {B, D, E}"],
        [['A', 'B', 'C'], ['A', 'B', 'D', 'E']]):
    ax.axis('off')
    ax.set_title(title, fontweight='bold', color=INK, pad=5, fontfamily=FONT)
    inside = [(u, v) for u, v in G41.edges if u in hilite and v in hilite]
    outside = [e for e in G41.edges if e not in inside]
    nx.draw_networkx_edges(G41, pos41, ax=ax, edgelist=outside,
                           edge_color=EDGE, width=0.8)
    nx.draw_networkx_edges(G41, pos41, ax=ax, edgelist=inside,
                           edge_color=EDGE_HL, width=1.6)
    fills = {n: (FOCUS if n == 'A' else
                 MIDWAY if n in hilite else DISTANT) for n in G41.nodes}
    nodes(ax, pos41, fills,
          ring={n: (1.4 if n == 'A' else 0.7) for n in G41.nodes})
    label(ax, pos41, {n: n for n in G41.nodes})
    ax.margins(0.10)

fig.legend(handles=[
    plt.matplotlib.patches.Patch(facecolor=FOCUS, edgecolor=G3,
                                 label='Source node A'),
    plt.matplotlib.patches.Patch(facecolor=MIDWAY, edgecolor=G3,
                                 label='Neighborhood N(A)'),
    plt.matplotlib.patches.Patch(facecolor=DISTANT, edgecolor=G3,
                                 label='Other nodes'),
], loc='outside lower center', ncols=3, fontsize=8, framealpha=1.0,
    edgecolor=G6, handlelength=1.4, columnspacing=1.2)

save(fig, "fig4_1_neighborhood.png")


# =============================================================================
# Figure 4.2 - the transition probabilities out of v
# =============================================================================
print("Figure 4.2 ...")

# As in Figure 2 of the Node2Vec paper: the walk has just moved from t to v.
# Going back to t itself is d(t, x) = 0 (alpha = 1/p); x1 is a neighbor of
# both v and t, d = 1 (alpha = 1); x2 and x3 are not neighbors of t, d = 2
# (alpha = 1/q). The same roles and fills as Figure 4.3.
pos42 = {'t': (0.70, 1.40), 'v': (2.00, 1.40), 'x1': (3.35, 2.45),
         'x2': (3.35, 1.40), 'x3': (3.35, 0.35)}
fills42 = {'t': DISTANT, 'v': FOCUS, 'x1': MIDWAY, 'x2': DISTANT,
           'x3': DISTANT}
rings42 = {'t': 0.7, 'v': 1.6, 'x1': 0.7, 'x2': 0.7, 'x3': 0.7}
texts42 = {'t': 't', 'v': 'v', 'x1': 'x$_1$', 'x2': 'x$_2$', 'x3': 'x$_3$'}

fig, ax = figure(aspect=0.56)
ax.axis('off')
ax.set_xlim(-0.75, 5.45)
ax.set_ylim(0.00, 2.90)

for a, b in [('t', 'v'), ('v', 'x1'), ('v', 'x2'), ('v', 'x3')]:
    ax.plot([pos42[a][0], pos42[b][0]], [pos42[a][1], pos42[b][1]],
            color=EDGE_HL, lw=1.0, zorder=1)

# The t-x1 edge is what puts x1 at distance 1 from t. Bowed upward so that it
# clears both the alpha tag on the v-x1 edge and the walk-direction note.
ax.add_patch(plt.matplotlib.patches.FancyArrowPatch(
    pos42['t'], pos42['x1'], arrowstyle='-', color=EDGE, lw=1.0,
    connectionstyle='arc3,rad=-0.34', shrinkA=9, shrinkB=9, zorder=1))

nodes(ax, pos42, fills42, size=300, ring=rings42)
label(ax, pos42, texts42, fontsize=9)

# The walk arrives at v from t; without the arrow nothing in the picture says
# which of the two is the previous node.
arrow(ax, pos42['t'], pos42['v'], color=INK, lw=1.2, scale=10)
ax.text(1.35, 1.52, "walk direction", ha='center', fontsize=8,
        color=INK_SOFT, fontstyle='italic', fontfamily=FONT)

# Returning to t: a second arrow from v back to t, bowed below the first.
ax.add_patch(plt.matplotlib.patches.FancyArrowPatch(
    pos42['v'], pos42['t'], arrowstyle='-|>', color=INK, lw=1.0,
    mutation_scale=9, connectionstyle='arc3,rad=-0.45', shrinkA=9, shrinkB=9,
    zorder=2))
tag(ax, 1.35, 1.10, r'$\alpha = 1/p$')

ax.text(0.70, 0.90, "previous", ha='center', fontsize=8, color=INK_SOFT,
        fontfamily=FONT)
ax.text(2.00, 0.90, "current", ha='center', fontsize=8, color=INK_SOFT,
        fontfamily=FONT)
ax.text(0.42, 1.40, "d(t, x) = 0\nreturn", ha='right', va='center',
        fontsize=8, color=INK, fontfamily=FONT, linespacing=1.3)

for key, alpha, note in [
        ('x1', r'$\alpha = 1$', 'd(t, x) = 1   neighbor of t'),
        ('x2', r'$\alpha = 1/q$', 'd(t, x) = 2   explore'),
        ('x3', r'$\alpha = 1/q$', 'd(t, x) = 2   explore')]:
    xv, yv = pos42['v']
    xn, yn = pos42[key]
    offset = 0.20 if yn > yv else (-0.20 if yn < yv else -0.20)
    tag(ax, (xv + xn) / 2 + 0.10, (yv + yn) / 2 + offset, alpha)
    ax.text(xn + 0.26, yn, note, ha='left', va='center', fontsize=8,
            color=INK, fontfamily=FONT)

save(fig, "fig4_2_transition_probs.png")


# =============================================================================
# Figure 4.3 - the same probabilities, worked out
# =============================================================================
print("Figure 4.3 ...")

pos43 = {'t': (0.00, 0.00), 'v': (1.20, 0.00), 'A': (2.40, 0.90),
         'B': (2.40, 0.00), 'C': (2.40, -0.90)}
edges43 = [('t', 'v'), ('v', 'A'), ('v', 'B'), ('v', 'C')]

fig, axes = figure(aspect=0.50, ncols=2)

for ax, (p, q, title) in zip(axes, [
        (1.0, 1.0, 'p = 1, q = 1\n(DeepWalk)'),
        (0.5, 2.0, 'p = 0.5, q = 2\n(returns, stays local)')]):
    ax.axis('off')
    ax.set_title(title, fontweight='bold', color=INK, pad=5, fontfamily=FONT)
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(-1.55, 1.35)

    raw = {'t': 1 / p, 'A': 1 / q, 'B': 1.0, 'C': 1 / q}
    total = sum(raw.values())
    probs = {k: v / total for k, v in raw.items()}

    for a, b in edges43:
        ax.plot([pos43[a][0], pos43[b][0]], [pos43[a][1], pos43[b][1]],
                color=EDGE, lw=0.9, zorder=1)

    # t-B is collinear with v and would vanish under the two arrows drawn over
    # it. It is the edge that puts B at distance 1 from t, which is what gives
    # it alpha = 1, so it is bowed clear of them.
    ax.add_patch(plt.matplotlib.patches.FancyArrowPatch(
        pos43['t'], pos43['B'], arrowstyle='-', color=EDGE, lw=0.9,
        connectionstyle='arc3,rad=0.30', shrinkA=8, shrinkB=8, zorder=1))

    for nbr in ['t', 'A', 'B', 'C']:
        arrow(ax, pos43['v'], pos43[nbr], color=INK,
              lw=0.7 + probs[nbr] * 3.0, scale=9)
        xv, yv = pos43['v']
        xn, yn = pos43[nbr]
        offset = 0.26 if yn > yv else (-0.26 if yn < yv else 0.24)
        tag(ax, (xv + xn) / 2 + 0.05, (yv + yn) / 2 + offset,
            f'{probs[nbr]:.3g}')

    fills = {'t': DISTANT, 'v': FOCUS, 'A': DISTANT, 'B': MIDWAY,
             'C': DISTANT}
    nodes(ax, pos43, fills, size=280,
          ring={n: (1.6 if n == 'v' else 0.7) for n in pos43})
    label(ax, pos43, {n: n for n in pos43}, fontsize=9)

    ax.text(0.00, -1.32, 'previous (t)', ha='center', fontsize=8,
            color=INK_SOFT, fontfamily=FONT)
    ax.text(1.20, -1.32, 'current (v)', ha='center', fontsize=8,
            color=INK_SOFT, fontfamily=FONT)

save(fig, "fig4_3_prob_example.png")

print("Figures 4.1-4.3 done.")