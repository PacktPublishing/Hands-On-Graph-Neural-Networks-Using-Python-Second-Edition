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

FONT = 'DejaVu Sans'
BG = 'white'
G0 = '#111111'; G1 = '#333333'; G2 = '#555555'
G3 = '#777777'; G4 = '#999999'; G5 = '#BBBBBB'; G6 = '#DDDDDD'
plt.rcParams['font.family'] = FONT


def save(fig, name, dpi=200):
    fig.savefig(os.path.join(OUT, name), dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f'  saved {name}')


def rbox(ax, x, y, w, h, label, fill=G2, fc='white', fs=9, r=0.04):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                boxstyle=f'round,pad=0.02,rounding_size={r}',
                                facecolor=fill, edgecolor=G4, linewidth=1.1,
                                zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=fc,
            fontweight='bold', linespacing=1.35, multialignment='center',
            zorder=4)


def arr(ax, x1, y1, x2, y2, color=G2, lw=1.4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                mutation_scale=12), zorder=2)


print('Figure 15.1 ...')

fig, ax = plt.subplots(figsize=(13, 4.2))
fig.patch.set_facecolor(BG)
ax.axis('off')
ax.set_xlim(0, 13)
ax.set_ylim(2.2, 6.0)

rbox(ax, 6.5, 5.4, 3.0, 0.8, 'GNN Explanation\nTechniques',
     fill=G0, fs=11)

for x, label, fill in [
    (1.5,  'Gradient-based\ne.g. Integrated\nGradients', G1),
    (4.5,  'Perturbation-\nbased\ne.g. GNNExplainer', G2),
    (8.5,  'Decomposition\ne.g. GNN-LRP', G3),
    (11.5, 'Surrogate\ne.g. GraphLIME', G4),
]:
    rbox(ax, x, 3.5, 2.2, 1.4, label, fill=fill, fs=9)
    arr(ax, 6.5, 5.0, x, 4.2, color=fill)

for x in (1.5, 4.5):
    ax.text(x, 2.55, '\u2713 Chapter 15', ha='center', fontsize=8.5,
            color=G0, fontweight='bold')

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
SHADES = [G5, G4, G3, G2, G1]


def shade(value):
    return SHADES[min(int(value * len(SHADES)), len(SHADES) - 1)]


def draw_nodes(ax, nodes):
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax,
                           node_color=[G0 if n == target else G3
                                       for n in nodes],
                           node_size=400, edgecolors='white', linewidths=1.2)
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in nodes}, ax=ax,
                            font_size=10, font_color='white',
                            font_weight='bold', font_family=FONT)


fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.patch.set_facecolor(BG)

ax = axes[0]
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=G5, width=1.0)
draw_nodes(ax, list(G.nodes()))
ax.set_title('Full graph', fontsize=11, fontweight='bold', color=G0)

ax = axes[1]
edges = list(mask)
nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax,
                       edge_color=[shade(mask[e]) for e in edges],
                       width=[0.5 + 3.5 * mask[e] for e in edges])
draw_nodes(ax, list(G.nodes()))
ax.set_title('Edge mask', fontsize=11, fontweight='bold', color=G0)

ax = axes[2]
nx.draw_networkx_edges(G, pos, edgelist=kept, ax=ax, edge_color=G1, width=3.5)
draw_nodes(ax, kept_nodes)
ax.set_title('Explanation subgraph', fontsize=11, fontweight='bold', color=G0)

xs = [p[0] for p in pos.values()]
ys = [p[1] for p in pos.values()]
for ax in axes:
    ax.set_facecolor(BG)
    ax.axis('off')
    ax.set_xlim(min(xs) - 0.2, max(xs) + 0.2)
    ax.set_ylim(min(ys) - 0.2, max(ys) + 0.2)

fig.tight_layout()
save(fig, 'fig15_2_gnnexplainer_schema.png')

print(f'\nFigures saved to {OUT}')