"""
Chapter 11: conceptual diagrams.

Saves, in the same folder as this script:
    fig11_1_isomorphic.png
    fig11_2_wl_failure.png
    fig11_3_wl_test.png
    fig11_4_injective.png

The figures derived from the PROTEINS dataset (11.5, 11.6, 11.7) are
produced by run.py.

Run: python generate_figures.py
"""

import os
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

SEED = 0
random.seed(SEED)
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


# =============================================================================
# PART 1: Figure 11.1, two isomorphic graphs
# =============================================================================

print("Figure 11.1 …")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor(BG)

edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
G_a = nx.Graph(); G_a.add_edges_from(edges)
G_b = nx.Graph(); G_b.add_edges_from(edges)

pos_a = nx.circular_layout(G_a)
pos_b = {i: pos_a[(i + 2) % 5] for i in range(5)}

for ax, G, pos, title in zip(
    axes, [G_a, G_b], [pos_a, pos_b],
    ["Graph A", "Graph B  (same structure, relabeled nodes)"]
):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold',
                 color=G0, fontfamily=FONT, pad=10)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=G2, width=2.0)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=G1,
                           node_size=700, edgecolors='white', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=13,
                            font_color='white', font_weight='bold',
                            font_family=FONT)

fig.suptitle("Two isomorphic graphs — same structure, different node labels",
             fontsize=11, color=G3, fontstyle='italic', fontfamily=FONT, y=0.04)
fig.text(0.50, 0.50, "\u2245", ha='center', va='center',
         fontsize=36, color=G2, fontfamily=FONT)
fig.tight_layout()
save(fig, "fig11_1_isomorphic.png")


# =============================================================================
# PART 2: Figure 11.3, the WL test step by step
# =============================================================================

print("Figure 11.3 …")

G_wl = nx.Graph()
G_wl.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])
pos_wl = {0: (0, 1), 1: (1, 2), 2: (1, 0), 3: (2, 1), 4: (3, 1)}


def wl_steps(G, n_steps):
    col = {n: 0 for n in G.nodes()}
    steps = [dict(col)]
    for _ in range(n_steps):
        sig = {n: (col[n], tuple(sorted(col[m] for m in G[n]))) for n in G.nodes()}
        uniq = {s: i for i, s in enumerate(sorted(set(sig.values())))}
        col = {n: uniq[sig[n]] for n in G.nodes()}
        steps.append(dict(col))
    return steps


step_colors = wl_steps(G_wl, 2)
SHADES = [G6, G4, G2, G0]
LABELCOL = [G0, "white", "white", "white"]
step_titles = [
    "Step 0\n(initialize — every node gets the same color)",
    "Step 1\n(aggregate + hash — nodes split by degree)",
    "Step 2\n(aggregate + hash — stable coloring reached)",
]


def partition_label(colors):
    groups = {}
    for n, c in colors.items():
        groups.setdefault(c, []).append(n)
    return "   ".join(f"c{c}: {{{', '.join(str(n) for n in sorted(v))}}}"
                      for c, v in sorted(groups.items()))


fig, axes = plt.subplots(1, 3, figsize=(13, 5.0))
fig.patch.set_facecolor(BG)

for ax, colors, title in zip(axes, step_colors, step_titles):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_title(title, fontsize=10.5, fontweight='bold',
                 color=G0, fontfamily=FONT, pad=8)
    nx.draw_networkx_edges(G_wl, pos_wl, ax=ax, edge_color=G3, width=1.8)
    nc = [SHADES[colors[n]] for n in G_wl.nodes()]
    nx.draw_networkx_nodes(G_wl, pos_wl, ax=ax, node_color=nc,
                           node_size=700, edgecolors=G0, linewidths=1.2)
    for n, (x, y) in pos_wl.items():
        ax.text(x, y, str(n), ha='center', va='center', fontsize=12,
                fontweight='bold', color=LABELCOL[colors[n]],
                fontfamily=FONT, zorder=5)
    for n, (x, y) in pos_wl.items():
        ax.text(x + 0.16, y + 0.20, f"c{colors[n]}", fontsize=10,
                fontweight='bold', color=G0, fontfamily=FONT,
                ha='left', va='bottom',
                bbox=dict(boxstyle="round,pad=0.15", fc='white',
                          ec=G5, lw=0.6))
    ax.set_xlim(-0.5, 3.8); ax.set_ylim(-0.35, 2.65)
    ax.text(0.5, 0.01, partition_label(colors), transform=ax.transAxes,
            ha='center', va='bottom', fontsize=9.5, color=G1, fontfamily=FONT)

handles = [mpatches.Patch(facecolor=SHADES[i], edgecolor=G0, label=f"color c{i}")
           for i in range(4)]
fig.tight_layout()
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
           fontsize=9.5, bbox_to_anchor=(0.5, -0.04))
save(fig, "fig11_3_wl_test.png")


# =============================================================================
# PART 3: Figure 11.2, the 1-WL failure case (C6 versus two triangles)
# =============================================================================

print("Figure 11.2 …")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor(BG)

C6 = nx.cycle_graph(6)
pos_c6 = nx.circular_layout(C6)

two_C3 = nx.compose(nx.cycle_graph(3), nx.cycle_graph(range(3, 6)))
pos_2c3 = {
    0: (-0.6, 0.3),
    1: (-1.0, -0.3),
    2: (-0.2, -0.3),
    3: (0.6, 0.3),
    4: (0.2, -0.3),
    5: (1.0, -0.3),
}

titles = [
    "Cycle of 6 nodes (C6)\nAll nodes: degree 2",
    "Two disconnected triangles (2 \u00d7 C3)\nAll nodes: degree 2",
]

for ax, G, pos, title in zip(axes, [C6, two_C3], [pos_c6, pos_2c3], titles):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold',
                 color=G0, fontfamily=FONT, pad=10)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=G2, width=2.0)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=G1,
                           node_size=700, edgecolors='white', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=13,
                            font_color='white', font_weight='bold',
                            font_family=FONT)

fig.suptitle("1-WL failure case — both graphs are 2-regular,\n"
             "so the WL test assigns identical labels and cannot distinguish them",
             fontsize=11, color=G3, fontstyle='italic', fontfamily=FONT, y=0.02)
fig.text(0.50, 0.50, "\u2247", ha='center', va='center',
         fontsize=36, color=G2, fontfamily=FONT)
fig.tight_layout()
save(fig, "fig11_2_wl_failure.png")


# =============================================================================
# PART 4: Figure 11.4, injective versus non-injective functions
# =============================================================================

print("Figure 11.4 …")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor(BG)


def draw_mapping(ax, title, inputs, outputs, mapping, injective, top):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_xlim(0, 4); ax.set_ylim(-0.7, top + 0.9)
    ax.set_title(title, fontsize=12, fontweight='bold',
                 color=G0, fontfamily=FONT, pad=10)

    for i, lbl in enumerate(inputs):
        y = len(inputs) - i
        ax.add_patch(plt.Circle((0.8, y), 0.28, color=G1, zorder=3))
        ax.text(0.8, y, lbl, ha='center', va='center', fontsize=11,
                color='white', fontweight='bold', fontfamily=FONT, zorder=4)

    for i, lbl in enumerate(outputs):
        y = len(outputs) - i
        ax.add_patch(plt.Circle((3.2, y), 0.28, color=G3, zorder=3))
        ax.text(3.2, y, lbl, ha='center', va='center', fontsize=11,
                color='white', fontweight='bold', fontfamily=FONT, zorder=4)

    for src_i, dst_i in mapping:
        src_y = len(inputs) - src_i
        dst_y = len(outputs) - dst_i
        ax.annotate("", xy=(2.92, dst_y), xytext=(1.08, src_y),
                    arrowprops=dict(arrowstyle="-|>", color=G0,
                                    lw=1.5, mutation_scale=14))

    ax.text(0.8, 0.3, "Input", ha='center', fontsize=9,
            color=G3, fontfamily=FONT)
    ax.text(3.2, 0.3, "Output", ha='center', fontsize=9,
            color=G3, fontfamily=FONT)
    verdict = "\u2713 Injective" if injective else "\u2717 Non-injective"
    ax.text(2.0, -0.1, verdict, ha='center', fontsize=11,
            color=G0 if injective else G3, fontweight='bold', fontfamily=FONT)


draw_mapping(axes[0], "Injective function\n(distinct inputs \u2192 distinct outputs)",
             ["A", "B", "C"], ["1", "2", "3", "4"],
             [(0, 0), (1, 1), (2, 3)], injective=True, top=4)

draw_mapping(axes[1], "Non-injective function\n(two inputs share the same output)",
             ["A", "B", "C"], ["1", "2", "3"],
             [(0, 0), (1, 1), (2, 1)], injective=False, top=4)

fig.tight_layout()
save(fig, "fig11_4_injective.png")

print(f"\nAll figures saved to {OUT}")