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
import sys

import numpy as np
import matplotlib.patches as mpatches
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, FILLS, G1, G2, G3, G4, G5, G6)

plt = apply()
G0 = "#111111"
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

OUT = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))
    print(f"  saved {name}")


# =============================================================================
# PART 1: Figure 11.1, two isomorphic graphs
# =============================================================================

print("Figure 11.1 …")

fig, axes = figure(aspect=0.56, ncols=2)

edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
G_a = nx.Graph(); G_a.add_edges_from(edges)
G_b = nx.Graph(); G_b.add_edges_from(edges)

pos_a = nx.circular_layout(G_a)
pos_b = {i: pos_a[(i + 2) % 5] for i in range(5)}

for ax, G, pos, title in zip(
    axes, [G_a, G_b], [pos_a, pos_b],
    ["Graph A", "Graph B\n(same structure, relabeled nodes)"]
):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_title(title, fontsize=9, fontweight='bold', color=INK,
                 fontfamily=FONT)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=G3, width=1.2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=G5,
                           node_size=280, edgecolors=G3, linewidths=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                            font_color=INK, font_weight='bold',
                            font_family=FONT)

fig.supxlabel("Two isomorphic graphs: same structure, different node labels",
              fontsize=8, color=INK_SOFT, fontstyle='italic', fontfamily=FONT)
fig.text(0.50, 0.55, "\u2245", ha='center', va='center',
         fontsize=18, color=INK, fontfamily=FONT)
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
# Light fills with dark numbers: white digits on the darker two closed up
# at this size.
SHADES = ['#F2F2F2', G6, G5, G4]
LABELCOL = [INK, INK, INK, INK]
step_titles = [
    "Step 0: every node starts with the same colour",
    "Step 1: aggregate and hash, nodes split by degree",
    "Step 2: aggregate and hash, the colouring is stable",
]


def partition_label(colors):
    groups = {}
    for n, c in colors.items():
        groups.setdefault(c, []).append(n)
    return "   ".join(f"c{c}: {{{', '.join(str(n) for n in sorted(v))}}}"
                      for c, v in sorted(groups.items()))


# The three steps are stacked, not set side by side: across 5.5 inches each
# panel would be 1.8 wide, and the step headings alone are wider than that.
fig, axes = figure(aspect=1.12, nrows=3)

for ax, colors, title in zip(axes, step_colors, step_titles):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_title(title, fontsize=9, fontweight='bold', color=INK,
                 fontfamily=FONT)
    nx.draw_networkx_edges(G_wl, pos_wl, ax=ax, edge_color=G3, width=1.0)
    nc = [SHADES[colors[n]] for n in G_wl.nodes()]
    nx.draw_networkx_nodes(G_wl, pos_wl, ax=ax, node_color=nc,
                           node_size=300, edgecolors=G3, linewidths=0.8)
    for n, (x, y) in pos_wl.items():
        ax.text(x, y, str(n), ha='center', va='center', fontsize=8,
                fontweight='bold', color=LABELCOL[colors[n]],
                fontfamily=FONT, zorder=5)
        ax.text(x + 0.12, y + 0.16, f"c{colors[n]}", fontsize=8,
                fontweight='bold', color=INK, fontfamily=FONT,
                ha='left', va='bottom',
                bbox=dict(boxstyle="round,pad=0.12", fc=BG, ec=G5, lw=0.5))
    ax.set_xlim(-0.5, 5.6); ax.set_ylim(-0.95, 2.75)   # room for the legend
    # The partition goes to the right of the graph, where the layout leaves
    # room; under it, it ran into the next panel's heading.
    ax.text(0.98, 0.5, partition_label(colors).replace("   ", "\n"),
            transform=ax.transAxes, ha='right', va='center', fontsize=8,
            color=INK_SOFT, fontfamily=FONT, linespacing=1.6)

handles = [mpatches.Patch(facecolor=SHADES[i], edgecolor=G3, label=f"c{i}")
           for i in range(4)]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
           fontsize=8, handlelength=1.2, handletextpad=0.4)
save(fig, "fig11_3_wl_test.png")


# =============================================================================
# PART 3: Figure 11.2, the 1-WL failure case (C6 versus two triangles)
# =============================================================================

print("Figure 11.2 …")

fig, axes = figure(aspect=0.50, ncols=2)

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
    ax.set_title(title, fontsize=9, fontweight='bold', color=INK,
                 fontfamily=FONT)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=G3, width=1.2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=G5,
                           node_size=280, edgecolors=G3, linewidths=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                            font_color=INK, font_weight='bold',
                            font_family=FONT)

fig.supxlabel("Both graphs are 2-regular, so 1-WL gives them identical "
              "labels\nand cannot tell them apart", fontsize=8,
              color=INK_SOFT, fontstyle='italic', fontfamily=FONT,
              linespacing=1.4)
fig.text(0.50, 0.58, "\u2247", ha='center', va='center',
         fontsize=18, color=INK, fontfamily=FONT)
save(fig, "fig11_2_wl_failure.png")


# =============================================================================
# PART 4: Figure 11.4, injective versus non-injective functions
# =============================================================================

print("Figure 11.4 …")

fig, axes = figure(aspect=0.54, ncols=2)


def draw_mapping(ax, title, inputs, outputs, mapping, injective, top):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_xlim(0, 4); ax.set_ylim(-0.7, top + 0.9)
    ax.set_aspect('equal')          # or the circles come out as ovals
    ax.set_title(title, fontsize=9, fontweight='bold', color=INK,
                 fontfamily=FONT)

    for i, lbl in enumerate(inputs):
        y = len(inputs) - i
        ax.add_patch(plt.Circle((0.8, y), 0.26, facecolor=G5, edgecolor=G3,
                                linewidth=0.8, zorder=3))
        ax.text(0.8, y, lbl, ha='center', va='center', fontsize=8,
                color=INK, fontweight='bold', fontfamily=FONT, zorder=4)

    for i, lbl in enumerate(outputs):
        y = len(outputs) - i
        ax.add_patch(plt.Circle((3.2, y), 0.26, facecolor=G6, edgecolor=G3,
                                linewidth=0.8, zorder=3))
        ax.text(3.2, y, lbl, ha='center', va='center', fontsize=8,
                color=INK, fontweight='bold', fontfamily=FONT, zorder=4)

    for src_i, dst_i in mapping:
        src_y = len(inputs) - src_i
        dst_y = len(outputs) - dst_i
        ax.annotate("", xy=(2.92, dst_y), xytext=(1.08, src_y),
                    arrowprops=dict(arrowstyle="-|>", color=G2,
                                    lw=1.0, mutation_scale=9))

    ax.text(0.8, 0.3, "Input", ha='center', fontsize=8,
            color=INK_SOFT, fontfamily=FONT)
    ax.text(3.2, 0.3, "Output", ha='center', fontsize=8,
            color=INK_SOFT, fontfamily=FONT)
    verdict = "\u2713 Injective" if injective else "\u2717 Not injective"
    ax.text(2.0, -0.25, verdict, ha='center', fontsize=8.5, color=INK,
            fontweight='bold', fontfamily=FONT)


draw_mapping(axes[0], "Injective\ndistinct inputs,\ndistinct outputs",
             ["A", "B", "C"], ["1", "2", "3", "4"],
             [(0, 0), (1, 1), (2, 3)], injective=True, top=4)

draw_mapping(axes[1], "Not injective\ntwo inputs share\none output",
             ["A", "B", "C"], ["1", "2", "3"],
             [(0, 0), (1, 1), (2, 1)], injective=False, top=4)

save(fig, "fig11_4_injective.png")

print(f"\nAll figures saved to {OUT}")