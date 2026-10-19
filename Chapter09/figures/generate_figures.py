"""
Chapter 9 - Figures 9.1 and 9.2, drawn at the width they are printed.

Both are conceptual diagrams: no data, nothing measured. Titles and in-figure
captions are omitted on purpose, because the chapter carries that text in the
caption below each image.

    python figures/generate_figures.py
"""

import os
import sys

import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, FILLS, G3, G4, G5, G6)

plt = apply()

# The figures belong beside this script, where the index looks for them. The
# earlier "./output" put them wherever the script happened to be launched.
OUT = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))
    print(f"  saved {name}")


def canvas(aspect, **kwargs):
    fig, ax = figure(aspect=aspect, **kwargs)
    for one in np.atleast_1d(ax).ravel():
        one.set_facecolor(BG); one.axis('off')
        one.set_xlim(0, 1); one.set_ylim(0, 1)
    return fig, ax


def box(ax, x, y, w, h, label, fill=G6, fs=8, weight='bold', edge=G3):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fill,
                           edgecolor=edge, linewidth=0.8, zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=INK,
            fontweight=weight, fontfamily=FONT, linespacing=1.5, zorder=4)


def arr(ax, x1, y1, x2, y2, color=G3, lw=1.2, scale=9):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=scale), zorder=2)


# ── Fig 9.1: from OGB to Neo4j to GDS to PyG ─────────────────────────────────
print("Figure 9.1 …")

# The pipeline runs down the page, not across it: five stages side by side
# leave each an inch, and "PygNodePropPredDataset" alone is wider than that.
STAGES = [
    ("OGB",   "PygNodePropPredDataset"),
    ("CSV",   "papers.csv, citations.csv"),
    ("Neo4j", "LOAD CSV → :Paper, :CITES"),
    ("GDS",   "in-memory projected graph"),
    ("PyG",   "Data(x, y, edge_index)"),
]

fig, ax = canvas(0.78)
TOP, STEP, BH = 0.90, 0.195, 0.115
for i, (name, detail) in enumerate(STAGES):
    y = TOP - i * STEP
    box(ax, 0.17, y, 0.26, BH, name, fill=FILLS[1], fs=9)
    box(ax, 0.63, y, 0.60, BH, detail, fill=G6, fs=8, weight='normal')
    if i:
        arr(ax, 0.17, y + STEP - BH / 2, 0.17, y + BH / 2)

save(fig, "fig09_1_ingestion_pipeline.png")


# ── Fig 9.2: the PyG remote backend ──────────────────────────────────────────
print("Figure 9.2 …")

fig, ax = canvas(0.72)

box(ax, 0.50, 0.08, 0.44, 0.11, "Graph database (e.g. Kùzu)",
    fill=FILLS[2], fs=8.5)
box(ax, 0.24, 0.38, 0.40, 0.14, "FeatureStore\nget_tensor(node_ids)",
    fill=FILLS[1], fs=8)
box(ax, 0.76, 0.38, 0.40, 0.14, "GraphStore\nget_edge_index(nodes)",
    fill=FILLS[1], fs=8)
box(ax, 0.50, 0.68, 0.56, 0.15,
    "NeighborLoader\nsamples seeds, assembles batches", fill=FILLS[0], fs=8)
box(ax, 0.50, 0.92, 0.34, 0.10, "PyG mini-batch", fill=G5, fs=8.5)

arr(ax, 0.40, 0.14, 0.28, 0.30)
arr(ax, 0.60, 0.14, 0.72, 0.30)
ax.text(0.30, 0.22, "features", ha='right', va='center', fontsize=8,
        color=INK_SOFT, fontfamily=FONT, fontstyle='italic')
ax.text(0.70, 0.22, "topology", ha='left', va='center', fontsize=8,
        color=INK_SOFT, fontfamily=FONT, fontstyle='italic')

arr(ax, 0.28, 0.46, 0.38, 0.60)
arr(ax, 0.72, 0.46, 0.62, 0.60)
arr(ax, 0.50, 0.76, 0.50, 0.86)

save(fig, "fig09_2_remote_backend.png")
print("Done.")
