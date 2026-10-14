"""
Chapter 1 – Figures 1.2 to 1.5, redrawn from the manuscript.

The versions in the manuscript were supplied at 144 dpi and in colour. They
are redrawn here grey, at the width they are printed, from the content their
captions describe:

    fig1_2_image_as_graph.png   an image and the same image as a graph
    fig1_3_tabular_vs_graph.png a family as a table and as a graph
    fig1_4_gnn_tasks.png        node classification, link prediction, graph
                                classification
    fig1_5_computation_graph.png an input graph and the computation graph a
                                GNN unrolls from it

The chapter has no run.py: nothing here is measured, every figure is drawn.

    python figures/generate_figures.py
"""

import os
import sys

import numpy as np
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, FILLS, G1, G2, G3, G4, G5, G6)

plt = apply()
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)


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


def node(ax, x, y, r, label, fill=G5, fs=7.5, aspect=1.0):
    """A round node. `aspect` corrects the radius so it is not an ellipse."""
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r / aspect, facecolor=fill,
                         edgecolor=INK, linewidth=0.8, zorder=3))
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=fs,
                color=INK, fontweight='bold', fontfamily=FONT, zorder=4)


def edge(ax, p, q, lw=0.9, color=G2):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, zorder=1)


def arrow(ax, x1, y1, x2, y2, color=G2, lw=1.2, scale=9):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=scale), zorder=2)


def title(ax, text, fs=10.5):
    ax.set_title(text, fontsize=fs, fontweight='bold', color=INK,
                 fontfamily=FONT)


# ── Figure 1.2 — an image, and the same image as a graph ─────────────────────
print("Figure 1.2 …")

# The classic 11x8 invader of the manuscript's figure.
SPRITE = [
    "..X.....X..",
    "...X...X...",
    "..XXXXXXX..",
    ".XX.XXX.XX.",
    "XXXXXXXXXXX",
    "X.XXXXXXX.X",
    "X.X.....X.X",
    "...XX.XX...",
]
BITS = np.array([[c == "X" for c in row] for row in SPRITE])
ROWS, COLS = BITS.shape

fig, (ax_img, ax_graph) = canvas(0.46, ncols=2)

# Left: the picture, as pixels.
for r in range(ROWS):
    for c in range(COLS):
        if BITS[r, c]:
            ax_img.add_patch(Rectangle((c / COLS, 1 - (r + 1) / ROWS),
                                       1 / COLS, 1 / ROWS,
                                       facecolor=G2, edgecolor='none'))
ax_img.set_xlim(-0.08, 1.08); ax_img.set_ylim(-0.12, 1.12)
ax_img.set_aspect('equal')
ax_img.text(0.5, -0.09, "An image: a grid of pixels", ha='center', va='top',
            fontsize=8.5, color=INK_SOFT, fontfamily=FONT, fontstyle='italic')

# Right: the same grid, as nodes. One margin of empty nodes shows that the
# background pixels are nodes too, not absences.
PAD = 1
for r in range(-PAD, ROWS + PAD):
    for c in range(-PAD, COLS + PAD):
        on = 0 <= r < ROWS and 0 <= c < COLS and BITS[r, c]
        ax_graph.add_patch(Circle((c + 0.5, ROWS - r - 0.5), 0.40,
                                  facecolor=G2 if on else BG,
                                  edgecolor=INK if on else G4,
                                  linewidth=0.5, zorder=3))
ax_graph.set_xlim(-PAD - 0.4, COLS + PAD + 0.4)
ax_graph.set_ylim(-PAD - 0.4, ROWS + PAD + 0.4)
ax_graph.set_aspect('equal')
ax_graph.text(0.5, -0.02, "The same image: one node per pixel",
              transform=ax_graph.transAxes, ha='center', va='top',
              fontsize=8.5, color=INK_SOFT, fontfamily=FONT, fontstyle='italic')

save(fig, "fig1_2_image_as_graph.png")


# ── Figure 1.3 — the same family as a table and as a graph ───────────────────
print("Figure 1.3 …")

PEOPLE = [("1", "Mary", 76, "Female"), ("2", "John", 75, "Male"),
          ("3", "Kate", 46, "Female"), ("4", "Robert", 47, "Male"),
          ("5", "Louise", 18, "Female")]
LINKS = [("1", "3"), ("2", "3"), ("3", "5"), ("4", "5")]

fig, (ax_tab, ax_graph) = canvas(0.52, ncols=2)

rows = [[i, n, str(a), g] for i, n, a, g in PEOPLE]
table = ax_tab.table(cellText=rows, colLabels=["ID", "Name", "Age", "Gender"],
                     cellLoc='center', colWidths=[0.16, 0.30, 0.20, 0.34],
                     bbox=[0.0, 0.08, 1.0, 0.76])
table.auto_set_font_size(False)
table.set_fontsize(8)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor(G4)
    cell.set_linewidth(0.6)
    cell.get_text().set_fontfamily(FONT)
    cell.get_text().set_color(INK)
    if r == 0:
        cell.set_facecolor(G5)
        cell.get_text().set_fontweight('bold')
    else:
        cell.set_facecolor(BG)
ax_tab.set_title("A tabular dataset", fontsize=9.5, fontweight='bold',
                 color=INK, fontfamily=FONT)

POS = {"1": (0.17, 0.80), "2": (0.83, 0.80), "3": (0.50, 0.54),
       "4": (0.83, 0.26), "5": (0.33, 0.24)}
# Each of the two panels is half the figure wide, so its own height-to-width
# ratio is the figure's aspect times the number of columns, not the aspect.
ASP = 0.52 * 2
# Where each caption sits relative to its node, so none lands on an edge.
LABEL_AT = {"1": (0.0, -1), "2": (0.0, -1), "3": (0.0, 1),
            "4": (0.0, -1), "5": (0.0, -1)}
for a, b in LINKS:
    edge(ax_graph, POS[a], POS[b])
for ident, name, age, gender in PEOPLE:
    x, y = POS[ident]
    node(ax_graph, x, y, 0.052, ident, fill=G5, aspect=ASP)
    dx, dy = LABEL_AT[ident]
    ax_graph.text(x + dx, y + dy * 0.10, f"{name}, {age}\n{gender}",
                  ha='center', va='bottom' if dy > 0 else 'top',
                  fontsize=7.5, color=INK_SOFT, fontfamily=FONT,
                  linespacing=1.3)
ax_graph.set_title("The same data as a graph", fontsize=9.5, fontweight='bold',
                   color=INK, fontfamily=FONT)

save(fig, "fig1_3_tabular_vs_graph.png")


# ── Figure 1.4 — what a GNN is asked to predict ─────────────────────────
print("Figure 1.4 \u2026")

# One small graph, reused on both sides of every row so the reader compares
# the labelling and not the shape.
SMALL_POS = [(0.30, 0.86), (0.78, 0.92), (0.52, 0.58),
             (0.12, 0.42), (0.86, 0.46), (0.40, 0.16), (0.74, 0.12)]
SMALL_EDGES = [(0, 2), (1, 2), (2, 3), (2, 4), (3, 5), (4, 6), (5, 6)]


def mini_graph(ax, x0, y0, w, h, fills, edges=SMALL_EDGES, dashed=(),
               aspect=1.0, r=0.030):
    """Draw the shared small graph inside the box (x0, y0, w, h)."""
    pts = [(x0 + px * w, y0 + py * h) for px, py in SMALL_POS]
    for i, (a, b) in enumerate(edges):
        style = dict(color=G2, lw=0.9)
        if i in dashed:
            style = dict(color=G4, lw=0.9, linestyle=(0, (2.2, 1.8)))
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                zorder=1, **style)
    for i, (px, py) in enumerate(pts):
        node(ax, px, py, r, "", fill=fills[i % len(fills)], aspect=aspect)
    return pts


fig, axes = canvas(0.92, nrows=3)
# Stacked rows: each axes is the full width but a third of the height, so
# its own height-to-width ratio is the figure's aspect divided by the rows.
# (For side-by-side columns it is multiplied by them instead.)
ASP3 = 0.92 / 3

ROWS = [
    ("a. Node classification",
     "every node unlabelled", "each node given a class"),
    ("b. Link prediction",
     "an edge that is missing", "the edge the model proposes"),
    ("c. Graph classification",
     "a graph with no label", "the graph given a class"),
]
for ax, (heading, left_note, right_note) in zip(axes, ROWS):
    ax.text(0.0, 0.97, heading, ha='left', va='top', fontsize=9,
            fontweight='bold', color=INK, fontfamily=FONT)
    ax.text(0.145, 0.03, left_note, ha='center', va='bottom', fontsize=7.5,
            color=INK_SOFT, fontfamily=FONT, fontstyle='italic')
    ax.text(0.855, 0.03, right_note, ha='center', va='bottom', fontsize=7.5,
            color=INK_SOFT, fontfamily=FONT, fontstyle='italic')
    # the GNN in the middle
    ax.add_patch(Rectangle((0.42, 0.42), 0.16, 0.22, facecolor=G6,
                           edgecolor=G3, linewidth=0.8, zorder=3))
    ax.text(0.50, 0.53, "GNN", ha='center', va='center', fontsize=9,
            fontweight='bold', color=INK, fontfamily=FONT, zorder=4)
    arrow(ax, 0.33, 0.53, 0.41, 0.53)
    arrow(ax, 0.59, 0.53, 0.67, 0.53)

# a. node classification: one fill in, three out
mini_graph(axes[0], 0.02, 0.18, 0.28, 0.66, [BG], aspect=ASP3)
mini_graph(axes[0], 0.70, 0.18, 0.28, 0.66,
           [G4, G6, G4, G5, G6, G5, G4], aspect=ASP3)

# b. link prediction: the last edge is missing on the left, proposed on the right
mini_graph(axes[1], 0.02, 0.18, 0.28, 0.66, [BG],
           edges=SMALL_EDGES[:-1], aspect=ASP3)
mini_graph(axes[1], 0.70, 0.18, 0.28, 0.66, [BG],
           dashed={len(SMALL_EDGES) - 1}, aspect=ASP3)

# c. graph classification: four little graphs, unlabelled then labelled
# Two graphs a side, not four: at a third of the height, four would be half
# a node apart.
PAIRS = [(0.015, 0.20), (0.175, 0.20)]
for i, (qx, qy) in enumerate(PAIRS):
    mini_graph(axes[2], qx, qy, 0.115, 0.56, [BG], aspect=ASP3, r=0.017)
    mini_graph(axes[2], qx + 0.685, qy, 0.115, 0.56,
               [[G3, G5][i]], aspect=ASP3, r=0.017)

save(fig, "fig1_4_gnn_tasks.png")


# ── Figure 1.5 — the computation graph a GNN unrolls ────────────────────
print("Figure 1.5 \u2026")

fig, (ax_in, ax_comp) = canvas(0.46, ncols=2)
ASP2 = 0.46 * 2

for a, b in LINKS:
    edge(ax_in, POS[a], POS[b])
for ident, *_ in PEOPLE:
    x, y = POS[ident]
    node(ax_in, x, y, 0.052, ident, fill=G5, aspect=ASP2)
ax_in.set_title("The input graph", fontsize=9.5, fontweight='bold',
                color=INK, fontfamily=FONT)

# The computation graph for node 5: its neighbours, and theirs. It is read
# right to left, the direction the manuscript drew it: inputs on the right,
# the target node on the left.
def layer(ax, x, y, h=0.26, w=0.055):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=G6,
                           edgecolor=G3, linewidth=0.8, zorder=3))
    ax.text(x, y, "GNN layer", ha='center', va='center', fontsize=8,
            color=INK, fontfamily=FONT, rotation=90, zorder=4)

R = 0.048
node(ax_comp, 0.07, 0.50, R, "5", fill=G5, aspect=ASP2)
layer(ax_comp, 0.24, 0.50, h=0.44)
node(ax_comp, 0.42, 0.70, R, "3", fill=G5, aspect=ASP2)
node(ax_comp, 0.42, 0.24, R, "4", fill=G5, aspect=ASP2)
arrow(ax_comp, 0.205, 0.50, 0.125, 0.50)
arrow(ax_comp, 0.372, 0.70, 0.275, 0.58)
arrow(ax_comp, 0.372, 0.24, 0.275, 0.42)

layer(ax_comp, 0.60, 0.70, h=0.40)
for y in (0.88, 0.70, 0.52):
    arrow(ax_comp, 0.735, y, 0.635, 0.70)
for y, lbl in [(0.88, "1"), (0.70, "2"), (0.52, "5")]:
    node(ax_comp, 0.79, y, R, lbl, fill=G6, aspect=ASP2)
arrow(ax_comp, 0.565, 0.70, 0.47, 0.70)

layer(ax_comp, 0.60, 0.24, h=0.32)
node(ax_comp, 0.79, 0.24, R, "5", fill=G6, aspect=ASP2)
arrow(ax_comp, 0.735, 0.24, 0.635, 0.24)
arrow(ax_comp, 0.565, 0.24, 0.47, 0.24)

ax_comp.set_title("The computation graph for node 5", fontsize=9.5,
                  fontweight='bold', color=INK, fontfamily=FONT)

save(fig, "fig1_5_computation_graph.png")
print("Done.")
