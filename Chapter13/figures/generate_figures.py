"""
Chapter 13 – All figures in grayscale, drawn at the width they are printed.

Figs 13.1–13.6: pure matplotlib diagrams, no external downloads.

Every figure is laid out in fractions of its axes rather than in inches, so
the layout does not move when the printed width changes; bookstyle sets that
width, the point sizes and the greys that may carry text.
"""

import os
import sys

import numpy as np
from matplotlib.patches import Ellipse, Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, FILLS, G1, G2, G3, G4, G5, G6)

plt = apply()
np.random.seed(0)

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    save_fig(fig, f"{OUT}/{name}")
    print(f"  saved {name}")


def canvas(aspect, xlim=(0, 1), ylim=(0, 1), **kwargs):
    """A blank axes at the printed width, with no frame."""
    fig, ax = figure(aspect=aspect, **kwargs)
    for one in np.atleast_1d(ax).ravel():
        one.set_facecolor(BG)
        one.axis('off')
        one.set_xlim(*xlim)
        one.set_ylim(*ylim)
    return fig, ax


def box(ax, x, y, w, h, label, fill=G6, fs=8.5, edge=G3):
    """A square-cornered box with dark type on a light fill."""
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fill,
                           edgecolor=edge, linewidth=0.8, zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=INK,
            fontweight='bold', fontfamily=FONT, linespacing=1.5, zorder=4)


def arr(ax, x1, y1, x2, y2, color=INK_SOFT, lw=1.2, scale=9):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=scale), zorder=2)


def note(ax, x, y, text, fs=8.5, italic=True, **kwargs):
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=INK_SOFT,
            fontfamily=FONT, fontstyle='italic' if italic else 'normal',
            linespacing=1.4, **kwargs)


def title(ax, text, fs=10.5):
    ax.set_title(text, fontsize=fs, fontweight='bold', color=INK,
                 fontfamily=FONT)


# ── Fig 13.1: MPNN framework ──────────────────────────────────────────────────
print("Figure 13.1 …")

fig, ax = canvas(0.36)
BOX_W, BOX_H, ROW_Y = 0.17, 0.34, 0.68
CENTRES = (0.29, 0.50, 0.71)

for x, label, fill in zip(CENTRES,
                          ["Message\nfunction m()",
                           "Aggregate\nfunction ⊕()",
                           "Update\nfunction U()"], FILLS):
    box(ax, x, ROW_Y, BOX_W, BOX_H, label, fill=fill)

arr(ax, CENTRES[0] + BOX_W / 2, ROW_Y, CENTRES[1] - BOX_W / 2, ROW_Y)
arr(ax, CENTRES[1] + BOX_W / 2, ROW_Y, CENTRES[2] - BOX_W / 2, ROW_Y)

note(ax, 0.095, ROW_Y, "Node and\nedge\nfeatures", italic=False)
arr(ax, 0.165, ROW_Y, CENTRES[0] - BOX_W / 2, ROW_Y)
arr(ax, CENTRES[2] + BOX_W / 2, ROW_Y, 0.835, ROW_Y)
note(ax, 0.905, ROW_Y, "Updated\nembedding\nh_v", italic=False)

for x, caption in zip(CENTRES, ["h_v, h_u, e_uv\n→ message\nm_uv",
                                "sum, mean\nor max\nover N(v)",
                                "h_v +\naggregated\n→ new h_v"]):
    note(ax, x, 0.18, caption)

title(ax, "The Message Passing Neural Network (MPNN) framework")
save(fig, "fig13_1_mpnn.png")


# ── Fig 13.2: Heterogeneous graph ─────────────────────────────────────────────
print("Figure 13.2 …")

fig, ax = canvas(0.68)

nodes = {'U1': (0.14, 0.78), 'U2': (0.14, 0.52), 'U3': (0.14, 0.26),
         'G1': (0.50, 0.70), 'G2': (0.50, 0.34),
         'D1': (0.86, 0.70), 'D2': (0.86, 0.34)}
fills = {'U1': FILLS[0], 'U2': FILLS[0], 'U3': FILLS[0],
         'G1': FILLS[1], 'G2': FILLS[1], 'D1': FILLS[2], 'D2': FILLS[2]}
labels = {'U1': 'User 1', 'U2': 'User 2', 'U3': 'User 3',
          'G1': 'Game 1', 'G2': 'Game 2', 'D1': 'Dev 1', 'D2': 'Dev 2'}

edges = [('U1', 'U2', 'follows'), ('U2', 'U3', 'follows'),
         ('U1', 'G1', 'plays'), ('U2', 'G1', 'plays'),
         ('U2', 'G2', 'plays'), ('U3', 'G2', 'plays'),
         ('D1', 'G1', 'develops'), ('D2', 'G2', 'develops')]

# A circle in x-fractions is an ellipse once the axes are wider than tall,
# so the vertical radius is corrected by the aspect ratio.
ASPECT = 0.68
RX, RY = 0.075, 0.075 / ASPECT

for src, dst, rel in edges:
    x1, y1 = nodes[src]; x2, y2 = nodes[dst]
    # Stop the arrow on the node's outline, or its head disappears underneath.
    # t solves ((t*dx)/RX)^2 + ((t*dy)/RY)^2 = 1, the ellipse intersection.
    dx, dy = x2 - x1, y2 - y1
    t = 1.0 / ((dx / RX) ** 2 + (dy / RY) ** 2) ** 0.5
    arr(ax, x1 + dx * t, y1 + dy * t, x2 - dx * t, y2 - dy * t,
        color=G3, lw=1.0, scale=8)
    # A label on a vertical edge would sit on the line, so it steps aside.
    off = 0.055 if abs(dx) < 1e-6 else 0.0
    ax.text((x1 + x2) / 2 + off, (y1 + y2) / 2, rel,
            ha='left' if off else 'center', va='center',
            fontsize=8, color=INK_SOFT, fontfamily=FONT, fontstyle='italic',
            bbox=dict(facecolor=BG, edgecolor='none', pad=1), zorder=5)

for key, (x, y) in nodes.items():
    ax.add_patch(Ellipse((x, y), 2 * RX, 2 * RY, facecolor=fills[key],
                         edgecolor=G3, linewidth=0.8, zorder=3))
    ax.text(x, y, labels[key], ha='center', va='center', fontsize=8,
            color=INK, fontweight='bold', fontfamily=FONT, zorder=4)

for x, label in [(0.14, "Users"), (0.50, "Games"), (0.86, "Developers")]:
    ax.text(x, 0.95, label, ha='center', va='center', fontsize=9,
            fontweight='bold', color=INK, fontfamily=FONT)

note(ax, 0.50, 0.05, "3 node types (users, games, developers)   ·   "
                     "3 edge types (follows, plays, develops)", fs=8)
title(ax, "A heterogeneous graph")
save(fig, "fig13_2_hetero_graph.png")


# ── Fig 13.3: DBLP node type relationships ────────────────────────────────────
print("Figure 13.3 …")

fig, ax = canvas(0.56)

W, H = 0.24, 0.20
places = [(0.14, 0.74, "Author\n(4,057)", FILLS[0]),
          (0.50, 0.74, "Paper\n(14,328)", FILLS[1]),
          (0.86, 0.74, "Conference\n(20)", FILLS[2]),
          (0.50, 0.26, "Term\n(7,723)", G5)]
for x, y, label, fill in places:
    box(ax, x, y, W, H, label, fill=fill)

# Each relation label sits over the midpoint of its own pair of arrows.
pairs = [(0.14, 0.74, 0.50, 0.74, "write /\nwritten by", 0.32, 0.93),
         (0.50, 0.74, 0.86, 0.74, "published in /\ncontains", 0.68, 0.93),
         (0.50, 0.74, 0.50, 0.26, "contains / in", 0.66, 0.50)]
for x1, y1, x2, y2, label, lx, ly in pairs:
    if y1 == y2:
        arr(ax, x1 + W / 2, y1, x2 - W / 2, y2, color=G3)
        arr(ax, x2 - W / 2, y1 - 0.05, x1 + W / 2, y2 - 0.05, color=G3)
    else:
        arr(ax, x1 - 0.03, y1 - H / 2, x2 - 0.03, y2 + H / 2, color=G3)
        arr(ax, x1 + 0.03, y2 + H / 2, x2 + 0.03, y1 - H / 2, color=G3)
    note(ax, lx, ly, label, fs=8)

title(ax, "Node types in DBLP, and how they relate")
note(ax, 0.5, 0.04, "Goal: classify authors into DB / DM / AI / IR", fs=8)
save(fig, "fig13_3_dblp.png")


# ── Fig 13.4: Homogeneous vs Heterogeneous GAT ───────────────────────
print("Figure 13.4 …")

# Two panels stacked, not side by side: at 5.5 inches a pair of columns would
# leave each architecture 2.6 inches, too little for six relation names. The
# homogeneous panel holds one row, so it is given a quarter of the height.
fig, axes = canvas(1.05, nrows=2, gridspec_kw={'height_ratios': [1, 3]})


def draw_gat(ax, heading, columns, box_h, x_half=0.13):
    ax.text(0.5, 0.95, heading, ha='center', va='top', fontsize=8.5,
            fontweight='bold', color=INK, fontfamily=FONT, linespacing=1.4)
    xs = [0.16, 0.50, 0.85]
    for x, specs in zip(xs, columns):
        for y, label, fill in specs:
            box(ax, x, y, 2 * x_half, box_h, label, fill=fill, fs=7.5)
    for i in range(len(columns) - 1):
        # One height for both ends, so the connector is horizontal. Taking the
        # centre of each column separately tilts it whenever the two columns
        # hold a different number of boxes.
        y = float(np.mean([spec[0] for spec in columns[i] + columns[i + 1]]))
        arr(ax, xs[i] + x_half + 0.01, y, xs[i + 1] - x_half - 0.01, y,
            color=G3)


draw_gat(axes[0], "Homogeneous GAT (single relation type)", [
    [(0.42, "Author\nfeatures", FILLS[0])],
    [(0.42, "GAT layer\n(1 shared)", FILLS[1])],
    [(0.42, "Author\nclassification", FILLS[2])],
], box_h=0.42)

draw_gat(axes[1], "Heterogeneous GAT via to_hetero()\n(one layer per relation)", [
    [(0.70, "Author\nfeatures", FILLS[0]),
     (0.52, "Paper\nfeatures", FILLS[0]),
     (0.34, "Term\nfeatures", FILLS[0]),
     (0.16, "Conf.\nfeatures", FILLS[0])],
    [(0.74, "author→paper", FILLS[1]),
     (0.62, "paper→author", FILLS[1]),
     (0.50, "paper→term", FILLS[1]),
     (0.38, "term→paper", FILLS[1]),
     (0.26, "conf.→paper", FILLS[1]),
     (0.14, "paper→conf.", FILLS[1])],
    [(0.44, "Aggregate\n+ classify", FILLS[2])],
], box_h=0.095)
save(fig, "fig13_4_homo_vs_hetero.png")


# ── Fig 13.5: HAN architecture ────────────────────────────────────────────────
print("Figure 13.5 …")

fig, ax = canvas(0.62)

mp = [("Meta-path\nAPA", 0.80), ("Meta-path\nAPCPA", 0.55), ("Meta-path\nAPTPA", 0.30)]
for label, y in mp:
    box(ax, 0.13, y, 0.20, 0.16, label, fill=FILLS[0], fs=7.5)
    box(ax, 0.39, y, 0.20, 0.16, "Node-level\nattention", fill=FILLS[1], fs=7.5)
    arr(ax, 0.23, y, 0.29, y, color=G3)
    arr(ax, 0.49, y, 0.57, 0.55, color=G3)

box(ax, 0.68, 0.55, 0.20, 0.46, "Semantic-\nlevel\nattention\n(meta-path\nweights)",
    fill=G5, fs=7.5)
# The arrow leaves the semantic block for the embedding, not for the gap
# between the two output boxes.
arr(ax, 0.78, 0.55, 0.81, 0.68, color=G3)
box(ax, 0.89, 0.68, 0.16, 0.15, "Final\nembedding Z", fill=FILLS[1], fs=7.5)
box(ax, 0.89, 0.40, 0.16, 0.15, "Classifier\n(Linear)", fill=FILLS[2], fs=7.5)
arr(ax, 0.89, 0.605, 0.89, 0.48, color=G3)

for x, label in [(0.13, "Input"), (0.39, "Node level"),
                 (0.68, "Semantic level"), (0.89, "Output")]:
    note(ax, x, 0.05, label, fs=8)

title(ax, "HAN: node-level and semantic-level attention")
save(fig, "fig13_5_han.png")


# ── Fig 13.6: Comparison table ────────────────────────────────────────────────
print("Figure 13.6 …")

rows = [["GAT (homogeneous)", "Author-Paper-Author\nonly", "71.60%", "baseline"],
        ["Het-GAT (to_hetero)", "all 6 relation types", "80.01%", "+8.41%"],
        ["HAN", "node + semantic\nattention", "81.76%", "+10.16%"]]
cols = ["Model", "Information used", "Test acc.", "vs base"]

fig, ax = canvas(0.36)
# bbox keeps the table clear of the title; loc='center' alone lets the two
# collide, because the table fills the axes.
table = ax.table(cellText=rows, colLabels=cols, cellLoc='center',
                 colWidths=[0.30, 0.34, 0.18, 0.18],
                 bbox=[0.0, 0.0, 1.0, 0.88])
table.auto_set_font_size(False)
table.set_fontsize(8.5)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor(G4)
    cell.set_linewidth(0.6)
    cell.get_text().set_fontfamily(FONT)
    if row == 0:
        cell.set_facecolor(G5)
        cell.get_text().set_color(INK)
        cell.get_text().set_fontweight('bold')
    else:
        cell.set_facecolor(G6 if col == 2 else BG)
        cell.get_text().set_color(INK)

title(ax, "Author classification on DBLP: 100 epochs, single run", fs=9)
save(fig, "fig13_6_comparison.png")

print("Done.")
