"""
Chapter 18 - Conceptual diagrams, in grayscale.

Figure 18.5: collaborative filtering on a user-item bipartite graph.
Figure 18.6: LightGCN architecture (light graph convolution and layer
combination).

The data-derived figures (18.1-18.4 and 18.7) are produced by run.py.
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

SEED = 0
np.random.seed(SEED)

OUT = os.path.dirname(os.path.abspath(__file__))

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, FILLS)

apply()
BG = "white"
G0 = "#111111"; G1 = "#333333"; G2 = "#555555"
G3 = "#777777"; G4 = "#999999"; G5 = "#BBBBBB"; G6 = "#DDDDDD"

plt.rcParams['font.family'] = FONT
plt.rcParams['mathtext.fontset'] = 'dejavusans'


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))
    print(f"  saved {name}")


from matplotlib.patches import Rectangle


def rbox(ax, x, y, w, h, label, fill=G6, fc=None, fs=8, r=0.04, ls=None):
    """Square corners, dark type on a light fill; `fc` and `ls` are ignored."""
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fill,
                           edgecolor=G3, linewidth=0.8, zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=INK,
            fontweight='bold', linespacing=1.5, zorder=4)


def arr(ax, x1, y1, x2, y2, color=G3, lw=1.1):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=9), zorder=2)


# Figure 18.5 - Collaborative filtering on a bipartite graph
print("Figure 18.5 ...")

fig, ax = plt.subplots(figsize=(5.50, 3.02))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0.4, 5.5)

R = 0.42
users = {"1": (1.5, 4.2), "2": (1.5, 2.7), "3": (1.5, 1.2)}
items = {"A": (8.5, 4.8), "B": (8.5, 3.6), "C": (8.5, 2.4), "D": (8.5, 1.2)}

for name, (x, y) in users.items():
    ax.add_patch(plt.Circle((x, y), R, color=G1, zorder=3))
    ax.text(x, y, name, ha='center', va='center', fontsize=10,
            color='white', fontweight='bold', zorder=4)
    ax.text(x - 0.7, y, f"User {name}", ha='right', va='center',
            fontsize=10, color=G0)

for name, (x, y) in items.items():
    ax.add_patch(plt.Circle((x, y), R, color=INK_SOFT, zorder=3))
    ax.text(x, y, name, ha='center', va='center', fontsize=10,
            color='white', fontweight='bold', zorder=4)
    ax.text(x + 0.7, y, f"Item {name}", ha='left', va='center',
            fontsize=10, color=G0)

known = [("1", "A"), ("1", "B"), ("2", "A"), ("2", "C"),
         ("3", "B"), ("3", "D")]
for u, i in known:
    (x1, y1), (x2, y2) = users[u], items[i]
    ax.plot([x1 + R, x2 - R], [y1, y2], color=G0, linewidth=2.0, zorder=2)

(x1, y1), (x2, y2) = users["2"], items["B"]
ax.plot([x1 + R, x2 - R], [y1, y2], color=G2, linewidth=2.2,
        linestyle='--', zorder=2)

handles = [
    Line2D([0], [0], color=G0, linewidth=2.0, label='Known interaction'),
    Line2D([0], [0], color=G2, linewidth=2.2, linestyle='--',
           label='Recommendation'),
]
ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.08),
          ncol=2, fontsize=9, framealpha=0.95, edgecolor=G5)
ax.set_title("Bipartite graph for collaborative filtering\n"
             "User 2 shares Item A with User 1, who also liked Item B",
             fontsize=11, fontweight='bold', color=G0, pad=10)
save(fig, "fig18_5_bipartite_cf.png")


# Figure 18.6 - LightGCN architecture
print("Figure 18.6 ...")

# The original ran 14 inches across: four conv layers, the combination and
# the score all on one row. At 5.5 the row is folded, with the convolution
# stack on top and the layer combination and score under it.
fig, ax = figure(aspect=0.80)
ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

ax.text(0.5, 0.985, "Light graph convolution (no feature transform,\n"
        "no non-linearity)", ha='center', va='top', fontsize=8,
        color=INK_SOFT, fontstyle='italic', fontfamily=FONT, linespacing=1.4)

rbox(ax, 0.12, 0.78, 0.20, 0.10, "User\n$e_u^{(0)}$", fill=FILLS[0], fs=8)
rbox(ax, 0.12, 0.60, 0.20, 0.10, "Item\n$e_i^{(0)}$", fill=FILLS[0], fs=8)

LAYER_X = [0.36, 0.53, 0.70, 0.87]
for k, x in enumerate(LAYER_X, start=1):
    rbox(ax, x, 0.69, 0.145, 0.12, f"LGConv\nlayer {k}", fill=FILLS[1], fs=8)
    if k > 1:
        arr(ax, LAYER_X[k-2] + 0.073, 0.69, x - 0.073, 0.69)
arr(ax, 0.22, 0.78, 0.287, 0.71)
arr(ax, 0.22, 0.60, 0.287, 0.67)

# every layer's output feeds the combination, through a bus under the stack
ax.plot([0.12, 0.87], [0.52, 0.52], color=G3, lw=0.9, zorder=1)
for x in [0.12] + LAYER_X:
    ax.plot([x, x], [0.555 if x in LAYER_X else 0.555, 0.52], color=G3,
            lw=0.9, zorder=1)
ax.text(0.5, 0.485, "$e^{(0)},\\ e^{(1)},\\ e^{(2)},\\ e^{(3)},\\ e^{(4)}$",
        ha='center', va='top', fontsize=8.5, color=INK, fontfamily=FONT)

ax.text(0.5, 0.39, "Weighted sum, equally weighted "
        "($\\alpha_k = 1/(K+1)$)", ha='center', va='top', fontsize=8,
        color=INK_SOFT, fontstyle='italic', fontfamily=FONT)
arr(ax, 0.5, 0.355, 0.5, 0.305)
rbox(ax, 0.5, 0.245, 0.30, 0.10, "Layer combination\n(mean)", fill=FILLS[2],
     fs=8)
arr(ax, 0.42, 0.195, 0.30, 0.135)
arr(ax, 0.58, 0.195, 0.70, 0.135)
rbox(ax, 0.26, 0.085, 0.16, 0.085, "$e_u^{*}$", fill=FILLS[1], fs=9)
rbox(ax, 0.74, 0.085, 0.16, 0.085, "$e_i^{*}$", fill=FILLS[1], fs=9)
ax.text(0.5, 0.085, "$\\hat{y}_{ui} = e_u^{*} \\cdot e_i^{*}$", ha='center',
        va='center', fontsize=9, color=INK, fontfamily=FONT)

ax.set_title("LightGCN: light graph convolution and layer combination",
             fontsize=10.5, fontweight='bold', color=INK, fontfamily=FONT)
save(fig, "fig18_6_lightgcn_arch.png")

print(f"\nDone: 2 diagrams saved to {OUT}")