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

FONT = "DejaVu Sans"
BG = "white"
G0 = "#111111"; G1 = "#333333"; G2 = "#555555"
G3 = "#777777"; G4 = "#999999"; G5 = "#BBBBBB"; G6 = "#DDDDDD"

plt.rcParams['font.family'] = FONT
plt.rcParams['mathtext.fontset'] = 'dejavusans'


def save(fig, name, dpi=200):
    fig.savefig(os.path.join(OUT, name), dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"  saved {name}")


def rbox(ax, x, y, w, h, label, fill=G2, fc='white', fs=9, r=0.04, ls=0.22):
    rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle=f"round,pad=0.02,rounding_size={r}",
                          facecolor=fill, edgecolor=G4, linewidth=1.1,
                          zorder=3)
    ax.add_patch(rect)
    lines = label.split('\n')
    for li, line in enumerate(lines):
        off = ls * (li - (len(lines) - 1) / 2)
        ax.text(x, y - off, line, ha='center', va='center', fontsize=fs,
                color=fc, fontweight='bold', zorder=4)


def arr(ax, x1, y1, x2, y2, color=G2, lw=1.4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12), zorder=2)


# Figure 18.5 - Collaborative filtering on a bipartite graph
print("Figure 18.5 ...")

fig, ax = plt.subplots(figsize=(10, 5.5))
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
    ax.add_patch(plt.Circle((x, y), R, color=G3, zorder=3))
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
fig.tight_layout()
save(fig, "fig18_5_bipartite_cf.png")


# Figure 18.6 - LightGCN architecture
print("Figure 18.6 ...")

fig, ax = plt.subplots(figsize=(14, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis('off')
ax.set_xlim(0, 15.2)
ax.set_ylim(0.3, 5.6)

Y = 3.15
BUS = 1.1
LW, LH = 1.2, 1.4
layers_x = [3.5, 5.2, 6.9, 8.6]
COMB_X, COMB_W, COMB_H = 10.5, 1.6, 1.5
FIN_X = 12.6

rbox(ax, 1.2, 3.9, 1.8, 0.75, "User emb.\n$e_u^{(0)}$", fill=G1, fs=9)
rbox(ax, 1.2, 2.4, 1.8, 0.75, "Item emb.\n$e_i^{(0)}$", fill=G3, fs=9)

JX = 2.55
ax.plot([2.1, JX], [3.9, Y], color=G1, linewidth=1.4, zorder=2)
ax.plot([2.1, JX], [2.4, Y], color=G3, linewidth=1.4, zorder=2)
ax.plot([JX], [Y], marker='o', color=G2, markersize=5, zorder=3)
arr(ax, JX, Y, layers_x[0] - LW / 2, Y, color=G2)

for k, x in enumerate(layers_x):
    rbox(ax, x, Y, LW, LH, f"LGConv\nlayer {k + 1}", fill=G2, fs=8.5)
    if k > 0:
        arr(ax, layers_x[k - 1] + LW / 2, Y, x - LW / 2, Y, color=G2)

taps = [(JX, Y)] + [(x, Y - LH / 2) for x in layers_x]
for x, y in taps:
    ax.plot([x, x], [y, BUS], color=G4, linewidth=1.2, zorder=1)
ax.plot([JX, COMB_X], [BUS, BUS], color=G4, linewidth=1.2, zorder=1)
arr(ax, COMB_X, BUS, COMB_X, Y - COMB_H / 2, color=G4, lw=1.2)
ax.text((JX + layers_x[-1]) / 2, BUS - 0.35,
        r"$e^{(0)},\ e^{(1)},\ e^{(2)},\ e^{(3)},\ e^{(4)}$",
        ha='center', va='center', fontsize=10, color=G2)

rbox(ax, COMB_X, Y, COMB_W, COMB_H, "Layer\ncombination\n(mean)",
     fill=G0, fs=9)

rbox(ax, FIN_X, 4.1, 1.2, 0.65, "$e_u^{*}$", fill=G1, fs=11)
rbox(ax, FIN_X, 2.2, 1.2, 0.65, "$e_i^{*}$", fill=G3, fs=11)
arr(ax, COMB_X + COMB_W / 2, Y + 0.35, FIN_X - 0.6, 4.1, color=G1)
arr(ax, COMB_X + COMB_W / 2, Y - 0.35, FIN_X - 0.6, 2.2, color=G3)
arr(ax, FIN_X + 0.6, 4.1, 13.6, Y + 0.25, color=G0)
arr(ax, FIN_X + 0.6, 2.2, 13.6, Y - 0.25, color=G0)
ax.text(13.7, Y, r"$\hat{y}_{ui} = e_u^{*} \cdot e_i^{*}$",
        ha='left', va='center', fontsize=12, color=G0)

ax.text(1.2, 5.0, "Input embeddings", ha='center', fontsize=9,
        color=G3, fontstyle='italic')
ax.text((layers_x[0] + layers_x[-1]) / 2, 5.0,
        "Light graph convolution (no feature transformation, "
        "no activation)", ha='center', fontsize=9, color=G3,
        fontstyle='italic')
ax.text(COMB_X, 5.0, r"Weighted sum, $\alpha_k = 1/(K+1)$",
        ha='center', fontsize=9, color=G3, fontstyle='italic')
ax.text(FIN_X + 0.9, 5.0, "Final embeddings and score", ha='center',
        fontsize=9, color=G3, fontstyle='italic')

ax.set_title("LightGCN architecture: light graph convolution and layer "
             "combination", fontsize=12, fontweight='bold', color=G0, pad=10)
fig.tight_layout()
save(fig, "fig18_6_lightgcn_arch.png")

print(f"\nDone: 2 diagrams saved to {OUT}")