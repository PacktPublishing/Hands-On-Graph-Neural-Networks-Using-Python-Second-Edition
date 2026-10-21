"""
Chapter 10 - All figures in grayscale.
Fig 10.1: local vs global attention (conceptual).
Fig 10.2: anatomy of a GraphGPS block (conceptual).
Fig 10.3: MAE by molecule size, GINE vs GraphGPS, from a run of run.py.
Fig 10.4: ablation table, from a run of ablation.ipynb.
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
G0 = "#111111"
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))
    print(f"  saved {name}")


def rbox(ax, x, y, w, h, label, fill=G6, fc=None, fs=8, r=0.04):
    """Square corners, dark type on a light fill. `fc` is accepted and
    ignored: the colour of the text follows from the fill, not the caller."""
    ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, facecolor=fill,
                           edgecolor=G3, linewidth=0.8, zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=INK,
            fontweight='bold', fontfamily=FONT, linespacing=1.5, zorder=4)


def arr(ax, x1, y1, x2, y2, color=G3, lw=1.1, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=9), zorder=2)


# ── Fig 10.1: local vs global attention ─────────────────────────────────────
print("Figure 10.1 …")
fig, axes = figure(aspect=0.62, ncols=2)
ASP1 = 0.62 * 2   # two panels: each is half the width

# Node positions on a small graph. Same layout in both panels for
# a direct visual comparison.
np.random.seed(0)
positions = {
    0: (0.5, 0.55),
    1: (0.35, 0.7),
    2: (0.65, 0.7),
    3: (0.2, 0.55),
    4: (0.8, 0.55),
    5: (0.35, 0.35),
    6: (0.65, 0.35),
    7: (0.15, 0.2),
    8: (0.85, 0.2),
}
edges = [(0,1), (0,2), (0,5), (0,6),
         (1,3), (2,4), (5,7), (6,8),
         (3,7), (4,8), (1,2), (5,6)]
target = 0

def draw_graph(ax, edges_highlight=None, dashed_edges=None,
               title="", highlight_center=True):
    ax.set_facecolor(BG)
    ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # Edges
    for (a, b) in edges:
        x1, y1 = positions[a]; x2, y2 = positions[b]
        ax.plot([x1, x2], [y1, y2], color=G5, linewidth=0.8, zorder=1)
    # Extra edges (dashed) — represent attention over non-neighbours
    if dashed_edges:
        for (a, b) in dashed_edges:
            x1, y1 = positions[a]; x2, y2 = positions[b]
            ax.plot([x1, x2], [y1, y2], color=G2, linewidth=1.1,
                    linestyle='--', zorder=1.5)
    # Highlighted edges — the ones used by the aggregation
    if edges_highlight:
        for (a, b) in edges_highlight:
            x1, y1 = positions[a]; x2, y2 = positions[b]
            ax.plot([x1, x2], [y1, y2], color=G0, linewidth=2.2, zorder=2)
    # Nodes
    for i, (x, y) in positions.items():
        # A circle given one radius in axes fractions comes out an ellipse,
        # because the panel is not square.
        if i == target and highlight_center:
            r = 0.050
            c = Ellipse((x, y), 2*r, 2*r/ASP1, facecolor=G4, edgecolor=INK,
                        linewidth=1.2, zorder=4)
        else:
            r = 0.040
            c = Ellipse((x, y), 2*r, 2*r/ASP1, facecolor=G6, edgecolor=G3,
                        linewidth=0.8, zorder=4)
        fc = INK
        ax.add_patch(c)
        ax.text(x, y, str(i), ha='center', va='center',
                fontsize=8, color=fc, fontweight='bold',
                fontfamily=FONT, zorder=5)
    ax.set_title(title, fontsize=9, fontweight='bold',
                 color=INK, fontfamily=FONT)


# Left: local aggregation (GAT-style, one layer)
neighbours_of_target = [1, 2, 5, 6]
draw_graph(axes[0],
           edges_highlight=[(target, n) for n in neighbours_of_target],
           title="Message passing\n(one layer): reach 1 hop")

# Right: global attention. All non-neighbours are dashed = attended to.
non_neighbours = [n for n in positions if n != target
                                       and n not in neighbours_of_target]
draw_graph(axes[1],
           edges_highlight=[(target, n) for n in neighbours_of_target],
           dashed_edges=[(target, n) for n in non_neighbours],
           title="Global attention\nreach: the whole graph")

# Bottom caption
fig.text(0.5, 0.015,
         "The same graph twice. Solid lines are the edges node 0 uses;\n"
         "dashed lines are the extra reach global attention adds.",
         ha='center', va='bottom', fontsize=8, color=INK_SOFT,
         fontfamily=FONT, fontstyle='italic', linespacing=1.4)

fig.suptitle("Local message passing vs global attention",
             fontsize=10.5, fontweight='bold', color=INK, fontfamily=FONT)
save(fig, "fig10_1_local_vs_global_attention.png")


# ── Fig 10.2: anatomy of a GraphGPS block ───────────────────────────────────
print("Figure 10.2 …")
fig, ax = figure(aspect=0.70)
ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 11); ax.set_ylim(0, 7.0)

# Input at the bottom
# Two lines: on one, this label is as wide as the box that holds it.
rbox(ax, 5.5, 0.6, 3.4, 0.9,
     "Node features\n+ positional encodings", fill=G6, fs=8)

# Split into two branches
arr(ax, 4.6, 1.05, 3.2, 1.9)
arr(ax, 6.4, 1.05, 7.8, 1.9)

# Two branches. What each one does with the graph is written inside the box:
# set below it, the note landed on the arrow feeding the box.
rbox(ax, 2.55, 2.75, 4.5, 1.5,
     "Message passing branch\nGINEConv (adjacency)\nrespects the graph",
     fill=FILLS[1], fs=8)
rbox(ax, 8.45, 2.75, 4.5, 1.5,
     "Attention branch\nMultiHeadAttention (all nodes)\nignores the graph",
     fill=FILLS[2], fs=8)

# Converge into sum
arr(ax, 2.55, 3.55, 4.6, 4.45)
arr(ax, 8.45, 3.55, 6.4, 4.45)
rbox(ax, 5.5, 4.8, 1.8, 0.65, "sum", fill=FILLS[1], fs=8.5)

# Feedforward
arr(ax, 5.5, 5.15, 5.5, 5.55)
rbox(ax, 5.5, 5.9, 3.8, 0.6, "Feedforward + LayerNorm", fill=FILLS[1], fs=8)

# Output
arr(ax, 5.5, 6.25, 5.5, 6.55)
ax.text(5.5, 6.75, "output node features", ha='center', va='center',
        fontsize=8, color=INK_SOFT, fontfamily=FONT, fontstyle='italic')

ax.set_title("Anatomy of a GraphGPS block",
             fontsize=10.5, fontweight='bold', color=INK, fontfamily=FONT)
save(fig, "fig10_2_graphgps_block.png")


# ── Fig 10.3: MAE by molecule size ──────────────────────────────────────────
print("Figure 10.3 …")
fig, ax = figure(aspect=0.55)
ax.set_facecolor(BG)

bins   = ['≤15', '16–20', '21–25', '26–30', '>30']
# Stratified means printed by run.py. Replace these two arrays if you
# re-run the chapter code; do not round them by hand.
gine = [0.2792, 0.2462, 0.2121, 0.2700, 0.2939]
gps  = [0.2124, 0.2175, 0.1564, 0.2555, 0.2452]

x = np.arange(len(bins))
w = 0.36

b1 = ax.bar(x - w/2, gine, width=w, color=G6, edgecolor=G2,
            linewidth=0.8, label="GINE (Section 2)")
b2 = ax.bar(x + w/2, gps,  width=w, color=G4, edgecolor=G2,
            linewidth=0.8, label="GraphGPS (Section 4)")

for bars in (b1, b2):
    for r in bars:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + 0.008, f"{h:.2f}",
                ha='center', va='bottom', fontsize=8, rotation=90,
                color=INK, fontfamily=FONT)

ax.set_xticks(x)
ax.set_xticklabels(bins, fontfamily=FONT, fontsize=8.5, color=INK)
ax.set_ylabel("Test MAE (lower is better)", fontsize=9, color=INK,
              fontfamily=FONT)
ax.set_xlabel("Number of atoms per molecule", fontsize=9, color=INK,
              fontfamily=FONT)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(G3)
ax.spines['bottom'].set_color(G3)
ax.tick_params(colors=INK, labelsize=8.5)
ax.set_ylim(0, 0.60)
ax.grid(axis='y', color=G6, linestyle='-', linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

ax.legend(loc='upper left', frameon=False, ncol=2,
          prop={'family': FONT, 'size': 8.5})

ax.text(2.0, 0.50, "ZINC-subset test set", ha='center', fontsize=8,
        color=INK_SOFT, fontfamily=FONT, fontstyle='italic')

ax.set_title("Test MAE by molecule size: GINE vs GraphGPS",
             fontsize=10.5, fontweight='bold', color=INK, fontfamily=FONT)
save(fig, "fig10_3_mae_by_size.png")


# ── Fig 10.4: ablation — where does the gain come from? ─────────────────────
print("Figure 10.4 …")

rows = [
    ["GINE baseline (4 layers, 64 hidden)", "40,069",  "0.2551 ± 0.0286"],
    ["Local only, no PE",                   "349,447", "0.1920 ± 0.0077"],
    ["Local only + PE",                     "350,311", "0.1898 ± 0.0023"],
    ["Local only, widened",                 "570,604", "0.1795 ± 0.0011"],
    ["Attention, no PE",                    "574,087", "0.2210 ± 0.0062"],
    ["Attention + PE (full GraphGPS)",      "574,951", "0.1941 ± 0.0021"],
]
cols = ["Variant", "Parameters", "Test MAE"]
cclr = [
    ['white', 'white', G6],
    ['white', 'white', G6],
    ['white', 'white', G6],
    ['white', 'white', G4],
    ['white', 'white', G6],
    ['white', 'white', G6],
]

fig, ax = figure(aspect=0.46)
ax.axis('off')
tbl = ax.table(cellText=rows, colLabels=cols, cellLoc='center',
               colWidths=[0.46, 0.26, 0.28], cellColours=cclr,
               bbox=[0.0, 0.0, 1.0, 0.92])
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(G4); cell.set_linewidth(0.6)
    cell.get_text().set_fontfamily(FONT)
    cell.get_text().set_color(INK)
    if r == 0:
        cell.set_facecolor(G5)
        cell.get_text().set_fontweight('bold')
save(fig, "fig10_4_ablation.png")

print(f"\nAll figures saved to {OUT}")
