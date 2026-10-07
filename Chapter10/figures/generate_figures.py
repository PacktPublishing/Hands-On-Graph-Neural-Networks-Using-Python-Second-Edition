"""
Chapter 10 - All figures in grayscale.
Fig 10.1: local vs global attention (conceptual).
Fig 10.2: anatomy of a GraphGPS block (conceptual).
Fig 10.3: MAE by molecule size, GINE vs GraphGPS, from a run of run.py.
Fig 10.4: ablation table, from a run of ablation.ipynb.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
import os

OUT  = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

FONT = "DejaVu Sans"
BG   = "white"
G0   = "#111111"; G1 = "#333333"; G2 = "#555555"
G3   = "#777777"; G4 = "#999999"; G5 = "#BBBBBB"; G6 = "#DDDDDD"


def save(fig, name, dpi=200):
    fig.savefig(f"{OUT}/{name}", dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"  saved {name}")


def rbox(ax, x, y, w, h, label, fill=G2, fc='white', fs=9, r=0.04):
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle=f"round,pad=0.02,rounding_size={r}",
                          facecolor=fill, edgecolor=G4,
                          linewidth=1.1, zorder=3)
    ax.add_patch(rect)
    for li, line in enumerate(label.split('\n')):
        offset = 0.11*(li - label.count('\n')/2)
        ax.text(x, y-offset, line, ha='center', va='center',
                fontsize=fs, color=fc, fontweight='bold',
                fontfamily=FONT, zorder=4)


def arr(ax, x1, y1, x2, y2, color=G2, lw=1.4, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=12), zorder=2)


# ── Fig 10.1: local vs global attention ─────────────────────────────────────
print("Figure 10.1 …")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor(BG)

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
        ax.plot([x1, x2], [y1, y2], color=G5, linewidth=1.0, zorder=1)
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
        if i == target and highlight_center:
            c = Circle((x, y), 0.045, facecolor=G0, edgecolor=G0,
                       linewidth=1.5, zorder=4)
            fc = 'white'
        else:
            c = Circle((x, y), 0.035, facecolor=G6, edgecolor=G2,
                       linewidth=1.0, zorder=4)
            fc = G0
        ax.add_patch(c)
        ax.text(x, y, str(i), ha='center', va='center',
                fontsize=8.5, color=fc, fontweight='bold',
                fontfamily=FONT, zorder=5)
    ax.set_title(title, fontsize=11, fontweight='bold',
                 color=G0, fontfamily=FONT, pad=8)


# Left: local aggregation (GAT-style, one layer)
neighbours_of_target = [1, 2, 5, 6]
draw_graph(axes[0],
           edges_highlight=[(target, n) for n in neighbours_of_target],
           title="Message passing (one layer): reach = 1 hop")

# Right: global attention. All non-neighbours are dashed = attended to.
non_neighbours = [n for n in positions if n != target
                                       and n not in neighbours_of_target]
draw_graph(axes[1],
           edges_highlight=[(target, n) for n in neighbours_of_target],
           dashed_edges=[(target, n) for n in non_neighbours],
           title="Global attention: reach = whole graph")

# Bottom caption
fig.text(0.5, 0.02,
         "Both panels show the same graph. Solid black lines are the "
         "edges the target node (0) uses. Dashed lines are additional "
         "connections\ncreated by global attention, independent of the "
         "adjacency.",
         ha='center', fontsize=9, color=G3, fontfamily=FONT,
         fontstyle='italic')

fig.suptitle("Local message passing vs global attention",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT,
             y=1.02)
fig.tight_layout(rect=[0, 0.05, 1, 1])
save(fig, "fig10_1_local_vs_global_attention.png")


# ── Fig 10.2: anatomy of a GraphGPS block ───────────────────────────────────
print("Figure 10.2 …")
fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 11); ax.set_ylim(0, 6.5)

# Input at the bottom
rbox(ax, 5.5, 0.5, 3.2, 0.6,
     "Node features + positional encodings", fill=G3, fc='white', fs=10)

# Split into two branches
arr(ax, 4.3, 0.8, 2.7, 1.8, color=G2, lw=1.6)
arr(ax, 6.7, 0.8, 8.3, 1.8, color=G2, lw=1.6)

# Two branches
rbox(ax, 2.5, 2.4, 3.0, 1.1,
     "Message passing branch\nGINEConv (adjacency)",
     fill=G2, fc='white', fs=9)
rbox(ax, 8.5, 2.4, 3.0, 1.1,
     "Attention branch\nMultiHeadAttention\n(all nodes)",
     fill=G0, fc='white', fs=9)

# Converge into sum
arr(ax, 2.5, 3.0, 4.8, 4.0, color=G2, lw=1.5)
arr(ax, 8.5, 3.0, 6.2, 4.0, color=G2, lw=1.5)
rbox(ax, 5.5, 4.4, 1.6, 0.7, "sum", fill=G4, fc='white', fs=10)

# Feedforward
arr(ax, 5.5, 4.75, 5.5, 5.2, color=G2, lw=1.6)
rbox(ax, 5.5, 5.55, 2.8, 0.6, "Feedforward + LayerNorm",
     fill=G1, fc='white', fs=10)

# Output arrow
arr(ax, 5.5, 5.9, 5.5, 6.2, color=G2, lw=1.6)
ax.text(5.5, 6.35, "output node features",
        ha='center', fontsize=9, color=G3, fontfamily=FONT,
        fontstyle='italic')

# Side labels
ax.text(2.5, 1.65, "respects the graph",
        ha='center', fontsize=8.5, color=G3,
        fontfamily=FONT, fontstyle='italic')
ax.text(8.5, 1.65, "ignores the graph",
        ha='center', fontsize=8.5, color=G3,
        fontfamily=FONT, fontstyle='italic')

ax.set_title("Anatomy of a GraphGPS block",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig10_2_graphgps_block.png")


# ── Fig 10.3: MAE by molecule size ──────────────────────────────────────────
print("Figure 10.3 …")
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

bins   = ['≤15', '16–20', '21–25', '26–30', '>30']
# Stratified means printed by run.py. Replace these two arrays if you
# re-run the chapter code; do not round them by hand.
gine = [0.2792, 0.2462, 0.2121, 0.2700, 0.2939]
gps  = [0.2124, 0.2175, 0.1564, 0.2555, 0.2452]

x = np.arange(len(bins))
w = 0.36

b1 = ax.bar(x - w/2, gine, width=w, color=G3, edgecolor=G1,
            linewidth=0.8, label="GINE (Section 2)")
b2 = ax.bar(x + w/2, gps,  width=w, color=G0, edgecolor=G1,
            linewidth=0.8, label="GraphGPS (Section 4)")

for bars in (b1, b2):
    for r in bars:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + 0.008, f"{h:.2f}",
                ha='center', va='bottom', fontsize=8.5,
                color=G1, fontfamily=FONT)

ax.set_xticks(x)
ax.set_xticklabels(bins, fontfamily=FONT, fontsize=10, color=G1)
ax.set_ylabel("Test MAE (lower is better)", fontsize=10, color=G1,
              fontfamily=FONT)
ax.set_xlabel("Number of atoms per molecule", fontsize=10, color=G1,
              fontfamily=FONT)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(G3)
ax.spines['bottom'].set_color(G3)
ax.tick_params(colors=G3)
ax.set_ylim(0, 0.60)
ax.grid(axis='y', color=G6, linestyle='-', linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

ax.legend(loc='upper left', frameon=False,
          prop={'family': FONT, 'size': 10})

ax.text(2.0, 0.56, "Test MAE by molecule size on the ZINC-subset test set",
        ha='center', fontsize=8.5, color=G4,
        fontfamily=FONT, fontstyle='italic')

ax.set_title("Test MAE by molecule size: GINE vs GraphGPS",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT,
             pad=10)
fig.tight_layout()
save(fig, "fig10_3_mae_by_size.png")


# ── Fig 10.4: ablation — where does the gain come from? ─────────────────────
print("Figure 10.4 …")

rows = [
    ["Message passing only",              "125,383", "0.3018 ± 0.0160"],
    ["Message passing + PE",              "126,247", "0.3032 ± 0.0089"],
    ["Message passing only, widened",     "574,503", "0.3054 ± 0.0032"],
    ["Attention, no PE",                  "574,087", "0.2140 ± 0.0086"],
    ["Attention + PE (full GraphGPS)",    "574,951", "0.2064 ± 0.0105"],
]
cols = ["Variant", "Parameters", "Test MAE"]
cclr = [
    ['white', 'white', G6],
    ['white', 'white', G6],
    ['white', 'white', G6],
    ['white', 'white', G5],
    ['white', 'white', G4],
]

fig, ax = plt.subplots(figsize=(11, 3.4))
fig.patch.set_facecolor(BG); ax.axis('off')
tbl = ax.table(cellText=rows, colLabels=cols,
               cellLoc='center', loc='center', cellColours=cclr)
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.2)
for ci in range(len(cols)):
    tbl[0, ci].set_facecolor(G1)
    tbl[0, ci].set_text_props(color='white', fontweight='bold')
fig.tight_layout()
save(fig, "fig10_4_ablation.png")

print(f"\nAll figures saved to {OUT}")