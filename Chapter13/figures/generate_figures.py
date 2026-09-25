"""
Chapter 13 (ex-12) – All figures in grayscale.
Figs 13.1–13.6: pure matplotlib diagrams, no external downloads.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import os

np.random.seed(0)

OUT  = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

FONT = "DejaVu Sans"
BG   = "white"
G0   = "#111111"
G1   = "#333333"
G2   = "#555555"
G3   = "#777777"
G4   = "#999999"
G5   = "#BBBBBB"
G6   = "#DDDDDD"

def save(fig, name, dpi=200):
    fig.savefig(f"{OUT}/{name}", dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"  saved {name}")

def rbox(ax, x, y, w, h, label, fill=G2, fc='white', fs=10, r=0.04):
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle=f"round,pad=0.02,rounding_size={r}",
                          facecolor=fill, edgecolor=G4,
                          linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    for li, line in enumerate(label.split('\n')):
        offset = 0.12*(li - label.count('\n')/2)
        ax.text(x, y-offset, line, ha='center', va='center',
                fontsize=fs, color=fc, fontweight='bold',
                fontfamily=FONT, zorder=4)

def arr(ax, x1, y1, x2, y2, color=G2, lw=1.5, label=""):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=13), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.08, my, label, fontsize=8, color=G3,
                fontfamily=FONT, fontstyle='italic', zorder=5)


# ── Fig 13.1: MPNN framework ──────────────────────────────────────────────────
print("Figure 13.1 …")

fig, ax = plt.subplots(figsize=(13, 4.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 13); ax.set_ylim(0, 4.5)

# Three step boxes
steps = [
    (2.0, 2.25, "Message\nfunction m()", G1),
    (6.5, 2.25, "Aggregate\nfunction □()", G3),
    (11.0,2.25, "Update\nfunction U()", G2),
]
for x, y, lbl, fill in steps:
    rbox(ax, x, y, 3.0, 1.6, lbl, fill=fill, fc='white', fs=11)

# Arrows between steps
arr(ax, 3.5, 2.25, 5.0, 2.25, color=G1, lw=2.0)
arr(ax, 8.0, 2.25, 9.5, 2.25, color=G3, lw=2.0)

# Sub-labels
ax.text(2.0, 0.9,
        "h_v, h_u, e_uv\n→ message m_uv",
        ha='center', fontsize=8.5, color=G3, fontfamily=FONT,
        fontstyle='italic')
ax.text(6.5, 0.9,
        "sum / mean / max\nover N(v)",
        ha='center', fontsize=8.5, color=G3, fontfamily=FONT,
        fontstyle='italic')
ax.text(11.0, 0.9,
        "h_v + aggregated\n→ new h_v",
        ha='center', fontsize=8.5, color=G3, fontfamily=FONT,
        fontstyle='italic')

# Input / output labels
ax.text(0.4, 2.25, "Node &\nedge\nfeatures",
        ha='center', va='center', fontsize=9, color=G3, fontfamily=FONT)
ax.annotate("", xy=(0.65, 2.25), xytext=(0.0, 2.25),
            arrowprops=dict(arrowstyle="-|>", color=G2, lw=1.5,
                            mutation_scale=13))
ax.text(12.8, 2.25, "Updated\nembedding\nh_v",
        ha='center', va='center', fontsize=9, color=G3, fontfamily=FONT)

ax.set_title("The Message Passing Neural Network (MPNN) framework",
             fontsize=13, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig13_1_mpnn.png")


# ── Fig 13.2: Heterogeneous graph ─────────────────────────────────────────────
print("Figure 13.2 …")

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')

# Node positions
nodes = {
    # Users
    'U1': (1.0, 5.5), 'U2': (1.0, 4.0), 'U3': (1.0, 2.5),
    # Games
    'G1': (5.0, 5.0), 'G2': (5.0, 3.0),
    # Devs
    'D1': (9.0, 5.0), 'D2': (9.0, 3.0),
}

node_shapes = {
    'U1': (G0,'User 1'), 'U2': (G0,'User 2'), 'U3': (G0,'User 3'),
    'G1': (G2,'Game 1'), 'G2': (G2,'Game 2'),
    'D1': (G4,'Dev 1'),  'D2': (G4,'Dev 2'),
}

# Draw edges first
edges = [
    # follows
    ('U1','U2', 'follows', 0.6, G1),
    ('U2','U3', 'follows', 0.6, G1),
    # plays
    ('U1','G1', 'plays',   0.5, G2),
    ('U2','G1', 'plays',   0.5, G2),
    ('U2','G2', 'plays',   0.5, G2),
    ('U3','G2', 'plays',   0.5, G2),
    # develops
    ('D1','G1', 'develops',0.5, G4),
    ('D2','G2', 'develops',0.5, G4),
]

for src, dst, rel, lw, col in edges:
    x1,y1 = nodes[src]; x2,y2 = nodes[dst]
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=lw, mutation_scale=14,
                                connectionstyle="arc3,rad=0.05"),
                zorder=1)
    mx,my = (x1+x2)/2+0.1, (y1+y2)/2
    ax.text(mx, my, rel, ha='center', va='bottom', fontsize=8,
            color=col, fontfamily=FONT, fontstyle='italic',
            bbox=dict(facecolor='white', edgecolor='none', pad=1))

# Draw nodes
for key, (x,y) in nodes.items():
    col, lbl = node_shapes[key]
    circle = plt.Circle((x,y), 0.42, color=col, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, lbl, ha='center', va='center', fontsize=9,
            color='white', fontweight='bold', fontfamily=FONT, zorder=4)

# Group labels
for x, y, lbl in [(1.0,6.4,"Users"), (5.0,6.4,"Games"), (9.0,6.4,"Developers")]:
    ax.text(x, y, lbl, ha='center', fontsize=11, fontweight='bold',
            color=G0, fontfamily=FONT,
            bbox=dict(facecolor=G6, edgecolor=G4,
                      boxstyle='round,pad=0.4'))

# Legend
handles = [
    mpatches.Patch(color=G0, label='User nodes'),
    mpatches.Patch(color=G2, label='Game nodes'),
    mpatches.Patch(color=G4, label='Developer nodes'),
]
ax.legend(handles=handles, loc='lower right', fontsize=9,
          framealpha=0.95, edgecolor=G5)
ax.set_xlim(-0.2, 10.8); ax.set_ylim(1.5, 7.2)
ax.set_title("A heterogeneous graph with 3 node types and 3 edge types\n"
             "(users, games, developers  /  follows, plays, develops)",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig13_2_hetero_graph.png")


# ── Fig 13.3: DBLP node type relationships ────────────────────────────────────
print("Figure 13.3 …")

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 9); ax.set_ylim(0, 5.5)

node_types = [
    (1.5, 4.0, "Author\n(4,057)", G0),
    (4.5, 4.0, "Paper\n(14,328)", G2),
    (7.5, 4.0, "Conference\n(20)", G4),
    (4.5, 1.5, "Term\n(7,723)", G3),
]
for x, y, lbl, fill in node_types:
    rbox(ax, x, y, 2.0, 1.2, lbl, fill=fill, fc='white', fs=10)

# Directed edges with relation labels
rels = [
    (1.5, 4.0, 4.5, 4.0, "write / written by", 0),
    (4.5, 4.0, 7.5, 4.0, "published in / contains", 0),
    (4.5, 4.0, 4.5, 1.5, "contains / in", 0),
]
for x1,y1,x2,y2,lbl,_ in rels:
    # forward arrow
    ax.annotate("", xy=(x2-1.05,y2), xytext=(x1+1.05,y1),
                arrowprops=dict(arrowstyle="-|>", color=G1,
                                lw=1.6, mutation_scale=13,
                                connectionstyle="arc3,rad=0.15"))
    # reverse arrow
    ax.annotate("", xy=(x1+1.05,y1), xytext=(x2-1.05,y2),
                arrowprops=dict(arrowstyle="-|>", color=G1,
                                lw=1.6, mutation_scale=13,
                                connectionstyle="arc3,rad=0.15"))
    mx,my = (x1+x2)/2, (y1+y2)/2 + 0.32
    ax.text(mx, my, lbl, ha='center', fontsize=8.5,
            color=G2, fontfamily=FONT, fontstyle='italic',
            bbox=dict(facecolor='white', edgecolor='none', pad=1))

ax.set_title("Node type relationships in the DBLP dataset\n"
             "Goal: classify authors into 4 categories (DB / DM / AI / IR)",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig13_3_dblp.png")


# ── Fig 13.4: Homogeneous vs Heterogeneous GAT ────────────────────────────────
print("Figure 13.4 …")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.patch.set_facecolor(BG)

def draw_gat_arch(ax, title, layers_spec):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_xlim(0, 5); ax.set_ylim(0, 6)
    ax.set_title(title, fontsize=12, fontweight='bold',
                 color=G0, fontfamily=FONT, pad=8)
    xs = [1.0, 2.5, 4.0]
    for (x, specs) in zip(xs, layers_spec):
        for i, (y, lbl, fill) in enumerate(specs):
            rbox(ax, x, y, 1.6, 0.7, lbl, fill=fill, fc='white', fs=8)
    # arrows between layer columns
    for i in range(len(layers_spec)-1):
        x1, x2 = xs[i]+0.8, xs[i+1]-0.8
        mid_y = np.mean([s[0] for s in layers_spec[i]])
        arr(ax, x1, mid_y, x2, mid_y, color=G2, lw=1.4)

# Homogeneous: one input, one GAT layer, one output
draw_gat_arch(axes[0], "Homogeneous GAT\n(single relation type)", [
    [(3.0, "Author\nfeatures", G0)],
    [(3.0, "GAT layer\n(1 shared)", G2)],
    [(3.0, "Author\nclassification", G4)],
])

# Heterogeneous: multiple input types, 6 separate GAT layers, aggregation
draw_gat_arch(axes[1], "Heterogeneous GAT via to_hetero()\n(one layer per relation)", [
    [
        (5.0, "Author\nfeatures", G0),
        (4.0, "Paper\nfeatures", G1),
        (3.0, "Term\nfeatures", G2),
        (2.0, "Conf.\nfeatures", G3),
    ],
    [
        (5.0, "Layer\nauthor→paper", G1),
        (4.2, "Layer\npaper→author", G1),
        (3.4, "Layer\npaper→term", G2),
        (2.6, "Layer\nterm→paper", G2),
        (1.8, "Layer\nconf.→paper", G3),
        (1.0, "Layer\npaper→conf.", G3),
    ],
    [(3.0, "Aggregate\n+ classify", G0)],
])

fig.tight_layout()
save(fig, "fig13_4_homo_vs_hetero.png")


# ── Fig 13.5: HAN architecture ────────────────────────────────────────────────
print("Figure 13.5 …")

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 13); ax.set_ylim(0, 6)

# Column 1: input meta-paths
mp_labels = ["Meta-path\nAPA", "Meta-path\nAPCPA", "Meta-path\nAPTPA"]
mp_y      = [4.8, 3.0, 1.2]
mp_fill   = [G1, G2, G3]
for y, lbl, fill in zip(mp_y, mp_labels, mp_fill):
    rbox(ax, 1.5, y, 2.0, 1.0, lbl, fill=fill, fc='white', fs=9)

# Column 2: node-level attention (per meta-path)
for y, fill in zip(mp_y, mp_fill):
    rbox(ax, 4.5, y, 2.0, 0.9, "Node-level\nattention", fill=fill, fc='white', fs=9)
    arr(ax, 2.5, y, 3.5, y, color=fill, lw=1.4)

# Column 3: semantic-level attention
rbox(ax, 7.8, 3.0, 2.2, 3.6, "Semantic-level\nattention\n(meta-path\nweights)",
     fill=G0, fc='white', fs=9)
for y in mp_y:
    arr(ax, 5.5, y, 6.7, 3.0, color=G2, lw=1.2)

# Column 4: final embedding + classifier
rbox(ax, 10.8, 3.0, 2.0, 1.0, "Final\nembedding Z",  fill=G2, fc='white', fs=9)
rbox(ax, 10.8, 1.4, 2.0, 0.9, "Classifier\n(Linear)", fill=G4, fc='white', fs=9)
arr(ax, 8.9, 3.0, 9.8, 3.0, color=G0, lw=1.8)
arr(ax, 10.8, 2.5, 10.8, 1.85, color=G2, lw=1.4)

# Labels
ax.text(1.5, 5.9, "Input\n(meta-paths)", ha='center',
        fontsize=9, color=G3, fontfamily=FONT, fontstyle='italic')
ax.text(4.5, 5.9, "Node-level\nattention", ha='center',
        fontsize=9, color=G3, fontfamily=FONT, fontstyle='italic')
ax.text(7.8, 5.9, "Semantic-level\nattention", ha='center',
        fontsize=9, color=G3, fontfamily=FONT, fontstyle='italic')
ax.text(10.8, 5.9, "Output", ha='center',
        fontsize=9, color=G3, fontfamily=FONT, fontstyle='italic')

ax.set_title("HAN architecture — node-level and semantic-level attention",
             fontsize=13, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig13_5_han.png")


# ── Fig 13.6: Comparison table ────────────────────────────────────────────────
print("Figure 13.6 …")

rows = [
    ["GAT (homogeneous)\n+ meta-path",
     "Author-Paper-Author only",
     "71.60%", "Baseline"],
    ["Het-GAT (to_hetero())",
     "All 6 relation types",
     "80.01%", "+8.41%"],
    ["HAN",
     "Node + semantic attention\n(auto meta-path weighting)",
     "81.76%", "+10.16%"],
]
cols = ["Model", "Heterogeneous information used",
        "Test accuracy", "vs baseline"]
cclr = [
    ['white','white', G5,      'white'],
    ['white','white', "#DDDDDD","white"],
    ['white','white', "#AAAAAA","white"],
]

fig, ax = plt.subplots(figsize=(12, 3.2))
fig.patch.set_facecolor(BG); ax.axis('off')
tbl = ax.table(cellText=rows, colLabels=cols,
               cellLoc='center', loc='center', cellColours=cclr)
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.6)
for ci in range(len(cols)):
    tbl[0,ci].set_facecolor(G1)
    tbl[0,ci].set_text_props(color='white', fontweight='bold')
ax.set_title(
    "Author classification on DBLP: 100 epochs, single run",
    fontsize=10, color=G3, fontstyle='italic', fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig13_6_comparison.png")

print(f"\nAll figures saved to {OUT}")