"""
Generate all Chapter 5 figures — pure matplotlib/networkx, no PyTorch required.
Figures 5.1 and 5.2 use synthetic graphs that match the published properties
of Cora and Facebook Page-Page (node count, class structure, density).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import random, os

random.seed(42)
np.random.seed(42)

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

FONT   = "DejaVu Sans"
BG     = "white"

# Shared grayscale ramp used from Chapter 6 onwards for non-categorical marks
# (fills, headers, arrows): G0 is almost black, G6 almost white.
G0, G1, G2, G3 = "#111111", "#333333", "#555555", "#777777"
G4, G5, G6     = "#999999", "#BBBBBB", "#DDDDDD"

GRAY_EDGE = "#595959"   # graph edges, connector arrows and caption text
LGRAY     = "#EDEDED"   # legend borders and "…" filler cells
BLACK     = "#1A1A1A"   # titles

# Categorical tones for the graph figures: evenly spaced lightness steps so the
# classes stay separable when the page is printed in black and white.
GRAY7 = ["#1A1A1A", "#404040", "#666666", "#8C8C8C", "#ABABAB", "#C8C8C8", "#E0E0E0"]
GRAY4 = ["#1A1A1A", "#606060", "#A0A0A0", "#D0D0D0"]

# Table accents: the highlighted columns must differ from LGRAY filler cells.
HDR_FILL   = G1   # header row, dark enough for white bold text
HL_STRONG  = G5   # primary highlighted column (label / GNN accuracy)
HL_SOFT    = G6   # secondary highlighted column (improvement)

# Architecture boxes, lightening with depth; all still carry white text.
BOX_IN, BOX_HIDDEN, BOX_OUT = G0, G2, G3
ADJ = G0          # adjacency-matrix arrows, near-black against GRAY_EDGE flow

def save(fig, name, dpi=200):
    fig.savefig(f"{OUT}/{name}", dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"  saved {name}")


# ── Fig 5.1: Cora-like citation network ──────────────────────────────────────
print("Figure 5.1 …")

n_per_class = 30
n_classes   = 7
class_names = ["Theory", "Reinf. Learning", "Genetic Alg.",
               "Neural Nets", "Probabilistic", "Case Based", "Rule Learning"]

G = nx.Graph()
node_cls = {}
node_id  = 0
class_nodes = []

for c in range(n_classes):
    nodes = list(range(node_id, node_id + n_per_class))
    class_nodes.append(nodes)
    for n in nodes:
        G.add_node(n)
        node_cls[n] = c
    for i in nodes:
        for j in nodes:
            if i < j and random.random() < 0.18:
                G.add_edge(i, j)
    node_id += n_per_class

for c1 in range(n_classes):
    for c2 in range(c1+1, n_classes):
        for _ in range(4):
            G.add_edge(random.choice(class_nodes[c1]),
                       random.choice(class_nodes[c2]))

pos = {}
for c, nodes in enumerate(class_nodes):
    angle = 2*np.pi*c/n_classes
    cx, cy = 3.0*np.cos(angle), 3.0*np.sin(angle)
    sub = G.subgraph(nodes)
    sp  = nx.spring_layout(sub, seed=c, k=0.6)
    for n, (x, y) in sp.items():
        pos[n] = (cx + 0.9*x, cy + 0.9*y)

deg  = dict(G.degree())
nc   = [GRAY7[node_cls[n]] for n in G.nodes()]
ns   = [20 + deg[n]*8 for n in G.nodes()]

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.18, edge_color=GRAY_EDGE, width=0.7)
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=ns,
                       edgecolors='white', linewidths=0.5)
handles = [mpatches.Patch(color=GRAY7[i], label=class_names[i])
           for i in range(n_classes)]
ax.legend(handles=handles, loc='lower right', fontsize=9,
          framealpha=0.95, edgecolor=LGRAY,
          title="Category (7 classes)", title_fontsize=9)
ax.set_title("Cora dataset – citation network (2,708 nodes, 7 classes)\n"
             "Illustrative subgraph coloured by research category",
             fontsize=12, fontweight='bold', color=BLACK, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig5_1_cora.png")


# ── Fig 5.2: Facebook Page-Page-like social graph ────────────────────────────
print("Figure 5.2 …")

fb_class_names = ["Politicians", "Companies", "TV Shows", "Gov. Organizations"]
n_hubs = 6
G2 = nx.Graph()
fb_cls = {}
node_id = 0
class_hubs = []

for c in range(4):
    hubs = list(range(node_id, node_id + n_hubs))
    class_hubs.append(hubs)
    for h in hubs:
        G2.add_node(h); fb_cls[h] = c
    for i in hubs:
        for j in hubs:
            if i < j and random.random() < 0.5:
                G2.add_edge(i, j)
    node_id += n_hubs
    for h in hubs:
        for _ in range(random.randint(3, 8)):
            G2.add_node(node_id); fb_cls[node_id] = c
            G2.add_edge(h, node_id)
            node_id += 1

for c1 in range(4):
    for c2 in range(c1+1, 4):
        for _ in range(3):
            G2.add_edge(random.choice(class_hubs[c1]),
                        random.choice(class_hubs[c2]))

pos2 = nx.spring_layout(G2, seed=7, k=0.55)
deg2 = dict(G2.degree())
nc2  = [GRAY4[fb_cls[n]] for n in G2.nodes()]
ns2  = [30 + deg2[n]*12 for n in G2.nodes()]

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
nx.draw_networkx_edges(G2, pos2, ax=ax, alpha=0.15, edge_color=GRAY_EDGE, width=0.7)
nx.draw_networkx_nodes(G2, pos2, ax=ax, node_color=nc2, node_size=ns2,
                       edgecolors='white', linewidths=0.5)
handles2 = [mpatches.Patch(color=GRAY4[i], label=fb_class_names[i])
            for i in range(4)]
ax.legend(handles=handles2, loc='lower right', fontsize=10,
          framealpha=0.95, edgecolor=LGRAY,
          title="Category (4 classes)", title_fontsize=9)
ax.set_title("Facebook Page-Page dataset – social network (22,470 nodes, 4 classes)\n"
             "Illustrative subgraph; node size proportional to degree",
             fontsize=12, fontweight='bold', color=BLACK, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig5_2_facebook.png")


# ── Fig 5.3: Tabular representation ──────────────────────────────────────────
print("Figure 5.3 …")

np.random.seed(0)
sample = (np.random.rand(5, 1433) < 0.02).astype(int)
lbls   = [3, 4, 4, 1, 3]

disp_cols = [str(i) for i in range(7)] + ['…', '1432', 'label']
disp_data = []
for i in range(5):
    row = ([str(v) for v in sample[i, :7]] + ['…'] +
           [str(sample[i, -1])] + [str(lbls[i])])
    disp_data.append(row)
disp_data.append(['…'] * len(disp_cols))
row_labels = ['0','1','2','3','4','…']

cell_colors = []
for ri, row in enumerate(disp_data):
    row_c = []
    for ci, col in enumerate(disp_cols):
        if ri == 5 or col == '…':
            row_c.append(LGRAY)
        elif col == 'label':
            row_c.append(HL_STRONG)
        else:
            row_c.append('white')
    cell_colors.append(row_c)

fig, ax = plt.subplots(figsize=(12, 3.0))
fig.patch.set_facecolor(BG); ax.axis('off')
tbl = ax.table(cellText=disp_data, rowLabels=row_labels,
               colLabels=disp_cols, cellLoc='center', loc='center',
               cellColours=cell_colors)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.0, 1.8)
for ci in range(len(disp_cols)):
    tbl[0, ci].set_facecolor(HDR_FILL)
    tbl[0, ci].set_text_props(color='white', fontweight='bold')
ax.set_title(
    "Tabular representation of the Cora dataset  "
    "(5 of 2,708 nodes; 7 of 1,433 features shown; label column highlighted)",
    fontsize=10, color=GRAY_EDGE, fontstyle='italic', fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig5_3_tabular.png")


# ── Fig 5.4: MLP vs Vanilla GNN architecture diagram ─────────────────────────
print("Figure 5.4 …")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor(BG)

def draw_arch(ax, title, layers, show_adj=False):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=13, fontweight='bold',
                 color=BLACK, fontfamily=FONT, pad=12)
    n  = len(layers)
    xs = np.linspace(0.14, 0.86, n)
    for i, (x, layer) in enumerate(zip(xs, layers)):
        h  = layer['h']
        y0 = 0.5 - h/2
        rect = FancyBboxPatch((x-0.10, y0), 0.20, h,
                              boxstyle="round,pad=0.015,rounding_size=0.03",
                              facecolor=layer['color'], edgecolor='none', zorder=3)
        ax.add_patch(rect)
        lines = layer['label'].split('\n')
        n_lines = len(lines)
        for li, line in enumerate(lines):
            offset = 0.045*(li-(n_lines-1)/2)
            ax.text(x, 0.5-offset, line, ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold',
                    fontfamily=FONT, zorder=4)
        ax.text(x, y0-0.07, layer.get('sub',''), ha='center',
                fontsize=8, color=GRAY_EDGE, fontfamily=FONT)
        if i < n-1:
            ax.annotate("", xy=(xs[i+1]-0.10, 0.5), xytext=(x+0.10, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=GRAY_EDGE,
                                        lw=1.5, mutation_scale=15), zorder=2)
    if show_adj:
        for j, xi in enumerate(xs[1:], 1):
            ax.annotate("",
                xy=(xi-0.10, 0.22), xytext=(xs[0]+0.10, 0.22),
                arrowprops=dict(arrowstyle="-|>", color=ADJ, lw=1.4,
                                mutation_scale=12,
                                connectionstyle=f"arc3,rad={-0.25-0.1*j}"),
                zorder=2)
        ax.text(0.50, 0.08, "Adjacency matrix Ã  (topology at every layer)",
                ha='center', fontsize=8.5, color=ADJ,
                fontfamily=FONT, fontstyle='italic')

draw_arch(axes[0], "MLP  (topology-agnostic)", [
    {'label':'Node\nfeatures\nx',  'h':0.56, 'color':BOX_IN,
     'sub':'Input  (1,433 dims)'},
    {'label':'Linear\n+ ReLU',     'h':0.36, 'color':BOX_HIDDEN,
     'sub':'Hidden  (16 dims)'},
    {'label':'Linear\n+ Softmax',  'h':0.28, 'color':BOX_OUT,
     'sub':'Output  (7 classes)'},
], show_adj=False)
axes[0].text(0.50, 0.93, "H = σ( X · W )",
             ha='center', fontsize=11, color=GRAY_EDGE,
             fontfamily=FONT, fontstyle='italic')

draw_arch(axes[1], "Vanilla GNN  (topology-aware)", [
    {'label':'Node\nfeatures\nx',  'h':0.56, 'color':BOX_IN,
     'sub':'Input  (1,433 dims)'},
    {'label':'Graph\nlayer 1',     'h':0.36, 'color':BOX_HIDDEN,
     'sub':'Ã · X · W₁  (16 dims)'},
    {'label':'Graph\nlayer 2',     'h':0.28, 'color':BOX_OUT,
     'sub':'Ã · H · W₂  (7 classes)'},
], show_adj=True)
axes[1].text(0.50, 0.93, "H = σ( Ã · X · W )",
             ha='center', fontsize=11, color=GRAY_EDGE,
             fontfamily=FONT, fontstyle='italic')

fig.tight_layout(pad=2.0)
save(fig, "fig5_4_architectures.png")


# ── Fig 5.5: Results table ────────────────────────────────────────────────────
print("Figure 5.5 …")

data_rows = [
    ["Cora",     "53.47%  (±1.81%)", "74.98%  (±1.50%)", "+21.51%"],
    ["Facebook", "75.21%  (±0.40%)", "84.85%  (±1.68%)", "+9.64%"],
]
col_labels = ["Dataset", "MLP accuracy", "Vanilla GNN accuracy", "Improvement"]
row_colors = [
    ['white', 'white', HL_STRONG, HL_SOFT],
    ['white', 'white', HL_STRONG, HL_SOFT],
]

fig, ax = plt.subplots(figsize=(9, 2.8))
fig.patch.set_facecolor(BG); ax.axis('off')
tbl = ax.table(cellText=data_rows, colLabels=col_labels,
               cellLoc='center', loc='center', cellColours=row_colors)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 2.2)
for ci in range(len(col_labels)):
    tbl[0, ci].set_facecolor(HDR_FILL)
    tbl[0, ci].set_text_props(color='white', fontweight='bold')
ax.set_title(
    "Mean test accuracy over 20 runs  —  MLP vs Vanilla GNN\n"
    "(run chapter5.py locally to reproduce exact values)",
    fontsize=10, color=GRAY_EDGE, fontstyle='italic', fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig5_5_results.png")

print(f"\nAll figures saved to {OUT}")
