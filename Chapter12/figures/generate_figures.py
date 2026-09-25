"""
Chapter 12 - All figures in grayscale.
Figs 12.1-12.4: pure matplotlib/numpy diagrams.
No external dataset download required.

Fig 12.4 reports the VGAE and SEAL test results. Those numbers are not
computed here: set them in the RESULTS block below, taking them from a
run of run.py. They changed when the double-sigmoid bug in DGCNN was
fixed, so do not reuse the first-edition values.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
import os

np.random.seed(0)

# ── RESULTS (fill in from a run of run.py) ───────────────────────────────────
VGAE_AUC = "0.8801"
VGAE_AP  = "0.8799"
SEAL_AUC = "0.8744"
SEAL_AP  = "0.9013"

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
                          facecolor=fill, edgecolor=G3,
                          linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    for li, line in enumerate(label.split('\n')):
        offset = 0.12*(li - label.count('\n')/2)
        ax.text(x, y-offset, line, ha='center', va='center',
                fontsize=fs, color=fc, fontweight='bold',
                fontfamily=FONT, zorder=4)

def arr(ax, x1, y1, x2, y2, color=G2, lw=1.5, style="-|>"):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=13), zorder=2)


# ── Fig 12.1: Graph with 1-hop, 2-hop, 3-hop neighbours ──────────────────────
print("Figure 12.1 …")

edges = [
    (0,1),(0,2),(0,3),
    (1,4),(1,5),(2,6),(3,7),(3,8),
    (4,9),(5,10),(6,11),(7,12),(8,13),(9,14)
]
G10 = nx.Graph(); G10.add_edges_from(edges)

hop1 = {1,2,3}
hop2 = {4,5,6,7,8}
hop3 = {9,10,11,12,13,14}

pos10 = {
    0:  (0.0,  0.0),
    1:  (-1.6, -1.0), 2: (0.0, -1.1), 3: (1.6, -1.0),
    4:  (-2.6, -2.2), 5:(-1.0,-2.2),  6:(0.5,-2.3), 7:(1.4,-2.2), 8:(2.5,-2.1),
    9:  (-3.2,-3.4), 10:(-0.8,-3.4), 11:(0.8,-3.4),
    12: (1.2,-3.4),  13:(2.8,-3.4),  14:(-3.6,-3.4),
}

nc10  = [G0 if n==0 else G2 if n in hop1 else G4 if n in hop2 else G6
         for n in G10.nodes()]
ns10  = [900 if n==0 else 600 if n in hop1 else 500 if n in hop2 else 380
         for n in G10.nodes()]
ec10  = [G1 if (0 in (u,v)) else
         G3 if (u in hop1 or v in hop1) else G5
         for u,v in G10.edges()]
ew10  = [2.0 if (0 in (u,v)) else
         1.5 if (u in hop1 or v in hop1) else 0.8
         for u,v in G10.edges()]

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
nx.draw_networkx_edges(G10, pos10, ax=ax, edge_color=ec10, width=ew10)
nx.draw_networkx_nodes(G10, pos10, ax=ax, node_color=nc10, node_size=ns10,
                       edgecolors='white', linewidths=1.2)
nx.draw_networkx_labels(G10, pos10, ax=ax,
                        labels={n:str(n) for n in G10.nodes()},
                        font_size=9, font_color='white',
                        font_weight='bold', font_family=FONT)
handles = [
    mpatches.Patch(color=G0, label='Target node (0)'),
    mpatches.Patch(color=G2, label='1-hop neighbours'),
    mpatches.Patch(color=G4, label='2-hop neighbours'),
    mpatches.Patch(color=G6, label='3-hop neighbours'),
]
ax.legend(handles=handles, loc='lower right', fontsize=9,
          framealpha=0.95, edgecolor=G5)
ax.set_title("Graph with 1-hop, 2-hop, and 3-hop neighbours of node 0",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig12_1_hops.png")


# ── Fig 12.2: Matrix factorization visualisation ──────────────────────────────
print("Figure 12.2 …")

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 11); ax.set_ylim(0, 5.5)

n, d = 5, 3   # 5 nodes, 3-dim embeddings

def draw_matrix(ax, x0, y0, rows, cols, label, shade_fn,
                cell_w=0.55, cell_h=0.42):
    """Draw a matrix with shaded cells."""
    np.random.seed(42)
    for r in range(rows):
        for c in range(cols):
            gray = shade_fn(r, c)
            rect = plt.Rectangle((x0 + c*cell_w, y0 + (rows-1-r)*cell_h),
                                  cell_w, cell_h,
                                  facecolor=str(gray), edgecolor='white',
                                  linewidth=0.8)
            ax.add_patch(rect)
    # Border
    border = plt.Rectangle((x0, y0), cols*cell_w, rows*cell_h,
                            facecolor='none', edgecolor=G1, linewidth=1.5)
    ax.add_patch(border)
    # Label
    ax.text(x0 + cols*cell_w/2, y0 + rows*cell_h + 0.22, label,
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=G0, fontfamily=FONT)

# A matrix (n×n)
draw_matrix(ax, 0.3, 0.8, n, n, "A  (n × n)",
            lambda r,c: 0.85 if r==c else (0.2 if np.random.rand()<0.3 else 0.85))

# ≈ symbol
ax.text(3.4, 2.85, "≈", ha='center', va='center', fontsize=30,
        color=G0, fontfamily=FONT)

# Z matrix (n×d)
draw_matrix(ax, 4.1, 1.6, n, d, "Z  (n × d)",
            lambda r,c: 0.15 + 0.7*np.random.rand())

# × symbol
ax.text(6.1, 2.85, "×", ha='center', va='center', fontsize=26,
        color=G0, fontfamily=FONT)

# Z^T matrix (d×n)
draw_matrix(ax, 6.7, 2.3, d, n, "Zᵀ  (d × n)",
            lambda r,c: 0.15 + 0.7*np.random.rand())

# Dimension annotations
ax.text(1.65, 0.45, f"n = number of nodes", ha='center',
        fontsize=9, color=G3, fontfamily=FONT, fontstyle='italic')
ax.text(4.9,  1.30, f"n × d", ha='center',
        fontsize=9, color=G3, fontfamily=FONT, fontstyle='italic')
ax.text(9.45, 2.05, f"d × n", ha='center',
        fontsize=9, color=G3, fontfamily=FONT, fontstyle='italic')

ax.text(5.5, 0.18,
        "d = embedding dimension  (d ≪ n for a compressed representation)",
        ha='center', fontsize=9, color=G3, fontfamily=FONT, fontstyle='italic')

ax.set_title("Matrix factorization: A ≈ Z × Zᵀ\n"
             "The adjacency matrix is approximated by the product of node embeddings",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig12_2_matrix_factorization.png")


# ── Fig 12.3: SEAL framework diagram ─────────────────────────────────────────
print("Figure 12.3 …")

fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 13); ax.set_ylim(0, 5.5)

# Step boxes
steps = [
    (1.5, 2.75, "Step 1\nEnclosing subgraph\nextraction",    G1),
    (5.5, 2.75, "Step 2\nNode information\nmatrix construction", G3),
    (9.5, 2.75, "Step 3\nGNN training\n(DGCNN)",             G2),
]
for x, y, lbl, fill in steps:
    rbox(ax, x, y, 2.5, 2.0, lbl, fill=fill, fc='white', fs=10)

# Arrows between steps
arr(ax, 2.75, 2.75, 4.25, 2.75, color=G2, lw=2.0)
arr(ax, 6.75, 2.75, 8.25, 2.75, color=G2, lw=2.0)

# Sub-labels below each box
sub_texts = [
    (1.5, 1.25,
     "Positive edges (real links)\n+ negative edges (fake links)"),
    (5.5, 1.25,
     "Node labels (DRNL)\n+ embeddings + features"),
    (9.5, 1.25,
     "Sort pooling\n→ link probability"),
]
for x, y, txt in sub_texts:
    ax.text(x, y, txt, ha='center', va='center', fontsize=8.5,
            color=G3, fontfamily=FONT, fontstyle='italic')

# Input graph on the left
G_s = nx.path_graph(4); G_s.add_edge(0,2)
pos_s = nx.spring_layout(G_s, seed=1)
pos_s = {k: (v[0]*0.6 + 0.4, v[1]*0.6 + 2.75) for k,v in pos_s.items()}

# Mark target link (0-3) in dark, rest lighter
nc_s = [G0 if n in (0,3) else G4 for n in G_s.nodes()]
ec_s = [G0 if set((u,v))=={0,3} else G5 for u,v in G_s.edges()]
ew_s = [2.5 if set((u,v))=={0,3} else 0.8 for u,v in G_s.edges()]
nx.draw_networkx_edges(G_s, pos_s, ax=ax, edge_color=ec_s, width=ew_s, style='dashed')
nx.draw_networkx_nodes(G_s, pos_s, ax=ax, node_color=nc_s, node_size=200,
                       edgecolors='white', linewidths=0.8)
ax.text(0.4, 4.8, "Input graph\n(dashed = target link)",
        ha='center', fontsize=8, color=G3, fontfamily=FONT, fontstyle='italic')

# Output arrow
arr(ax, 10.75, 2.75, 11.6, 2.75, color=G2, lw=2.0)
ax.text(12.2, 2.75, "P(link)\n∈ [0,1]",
        ha='center', va='center', fontsize=10, color=G0,
        fontweight='bold', fontfamily=FONT)

ax.set_title("The SEAL framework: three steps from graph to link probability",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig12_3_seal_framework.png")


# ── Fig 12.4: VGAE vs SEAL comparison table ───────────────────────────────────
print("Figure 12.4 …")

rows = [
    ["VGAE",  "Node embeddings\n(matrix factorization)",
     VGAE_AUC, VGAE_AP, "Fast (no preprocessing)"],
    ["SEAL",  "Enclosing subgraphs\n(subgraph representation)",
     SEAL_AUC, SEAL_AP, "Slower (subgraph\npreprocessing dominates)"],
]
cols = ["Model", "Approach", "Test AUC", "Test AP", "Training time"]
cclr = [
    ['white', 'white', G5, G5, 'white'],
    ['white', 'white', G5, G5, 'white'],
]

fig, ax = plt.subplots(figsize=(12, 2.8))
fig.patch.set_facecolor(BG); ax.axis('off')
tbl = ax.table(cellText=rows, colLabels=cols,
               cellLoc='center', loc='center', cellColours=cclr)
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.5)
for ci in range(len(cols)):
    tbl[0, ci].set_facecolor(G1)
    tbl[0, ci].set_text_props(color='white', fontweight='bold')
ax.set_title("VGAE vs SEAL: test results on Cora (link prediction)\n"
             "Both models trained with 85/5/10 train/val/test split",
             fontsize=10, color=G3, fontstyle='italic',
             fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig12_4_comparison.png")

print(f"\nAll figures saved to {OUT}")