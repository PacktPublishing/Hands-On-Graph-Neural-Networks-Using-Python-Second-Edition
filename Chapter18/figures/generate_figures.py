"""
Chapter 18 – All figures in grayscale.
Only the three conceptual diagrams of the chapter live here: the bipartite
graph sketch (18.1), the collaborative-filtering example (18.5) and the
LightGCN architecture (18.6). Every data-derived figure is produced by run.py
from a real run, so that nothing printed in the chapter comes from invented
numbers.
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

FONT = "DejaVu Sans"; BG = "white"
G0="#111111"; G1="#333333"; G2="#555555"
G3="#777777"; G4="#999999"; G5="#BBBBBB"; G6="#DDDDDD"

def save(fig, name, dpi=200):
    fig.savefig(f"{OUT}/{name}", dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig); print(f"  saved {name}")

def rbox(ax, x, y, w, h, label, fill=G2, fc='white', fs=9, r=0.04):
    rect = FancyBboxPatch((x-w/2,y-h/2),w,h,
                          boxstyle=f"round,pad=0.02,rounding_size={r}",
                          facecolor=fill,edgecolor=G4,linewidth=1.1,zorder=3)
    ax.add_patch(rect)
    for li,line in enumerate(label.split('\n')):
        off=0.11*(li-label.count('\n')/2)
        ax.text(x,y-off,line,ha='center',va='center',fontsize=fs,
                color=fc,fontweight='bold',fontfamily=FONT,zorder=4)

def arr(ax,x1,y1,x2,y2,color=G2,lw=1.4):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>",color=color,
                                lw=lw,mutation_scale=12),zorder=2)


# ── Fig 18.1: Book-Crossing bipartite graph ───────────────────────────────────
print("Figure 18.1 …")

rng = np.random.default_rng(42)
N_USERS = 60; N_BOOKS = 80
N_EDGES = 200

G_bx = nx.Graph()
users_nodes = [f"U{i}" for i in range(N_USERS)]
books_nodes = [f"B{i}" for i in range(N_BOOKS)]
G_bx.add_nodes_from(users_nodes, bipartite=0)
G_bx.add_nodes_from(books_nodes, bipartite=1)

# Power-law degree distribution — some books are very popular
book_probs = np.array([(N_BOOKS-i)**1.5 for i in range(N_BOOKS)])
book_probs = book_probs / book_probs.sum()
for _ in range(N_EDGES):
    u = f"U{rng.integers(0, N_USERS)}"
    b = f"B{rng.choice(N_BOOKS, p=book_probs)}"
    G_bx.add_edge(u, b)

pos_bx = {}
for i, u in enumerate(users_nodes):
    pos_bx[u] = (0, i * (10/N_USERS))
for i, b in enumerate(books_nodes):
    pos_bx[b] = (2, i * (10/N_BOOKS))

deg_bx = dict(G_bx.degree())
u_sizes = [10 + deg_bx.get(u,0)*15 for u in users_nodes]
b_sizes = [10 + deg_bx.get(b,0)*15 for b in books_nodes]

fig, ax = plt.subplots(figsize=(9, 8))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
nx.draw_networkx_edges(G_bx, pos_bx, ax=ax, alpha=0.12,
                       edge_color=G3, width=0.5)
nx.draw_networkx_nodes(G_bx, pos_bx, ax=ax,
                       nodelist=users_nodes, node_color=G1,
                       node_size=u_sizes, edgecolors='white', linewidths=0.4)
nx.draw_networkx_nodes(G_bx, pos_bx, ax=ax,
                       nodelist=books_nodes, node_color=G4,
                       node_size=b_sizes, edgecolors='white', linewidths=0.4)

handles = [mpatches.Patch(color=G1, label='Users (darker = more ratings)'),
           mpatches.Patch(color=G4, label='Books (lighter = more ratings)')]
ax.legend(handles=handles, loc='lower right', fontsize=9,
          framealpha=0.95, edgecolor=G5)
ax.text(-0.15, 5.0, "Users", ha='center', fontsize=11,
        fontweight='bold', color=G0, fontfamily=FONT, rotation=90)
ax.text(2.15, 5.0, "Books", ha='center', fontsize=11,
        fontweight='bold', color=G0, fontfamily=FONT, rotation=90)
ax.set_title("Book-Crossing dataset — bipartite graph (illustrative subset)\n"
             "Node size proportional to number of connections",
             fontsize=11, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
ax.text(0.02, 0.02, "Illustrative — run run.py locally for real graph",
        transform=ax.transAxes, fontsize=7.5, color=G3, fontstyle='italic')
ax.set_xlim(-0.4, 2.5)
fig.tight_layout()
save(fig, "fig18_1_bookcrossing_graph.png")


# ── Fig 18.5: Bipartite graph example (collaborative filtering) ───────────────
print("Figure 18.5 …")

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)

# Users left, items right
users_cf = [(1.5, 4.2, "User 1"), (1.5, 2.7, "User 2"), (1.5, 1.2, "User 3")]
items_cf = [(8.5, 4.8, "Item A"), (8.5, 3.6, "Item B"),
            (8.5, 2.4, "Item C"), (8.5, 1.2, "Item D")]

for x, y, lbl in users_cf:
    circ = plt.Circle((x,y), 0.42, color=G1, zorder=3)
    ax.add_patch(circ)
    ax.text(x, y, lbl.split()[1], ha='center', va='center',
            fontsize=10, color='white', fontweight='bold', fontfamily=FONT, zorder=4)
    ax.text(x-0.7, y, lbl, ha='right', fontsize=10, color=G0, fontfamily=FONT)

for x, y, lbl in items_cf:
    circ = plt.Circle((x,y), 0.42, color=G3, zorder=3)
    ax.add_patch(circ)
    ax.text(x, y, lbl.split()[1], ha='center', va='center',
            fontsize=10, color='white', fontweight='bold', fontfamily=FONT, zorder=4)
    ax.text(x+0.7, y, lbl, ha='left', fontsize=10, color=G0, fontfamily=FONT)

# Known interactions (solid)
edges_known = [(1.5,4.2,8.5,4.8),(1.5,4.2,8.5,3.6),  # User1-A, User1-B
               (1.5,2.7,8.5,4.8),(1.5,2.7,8.5,2.4),  # User2-A, User2-C → User2 has A,C? no
               (1.5,1.2,8.5,3.6),(1.5,1.2,8.5,1.2)]  # User3-B, User3-D
for x1,y1,x2,y2 in edges_known:
    ax.plot([x1+0.42,x2-0.42],[y1,y2], color=G0, linewidth=2.0, zorder=2)

# Recommendation (dashed)
ax.plot([1.5+0.42, 8.5-0.42],[2.7, 3.6], color=G2, linewidth=2.2,
        linestyle='--', zorder=2)
ax.text(5.0, 3.35, "Recommend\n(collaborative\nfiltering)",
        ha='center', fontsize=9, color=G2, fontfamily=FONT, fontstyle='italic',
        bbox=dict(facecolor='white', edgecolor=G4, boxstyle='round,pad=0.3'))

handles = [
    plt.Line2D([0],[0], color=G0, linewidth=2.0, label='Known interaction'),
    plt.Line2D([0],[0], color=G2, linewidth=2.2, linestyle='--', label='Recommendation'),
]
ax.legend(handles=handles, loc='lower left', fontsize=9,
          framealpha=0.95, edgecolor=G5)
ax.set_title("Bipartite graph for collaborative filtering\n"
             "User 1 and User 3 both liked Item B — so we recommend Item B to User 2",
             fontsize=11, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig18_5_bipartite_cf.png")


# ── Fig 18.6: LightGCN architecture diagram ───────────────────────────────────
print("Figure 18.6 …")

fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 13); ax.set_ylim(0, 5.5)

# Initial embeddings
rbox(ax, 1.2, 3.8, 1.8, 0.7, "User emb.\ne_u^(0)", fill=G1, fc='white', fs=9)
rbox(ax, 1.2, 2.5, 1.8, 0.7, "Item emb.\ne_i^(0)", fill=G3, fc='white', fs=9)

# LGConv layers
for l, x in enumerate([3.3, 4.9, 6.5, 8.1]):
    lbl = f"LGConv\nlayer {l+1}"
    rbox(ax, x, 3.15, 1.35, 1.5, lbl, fill=G2, fc='white', fs=8.5)
    arr(ax, x-0.55, 3.15, x-0.75+0.55+1.0, 3.15 if l>0 else 3.15, color=G2)

arr(ax, 2.1, 3.8, 2.6, 3.5, color=G1)
arr(ax, 2.1, 2.5, 2.6, 2.8, color=G3)
arr(ax, 8.8, 3.15, 9.0, 3.15, color=G2)

# Layer combination
rbox(ax, 9.8, 3.15, 1.6, 1.5,
     "Layer\ncombi-\nnation\n(mean)", fill=G0, fc='white', fs=9)
arr(ax, 10.6, 3.15, 11.3, 3.15, color=G0, lw=1.6)

# Final embeddings + prediction
rbox(ax, 11.8, 4.0, 1.2, 0.6, "e_u^*", fill=G1, fc='white', fs=10)
rbox(ax, 11.8, 2.3, 1.2, 0.6, "e_i^*", fill=G3, fc='white', fs=10)
ax.text(10.6, 4.0, "", ha='center')
arr(ax, 10.6, 3.6, 11.2, 4.0, color=G1)
arr(ax, 10.6, 2.7, 11.2, 2.3, color=G3)
arr(ax, 12.4, 4.0, 12.9, 3.4, color=G0)
arr(ax, 12.4, 2.3, 12.9, 2.9, color=G0)
ax.text(13.0, 3.15, "ŷ =\ne_u·e_i",
        ha='left', va='center', fontsize=10,
        fontweight='bold', color=G0, fontfamily=FONT)

# Labels
ax.text(1.2, 5.0, "Input\nembeddings", ha='center', fontsize=8.5,
        color=G3, fontfamily=FONT, fontstyle='italic')
ax.text(5.7, 5.0, "Light Graph Convolution (no feature transform, no activation)",
        ha='center', fontsize=8.5, color=G3, fontfamily=FONT, fontstyle='italic')
ax.text(9.8, 5.0, "Weighted\nsum", ha='center', fontsize=8.5,
        color=G3, fontfamily=FONT, fontstyle='italic')

ax.set_title("LightGCN architecture — light graph convolution + layer combination",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig18_6_lightgcn_arch.png")


print(f"\nDone: 3 diagrams saved to {OUT}")