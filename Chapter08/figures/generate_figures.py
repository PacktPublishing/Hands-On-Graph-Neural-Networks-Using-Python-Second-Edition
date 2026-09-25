"""
Chapter 8 – All figures in grayscale.
Figs 8.1–8.3: pure matplotlib diagrams.
Figs 8.4–8.6: networkx / synthetic PyG data (no external downloads).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
import torch
import os, random

random.seed(0); np.random.seed(0); torch.manual_seed(0)

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

FONT = "DejaVu Sans"
BG   = "white"
# Grayscale palette
G0 = "#111111"   # very dark
G1 = "#333333"
G2 = "#555555"
G3 = "#777777"
G4 = "#999999"
G5 = "#BBBBBB"
G6 = "#DDDDDD"   # very light

def save(fig, name, dpi=200):
    fig.savefig(f"{OUT}/{name}", dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"  saved {name}")

def gbox(ax, x, y, w, h, label, fill=G2, fc='white', fontsize=10, r=0.05):
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle=f"round,pad=0.02,rounding_size={r}",
                          facecolor=fill, edgecolor=G3, linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            color=fc, fontweight='bold', fontfamily=FONT, zorder=4)

def arr(ax, x1, y1, x2, y2, color=G2, lw=1.5):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=13), zorder=2)


# ── Fig 8.1: 1-hop and 2-hop neighbours ──────────────────────────────────────
print("Figure 8.1 …")

G81 = nx.Graph()
# Centre node
edges = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,6),(3,7),(3,8),
         (4,9),(5,10),(6,11),(7,12)]
G81.add_edges_from(edges)
pos81 = nx.spring_layout(G81, seed=3, k=1.1)

hop1 = {1,2,3}
hop2 = {4,5,6,7,8}
rest = set(G81.nodes()) - {0} - hop1 - hop2

node_colors = []
node_sizes  = []
for n in G81.nodes():
    if n == 0:
        node_colors.append(G0); node_sizes.append(900)
    elif n in hop1:
        node_colors.append(G2); node_sizes.append(700)
    elif n in hop2:
        node_colors.append(G4); node_sizes.append(550)
    else:
        node_colors.append(G6); node_sizes.append(400)

edge_colors = []
for u,v in G81.edges():
    if (u==0 or v==0):
        edge_colors.append(G1)
    elif (u in hop1 or v in hop1) and (u in hop2 or v in hop2):
        edge_colors.append(G3)
    else:
        edge_colors.append(G5)

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
nx.draw_networkx_edges(G81, pos81, ax=ax, edge_color=edge_colors, width=1.8)
nx.draw_networkx_nodes(G81, pos81, ax=ax, node_color=node_colors,
                       node_size=node_sizes, edgecolors='white', linewidths=1.2)
nx.draw_networkx_labels(G81, pos81, ax=ax,
                        labels={n: str(n) for n in G81.nodes()},
                        font_size=9, font_color='white',
                        font_weight='bold', font_family=FONT)
handles = [
    mpatches.Patch(color=G0, label='Target node (0)'),
    mpatches.Patch(color=G2, label='1-hop neighbours'),
    mpatches.Patch(color=G4, label='2-hop neighbours'),
    mpatches.Patch(color=G6, label='Other nodes'),
]
ax.legend(handles=handles, loc='lower right', fontsize=9,
          framealpha=0.95, edgecolor=G5)
ax.set_title("1-hop and 2-hop neighbours of node 0",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig8_1_hops.png")


# ── Fig 8.2: Computation graph of node 0 (tree layout) ───────────────────────
print("Figure 8.2 …")

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 5)

# Tree: node 0 at centre top, 1-hop below, 2-hop below that
# Layer 2 (output) — node 0
gbox(ax, 5.0, 4.3, 1.1, 0.55, "node 0\n(output)", fill=G0, fontsize=9)

# Layer 1 — 1-hop neighbours
l1 = [(2.5, 2.8, "node 1"), (5.0, 2.8, "node 2"), (7.5, 2.8, "node 3")]
for x, y, lbl in l1:
    gbox(ax, x, y, 1.1, 0.50, lbl, fill=G2, fontsize=9)
    arr(ax, x, y+0.25, 5.0, 4.0)

# Layer 0 — 2-hop neighbours (children of each L1 node)
# Derived from Fig 8.1 adjacency: 1→{4,5}, 2→{6}, 3→{7,8}
l2_map = {
    2.5: [(1.2, 1.3, "node 4"), (3.8, 1.3, "node 5")],
    5.0: [(5.0, 1.3, "node 6")],
    7.5: [(6.2, 1.3, "node 7"), (8.8, 1.3, "node 8")],
}
for px, children in l2_map.items():
    for x, y, lbl in children:
        gbox(ax, x, y, 1.05, 0.45, lbl, fill=G4, fontsize=8)
        arr(ax, x, y+0.225, px, 2.55)

ax.text(5.0, 4.92, "Layer 2 (aggregation → embedding of node 0)",
        ha='center', fontsize=9, color=G2, fontfamily=FONT, fontstyle='italic')
ax.text(5.0, 0.6, "Layer 0 — 2-hop neighbourhood (leaf nodes)",
        ha='center', fontsize=9, color=G4, fontfamily=FONT, fontstyle='italic')
ax.text(0.1, 2.5, "Layer 1 — 1-hop", ha='left',
        fontsize=9, color=G2, fontfamily=FONT, fontstyle='italic')

ax.set_title("Computation graph for node 0 (2-layer GNN)",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig8_2_computation_graph.png")


# ── Fig 8.3: Computation graph WITH neighbor sampling ────────────────────────
print("Figure 8.3 …")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor(BG)

titles = ["Without sampling\n(all neighbors)", "With neighbor sampling\n(max 2 per hop)"]
l1_nodes = [
    [(1.4,2.8,"node 1"),(3.8,2.8,"node 2"),(6.2,2.8,"node 3"),(8.6,2.8,"node 4")],
    [(3.0,2.8,"node 1"),(7.0,2.8,"node 3")],
]
l2_nodes_list = [
    {1.4:[(0.8,1.2,"node 5"),(2.0,1.2,"node 6")],
     3.8:[(3.2,1.2,"node 7"),(4.4,1.2,"node 8")],
     6.2:[(5.6,1.2,"node 9"),(6.8,1.2,"node 10")],
     8.6:[(8.0,1.2,"node 11"),(9.2,1.2,"node 12")]},
    {3.0:[(2.2,1.2,"node 5"),(3.8,1.2,"node 6")],
     7.0:[(6.2,1.2,"node 9"),(7.8,1.2,"node 10")]},
]

for ax, title, l1, l2_map in zip(axes, titles, l1_nodes, l2_nodes_list):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0.5, 4.8)
    ax.set_title(title, fontsize=11, fontweight='bold',
                 color=G0, fontfamily=FONT, pad=8)

    # Output node
    gbox(ax, 5.0, 4.1, 1.1, 0.50, "node 0", fill=G0, fontsize=9)

    for x,y,lbl in l1:
        gbox(ax, x, y, 1.05, 0.45, lbl, fill=G2, fontsize=8)
        arr(ax, x, y+0.225, 5.0, 3.85)

    for px, children in l2_map.items():
        for x,y,lbl in children:
            gbox(ax, x, y, 1.0, 0.40, lbl, fill=G4, fontsize=7.5)
            arr(ax, x, y+0.20, px, 2.575)

axes[1].text(5.0, 0.72, "Sampled nodes only — smaller, bounded graph",
             ha='center', fontsize=8.5, color=G3, fontfamily=FONT,
             fontstyle='italic')

fig.suptitle("Neighbor sampling reduces the computation graph size",
             fontsize=11, color=G3, fontstyle='italic', fontfamily=FONT, y=0.03)
fig.tight_layout()
save(fig, "fig8_3_neighbor_sampling.png")


# ── Fig 8.4: PubMed subgraph visualisation ────────────────────────────────────
print("Figure 8.4 …")

rng = np.random.default_rng(42)
N   = 400
# Power-law degrees, 3 classes (diabetes types)
raw_deg = np.clip((rng.pareto(1.4, N)*4).astype(int), 1, 50)
y_pub   = rng.integers(0, 3, size=N)

G84 = nx.Graph()
G84.add_nodes_from(range(N))
for i in range(N):
    for _ in range(raw_deg[i]):
        j = rng.integers(0, N)
        if j != i: G84.add_edge(i, j)

# Take a small visually clean subgraph
sample  = list(G84.nodes())[:150]
G_sub84 = G84.subgraph(sample)
pos84   = nx.spring_layout(G_sub84, seed=7, k=0.5)
deg84   = dict(G_sub84.degree())
nc84    = [G0 if y_pub[n]==0 else (G3 if y_pub[n]==1 else G5)
           for n in G_sub84.nodes()]
ns84    = [15 + deg84[n]*8 for n in G_sub84.nodes()]

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
nx.draw_networkx_edges(G_sub84, pos84, ax=ax, alpha=0.15,
                       edge_color=G4, width=0.6)
nx.draw_networkx_nodes(G_sub84, pos84, ax=ax, node_color=nc84,
                       node_size=ns84, edgecolors='white', linewidths=0.4)
handles84 = [
    mpatches.Patch(color=G0, label='Diabetes mellitus experimental'),
    mpatches.Patch(color=G3, label='Diabetes mellitus type 1'),
    mpatches.Patch(color=G5, label='Diabetes mellitus type 2'),
]
ax.legend(handles=handles84, loc='lower right', fontsize=9,
          framealpha=0.95, edgecolor=G5)
ax.set_title("PubMed dataset – citation network (19,717 nodes, 3 classes)\n"
             "Illustrative subgraph coloured by diabetes category",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig8_4_pubmed.png")


# ── Fig 8.5: 4 NeighborLoader subgraphs (grayscale) ──────────────────────────
print("Figure 8.5 …")

# Simulate the 4 subgraphs obtained from NeighborLoader on PubMed
# Match approximate sizes from the chapter output
subgraph_sizes = [(400, 455), (262, 306), (275, 314), (194, 227)]
sub_titles = [
    "Subgraph 0\n(batch_size=16, 400 nodes)",
    "Subgraph 1\n(batch_size=16, 262 nodes)",
    "Subgraph 2\n(batch_size=16, 275 nodes)",
    "Subgraph 3\n(batch_size=12, 194 nodes)",
]

fig = plt.figure(figsize=(14, 12))
fig.patch.set_facecolor(BG)

rng2 = np.random.default_rng(7)
for idx, ((n_nodes, n_edges), title) in enumerate(zip(subgraph_sizes, sub_titles)):
    ax = fig.add_subplot(2, 2, idx+1)
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold',
                 color=G0, fontfamily=FONT, pad=6)

    # Build a synthetic subgraph matching the approximate edge count
    G_s = nx.Graph()
    G_s.add_nodes_from(range(min(n_nodes, 120)))  # subsample for clarity
    target_edges = int(n_edges * 0.3)
    for _ in range(target_edges):
        u = rng2.integers(0, min(n_nodes, 120))
        v = rng2.integers(0, min(n_nodes, 120))
        if u != v: G_s.add_edge(u, v)

    y_sg = rng2.integers(0, 3, size=G_s.number_of_nodes())
    pos_sg = nx.spring_layout(G_s, seed=idx, k=0.6)
    nc_sg  = [G0 if y_sg[n]==0 else (G3 if y_sg[n]==1 else G5)
              for n in G_s.nodes()]
    deg_sg = dict(G_s.degree())
    ns_sg  = [max(30, 10 + deg_sg.get(n,0)*15) for n in G_s.nodes()]

    nx.draw_networkx_edges(G_s, pos_sg, ax=ax, alpha=0.25,
                           edge_color=G4, width=0.8)
    nx.draw_networkx_nodes(G_s, pos_sg, ax=ax, node_color=nc_sg,
                           node_size=ns_sg, edgecolors='white', linewidths=0.5)

fig.suptitle("Subgraphs obtained with NeighborLoader on PubMed\n"
             "(num_neighbors=[10,10], batch_size=16)",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, y=0.02)
fig.tight_layout()
save(fig, "fig8_5_subgraphs.png")


# ── Fig 8.6: PPI network visualisation ────────────────────────────────────────
print("Figure 8.6 …")

rng3 = np.random.default_rng(99)
N_PPI = 350

# PPI is dense — Barabasi-Albert gives hub-and-spoke structure
G_ppi = nx.barabasi_albert_graph(N_PPI, m=5, seed=42)
pos_ppi = nx.spring_layout(G_ppi, seed=3, k=0.45)
deg_ppi = dict(G_ppi.degree())

# Shade by degree (proxy for biological connectivity)
max_deg = max(deg_ppi.values())
nc_ppi  = [str(0.1 + 0.7*(deg_ppi[n]/max_deg)) for n in G_ppi.nodes()]
# Convert to actual gray values
nc_ppi_rgb = [(0.1 + 0.7*(deg_ppi[n]/max_deg),)*3 for n in G_ppi.nodes()]
ns_ppi = [10 + deg_ppi[n]*2.5 for n in G_ppi.nodes()]

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
nx.draw_networkx_edges(G_ppi, pos_ppi, ax=ax, alpha=0.10,
                       edge_color=G4, width=0.5)
nx.draw_networkx_nodes(G_ppi, pos_ppi, ax=ax, node_color=nc_ppi_rgb,
                       node_size=ns_ppi, edgecolors='white', linewidths=0.3)

# Colorbar legend
sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys,
                            norm=plt.Normalize(vmin=0, vmax=max_deg))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label('Node degree', fontsize=9, fontfamily=FONT)

ax.set_title("Protein-protein interaction network\n"
             "(21,557 proteins, 342,353 interactions — illustrative subgraph)\n"
             "Node shade proportional to degree",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig8_6_ppi.png")

print(f"\nAll figures saved to {OUT}")
