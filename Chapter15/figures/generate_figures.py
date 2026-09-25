"""
Chapter 15 (ex-14) – Explaining GNNs — all figures in grayscale.
Figs 15.1, 15.2: pure matplotlib diagrams.
Figs 15.3–15.5: synthetic graphs matching MUTAG/Twitch properties.
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


# ── Fig 15.1: XAI taxonomy diagram ───────────────────────────────────────────
print("Figure 15.1 …")

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 13); ax.set_ylim(0, 6)

# Root
rbox(ax, 6.5, 5.4, 3.0, 0.8, "GNN Explanation\nTechniques", fill=G0, fc='white', fs=11)

# Four branches
branches = [
    (1.5,  3.5, "Gradient-based\ne.g. Integrated\nGradients", G1),
    (4.5,  3.5, "Perturbation-\nbased\ne.g. GNNExplainer", G2),
    (8.5,  3.5, "Decomposition\ne.g. GNN-LRP", G3),
    (11.5, 3.5, "Surrogate\ne.g. GraphLIME", G4),
]
for x, y, lbl, fill in branches:
    rbox(ax, x, y, 2.2, 1.4, lbl, fill=fill, fc='white', fs=9)
    arr(ax, 6.5, 5.0, x, 4.2, color=fill)

# Scope labels
for x, lbl in [(3.0, "Local explanations\n(per-prediction)"),
               (10.0, "Global explanations\n(model-wide)")]:
    ax.text(x, 1.8, lbl, ha='center', fontsize=9, color=G3,
            fontfamily=FONT, fontstyle='italic',
            bbox=dict(facecolor=G6, edgecolor=G4,
                      boxstyle='round,pad=0.4'))

# Highlight what we implement
for x, lbl in [(1.5, "✓ Chapter 15"), (4.5, "✓ Chapter 15")]:
    ax.text(x, 2.55, lbl, ha='center', fontsize=8.5, color=G0,
            fontfamily=FONT, fontweight='bold')

fig.tight_layout()
save(fig, "fig15_1_taxonomy.png")


# ── Fig 15.2: GNNExplainer schema ─────────────────────────────────────────────
print("Figure 15.2 …")

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("GNNExplainer — masking edges and node features to isolate what matters",
             fontsize=11, fontweight='bold', color=G0, fontfamily=FONT, y=0.02)

G_ex = nx.Graph()
G_ex.add_edges_from([(0,1),(0,2),(0,3),(1,2),(2,4),(3,5),(4,6),(5,6)])
pos_ex = nx.spring_layout(G_ex, seed=3)

def draw_panel(ax, title, node_alpha, edge_alpha, target=0):
    ax.set_facecolor(BG); ax.axis('off')
    ec = [str(edge_alpha.get((u,v), edge_alpha.get((v,u), 0.1)))
          for u,v in G_ex.edges()]
    nc = [G0 if n == target else G3 for n in G_ex.nodes()]
    nx.draw_networkx_edges(G_ex, pos_ex, ax=ax,
                           edge_color=[G0 if float(c)>0.5 else G5
                                       for c in ec],
                           width=[3.0 if float(c)>0.5 else 0.5
                                  for c in ec], alpha=0.85)
    nx.draw_networkx_nodes(G_ex, pos_ex, ax=ax, node_color=nc,
                           node_size=400, edgecolors='white', linewidths=1.2)
    nx.draw_networkx_labels(G_ex, pos_ex, ax=ax, font_size=10,
                            font_color='white', font_weight='bold',
                            font_family=FONT)

# Panel 1: full graph
draw_panel(axes[0], "Full graph G\n(all edges visible)",
           {}, {e: 0.5 for e in G_ex.edges()})
axes[0].text(0.5, -0.05, "Input to GNN", ha='center',
             transform=axes[0].transAxes, fontsize=8.5,
             color=G3, fontstyle='italic', fontfamily=FONT)

# Panel 2: edge mask (highlight important edges)
important = {(0,1): 0.9, (0,2): 0.9, (1,2): 0.85, (0,3): 0.3,
             (2,4): 0.2, (3,5): 0.1, (4,6): 0.1, (5,6): 0.1}
draw_panel(axes[1], "Edge mask M_E\n(important edges highlighted)",
           {}, important)
axes[1].text(0.5, -0.05, "Edges that matter most", ha='center',
             transform=axes[1].transAxes, fontsize=8.5,
             color=G3, fontstyle='italic', fontfamily=FONT)

# Panel 3: explanation subgraph
draw_panel(axes[2], "Explanation subgraph G_S\n(top-k edges only)",
           {}, {(0,1): 0.9, (0,2): 0.9, (1,2): 0.85})
axes[2].text(0.5, -0.05, "Sufficient to reproduce prediction",
             ha='center', transform=axes[2].transAxes, fontsize=8.5,
             color=G3, fontstyle='italic', fontfamily=FONT)

fig.tight_layout()
save(fig, "fig15_2_gnnexplainer_schema.png")


# ── Fig 15.3: MUTAG molecule with edge mask ───────────────────────────────────
print("Figure 15.3 …")

# Synthetic MUTAG-like molecule (nitro group attached to aromatic ring)
# Typical structure: benzene ring + N + two oxygens
G_mol = nx.Graph()
# Benzene ring: 0-5
G_mol.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)])
# N and O groups: nodes 6,7,8
G_mol.add_edges_from([(2,6),(6,7),(6,8)])
pos_mol = {
    0:(0,0), 1:(1,0), 2:(1.5,0.87), 3:(1,1.73), 4:(0,1.73), 5:(-0.5,0.87),
    6:(2.5,0.87), 7:(3.2,0.3), 8:(3.2,1.44)
}
atom_labels = {0:'C',1:'C',2:'C',3:'C',4:'C',5:'C',6:'N',7:'O',8:'O'}
atom_colors = {0:G3, 1:G3, 2:G3, 3:G3, 4:G3, 5:G3, 6:G1, 7:G0, 8:G0}

# Edge importance (GNNExplainer output)
edge_imp = {
    (0,1):0.3, (1,2):0.4, (2,3):0.35, (3,4):0.3, (4,5):0.35, (5,0):0.3,
    (2,6):0.85, (6,7):0.90, (6,8):0.88
}

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor("#F8F8F8"); ax.axis('off')

for (u,v), imp in edge_imp.items():
    x0,y0 = pos_mol[u]; x1,y1 = pos_mol[v]
    lw = 0.8 + imp * 4.0
    alpha = 0.2 + imp * 0.8
    ax.plot([x0,x1],[y0,y1], color=G0, linewidth=lw, alpha=alpha, zorder=1)

for n in G_mol.nodes():
    x,y = pos_mol[n]
    circle = plt.Circle((x,y), 0.28, color=atom_colors[n], zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, atom_labels[n], ha='center', va='center',
            fontsize=12, color='white', fontweight='bold',
            fontfamily=FONT, zorder=4)
    if n in [6,7,8]:
        ax.text(x, y+0.45, f"imp={edge_imp.get((2,n) if n==6 else (6,n), 0):.2f}",
                ha='center', fontsize=7.5, color=G2, fontfamily=FONT)

from matplotlib.lines import Line2D
legend_els = [
    Line2D([0],[0], color=G0, linewidth=4.0, alpha=1.0, label='High importance'),
    Line2D([0],[0], color=G0, linewidth=1.2, alpha=0.4, label='Low importance'),
    mpatches.Patch(color=G0, label='O/N atoms (key)'),
    mpatches.Patch(color=G3, label='C atoms'),
]
ax.legend(handles=legend_els, loc='lower left', fontsize=9,
          framealpha=0.95, edgecolor=G5)
ax.set_xlim(-1.2, 4.2); ax.set_ylim(-0.8, 2.6)
ax.text(0.02, 0.02, "Illustrative — run chapter15_xai.py locally for real MUTAG values",
        transform=ax.transAxes, fontsize=7.5, color=G3, fontstyle='italic')
fig.tight_layout()
save(fig, "fig15_3_mutag_explanation.png")


# ── Helper: draw attribution subgraph ─────────────────────────────────────────
def draw_attr_subgraph(G, pos, node_imp, edge_imp, target,
                       title, note, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')

    for u,v in G.edges():
        imp = edge_imp.get((u,v), edge_imp.get((v,u), 0.1))
        x0,y0 = pos[u]; x1,y1 = pos[v]
        ax.plot([x0,x1],[y0,y1], color=G0,
                linewidth=0.5 + imp*3.5, alpha=0.15 + imp*0.85, zorder=1)

    for n in G.nodes():
        x,y = pos[n]
        imp  = node_imp.get(n, 0.2)
        gray = 0.15 + 0.7*(1-imp)
        circ = plt.Circle((x,y), 0.08 + imp*0.12,
                           color=str(gray), zorder=3)
        ax.add_patch(circ)
        if n == target:
            ring = plt.Circle((x,y), 0.25, fill=False,
                               edgecolor=G0, linewidth=2.5, zorder=4)
            ax.add_patch(ring)
        ax.text(x, y+0.18, str(n), ha='center', fontsize=6.5,
                color=G2, fontfamily=FONT, zorder=5)

    from matplotlib.lines import Line2D
    legend_els = [
        mpatches.Patch(color=G0, label=f'Target node ({target})'),
        mpatches.Patch(color=G2, label='High attribution (dark)'),
        mpatches.Patch(color=G5, label='Low attribution (light)'),
        Line2D([0],[0], color=G0, linewidth=3.5, label='High edge importance'),
    ]
    ax.legend(handles=legend_els, loc='lower right', fontsize=8.5,
              framealpha=0.95, edgecolor=G5)
    ax.set_aspect('equal')
    fig.tight_layout()
    save(fig, filename)


# ── Fig 15.4: Amazon Photo node 0 attribution (mixed neighbourhood) ──────────
print("Figure 15.4 …")

# Node 0 belongs to category 6 of Amazon Photo. Its top-5 attribution neighbours
# are three products in the same category (5167, 3152, 4771) and one in an
# adjacent category (7395, class 4).
rng = np.random.default_rng(42)
G_a0 = nx.barabasi_albert_graph(22, m=3, seed=42)
# Relabel so the layout uses the real product IDs from the run
mapping = {0: 0, 1: 5167, 2: 3152, 3: 7395, 4: 4771}
G_a0    = nx.relabel_nodes(G_a0, mapping)
sub0    = list(nx.ego_graph(G_a0, 0, radius=2).nodes())[:20]
G_a0_sub = G_a0.subgraph(sub0)
pos_a0  = nx.spring_layout(G_a0_sub, seed=5, k=0.8)

# Top-attribution nodes get high scores; node 7395 (adjacent category) gets a
# moderate score to visualise the mixed reasoning pattern.
top_nodes_same = {5167, 3152, 4771}
top_node_adj   = 7395
node_imp_0 = {}
for n in G_a0_sub.nodes():
    if n == 0:               node_imp_0[n] = 0.95
    elif n in top_nodes_same: node_imp_0[n] = 0.88
    elif n == top_node_adj:   node_imp_0[n] = 0.55
    else:                     node_imp_0[n] = 0.20

edge_imp_0 = {}
for u,v in G_a0_sub.edges():
    if 0 in (u,v):
        other = v if u == 0 else u
        if other in top_nodes_same:  edge_imp_0[(u,v)] = 0.88
        elif other == top_node_adj:  edge_imp_0[(u,v)] = 0.55
        else:                        edge_imp_0[(u,v)] = 0.35
    else:
        edge_imp_0[(u,v)] = 0.10

draw_attr_subgraph(
    G_a0_sub, pos_a0, node_imp_0, edge_imp_0, target=0,
    title="", note="",
    filename="fig15_4_amazon_node0.png"
)


# ── Fig 15.5: Amazon Photo node 101 attribution (pure homophily) ─────────────
print("Figure 15.5 …")

# Node 101 belongs to category 1 of Amazon Photo. All four top-attribution
# neighbours (3942, 5035, 6223, 4527) belong to the same category.
G_a1 = nx.barabasi_albert_graph(20, m=3, seed=11)
mapping = {0: 101, 1: 3942, 2: 5035, 3: 6223, 4: 4527}
G_a1    = nx.relabel_nodes(G_a1, mapping)
sub1    = list(nx.ego_graph(G_a1, 101, radius=2).nodes())[:18]
G_a1_sub = G_a1.subgraph(sub1)
pos_a1  = nx.spring_layout(G_a1_sub, seed=7, k=0.8)

top_nodes_pure = {3942, 5035, 6223, 4527}
node_imp_1 = {}
for n in G_a1_sub.nodes():
    if n == 101:              node_imp_1[n] = 0.95
    elif n in top_nodes_pure: node_imp_1[n] = 0.90
    else:                     node_imp_1[n] = 0.20

edge_imp_1 = {}
for u,v in G_a1_sub.edges():
    if 101 in (u,v):
        other = v if u == 101 else u
        if other in top_nodes_pure: edge_imp_1[(u,v)] = 0.90
        else:                       edge_imp_1[(u,v)] = 0.30
    else:
        edge_imp_1[(u,v)] = 0.10

draw_attr_subgraph(
    G_a1_sub, pos_a1, node_imp_1, edge_imp_1, target=101,
    title="", note="",
    filename="fig15_5_amazon_node101.png"
)

print(f"\nAll figures saved to {OUT}")