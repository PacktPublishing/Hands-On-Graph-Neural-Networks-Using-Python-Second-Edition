"""
Generate grayscale versions of Chapter 5 figures 5.1 and 5.2.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import random, os

random.seed(42)
np.random.seed(42)

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

FONT = "DejaVu Sans"
BG   = "white"
GRAY_EDGE = "#595959"
BLACK = "#1A1A1A"

# 7 distinct grayscale tones for Cora
GRAY7 = ["#1A1A1A", "#404040", "#666666", "#8C8C8C", "#ABABAB", "#C8C8C8", "#E0E0E0"]
# 4 distinct grayscale tones for Facebook
GRAY4 = ["#1A1A1A", "#606060", "#A0A0A0", "#D0D0D0"]
# Hatching patterns to further distinguish classes
HATCH7 = ['', '///', '\\\\\\', '...', 'xxx', '+++', 'ooo']

def save(fig, name, dpi=200):
    fig.savefig(f"{OUT}/{name}", dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"  saved {name}")


# ── Fig 5.1: Cora-like citation network (grayscale) ──────────────────────────
print("Figure 5.1 (grayscale) …")

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
          framealpha=0.95, edgecolor='#EDEDED',
          title="Category (7 classes)", title_fontsize=9)
ax.set_title("Cora dataset – citation network (2,708 nodes, 7 classes)\n"
             "Illustrative subgraph coloured by research category",
             fontsize=12, fontweight='bold', color=BLACK, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig5_1_cora_gray.png")


# ── Fig 5.2: Facebook Page-Page-like social graph (grayscale) ─────────────────
print("Figure 5.2 (grayscale) …")

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
          framealpha=0.95, edgecolor='#EDEDED',
          title="Category (4 classes)", title_fontsize=9)
ax.set_title("Facebook Page-Page dataset – social network (22,470 nodes, 4 classes)\n"
             "Illustrative subgraph; node size proportional to degree",
             fontsize=12, fontweight='bold', color=BLACK, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig5_2_facebook_gray.png")

print(f"\nAll grayscale figures saved to {OUT}")
