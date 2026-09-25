import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import networkx as nx
import random, os

random.seed(0)
np.random.seed(0)

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

FONT  = "DejaVu Sans"
BG    = "white"

# Shared grayscale palette (same values as chapters 06+).
G0 = "#111111"; G1 = "#333333"; G2 = "#555555"
G3 = "#777777"; G4 = "#999999"; G5 = "#BBBBBB"; G6 = "#DDDDDD"

# Ordered gray ramps for categories (same values as Chapter05's grayscale figures).
GRAY7 = ["#1A1A1A", "#404040", "#666666", "#8C8C8C", "#ABABAB", "#C8C8C8", "#E0E0E0"]
GRAY4 = ["#1A1A1A", "#606060", "#A0A0A0", "#D0D0D0"]

# Semantic roles. The node2vec categories are ranked by the distance d(t,x),
# so they take an ordered ramp: d=0 darkest → d=2 lightest.
FOCUS  = GRAY7[0]   # source node A / current node v  (was orange)
RETURN = GRAY7[1]   # d(t,x)=0, alpha = 1/p          (was red)
NEAR   = GRAY7[2]   # d(t,x)=1, alpha = 1            (was blue)
FAR    = GRAY7[3]   # d(t,x)=2, alpha = 1/q          (was green)
PREV   = GRAY7[4]   # previous node t                (was mid gray)

INK     = G0   # titles and label text
MUTED   = G2   # secondary text and the walk-direction arrow
EDGE_HL = G2   # emphasised edges
FILL_BG = G5   # unhighlighted node fill
EDGE_BG = G6   # background edges and legend frames

# Karate club factions: two well-separated tones, as in the grayscale variant.
KARATE_DARK  = GRAY7[1]   # Mr. Hi's faction  (was blue)
KARATE_LIGHT = GRAY4[2]   # Officer's faction (was red)

def save(fig, name, dpi=200):
    fig.savefig(f"{OUT}/{name}", dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"  saved {name}")

# ── Fig 4.1: BFS vs DFS neighbourhood ────────────────────────────────────────
print("Figure 4.1 …")
nodes = ['A','B','C','D','E','F','G']
edges = [('A','B'),('A','C'),('B','D'),('B','E'),('C','F'),('F','G')]
G41 = nx.Graph(); G41.add_nodes_from(nodes); G41.add_edges_from(edges)
pos41 = {'A':(0,0),'B':(-0.8,-0.9),'C':(0.8,-0.9),
          'D':(-1.4,-1.8),'E':(-0.2,-1.8),'F':(1.4,-1.8),'G':(1.8,-2.7)}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.patch.set_facecolor(BG)

for ax, title, highlight in zip(
    axes,
    ["BFS neighbourhood\nN(A) = {B, C}", "DFS neighbourhood\nN(A) = {B, D, E}"],
    [['A','B','C'], ['A','B','D','E']]
):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', color=INK,
                 pad=10, fontfamily=FONT)
    nc = [FOCUS if n=='A' else (NEAR if n in highlight else FILL_BG) for n in G41.nodes()]
    ns = [900 if n=='A' else (700 if n in highlight else 500) for n in G41.nodes()]
    ec = [EDGE_HL if (u in highlight and v in highlight) else EDGE_BG for u,v in G41.edges()]
    ew = [2.5 if (u in highlight and v in highlight) else 1.0 for u,v in G41.edges()]
    nx.draw_networkx_edges(G41, pos41, ax=ax, edge_color=ec, width=ew)
    nx.draw_networkx_nodes(G41, pos41, ax=ax, node_color=nc, node_size=ns,
                           edgecolors='white', linewidths=1.5)
    nx.draw_networkx_labels(G41, pos41, ax=ax, font_size=11, font_color='white',
                            font_weight='bold', font_family=FONT)
    ax.legend(handles=[
        mpatches.Patch(color=FOCUS,   label='Source node A'),
        mpatches.Patch(color=NEAR,    label='Neighbourhood N(A)'),
        mpatches.Patch(color=FILL_BG, label='Other nodes'),
    ], loc='lower right', fontsize=9, framealpha=0.9, edgecolor=EDGE_BG)

fig.tight_layout()
save(fig, "fig4_1_neighborhood.png")

# ── Fig 4.2: p/q transition probability setup ─────────────────────────────────
print("Figure 4.2 …")
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 3.0)

node_pos = {'t':(0.8,1.5),'v':(2.2,1.5),'x1':(3.6,2.6),'x2':(3.6,1.5),'x3':(3.6,0.4)}
node_labels = {'t':'t\n(previous)','v':'v\n(current)',
               'x1':'x₁\n(d=0)','x2':'x₂\n(d=1)','x3':'x₃\n(d=2)'}
node_colors = {'t':PREV,'v':FOCUS,'x1':RETURN,'x2':NEAR,'x3':FAR}

for u, e in [('t','v'),('v','x1'),('v','x2'),('v','x3'),('t','x2')]:
    x1c,y1c = node_pos[u]; x2c,y2c = node_pos[e]
    lc = EDGE_HL if 'v' in (u,e) else EDGE_BG
    ax.plot([x1c,x2c],[y1c,y2c], color=lc, lw=1.8, zorder=1, alpha=0.7)

for name,(x,y) in node_pos.items():
    ax.add_patch(plt.Circle((x,y), 0.28, color=node_colors[name], zorder=3))
    lines = node_labels[name].split('\n')
    ax.text(x,y+0.07, lines[0], ha='center', va='center', fontsize=11,
            color='white', fontweight='bold', fontfamily=FONT, zorder=4)
    ax.text(x,y-0.12, lines[1], ha='center', va='center', fontsize=8,
            color='white', fontfamily=FONT, zorder=4)

for xname,(lbl,clr) in [('x1',('1/p',RETURN)),('x2',('1',NEAR)),('x3',('1/q',FAR))]:
    xp,yp = node_pos[xname]; vp = node_pos['v']
    mx,my = (xp+vp[0])/2, (yp+vp[1])/2
    oy = 0.18 if yp>vp[1] else (-0.18 if yp<vp[1] else 0)
    ax.text(mx+0.18, my+oy, f'α = {lbl}', ha='center', va='center',
            fontsize=10, color=INK, fontweight='bold', fontfamily=FONT,
            bbox=dict(facecolor='white', edgecolor=clr,
                      boxstyle='round,pad=0.2', linewidth=1.2))

ax.annotate("", xy=node_pos['v'],
            xytext=(node_pos['t'][0]+0.3, node_pos['t'][1]),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.0, mutation_scale=18))
ax.text(1.5,1.78,"walk direction", ha='center', fontsize=9,
        color=MUTED, fontstyle='italic', fontfamily=FONT)
ax.set_title("Node2Vec transition probabilities from current node v\n"
             "based on distance to previous node t\n"
             "d(t,x): shortest-path distance between previous node t and candidate x",
             fontsize=11, fontweight='bold', color=INK, pad=10, fontfamily=FONT)
ax.legend(handles=[
    mpatches.Patch(color=RETURN, label='d(t,x)=0  →  α = 1/p  (return)'),
    mpatches.Patch(color=NEAR,   label='d(t,x)=1  →  α = 1    (same distance)'),
    mpatches.Patch(color=FAR,    label='d(t,x)=2  →  α = 1/q  (explore)'),
], loc='lower left', fontsize=9, framealpha=0.95, edgecolor=EDGE_BG)
fig.tight_layout()
save(fig, "fig4_2_transition_probs.png")

# ── Fig 4.3: Concrete probability example ─────────────────────────────────────
print("Figure 4.3 …")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG)

G43 = nx.Graph()
G43.add_edges_from([('t','v'),('v','A'),('v','B'),('v','C'),('t','B')])
pos43 = {'t':(0,0),'v':(2,0),'A':(4,1.2),'B':(4,0),'C':(4,-1.2)}

for ax, (p, q, title) in zip(axes, [
    (1,   1,   'p=1, q=1  (DeepWalk)'),
    (0.5, 2.0, 'p=0.5, q=2  (prefer return, stay local)'),
]):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', color=INK,
                 pad=8, fontfamily=FONT)
    raw   = {'t':1/p, 'A':1/q, 'B':1.0, 'C':1/q}
    total = sum(raw.values())
    probs = {k:v/total for k,v in raw.items()}

    for u, e in G43.edges():
        x1c,y1c=pos43[u]; x2c,y2c=pos43[e]
        ax.plot([x1c,x2c],[y1c,y2c], color=EDGE_BG, lw=1.5, zorder=1)

    tgt_colors = {'t':RETURN,'A':FAR,'B':NEAR,'C':FAR}
    for nbr in ['t','A','B','C']:
        xv,yv=pos43['v']; xn,yn=pos43[nbr]
        clr = tgt_colors[nbr]
        ax.annotate("", xy=(xn,yn), xytext=(xv,yv),
                    arrowprops=dict(arrowstyle="-|>", color=clr,
                                   lw=1.5+probs[nbr]*4, mutation_scale=14, alpha=0.85))
        mx,my=(xv+xn)/2,(yv+yn)/2
        oy = 0.22 if yn>yv else (-0.22 if yn<yv else 0.22)
        ax.text(mx+0.15, my+oy, f'{probs[nbr]:.3g}', ha='center', va='center',
                fontsize=10, color=INK, fontweight='bold', fontfamily=FONT,
                bbox=dict(facecolor='white', edgecolor=clr,
                          boxstyle='round,pad=0.15', linewidth=1))

    ncolors={'t':PREV,'v':FOCUS,'A':FAR,'B':NEAR,'C':FAR}
    for n,(x,y) in pos43.items():
        ax.add_patch(plt.Circle((x,y),0.25,color=ncolors[n],zorder=3))
        ax.text(x,y, n, ha='center', va='center', fontsize=12,
                color='white', fontweight='bold', fontfamily=FONT, zorder=4)

    ax.set_xlim(-0.7,5.2); ax.set_ylim(-2.0,2.0)
    ax.text(0,-1.7,'previous (t)',ha='center',fontsize=8,color=MUTED,fontfamily=FONT)
    ax.text(2,-1.7,'current (v)', ha='center',fontsize=8,color=FOCUS,fontfamily=FONT)

fig.tight_layout()
save(fig, "fig4_3_prob_example.png")

# ── Fig 4.4: Zachary's Karate Club ────────────────────────────────────────────
print("Figure 4.4 …")
G44  = nx.karate_club_graph()
pos44 = nx.spring_layout(G44, seed=0)
labels44 = [1 if G44.nodes[n]['club']=='Officer' else 0 for n in G44.nodes]
colors44  = [KARATE_DARK if l==0 else KARATE_LIGHT for l in labels44]
fig, ax = plt.subplots(figsize=(8,7))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
nx.draw_networkx_edges(G44, pos44, ax=ax, edge_color=EDGE_BG, width=1.2, alpha=0.9)
nx.draw_networkx_nodes(G44, pos44, ax=ax, node_size=600, node_color=colors44,
                       edgecolors='white', linewidths=1.5)
nx.draw_networkx_labels(G44, pos44, ax=ax, font_size=9, font_color='white',
                        font_weight='bold', font_family=FONT)
ax.legend(handles=[mpatches.Patch(color=KARATE_DARK,  label="Mr. Hi's faction"),
                   mpatches.Patch(color=KARATE_LIGHT, label="Officer's faction")],
          loc='lower right', fontsize=10, framealpha=0.9, edgecolor=EDGE_BG)
fig.tight_layout()
save(fig, "fig4_4_karate_club.png")

print("Figs 4.1-4.4 done.")