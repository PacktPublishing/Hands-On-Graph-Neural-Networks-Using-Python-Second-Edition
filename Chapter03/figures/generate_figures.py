import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import random, os

random.seed(0)
np.random.seed(0)

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
FONT   = "DejaVu Sans"
BG     = "white"
BLUE   = "#2E75B6"
LBLUE  = "#BDD7EE"
ORANGE = "#C55A11"
LORNG  = "#F4B183"
GRAY   = "#595959"
LGRAY  = "#EDEDED"
BLACK  = "#1A1A1A"
RED    = "#C0392B"
LRED   = "#F1948A"

def save(fig, name, dpi=200):
    fig.savefig(f"{OUT}/{name}", dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"  saved {name}")

def box(ax, x, y, w, h, label, color, fontsize=10, textcolor="white", radius=0.04):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=f"round,pad=0.02,rounding_size={radius}",
                           facecolor=color, edgecolor="none", zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight='bold', zorder=4, fontfamily=FONT)

def arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.4, mutation_scale=14), zorder=2)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3.1 – CBOW vs Skip-gram
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 3.1 …")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.patch.set_facecolor(BG)

for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

# CBOW  (left)
ax = axes[0]
ax.set_title("CBOW", fontsize=13, fontweight='bold', color=BLACK, pad=10, fontfamily=FONT)
ctx_words = ["w(t-2)", "w(t-1)", "w(t+1)", "w(t+2)"]
ys = [0.82, 0.66, 0.34, 0.18]
for lbl, y in zip(ctx_words, ys):
    box(ax, 0.18, y, 0.28, 0.10, lbl, BLUE, fontsize=9)
# sum layer
box(ax, 0.50, 0.50, 0.18, 0.12, "Sum", ORANGE, fontsize=9)
for y in ys:
    arrow(ax, 0.33, y, 0.41, 0.50)
# output
box(ax, 0.82, 0.50, 0.28, 0.10, "w(t)", BLUE, fontsize=9)
arrow(ax, 0.59, 0.50, 0.68, 0.50)
ax.text(0.18, 0.94, "INPUT", ha='center', fontsize=8, color=GRAY, fontfamily=FONT)
ax.text(0.82, 0.94, "OUTPUT", ha='center', fontsize=8, color=GRAY, fontfamily=FONT)

# Skip-gram (right)
ax = axes[1]
ax.set_title("Skip-gram", fontsize=13, fontweight='bold', color=BLACK, pad=10, fontfamily=FONT)
box(ax, 0.18, 0.50, 0.28, 0.10, "w(t)", ORANGE, fontsize=9)
for lbl, y in zip(ctx_words, ys):
    box(ax, 0.82, y, 0.28, 0.10, lbl, BLUE, fontsize=9)
    arrow(ax, 0.33, 0.50, 0.68, y)
ax.text(0.18, 0.94, "INPUT", ha='center', fontsize=8, color=GRAY, fontfamily=FONT)
ax.text(0.82, 0.94, "OUTPUT", ha='center', fontsize=8, color=GRAY, fontfamily=FONT)

fig.tight_layout()
save(fig, "fig3_1_cbow_skipgram.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3.2 – Text to skip-grams
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 3.2 …")
words   = ["the", "quick", "brown", "fox", "jumps"]
target  = 2   # "brown"
ctx_sz  = 2

fig, ax = plt.subplots(figsize=(10, 3.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.axis('off')
ax.set_xlim(-0.5, len(words) - 0.5)
ax.set_ylim(-0.5, 1.8)

# top row: sentence words
for i, w in enumerate(words):
    is_target  = (i == target)
    is_context = (abs(i - target) <= ctx_sz and i != target)
    fc = ORANGE if is_target else (LBLUE if is_context else LGRAY)
    tc = "white" if is_target else (BLACK if is_context else GRAY)
    rect = FancyBboxPatch((i - 0.42, 0.85), 0.84, 0.42,
                           boxstyle="round,pad=0.02,rounding_size=0.04",
                           facecolor=fc, edgecolor="none")
    ax.add_patch(rect)
    ax.text(i, 1.06, w, ha='center', va='center', fontsize=11,
            color=tc, fontweight='bold' if is_target else 'normal', fontfamily=FONT)

# context size bracket
ax.annotate("", xy=(target - ctx_sz - 0.45, 0.78),
            xytext=(target + ctx_sz + 0.45, 0.78),
            arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.5))
ax.text(target, 0.68, f"context size = {ctx_sz}", ha='center',
        fontsize=9, color=BLUE, fontfamily=FONT)

# skip-gram pairs below
pairs = [(words[target], words[target + d])
         for d in range(-ctx_sz, ctx_sz + 1) if d != 0]
n = len(pairs)
xs = np.linspace(0.5, len(words) - 1.5, n)
for x, (tgt, ctx) in zip(xs, pairs):
    ax.text(x, 0.38, f"({tgt},", ha='right', fontsize=10,
            color=ORANGE, fontweight='bold', fontfamily=FONT)
    ax.text(x + 0.05, 0.38, f"{ctx})", ha='left', fontsize=10,
            color=BLUE, fontweight='bold', fontfamily=FONT)
    ax.plot([target, x + 0.02], [0.85, 0.50], color=GRAY,
            lw=0.8, linestyle='--', alpha=0.5)

ax.text(len(words)/2 - 0.5, 0.08,
        "Skip-gram pairs   (target, context)",
        ha='center', fontsize=9, color=GRAY, fontstyle='italic', fontfamily=FONT)

fig.tight_layout()
save(fig, "fig3_2_skipgrams.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3.3 – Word2Vec architecture
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 3.3 …")
fig, ax = plt.subplots(figsize=(11, 4.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 4)

# columns: input | W_embed | embedding | W_context | output
cols = [1.0, 2.8, 5.0, 7.2, 9.0]
labels_col = ["Input\n(one-hot)", "W_embed\n(V × d)", "Embedding\n(d-dim)", "W_context\n(d × V)", "Output\n(softmax)"]
colors_col = [LBLUE, BLUE, ORANGE, BLUE, LBLUE]
txt_colors = [BLACK, "white", "white", "white", BLACK]

for x, lbl, fc, tc in zip(cols, labels_col, colors_col, txt_colors):
    box(ax, x, 2.0, 1.2, 1.8, lbl, fc, fontsize=10, textcolor=tc)

for i in range(len(cols) - 1):
    arrow(ax, cols[i] + 0.6, 2.0, cols[i+1] - 0.6, 2.0)

# dimension labels
ax.text(cols[0], 0.5, "V", ha='center', fontsize=10, color=GRAY, fontfamily=FONT)
ax.text(cols[2], 0.5, "d", ha='center', fontsize=10, color=GRAY, fontfamily=FONT)
ax.text(cols[4], 0.5, "V", ha='center', fontsize=10, color=GRAY, fontfamily=FONT)
for x in [cols[0], cols[2], cols[4]]:
    ax.plot([x, x], [0.65, 0.95], color=LGRAY, lw=1)

ax.text(5.0, 3.7, "Word2Vec skip-gram architecture",
        ha='center', fontsize=12, fontweight='bold', color=BLACK, fontfamily=FONT)

fig.tight_layout()
save(fig, "fig3_3_word2vec_arch.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3.4 – Sentences as graphs
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 3.4 …")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.patch.set_facecolor(BG)

# LEFT: sentence
ax = axes[0]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title("Sentence", fontsize=12, fontweight='bold', color=BLACK, pad=8, fontfamily=FONT)
sent_words = ["graph", "neural", "network", "learning"]
xs = np.linspace(0.12, 0.88, len(sent_words))
y  = 0.55
for i, (x, w) in enumerate(zip(xs, sent_words)):
    box(ax, x, y, 0.17, 0.18, w, BLUE, fontsize=9)
    if i < len(sent_words) - 1:
        arrow(ax, x + 0.09, y, xs[i+1] - 0.09, y)
# window arrows
for i in range(len(sent_words) - 2):
    ax.annotate("", xy=(xs[i+2] - 0.085, y - 0.12),
                xytext=(xs[i] + 0.085, y - 0.12),
                arrowprops=dict(arrowstyle="<->", color=ORANGE,
                                lw=1.0, connectionstyle="arc3,rad=0.3"))
ax.text(0.50, 0.15, "context window", ha='center', fontsize=9,
        color=ORANGE, fontstyle='italic', fontfamily=FONT)

# RIGHT: graph / random walk
ax = axes[1]
ax.axis('off')
ax.set_title("Graph random walk", fontsize=12, fontweight='bold', color=BLACK, pad=8, fontfamily=FONT)

G_demo = nx.karate_club_graph()
pos    = nx.spring_layout(G_demo, seed=7)

# draw full graph faintly
nx.draw_networkx_edges(G_demo, pos, ax=ax, alpha=0.15, edge_color=GRAY, width=0.8)
nx.draw_networkx_nodes(G_demo, pos, ax=ax, node_size=80,
                       node_color=LGRAY, edgecolors=GRAY, linewidths=0.5)

# highlight a walk
walk_nodes = [0, 1, 2, 8, 33, 32, 28]
walk_edges = list(zip(walk_nodes[:-1], walk_nodes[1:]))
nx.draw_networkx_edges(G_demo, pos, edgelist=walk_edges, ax=ax,
                       edge_color=ORANGE, width=2.5, arrows=True,
                       arrowstyle='-|>', arrowsize=15)
nx.draw_networkx_nodes(G_demo, pos, nodelist=walk_nodes, ax=ax,
                       node_size=180, node_color=ORANGE, edgecolors="white", linewidths=1.2)
nx.draw_networkx_labels(G_demo, pos,
                        labels={n: str(n) for n in walk_nodes},
                        ax=ax, font_size=7, font_color="white", font_weight='bold')
ax.text(0.50, -0.05, "random walk = sequence of nodes",
        ha='center', fontsize=9, color=ORANGE,
        fontstyle='italic', fontfamily=FONT, transform=ax.transAxes)

fig.suptitle("Sentences and graphs share the same co-occurrence intuition",
             fontsize=11, color=GRAY, fontstyle='italic', fontfamily=FONT, y=0.02)
fig.tight_layout()
save(fig, "fig3_4_sentences_graphs.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3.5 – Random graph (Erdos-Renyi)
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 3.5 …")
random.seed(0); np.random.seed(0)
G5 = nx.erdos_renyi_graph(10, 0.3, seed=1, directed=False)
pos5 = nx.spring_layout(G5, seed=0)

fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')

nx.draw_networkx_edges(G5, pos5, ax=ax, edge_color=GRAY, width=1.5, alpha=0.7)
nx.draw_networkx_nodes(G5, pos5, ax=ax, node_size=700,
                       node_color=BLUE, edgecolors="white", linewidths=1.5)
nx.draw_networkx_labels(G5, pos5, ax=ax, font_size=12,
                        font_color="white", font_weight='bold', font_family=FONT)
fig.tight_layout()
save(fig, "fig3_5_random_graph.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3.6 – Zachary's Karate Club
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 3.6 …")
G6  = nx.karate_club_graph()
pos6 = nx.spring_layout(G6, seed=0)
labels6 = [1 if G6.nodes[n]['club'] == 'Officer' else 0 for n in G6.nodes]
colors6  = [BLUE if l == 0 else RED for l in labels6]

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')

nx.draw_networkx_edges(G6, pos6, ax=ax, edge_color=LGRAY, width=1.2, alpha=0.9)
nx.draw_networkx_nodes(G6, pos6, ax=ax, node_size=600,
                       node_color=colors6, edgecolors="white", linewidths=1.5)
nx.draw_networkx_labels(G6, pos6, ax=ax, font_size=9,
                        font_color="white", font_weight='bold', font_family=FONT)

patch0 = mpatches.Patch(color=BLUE, label="Mr. Hi's faction")
patch1 = mpatches.Patch(color=RED,  label="Officer's faction")
ax.legend(handles=[patch0, patch1], loc='lower right',
          fontsize=10, framealpha=0.9, edgecolor=LGRAY)
fig.tight_layout()
save(fig, "fig3_6_karate_club.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3.7 – t-SNE + UMAP of DeepWalk embeddings
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 3.7 — running DeepWalk embeddings …")
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import umap as umap_lib

random.seed(0); np.random.seed(0)

G7     = nx.karate_club_graph()
labels7 = np.array([1 if G7.nodes[n]['club'] == 'Officer' else 0 for n in G7.nodes])

def random_walk(G, start, length):
    walk = [str(start)]
    for _ in range(length):
        nbrs = list(G.neighbors(start))
        if not nbrs: break
        start = np.random.choice(nbrs)
        walk.append(str(start))
    return walk

walks = []
for node in G7.nodes:
    for _ in range(80):
        walks.append(random_walk(G7, node, 10))

model7 = Word2Vec(
    sentences   = walks,
    vector_size = 64,
    window      = 5,
    min_count   = 0,
    sg          = 1,
    hs          = 0,
    negative    = 5,
    workers     = 2,
    seed        = 0,
    epochs      = 30
)

nodes_wv = np.array([model7.wv[str(i)] for i in range(len(G7.nodes))])

tsne_proj = TSNE(n_components=2, init='pca',
                 learning_rate='auto', random_state=0,
                 perplexity=10).fit_transform(nodes_wv)

umap_proj = umap_lib.UMAP(n_components=2, random_state=0,
                           n_neighbors=8, min_dist=0.3).fit_transform(nodes_wv)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.patch.set_facecolor(BG)

for ax, proj, title in zip(axes,
                             [tsne_proj, umap_proj],
                             ["t-SNE", "UMAP"]):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', color=BLACK,
                 pad=10, fontfamily=FONT)
    sc = ax.scatter(proj[labels7 == 0, 0], proj[labels7 == 0, 1],
                    s=140, color=BLUE, edgecolors="white",
                    linewidths=0.8, label="Mr. Hi", zorder=3)
    sc = ax.scatter(proj[labels7 == 1, 0], proj[labels7 == 1, 1],
                    s=140, color=RED, edgecolors="white",
                    linewidths=0.8, label="Officer", zorder=3)
    for i, (x, y) in enumerate(proj):
        ax.text(x, y, str(i), ha='center', va='center',
                fontsize=6, color="white", fontweight='bold', zorder=4)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=LGRAY,
              loc='best', markerscale=0.9)

fig.suptitle("DeepWalk node embeddings – Zachary's Karate Club",
             fontsize=11, color=GRAY, fontstyle='italic',
             fontfamily=FONT, y=0.02)
fig.tight_layout()
save(fig, "fig3_7_tsne_umap.png")

print("\nAll figures saved to", OUT)
