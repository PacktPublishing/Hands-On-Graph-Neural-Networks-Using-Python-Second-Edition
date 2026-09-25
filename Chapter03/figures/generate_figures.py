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

# ── Shared style (grayscale) ──────────────────────────────────────────────────
FONT   = "DejaVu Sans"
BG     = "white"

# Shared gray ramp used across the book's figures, darkest to lightest. The
# drawing code below always goes through the role names, never through G0..G6
# directly: the figure sections reuse G5/G6/G7 as graph variables.
G0 = "#111111"; G1 = "#333333"; G2 = "#555555"
G3 = "#777777"; G4 = "#999999"; G5 = "#BBBBBB"; G6 = "#DDDDDD"

# Ordered ramp for categorical series that must stay apart in print.
GRAY4 = ["#1A1A1A", "#606060", "#A0A0A0", "#D0D0D0"]

# Roles: dark for the marks the reader should look at first, mid for the
# regular ones, light for structure that only has to stay in the background.
INK       = G0   # titles and dark labels
ACCENT    = G0   # highlighted marks (target word, sum layer, random walk)
PRIMARY   = G2   # ordinary filled boxes and nodes, white text on top
SECONDARY = G4   # light fills that carry dark text
MUTED     = G2   # arrows, graph edges and caption text
FAINT     = G6   # grid ticks, faint edges, legend frames

# Two-class encoding (karate-club factions). Two steps apart on GRAY4 so the
# fills differ by ~134 luminance levels and survive a black-and-white print.
CLASS_A = GRAY4[0]   # Mr. Hi's faction — takes white labels
CLASS_B = GRAY4[2]   # Officer's faction — light enough to need dark labels

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

def arrow(ax, x1, y1, x2, y2, color=MUTED):
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
ax.set_title("CBOW", fontsize=13, fontweight='bold', color=INK, pad=10, fontfamily=FONT)
ctx_words = ["w(t-2)", "w(t-1)", "w(t+1)", "w(t+2)"]
ys = [0.82, 0.66, 0.34, 0.18]
for lbl, y in zip(ctx_words, ys):
    box(ax, 0.18, y, 0.28, 0.10, lbl, PRIMARY, fontsize=9)
# sum layer
box(ax, 0.50, 0.50, 0.18, 0.12, "Sum", ACCENT, fontsize=9)
for y in ys:
    arrow(ax, 0.33, y, 0.41, 0.50)
# output
box(ax, 0.82, 0.50, 0.28, 0.10, "w(t)", PRIMARY, fontsize=9)
arrow(ax, 0.59, 0.50, 0.68, 0.50)
ax.text(0.18, 0.94, "INPUT", ha='center', fontsize=8, color=MUTED, fontfamily=FONT)
ax.text(0.82, 0.94, "OUTPUT", ha='center', fontsize=8, color=MUTED, fontfamily=FONT)

# Skip-gram (right)
ax = axes[1]
ax.set_title("Skip-gram", fontsize=13, fontweight='bold', color=INK, pad=10, fontfamily=FONT)
box(ax, 0.18, 0.50, 0.28, 0.10, "w(t)", ACCENT, fontsize=9)
for lbl, y in zip(ctx_words, ys):
    box(ax, 0.82, y, 0.28, 0.10, lbl, PRIMARY, fontsize=9)
    arrow(ax, 0.33, 0.50, 0.68, y)
ax.text(0.18, 0.94, "INPUT", ha='center', fontsize=8, color=MUTED, fontfamily=FONT)
ax.text(0.82, 0.94, "OUTPUT", ha='center', fontsize=8, color=MUTED, fontfamily=FONT)

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
    fc = ACCENT if is_target else (SECONDARY if is_context else FAINT)
    tc = "white" if is_target else (INK if is_context else MUTED)
    rect = FancyBboxPatch((i - 0.42, 0.85), 0.84, 0.42,
                           boxstyle="round,pad=0.02,rounding_size=0.04",
                           facecolor=fc, edgecolor="none")
    ax.add_patch(rect)
    ax.text(i, 1.06, w, ha='center', va='center', fontsize=11,
            color=tc, fontweight='bold' if is_target else 'normal', fontfamily=FONT)

# context size bracket
ax.annotate("", xy=(target - ctx_sz - 0.45, 0.78),
            xytext=(target + ctx_sz + 0.45, 0.78),
            arrowprops=dict(arrowstyle="<->", color=PRIMARY, lw=1.5))
ax.text(target, 0.68, f"context size = {ctx_sz}", ha='center',
        fontsize=9, color=PRIMARY, fontfamily=FONT)

# skip-gram pairs below
pairs = [(words[target], words[target + d])
         for d in range(-ctx_sz, ctx_sz + 1) if d != 0]
n = len(pairs)
xs = np.linspace(0.5, len(words) - 1.5, n)
for x, (tgt, ctx) in zip(xs, pairs):
    ax.text(x, 0.38, f"({tgt},", ha='right', fontsize=10,
            color=ACCENT, fontweight='bold', fontfamily=FONT)
    ax.text(x + 0.05, 0.38, f"{ctx})", ha='left', fontsize=10,
            color=PRIMARY, fontweight='bold', fontfamily=FONT)
    ax.plot([target, x + 0.02], [0.85, 0.50], color=MUTED,
            lw=0.8, linestyle='--', alpha=0.5)

ax.text(len(words)/2 - 0.5, 0.08,
        "Skip-gram pairs   (target, context)",
        ha='center', fontsize=9, color=MUTED, fontstyle='italic', fontfamily=FONT)

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
colors_col = [SECONDARY, PRIMARY, ACCENT, PRIMARY, SECONDARY]
txt_colors = [INK, "white", "white", "white", INK]

for x, lbl, fc, tc in zip(cols, labels_col, colors_col, txt_colors):
    box(ax, x, 2.0, 1.2, 1.8, lbl, fc, fontsize=10, textcolor=tc)

for i in range(len(cols) - 1):
    arrow(ax, cols[i] + 0.6, 2.0, cols[i+1] - 0.6, 2.0)

# dimension labels
ax.text(cols[0], 0.5, "V", ha='center', fontsize=10, color=MUTED, fontfamily=FONT)
ax.text(cols[2], 0.5, "d", ha='center', fontsize=10, color=MUTED, fontfamily=FONT)
ax.text(cols[4], 0.5, "V", ha='center', fontsize=10, color=MUTED, fontfamily=FONT)
for x in [cols[0], cols[2], cols[4]]:
    ax.plot([x, x], [0.65, 0.95], color=FAINT, lw=1)

ax.text(5.0, 3.7, "Word2Vec skip-gram architecture",
        ha='center', fontsize=12, fontweight='bold', color=INK, fontfamily=FONT)

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
ax.set_title("Sentence", fontsize=12, fontweight='bold', color=INK, pad=8, fontfamily=FONT)
sent_words = ["graph", "neural", "network", "learning"]
xs = np.linspace(0.12, 0.88, len(sent_words))
y  = 0.55
for i, (x, w) in enumerate(zip(xs, sent_words)):
    box(ax, x, y, 0.17, 0.18, w, PRIMARY, fontsize=9)
    if i < len(sent_words) - 1:
        arrow(ax, x + 0.09, y, xs[i+1] - 0.09, y)
# window arrows
for i in range(len(sent_words) - 2):
    ax.annotate("", xy=(xs[i+2] - 0.085, y - 0.12),
                xytext=(xs[i] + 0.085, y - 0.12),
                arrowprops=dict(arrowstyle="<->", color=ACCENT,
                                lw=1.0, connectionstyle="arc3,rad=0.3"))
ax.text(0.50, 0.15, "context window", ha='center', fontsize=9,
        color=ACCENT, fontstyle='italic', fontfamily=FONT)

# RIGHT: graph / random walk
ax = axes[1]
ax.axis('off')
ax.set_title("Graph random walk", fontsize=12, fontweight='bold', color=INK, pad=8, fontfamily=FONT)

G_demo = nx.karate_club_graph()
pos    = nx.spring_layout(G_demo, seed=7)

# draw full graph faintly
nx.draw_networkx_edges(G_demo, pos, ax=ax, alpha=0.15, edge_color=MUTED, width=0.8)
nx.draw_networkx_nodes(G_demo, pos, ax=ax, node_size=80,
                       node_color=FAINT, edgecolors=MUTED, linewidths=0.5)

# highlight a walk
walk_nodes = [0, 1, 2, 8, 33, 32, 28]
walk_edges = list(zip(walk_nodes[:-1], walk_nodes[1:]))
nx.draw_networkx_edges(G_demo, pos, edgelist=walk_edges, ax=ax,
                       edge_color=ACCENT, width=2.5, arrows=True,
                       arrowstyle='-|>', arrowsize=15)
nx.draw_networkx_nodes(G_demo, pos, nodelist=walk_nodes, ax=ax,
                       node_size=180, node_color=ACCENT, edgecolors="white", linewidths=1.2)
nx.draw_networkx_labels(G_demo, pos,
                        labels={n: str(n) for n in walk_nodes},
                        ax=ax, font_size=7, font_color="white", font_weight='bold')
ax.text(0.50, -0.05, "random walk = sequence of nodes",
        ha='center', fontsize=9, color=ACCENT,
        fontstyle='italic', fontfamily=FONT, transform=ax.transAxes)

fig.suptitle("Sentences and graphs share the same co-occurrence intuition",
             fontsize=11, color=MUTED, fontstyle='italic', fontfamily=FONT, y=0.02)
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

nx.draw_networkx_edges(G5, pos5, ax=ax, edge_color=MUTED, width=1.5, alpha=0.7)
nx.draw_networkx_nodes(G5, pos5, ax=ax, node_size=700,
                       node_color=PRIMARY, edgecolors="white", linewidths=1.5)
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
colors6  = [CLASS_A if l == 0 else CLASS_B for l in labels6]

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')

nx.draw_networkx_edges(G6, pos6, ax=ax, edge_color=FAINT, width=1.2, alpha=0.9)
nx.draw_networkx_nodes(G6, pos6, ax=ax, node_size=600,
                       node_color=colors6, edgecolors="white", linewidths=1.5)
# The two factions sit at opposite ends of the ramp, so the node ids are drawn
# in two passes: white on the dark faction, near-black on the light one.
for face, txt in ((0, "white"), (1, INK)):
    nx.draw_networkx_labels(G6, pos6, ax=ax, font_size=9,
                            labels={n: str(n) for n, l in zip(G6.nodes, labels6)
                                    if l == face},
                            font_color=txt, font_weight='bold', font_family=FONT)

patch0 = mpatches.Patch(color=CLASS_A, label="Mr. Hi's faction")
patch1 = mpatches.Patch(color=CLASS_B,  label="Officer's faction")
ax.legend(handles=[patch0, patch1], loc='lower right',
          fontsize=10, framealpha=0.9, edgecolor=FAINT)
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
    # gensim only honours `seed` with a single worker: with more threads the
    # order in which examples reach the model varies, so the embeddings — and
    # this figure — change from run to run.
    workers     = 1,
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
    ax.set_title(title, fontsize=13, fontweight='bold', color=INK,
                 pad=10, fontfamily=FONT)
    sc = ax.scatter(proj[labels7 == 0, 0], proj[labels7 == 0, 1],
                    s=140, color=CLASS_A, edgecolors="white",
                    linewidths=0.8, label="Mr. Hi", zorder=3)
    sc = ax.scatter(proj[labels7 == 1, 0], proj[labels7 == 1, 1],
                    s=140, color=CLASS_B, edgecolors="white",
                    linewidths=0.8, label="Officer", zorder=3)
    for i, (x, y) in enumerate(proj):
        # same trick as figure 3.6: the id follows the marker it sits on
        ax.text(x, y, str(i), ha='center', va='center', fontsize=6,
                color="white" if labels7[i] == 0 else INK,
                fontweight='bold', zorder=4)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=FAINT,
              loc='best', markerscale=0.9)

fig.suptitle("DeepWalk node embeddings – Zachary's Karate Club",
             fontsize=11, color=MUTED, fontstyle='italic',
             fontfamily=FONT, y=0.02)
fig.tight_layout()
save(fig, "fig3_7_tsne_umap.png")

print("\nAll figures saved to", OUT)
