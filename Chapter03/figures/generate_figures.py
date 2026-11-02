"""
Chapter 3 - conceptual figures (Figures 3.1 to 3.4).
Figures 3.5, 3.6 and 3.7 come from the data and are saved by run.py.

Usage:
    python figures/generate_figures.py
"""

import os
import sys

import numpy as np
import networkx as nx
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, G3, G4, G5, G6)

plt = apply()

np.random.seed(0)

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# Roles, mapped onto the three fills that carry dark type. The book is printed
# without colour, so a reversed white label on a mid grey is the one thing a
# 300 dpi file cannot rescue: at 8.5 pt the press closes up the counters.
HILITE = G4      # the mark the reader should look at first
PLAIN = G6       # ordinary boxes
CONTEXT = G5     # the context words around a target


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))
    print(f"  saved {name}")


def box(ax, x, y, w, h, label, fill=PLAIN, fontsize=8.5, weight='normal'):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=fill, edgecolor=G3, linewidth=0.7, zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            color=INK, fontweight=weight, zorder=4, fontfamily=FONT,
            linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color=INK_SOFT, lw=0.9):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=8), zorder=2)


# =============================================================================
# Figure 3.1 - CBOW versus skip-gram
# =============================================================================
print("Figure 3.1 ...")

# Two panels at column width are 2.75in each, so the four context boxes stack
# in a taller figure than the original 10x5: the labels keep their point size.
fig, axes = figure(aspect=0.60, ncols=2)

for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

ctx_words = ["w(t-2)", "w(t-1)", "w(t+1)", "w(t+2)"]
ys = [0.80, 0.63, 0.37, 0.20]

ax = axes[0]
ax.set_title("CBOW", fontweight='bold', color=INK, pad=6, fontfamily=FONT)
for lbl, y in zip(ctx_words, ys):
    box(ax, 0.17, y, 0.30, 0.13, lbl, PLAIN)
box(ax, 0.56, 0.50, 0.24, 0.15, "Average", HILITE, weight='bold')
for y in ys:
    arrow(ax, 0.33, y, 0.43, 0.50)
box(ax, 0.88, 0.50, 0.22, 0.13, "w(t)", PLAIN)
arrow(ax, 0.69, 0.50, 0.76, 0.50)
ax.text(0.17, 0.95, "INPUT", ha='center', fontsize=8, color=INK_SOFT,
        fontfamily=FONT)
ax.text(0.88, 0.95, "OUTPUT", ha='center', fontsize=8, color=INK_SOFT,
        fontfamily=FONT)

ax = axes[1]
ax.set_title("Skip-gram", fontweight='bold', color=INK, pad=6, fontfamily=FONT)
box(ax, 0.15, 0.50, 0.22, 0.13, "w(t)", HILITE, weight='bold')
for lbl, y in zip(ctx_words, ys):
    box(ax, 0.78, y, 0.30, 0.13, lbl, PLAIN)
    arrow(ax, 0.28, 0.50, 0.62, y)
ax.text(0.15, 0.95, "INPUT", ha='center', fontsize=8, color=INK_SOFT,
        fontfamily=FONT)
ax.text(0.78, 0.95, "OUTPUT", ha='center', fontsize=8, color=INK_SOFT,
        fontfamily=FONT)

save(fig, "fig3_1_cbow_skipgram.png")


# =============================================================================
# Figure 3.2 - Text to skip-grams
# =============================================================================
print("Figure 3.2 ...")

words = ["the", "quick", "brown", "fox", "jumps"]
target = 2      # "brown"
ctx_sz = 2

# The four pairs sat in one row across 10in. At 5.5in they go in two rows of
# two, which keeps them at 8 pt instead of shrinking to fit. The leader lines
# from the target to each pair are dropped: at this width they crossed each
# other and ran through the "context size" label.
fig, ax = figure(aspect=0.44)
ax.set_facecolor(BG)
ax.axis('off')
ax.set_xlim(-0.55, len(words) - 0.45)
ax.set_ylim(0.22, 1.80)

for i, w in enumerate(words):
    is_target = (i == target)
    fill = HILITE if is_target else CONTEXT
    ax.add_patch(FancyBboxPatch(
        (i - 0.40, 1.44), 0.80, 0.30,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=fill, edgecolor=G3, linewidth=0.7))
    ax.text(i, 1.59, w, ha='center', va='center', fontsize=8.5, color=INK,
            fontweight='bold' if is_target else 'normal', fontfamily=FONT)

ax.annotate("", xy=(target - ctx_sz - 0.42, 1.34),
            xytext=(target + ctx_sz + 0.42, 1.34),
            arrowprops=dict(arrowstyle="<->", color=INK_SOFT, lw=0.9,
                            mutation_scale=8))
ax.text(target, 1.19, f"context size = {ctx_sz}", ha='center', va='center',
        fontsize=8, color=INK_SOFT, fontfamily=FONT)

pairs = [(words[target], words[target + d])
         for d in range(-ctx_sz, ctx_sz + 1) if d != 0]
slots = [(0.30, 0.95), (2.35, 0.95), (0.30, 0.62), (2.35, 0.62)]
for (x, y), (tgt, ctx) in zip(slots, pairs):
    ax.text(x, y, f"({tgt}, {ctx})", ha='left', va='center', fontsize=8.5,
            color=INK, fontweight='bold', fontfamily=FONT)

ax.text((len(words) - 1) / 2, 0.31, "Skip-gram pairs   (target, context)",
        ha='center', va='center', fontsize=8, color=INK_SOFT,
        fontstyle='italic', fontfamily=FONT)

save(fig, "fig3_2_skipgrams.png")


# =============================================================================
# Figure 3.3 - Word2Vec architecture
# =============================================================================
print("Figure 3.3 ...")

# Five boxes across 5.5in leave 1.1in each, so the two-line labels need a
# taller box than the original 11x4.5 gave them.
fig, ax = figure(aspect=0.32)
ax.set_facecolor(BG)
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0.30, 2.60)

cols = [1.0, 3.0, 5.0, 7.0, 9.0]
labels_col = ["Input\n(one-hot)", "W_embed\n(V x d)", "Embedding\n(d-dim)",
              "W_context\n(d x V)", "Output\n(softmax)"]
fills_col = [PLAIN, CONTEXT, HILITE, CONTEXT, PLAIN]

for x, lbl, fill in zip(cols, labels_col, fills_col):
    box(ax, x, 1.80, 1.60, 1.00, lbl, fill, fontsize=8)

for i in range(len(cols) - 1):
    arrow(ax, cols[i] + 0.82, 1.80, cols[i + 1] - 0.82, 1.80)

for x, dim in [(cols[0], "V"), (cols[2], "d"), (cols[4], "V")]:
    ax.plot([x, x], [1.20, 0.92], color=G4, lw=0.8)
    ax.text(x, 0.62, dim, ha='center', va='center', fontsize=8,
            color=INK_SOFT, fontfamily=FONT)

ax.set_title("Word2Vec skip-gram architecture", fontweight='bold', color=INK,
             pad=6, fontfamily=FONT)

save(fig, "fig3_3_word2vec_arch.png")


# =============================================================================
# Figure 3.4 - Sentences as graphs
# =============================================================================
print("Figure 3.4 ...")

# Side by side, each panel would be 2.75in and the 34-node graph unreadable.
# Stacked, the graph gets the full column width.
fig, axes = plt.subplots(2, 1, figsize=(5.5, 5.5 * 0.86),
                         height_ratios=[1, 2.1])

ax = axes[0]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title("Sentence", fontweight='bold', color=INK, pad=4, fontfamily=FONT)

sent_words = ["graph", "neural", "network", "learning"]
xs = np.linspace(0.13, 0.87, len(sent_words))
y = 0.62
for i, (x, w) in enumerate(zip(xs, sent_words)):
    box(ax, x, y, 0.20, 0.30, w, PLAIN)
    if i < len(sent_words) - 1:
        arrow(ax, x + 0.105, y, xs[i + 1] - 0.105, y)
for i in range(len(sent_words) - 2):
    ax.annotate("", xy=(xs[i + 2] - 0.09, y - 0.20),
                xytext=(xs[i] + 0.09, y - 0.20),
                arrowprops=dict(arrowstyle="<->", color=INK_SOFT, lw=0.8,
                                mutation_scale=7,
                                connectionstyle="arc3,rad=0.35"))
ax.text(0.50, 0.04, "context window", ha='center', va='center', fontsize=8,
        color=INK_SOFT, fontstyle='italic', fontfamily=FONT)

ax = axes[1]
ax.axis('off')
ax.set_title("Graph random walk", fontweight='bold', color=INK, pad=4,
             fontfamily=FONT)

G_demo = nx.karate_club_graph()
# seed 7 put nodes 32 and 33, two adjacent hubs, on top of each other and their
# labels collided; this layout keeps every pair of walk nodes well apart.
pos = nx.spring_layout(G_demo, seed=8, k=0.8)

nx.draw_networkx_edges(G_demo, pos, ax=ax, edge_color=G6, width=0.6)
nx.draw_networkx_nodes(G_demo, pos, ax=ax, node_size=60,
                       node_color=G6, edgecolors=G4, linewidths=0.5)

walk_nodes = [0, 1, 2, 8, 33, 32, 28]
walk_edges = list(zip(walk_nodes[:-1], walk_nodes[1:]))
nx.draw_networkx_edges(G_demo, pos, edgelist=walk_edges, ax=ax,
                       edge_color=INK, width=1.6, arrows=True,
                       arrowstyle='-|>', arrowsize=9, node_size=230)
nx.draw_networkx_nodes(G_demo, pos, nodelist=walk_nodes, ax=ax,
                       node_size=230, node_color=G5, edgecolors=INK,
                       linewidths=0.8)
nx.draw_networkx_labels(G_demo, pos, labels={n: str(n) for n in walk_nodes},
                        ax=ax, font_size=7, font_color=INK,
                        font_weight='bold', font_family=FONT)

fig.supxlabel("Sentences and graphs share the same co-occurrence intuition",
              fontsize=8, color=INK_SOFT, fontstyle='italic',
              fontfamily=FONT)

save(fig, "fig3_4_sentences_graphs.png")

print("\nAll figures saved to", OUT)
