"""
Chapter 19 - Large Language Models Meet Graph Neural Networks
Conceptual diagrams, grayscale.

fig19_1_gretriever_pipeline.png   the four stages of G-Retriever
fig19_2_soft_prompting.png        answer generation with soft prompting

Figure 19.3 is derived from the test results and produced by run.py.
"""

import os
import sys

from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, FILLS, G1, G2, G3, G4, G5, G6)

plt = apply()
G0 = "#111111"
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))
    print(f"  saved {name}")


def rbox(ax, x, y, w, h, label, fill=G6, fc=None, fs=8, r=0.04):
    """Square corners, dark type on a light fill; `fc` is ignored."""
    ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, facecolor=fill,
                           edgecolor=G3, linewidth=0.8, zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=INK,
            fontweight='bold', fontfamily=FONT, linespacing=1.5,
            multialignment='center', zorder=4)


def arr(ax, x1, y1, x2, y2, color=G3, lw=1.1, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=9), zorder=2)


# ── Fig 19.1: G-Retriever pipeline (4 stages) ───────────────────────
# Four stages across needed 13 inches. They run down the page instead, each
# with what goes into it on its left.
print("Figure 19.1 \u2026")
fig, ax = figure(aspect=0.92)
ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

STAGES = [
    ("1. Indexing (offline)", "knowledge graph",
     "SentenceBERT embeddings\nof nodes and edges"),
    ("2. Retrieval", "user question",
     "cosine similarity:\nquery vs nodes and edges"),
    ("3. Subgraph construction", "scored nodes/edges",
     "Prize-Collecting\nSteiner Tree (pcst_fast)"),
    ("4. Answer generation", "connected subgraph",
     "GNN encoder + LLM\n(soft prompt)"),
]
TOP, STEP = 0.90, 0.215
for i, (phase, feed, stage) in enumerate(STAGES):
    y = TOP - i * STEP
    ax.text(0.0, y + 0.085, phase, ha='left', va='center', fontsize=8,
            color=INK_SOFT, fontfamily=FONT, fontstyle='italic')
    rbox(ax, 0.20, y, 0.36, 0.085, feed, fill=G6, fs=8)
    arr(ax, 0.39, y, 0.46, y)
    rbox(ax, 0.74, y, 0.48, 0.125, stage, fill=FILLS[1], fs=8)
    if i:
        arr(ax, 0.74, y + STEP - 0.063, 0.74, y + 0.063)

arr(ax, 0.74, 0.192, 0.74, 0.115)
ax.text(0.74, 0.07, "natural-language answer", ha='center', va='center',
        fontsize=8.5, color=INK, fontweight='bold', fontfamily=FONT)

ax.set_title("The four stages of G-Retriever", fontsize=10.5,
             fontweight='bold', color=INK, fontfamily=FONT)
save(fig, "fig19_1_gretriever_pipeline.png")


# ── Fig 19.2: Answer generation block with soft prompting ────────────────────
print("Figure 19.2 …")
fig, ax = figure(aspect=0.72)
ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 11); ax.set_ylim(0, 6.5)

# Input: retrieved subgraph
rbox(ax, 2.5, 5.3, 3.6, 1.0, "Retrieved subgraph\n(nodes + edges)",
     fill=G6, fs=8)

# Input: question
rbox(ax, 8.5, 5.3, 3.4, 1.0, "Question (text)",
     fill=G6, fs=8)

# GNN branch
rbox(ax, 2.5, 3.7, 3.6, 1.0, "GNN encoder\n(GAT + mean pool)",
     fill=FILLS[1], fs=8)
arr(ax, 2.5, 4.8, 2.5, 4.2, color=G2, lw=1.5)

# MLP projection
rbox(ax, 2.5, 2.2, 3.6, 0.9, "MLP projection\nto LLM hidden dim",
     fill=FILLS[1], fs=8)
arr(ax, 2.5, 3.2, 2.5, 2.65, color=G2, lw=1.5)

# Linearized subgraph (text)
rbox(ax, 6.2, 3.7, 2.8, 1.0, "Linearized subgraph\n(text triples)",
     fill=FILLS[1], fs=8)

# LLM
rbox(ax, 5.5, 0.9, 6.4, 1.0,
     "LLM (Qwen3-0.6B, LoRA-fine-tuned)",
     fill=FILLS[2], fs=8)

# Concatenation of three inputs at LLM
arr(ax, 2.5, 1.75, 3.2, 1.4, color=G2, lw=1.5)
arr(ax, 6.2, 3.2, 6.2, 1.4, color=G2, lw=1.5)
arr(ax, 8.5, 4.8, 8.5, 1.4, color=G2, lw=1.5)

# Labels on the three input paths
ax.text(2.9, 1.55, "soft token", ha='left', fontsize=8, color=INK_SOFT,
        fontfamily=FONT, fontstyle='italic')
ax.text(6.4, 2.3, "text context", ha='left', fontsize=8, color=INK_SOFT,
        fontfamily=FONT, fontstyle='italic')
ax.text(8.7, 3.5, "question tokens", ha='left', fontsize=8, color=INK_SOFT,
        fontfamily=FONT, fontstyle='italic')

# Output
arr(ax, 5.5, 0.4, 5.5, 0.1, color=G2, lw=1.5)
ax.text(5.5, -0.1, "natural-language answer",
        ha='center', fontsize=8.5, color=INK, fontfamily=FONT,
        fontweight='bold')

ax.set_title("Answer generation: GNN and LLM meet via soft prompting",
             fontsize=10.5, fontweight='bold', color=INK, fontfamily=FONT)
save(fig, "fig19_2_soft_prompting.png")

print(f"\nFigures saved to {OUT}")