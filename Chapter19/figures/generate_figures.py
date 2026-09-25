"""
Chapter 19 – All figures in grayscale.
Fig 19.1: the four stages of G-Retriever (conceptual pipeline).
Fig 19.2: architecture of the answer-generation block with soft prompting.
Fig 19.3: G-Retriever vs text-only RAG (side-by-side workflow comparison).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

OUT  = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

FONT = "DejaVu Sans"
BG   = "white"
G0   = "#111111"; G1 = "#333333"; G2 = "#555555"
G3   = "#777777"; G4 = "#999999"; G5 = "#BBBBBB"; G6 = "#DDDDDD"


def save(fig, name, dpi=200):
    fig.savefig(f"{OUT}/{name}", dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"  saved {name}")


def rbox(ax, x, y, w, h, label, fill=G2, fc='white', fs=9, r=0.04):
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle=f"round,pad=0.02,rounding_size={r}",
                          facecolor=fill, edgecolor=G4,
                          linewidth=1.1, zorder=3)
    ax.add_patch(rect)
    for li, line in enumerate(label.split('\n')):
        offset = 0.11*(li - label.count('\n')/2)
        ax.text(x, y-offset, line, ha='center', va='center',
                fontsize=fs, color=fc, fontweight='bold',
                fontfamily=FONT, zorder=4)


def arr(ax, x1, y1, x2, y2, color=G2, lw=1.4, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=12), zorder=2)


# ── Fig 19.1: G-Retriever pipeline (4 stages) ────────────────────────────────
print("Figure 19.1 …")
fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 13); ax.set_ylim(0, 5.5)

# Timeline labels
ax.text(2.0, 4.6, "1. Indexing (offline)", ha='center', fontsize=10,
        color=G3, fontfamily=FONT, fontstyle='italic')
ax.text(5.3, 4.6, "2. Retrieval", ha='center', fontsize=10,
        color=G3, fontfamily=FONT, fontstyle='italic')
ax.text(8.4, 4.6, "3. Subgraph construction", ha='center', fontsize=10,
        color=G3, fontfamily=FONT, fontstyle='italic')
ax.text(11.5, 4.6, "4. Answer generation", ha='center', fontsize=10,
        color=G3, fontfamily=FONT, fontstyle='italic')

# Stage boxes
rbox(ax, 2.0, 3.5, 2.6, 1.1, "SentenceBERT\nnode + edge\nembeddings",
     fill=G1, fc='white', fs=9)
rbox(ax, 5.3, 3.5, 2.6, 1.1, "cosine similarity\nquery vs\nnodes/edges",
     fill=G2, fc='white', fs=9)
rbox(ax, 8.4, 3.5, 2.6, 1.1, "Prize-Collecting\nSteiner Tree\n(pcst_fast)",
     fill=G0, fc='white', fs=9)
rbox(ax, 11.5, 3.5, 2.6, 1.1, "GNN encoder\n+ LLM\n(soft prompt)",
     fill=G1, fc='white', fs=9)

# Arrows between stages
arr(ax, 3.4, 3.5, 4.0, 3.5, color=G2, lw=1.6)
arr(ax, 6.7, 3.5, 7.1, 3.5, color=G2, lw=1.6)
arr(ax, 9.8, 3.5, 10.2, 3.5, color=G2, lw=1.6)

# Inputs/outputs (row below)
rbox(ax, 2.0, 1.6, 2.4, 0.7, "knowledge graph", fill=G6, fc=G0, fs=8.5)
rbox(ax, 5.3, 1.6, 2.4, 0.7, "user question", fill=G6, fc=G0, fs=8.5)
rbox(ax, 8.4, 1.6, 2.4, 0.7, "scored nodes/edges", fill=G6, fc=G0, fs=8.5)
rbox(ax, 11.5, 1.6, 2.4, 0.7, "connected subgraph", fill=G6, fc=G0, fs=8.5)

# Vertical connectors (inputs go up into the stage boxes)
arr(ax, 2.0, 1.95, 2.0, 2.85, color=G3, lw=1.2)
arr(ax, 5.3, 1.95, 5.3, 2.85, color=G3, lw=1.2)
arr(ax, 8.4, 1.95, 8.4, 2.85, color=G3, lw=1.2)
arr(ax, 11.5, 1.95, 11.5, 2.85, color=G3, lw=1.2)

# Final output arrow (bottom right, out of stage 4)
ax.text(11.5, 0.5, "natural-language answer",
        ha='center', fontsize=9.5, color=G1, fontfamily=FONT,
        fontweight='bold')

ax.set_title("The four stages of G-Retriever",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT,
             pad=10)
fig.tight_layout()
save(fig, "fig19_1_gretriever_pipeline.png")


# ── Fig 19.2: Answer generation block with soft prompting ────────────────────
print("Figure 19.2 …")
fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 11); ax.set_ylim(0, 6.5)

# Input: retrieved subgraph
rbox(ax, 2.5, 5.3, 3.6, 1.0, "Retrieved subgraph\n(nodes + edges)",
     fill=G6, fc=G0, fs=10)

# Input: question
rbox(ax, 8.5, 5.3, 3.4, 1.0, "Question (text)",
     fill=G6, fc=G0, fs=10)

# GNN branch
rbox(ax, 2.5, 3.7, 3.6, 1.0, "GNN encoder\n(GAT + mean pool)",
     fill=G1, fc='white', fs=10)
arr(ax, 2.5, 4.8, 2.5, 4.2, color=G2, lw=1.5)

# MLP projection
rbox(ax, 2.5, 2.2, 3.6, 0.9, "MLP projection\nto LLM hidden dim",
     fill=G2, fc='white', fs=9.5)
arr(ax, 2.5, 3.2, 2.5, 2.65, color=G2, lw=1.5)

# Linearized subgraph (text)
rbox(ax, 5.5, 3.7, 3.4, 1.0, "Linearized subgraph\n(text triples)",
     fill=G3, fc='white', fs=9.5)

# LLM
rbox(ax, 5.5, 0.9, 6.4, 1.0,
     "LLM (Qwen3-0.6B, LoRA-fine-tuned)",
     fill=G0, fc='white', fs=10)

# Concatenation of three inputs at LLM
arr(ax, 2.5, 1.75, 3.2, 1.4, color=G2, lw=1.5)
arr(ax, 5.5, 3.2, 5.5, 1.4, color=G2, lw=1.5)
arr(ax, 8.5, 4.8, 8.5, 1.4, color=G2, lw=1.5)

# Labels on the three input paths
ax.text(2.9, 1.55, "soft token", ha='left', fontsize=8.5, color=G3,
        fontfamily=FONT, fontstyle='italic')
ax.text(5.7, 2.3, "text context", ha='left', fontsize=8.5, color=G3,
        fontfamily=FONT, fontstyle='italic')
ax.text(8.7, 3.5, "question tokens", ha='left', fontsize=8.5, color=G3,
        fontfamily=FONT, fontstyle='italic')

# Output
arr(ax, 5.5, 0.4, 5.5, 0.1, color=G2, lw=1.5)
ax.text(5.5, -0.05, "natural-language answer",
        ha='center', fontsize=9.5, color=G1, fontfamily=FONT,
        fontweight='bold')

ax.set_title("Answer generation — GNN and LLM meet via soft prompting",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT,
             pad=10)
fig.tight_layout()
save(fig, "fig19_2_soft_prompting.png")


# ── Fig 19.3: G-Retriever vs text-only RAG side-by-side ──────────────────────
print("Figure 19.3 …")
fig, axes = plt.subplots(1, 2, figsize=(13, 8))
fig.patch.set_facecolor(BG)

# Fixed layout parameters so both panels use the same box size and the
# same vertical stride, regardless of how many stages each pipeline has.
BOX_W, BOX_H = 3.6, 0.95
STRIDE       = 1.55            # vertical distance between consecutive boxes
Y_TOP        = 9.5             # y of the first box in both panels


def _text_color_for(fill):
    """Dark text on light fills, white text on dark fills."""
    return G0 if fill in (G5, G6) else 'white'


def draw_pipeline(ax, title, stages, arrow_color=G2):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_xlim(0, 5); ax.set_ylim(0, 11)
    ax.set_title(title, fontsize=12.5, fontweight='bold', color=G0,
                 fontfamily=FONT, pad=12)
    for i, (label, fill) in enumerate(stages):
        y = Y_TOP - i * STRIDE
        rbox(ax, 2.5, y, BOX_W, BOX_H, label,
             fill=fill, fc=_text_color_for(fill), fs=10)
        if i < len(stages) - 1:
            arr(ax, 2.5, y - BOX_H/2 - 0.05,
                     2.5, y - STRIDE + BOX_H/2 + 0.05,
                color=arrow_color, lw=1.6)


# Shorter, single-line labels where possible, to avoid line wrapping
# that would collide with the arrows.
gr_stages = [
    ("Knowledge graph + SentenceBERT index",   G6),
    ("Embedding-based retrieval",              G2),
    ("PCST: connected subgraph",               G0),
    ("GNN encoder + soft prompt",              G1),
    ("LLM (Qwen3-0.6B) generates answer",      G0),
]

txt_stages = [
    ("Knowledge graph (triples as text)",      G6),
    ("BM25 keyword retrieval",                 G2),
    ("Top-k text chunks (no structure)",       G3),
    ("LLM (Qwen3-0.6B) generates answer",      G0),
]

draw_pipeline(axes[0], "G-Retriever", gr_stages)
draw_pipeline(axes[1], "Text-only RAG baseline", txt_stages)

# Caption placed well below the shorter panel (text-only ends higher up),
# so it never touches any box.
fig.text(0.5, 0.03,
         "Both pipelines share the same knowledge graph and the same LLM.\n"
         "They differ in whether the graph structure is preserved through "
         "the retrieval and encoding stages, or flattened into text.",
         ha='center', fontsize=9.5, color=G3, fontfamily=FONT,
         fontstyle='italic')

fig.suptitle("G-Retriever vs text-only RAG — same LLM, different pipelines",
             fontsize=13, fontweight='bold', color=G0, fontfamily=FONT,
             y=0.98)
fig.tight_layout(rect=[0, 0.08, 1, 0.95])
save(fig, "fig19_3_gretriever_vs_textonly.png")

print(f"\nAll figures saved to {OUT}")