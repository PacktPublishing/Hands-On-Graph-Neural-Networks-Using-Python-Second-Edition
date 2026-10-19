"""Conceptual diagram for Chapter 20 (no data values).

Figures produced:
    fig20_1_workflows.png               specialized workflow vs foundation workflow
    fig20_2_pretraining_families.png    the five families of pretraining approaches compared (table)

Figure 20.1 is drawn through bookstyle, at the width it is printed. Figure
20.2 is left as it was: it is not on the reviewers' list, and a six-column
table of sentences cannot be squeezed into 5.5 inches without deciding what
to cut, which is the author's call and not a layout fix.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, FILLS, G1, G2, G3, G4, G5, G6)

apply()
G0 = "#111111"
DARK = {G0, G1, G2, G3, G4}
OUT = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))
    print(f"Saved {name}")


def box(ax, cx, cy, w, h, text, fc, fontsize=11, bold=True):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                facecolor=fc, edgecolor=G4, linewidth=1.2))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            color='white' if fc in DARK else G0,
            fontweight='bold' if bold else 'normal', linespacing=1.4)


def arrow(ax, x0, y0, x1, y1, color=G2, lw=1.8):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=16))


# ── Figure 20.1 ──────────────────────────────────────────────
# The two workflows are stacked, not set side by side: at 5.5 inches a pair of
# panels leaves each 2.7, and the specialized one needs three columns of boxes.
fig, axes = figure(aspect=0.88, nrows=2, gridspec_kw={'height_ratios': [1, 1]})
for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); ax.set_facecolor(BG)


def flat(ax, cx, cy, w, h, text, fill, fs=8):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=fill,
                           edgecolor=G3, linewidth=0.8, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, color=INK,
            fontweight='bold', fontfamily=FONT, linespacing=1.4, zorder=4)


def thin(ax, x0, y0, x1, y1, lw=1.0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=G3, lw=lw,
                                mutation_scale=8), zorder=2)


# a. one model per graph, trained from scratch
ax = axes[0]
ax.text(0.5, 0.97, 'Specialized workflow (chapters 1\u201319)', ha='center',
        va='top', fontsize=9, fontweight='bold', color=INK, fontfamily=FONT)
for x, letter in zip([0.22, 0.50, 0.78], 'ABC'):
    flat(ax, x, 0.70, 0.20, 0.13, f'Graph {letter}', G6)
    thin(ax, x, 0.635, x, 0.525)
    flat(ax, x, 0.46, 0.20, 0.13, f'Model {letter}', FILLS[1])
    thin(ax, x, 0.395, x, 0.285)
    flat(ax, x, 0.22, 0.20, 0.13, f'Task {letter}', FILLS[2])
ax.text(0.5, 0.05, 'One model per graph, trained from scratch.', ha='center',
        va='center', fontsize=8, style='italic', color=INK_SOFT,
        fontfamily=FONT)

# b. one pretrained backbone, reused
ax = axes[1]
ax.text(0.5, 0.97, 'Foundation workflow (chapter 20)', ha='center', va='top',
        fontsize=9, fontweight='bold', color=INK, fontfamily=FONT)
flat(ax, 0.5, 0.76, 0.78, 0.13,
     'Pretraining corpus (many graphs, self-supervised)', G6)
thin(ax, 0.5, 0.695, 0.5, 0.585)
ax.text(0.53, 0.64, 'pretrain once', ha='left', va='center', fontsize=8,
        style='italic', color=INK_SOFT, fontfamily=FONT)
flat(ax, 0.5, 0.50, 0.42, 0.15, 'Foundation model\n(one pretrained backbone)',
     FILLS[1])
for x in [0.22, 0.50, 0.78]:
    thin(ax, 0.5 + (x - 0.5) * 0.55, 0.42, x, 0.295)
# Left of the fan-out, not right: on the right it lands on the diagonal
# arrow into Task C.
ax.text(0.03, 0.36, 'fine-tune\n(or zero-shot)', ha='left', va='center',
        fontsize=8, style='italic', color=INK_SOFT, fontfamily=FONT,
        linespacing=1.4)
for x, letter in zip([0.22, 0.50, 0.78], 'ABC'):
    flat(ax, x, 0.23, 0.20, 0.13, f'Task {letter}', FILLS[2])
ax.text(0.5, 0.05, 'One pretrained backbone reused across tasks and graphs.',
        ha='center', va='center', fontsize=8, style='italic', color=INK_SOFT,
        fontfamily=FONT)

save(fig, 'fig20_1_workflows.png')


# Figure 20.2: the five families compared, as a table (same table style as the other chapters:
# dark header with white bold text, white first column, light gray data cells)
import textwrap

HEADER_BG = "#2D2D2D"; FIRST_COL_BG = "#FFFFFF"; CELL_BG = "#D8D8D8"; EDGE = "#333333"

columns = ['Family', 'What the model learns', 'What it requires', 'Main strength',
           'Main limitation', 'Cross-domain transfer']
rows = [
    ['Contrastive learning',
     'Embeddings that stay similar across perturbed views of the same data',
     'Unlabeled graphs and a set of augmentations',
     'No labels needed',
     'Results depend on the augmentations, and the best choice varies by dataset',
     'Not addressed by the objective itself; requires compatible input features'],
    ['Masked node and edge modeling',
     'To reconstruct masked node features or edges',
     'Unlabeled graphs',
     'Direct objective, no augmentations to choose',
     'Tied to the feature representation of the training graphs',
     'Harder when feature spaces differ'],
    ['Text-attributed graphs and LLMs',
     'A graph model on top of language-model embeddings of node and edge text',
     'A text description for every node and edge',
     'Shared feature space across domains',
     'Only for graphs that can be described in text',
     'Across domains whose graphs can be described in text'],
    ['Models for arbitrary feature spaces',
     'To run on graphs whose features and labels were never seen',
     'No shared feature or label space with the target graph',
     'Applies to graphs with unseen features and labels',
     "Results come mostly from the authors' own evaluations",
     'The design goal; an independent evaluation found AnyGraph below tuned GNNs [21]'],
    ['Multi-task pretraining',
     'One encoder trained with several objectives at once',
     'Several pretext tasks and a way to weight them',
     'One encoder serves several downstream tasks',
     'Objectives can conflict',
     'Depends on the objectives combined'],
]

WRAP = 22
wrapped = [[textwrap.fill(c, WRAP) for c in r] for r in rows]
n_lines = [max(c.count('\n') + 1 for c in r) for r in wrapped]

fig, ax = plt.subplots(figsize=(14, 0.5 + 0.28 * (sum(n_lines) + 2 * len(rows) + 2)))
ax.axis('off')
tbl = ax.table(cellText=wrapped, colLabels=[textwrap.fill(c, 16) for c in columns], cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
total = sum(n + 1 for n in n_lines) + 3
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(EDGE)
    cell.set_linewidth(0.8)
    if r == 0:
        cell.set_facecolor(HEADER_BG)
        cell.get_text().set_color('white')
        cell.get_text().set_fontweight('bold')
        cell.set_height(3 / total)
    else:
        cell.set_facecolor(FIRST_COL_BG if c == 0 else CELL_BG)
        cell.get_text().set_color(G0)
        cell.set_height((n_lines[r - 1] + 1) / total)
save(fig, 'fig20_2_pretraining_families.png')