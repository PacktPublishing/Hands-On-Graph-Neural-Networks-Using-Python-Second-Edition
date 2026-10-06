"""Conceptual diagram for Chapter 20 (no data values).

Figures produced:
    fig20_1_workflows.png               specialized workflow vs foundation workflow
    fig20_2_pretraining_families.png    the five families of pretraining approaches compared (table)

Same grayscale palette, font and box style as the conceptual figures of Chapter 14.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

G0 = "#111111"; G1 = "#333333"; G2 = "#555555"; G3 = "#777777"
G4 = "#999999"; G5 = "#BBBBBB"; G6 = "#DDDDDD"
plt.rcParams['font.family'] = 'DejaVu Sans'
DARK = {G0, G1, G2, G3, G4}


def save(fig, name):
    fig.savefig(name, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
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


fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

ax.text(7, 5.85, 'From specialized GNNs to Graph Foundation Models',
        ha='center', va='center', fontsize=14, fontweight='bold', color=G0)

# Left panel: one model per graph per task
ax.text(3.5, 5.1, 'Specialized workflow (chapters 1–19)',
        ha='center', va='center', fontsize=13, fontweight='bold', color=G0)
for x, letter in zip([1.45, 3.5, 5.55], 'ABC'):
    box(ax, x, 4.2, 1.4, 0.4, f'Graph {letter}', G6, fontsize=10)
    arrow(ax, x, 4.0, x, 3.0, color=G3, lw=1.5)
    box(ax, x, 2.75, 1.4, 0.5, f'Model {letter}', G2, fontsize=11)
    arrow(ax, x, 2.5, x, 1.5, color=G3, lw=1.5)
    box(ax, x, 1.27, 1.4, 0.45, f'Task {letter}', G4, fontsize=10)
ax.text(3.5, 0.4, 'One model per graph, trained from scratch.',
        ha='center', va='center', fontsize=10, style='italic', color=G2)

# Right panel: one pretrained backbone reused across tasks
ax.text(10.5, 5.1, 'Foundation workflow (chapter 20)',
        ha='center', va='center', fontsize=13, fontweight='bold', color=G0)
box(ax, 10.5, 4.3, 5.4, 0.45, 'Pretraining corpus (many graphs, self-supervised)', G6, fontsize=10)
arrow(ax, 10.5, 4.07, 10.5, 3.42)
ax.text(10.75, 3.75, 'pretrain once', ha='left', va='center',
        fontsize=10, style='italic', color=G3)
box(ax, 10.5, 2.95, 3.1, 0.85, 'Foundation model\n(one pretrained backbone)', G0, fontsize=11)
for x in [8.45, 10.5, 12.55]:
    arrow(ax, 10.5 + (x - 10.5) * 0.63, 2.52, x, 1.5)
ax.text(10.65, 2.0, 'fine-tune\n(or zero-shot)', ha='left', va='center',
        fontsize=10, style='italic', color=G3, linespacing=1.3)
for x, letter in zip([8.45, 10.5, 12.55], 'ABC'):
    box(ax, x, 1.27, 1.4, 0.45, f'Task {letter}', G4, fontsize=10)
ax.text(10.5, 0.4, 'One pretrained backbone reused across tasks and graphs.',
        ha='center', va='center', fontsize=10, style='italic', color=G2)

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