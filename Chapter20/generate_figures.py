"""
Chapter 20 (theoretical version) – Figure.
Fig 20.1: Specialized vs Foundation graph learning workflows (conceptual).
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


# ── Fig 20.1: Specialized vs Foundation graph learning workflows ────────────
print("Figure 20.1 …")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG)

# LEFT PANEL: Specialized workflow
ax = axes[0]
ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_title("Specialized workflow (chapters 1–19)",
             fontsize=11.5, fontweight='bold', color=G0,
             fontfamily=FONT, pad=8)

# Three graphs, three models, three tasks
rbox(ax, 2.0, 8.5, 2.0, 0.8, "Graph A", fill=G6, fc=G0, fs=9)
rbox(ax, 5.0, 8.5, 2.0, 0.8, "Graph B", fill=G6, fc=G0, fs=9)
rbox(ax, 8.0, 8.5, 2.0, 0.8, "Graph C", fill=G6, fc=G0, fs=9)

rbox(ax, 2.0, 5.5, 2.0, 1.0, "Model A", fill=G2, fc='white', fs=10)
rbox(ax, 5.0, 5.5, 2.0, 1.0, "Model B", fill=G2, fc='white', fs=10)
rbox(ax, 8.0, 5.5, 2.0, 1.0, "Model C", fill=G2, fc='white', fs=10)

rbox(ax, 2.0, 2.5, 2.0, 0.9, "Task A", fill=G4, fc='white', fs=9.5)
rbox(ax, 5.0, 2.5, 2.0, 0.9, "Task B", fill=G4, fc='white', fs=9.5)
rbox(ax, 8.0, 2.5, 2.0, 0.9, "Task C", fill=G4, fc='white', fs=9.5)

# Arrows
for x in (2.0, 5.0, 8.0):
    arr(ax, x, 8.1, x, 6.0, color=G3, lw=1.3)
    arr(ax, x, 5.0, x, 2.95, color=G3, lw=1.3)

# Label
ax.text(5.0, 0.7, "One model per graph, trained from scratch.",
        ha='center', fontsize=9, color=G2, fontfamily=FONT,
        fontstyle='italic')

# RIGHT PANEL: Foundation workflow
ax = axes[1]
ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_title("Foundation workflow (chapter 20)",
             fontsize=11.5, fontweight='bold', color=G0,
             fontfamily=FONT, pad=8)

# Pretraining corpus (top, wide box)
rbox(ax, 5.0, 8.7, 6.5, 0.9,
     "Pretraining corpus (many graphs, self-supervised)",
     fill=G6, fc=G0, fs=9.5)

# Foundation model (center)
rbox(ax, 5.0, 6.2, 4.5, 1.2,
     "Foundation model\n(one pretrained backbone)",
     fill=G0, fc='white', fs=10.5)

# Arrow: pretraining
arr(ax, 5.0, 8.25, 5.0, 6.85, color=G2, lw=1.6)
ax.text(5.4, 7.55, "pretrain once", ha='left', fontsize=8.5,
        color=G3, fontfamily=FONT, fontstyle='italic')

# Three downstream tasks
rbox(ax, 2.0, 2.5, 2.0, 0.9, "Task A", fill=G4, fc='white', fs=9.5)
rbox(ax, 5.0, 2.5, 2.0, 0.9, "Task B", fill=G4, fc='white', fs=9.5)
rbox(ax, 8.0, 2.5, 2.0, 0.9, "Task C", fill=G4, fc='white', fs=9.5)

# Arrows from foundation to tasks
arr(ax, 4.0, 5.6, 2.4, 3.0, color=G2, lw=1.3)
arr(ax, 5.0, 5.6, 5.0, 3.0, color=G2, lw=1.3)
arr(ax, 6.0, 5.6, 7.6, 3.0, color=G2, lw=1.3)

ax.text(5.0, 4.1, "fine-tune (or zero-shot)",
        ha='center', fontsize=8.5, color=G3, fontfamily=FONT,
        fontstyle='italic')

# Label
ax.text(5.0, 0.7,
        "One pretrained backbone reused across tasks and graphs.",
        ha='center', fontsize=9, color=G2, fontfamily=FONT,
        fontstyle='italic')

fig.suptitle("From specialized GNNs to Graph Foundation Models",
             fontsize=12.5, fontweight='bold', color=G0, fontfamily=FONT,
             y=1.00)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
save(fig, "fig20_1_specialized_vs_foundation.png")

print(f"\nAll figures saved to {OUT}")
