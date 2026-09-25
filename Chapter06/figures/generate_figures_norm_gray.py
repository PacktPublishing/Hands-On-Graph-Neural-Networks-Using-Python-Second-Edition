import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

FONT="DejaVu Sans"; BG="white"
# Grayscale palette G0..G6
G0="#111111"; G2="#555555"; G3="#777777"; G5="#BBBBBB"; G6="#DDDDDD"
RECV=G0        # receiver node: darkest
SEND=G3        # sender node: mid grey
EDGE=G0        # main edge: black (thickness carries the weight)
FAINT=G6       # faint neighbour edges/nodes

d_i, d_j = 2, 6

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)
fig.patch.set_facecolor(BG)

def draw_panel(ax, title, weight_text, lw, receiver_only):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.set_title(title, fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=12)
    pos_i = (2.4, 4.0); pos_j = (7.6, 4.0)
    for k in range(d_i - 1):
        ang = np.pi*(0.6 + 0.5*k)
        p = (pos_i[0] + 1.35*np.cos(ang), pos_i[1] + 1.35*np.sin(ang))
        ax.plot([pos_i[0], p[0]], [pos_i[1], p[1]], color=FAINT, lw=1.5, zorder=1)
        ax.add_patch(Circle(p, 0.22, color=FAINT, zorder=2))
    for k in range(d_j - 1):
        ang = np.pi*(-0.5 + 0.28*k)
        p = (pos_j[0] + 1.5*np.cos(ang), pos_j[1] + 1.5*np.sin(ang))
        ax.plot([pos_j[0], p[0]], [pos_j[1], p[1]], color=FAINT, lw=1.5, zorder=1)
        ax.add_patch(Circle(p, 0.22, color=FAINT, zorder=2))
    ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]],
            color=EDGE, lw=lw, zorder=3, solid_capstyle='round')
    ax.text(5.0, 4.5, weight_text, ha='center', va='bottom', fontsize=13,
            color=G0, fontweight='bold', fontfamily=FONT,
            bbox=dict(facecolor='white', edgecolor=G2, boxstyle='round,pad=0.25', lw=1.3))
    ax.add_patch(Circle(pos_i, 0.42, color=RECV, zorder=4, ec='white', lw=1.5))
    ax.text(*pos_i, "i", ha='center', va='center', color='white', fontsize=14,
            fontweight='bold', fontfamily=FONT, zorder=5)
    ax.add_patch(Circle(pos_j, 0.42, color=SEND, zorder=4, ec='white', lw=1.5))
    ax.text(*pos_j, "j", ha='center', va='center', color='white', fontsize=14,
            fontweight='bold', fontfamily=FONT, zorder=5)
    ax.text(pos_i[0], pos_i[1]-0.95, f"receiver\ndegree $d_i$={d_i}", ha='center', va='top',
            fontsize=9.5, color=G0, fontfamily=FONT)
    ax.text(pos_j[0], pos_j[1]-0.95, f"sender\ndegree $d_j$={d_j}", ha='center', va='top',
            fontsize=9.5, color=G0, fontfamily=FONT)
    note = "scaled by the receiver's degree only" if receiver_only else "scaled by both endpoints' degrees"
    ax.text(5.0, 0.7, note, ha='center', fontsize=10, color=G2,
            fontstyle='italic', fontfamily=FONT)

draw_panel(axes[0], "Row normalisation  (asymmetric)",
           r"$\dfrac{1}{d_i}=\dfrac{1}{2}=0.50$", 6.0, True)
draw_panel(axes[1], "Symmetric normalisation",
           r"$\dfrac{1}{\sqrt{d_i\,d_j}}=\dfrac{1}{\sqrt{12}}\approx0.29$", 3.4, False)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_row_vs_symmetric_norm_gray.png"), dpi=200,
            bbox_inches='tight', facecolor=BG)
print("saved")
