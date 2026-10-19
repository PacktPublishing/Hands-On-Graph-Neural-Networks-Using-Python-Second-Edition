"""Conceptual diagrams for Chapter 14 (no data values).

Figures produced:
    fig14_1_snapshot_vs_event.png   snapshot-based vs event-based temporal modelling
    fig14_2_evolvegcn_overview.png  EvolveGCN: GCN weights evolved by an RNN
    fig14_3_evolvegcn_h.png         EvolveGCN-H weight update
    fig14_4_evolvegcn_o.png         EvolveGCN-O weight update
    fig14_10_mpnn_lstm.png          MPNN-LSTM architecture
    fig14_14_tgn.png                TGN modules

All data-derived figures are produced by run.py.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

G0 = "#111111"; G1 = "#333333"; G2 = "#555555"; G3 = "#777777"
G4 = "#999999"; G5 = "#BBBBBB"; G6 = "#DDDDDD"
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, FILLS, G4 as BG4, G5 as BG5, G6 as BG6)

apply()
OUT = os.path.dirname(os.path.abspath(__file__))
DARK = {G0, G1, G2, G3}


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))
    print(f"Saved {name}")


def box(ax, cx, cy, w, h, text, fc=None, fontsize=8, bold=True):
    """Square corners, dark type on a light fill; the caller's `fc` is mapped
    onto the light palette rather than used directly."""
    from matplotlib.patches import Rectangle
    light = {G0: BG4, G1: BG5, G2: BG5, G3: BG6}.get(fc, fc or BG6)
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=light,
                           edgecolor=G3, linewidth=0.8, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            color=INK, fontweight='bold' if bold else 'normal',
            linespacing=1.5, zorder=4)


def arrow(ax, x0, y0, x1, y1, color=G3, style='-|>', ls='-', rad=0.0, lw=1.1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}", mutation_scale=9))


def canvas(w, h, xlim, ylim):
    # The width the caller asks for is only a shape: the figure is drawn at
    # the printed width, with the same proportions.
    fig, ax = figure(aspect=h / w)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis('off')
    return fig, ax


# Taller than the original proportion: at 1.5 inches the italic note sat
# on the axis line it was meant to sit under.
fig, axes = figure(aspect=0.40, ncols=2)
snapshot_times = [1, 2, 3, 4, 5, 6, 7, 8, 9]
event_times = [0.6, 1.1, 1.3, 2.4, 2.6, 2.7, 3.9, 4.5, 4.6, 5.8, 6.1, 6.9, 7.0, 7.2, 8.3, 9.1, 9.4]
for ax, title, note in [
        (axes[0], 'Snapshot-based\n(EvolveGCN, MPNN-LSTM)', 'Graph observed at fixed intervals'),
        (axes[1], 'Event-based\n(TGN)', 'Model updated at every interaction event')]:
    ax.axhline(0, color=G5, linewidth=1.2)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel('Time', fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.spines[['left', 'top', 'right']].set_visible(False)
    ax.set_title(title, fontsize=9, fontweight='bold', color=INK)
    ax.text(5, 0.62, note, ha='center', fontsize=8, color=INK_SOFT,
            style='italic')
for t in snapshot_times:
    axes[0].axvline(t, color=G5, linewidth=1, zorder=0)
    axes[0].plot(t, 0, 'o', color=BG4, markeredgecolor=G3, markersize=7)
for t in event_times:
    axes[1].plot(t, 0, '|', color=G2, markersize=11, markeredgewidth=1.6)
save(fig, 'fig14_1_snapshot_vs_event.png')


fig, ax = canvas(13, 5.2, (0, 13), (0, 5.2))
xs = [1.6, 4.6, 7.6, 11.4]
labels = ['1', '2', '3', 'T']
# 2.6 units wide, not 2.2: at 5.5 inches the narrower box left "Node
# embeddings" hanging over both of its edges.
BW = 2.6
for x, t in zip(xs, labels):
    box(ax, x, 4.3, BW, 0.9, f'Graph $G_{t}$', G2)
    box(ax, x, 2.6, BW, 0.9, f'GCN with\nweights $W_{t}$', G0)
    box(ax, x, 0.9, BW, 1.0, f'Node\nembeddings $H_{t}$', G5)
    arrow(ax, x, 3.85, x, 3.07)
    arrow(ax, x, 2.15, x, 1.37)
for x0, x1 in zip(xs[:-1], xs[1:]):
    if x1 - x0 > 3.5:
        ax.text((x0 + x1) / 2, 2.6, '...', ha='center', va='center', fontsize=12, color=INK_SOFT)
        continue
    arrow(ax, x0 + BW / 2, 2.6, x1 - BW / 2 - 0.02, 2.6, color=G2, lw=1.4)
    ax.text((x0 + x1) / 2, 2.95, 'RNN', ha='center', fontsize=8, color=INK, fontweight='bold')
arrow(ax, xs[2] + BW / 2, 2.6, xs[2] + 1.75, 2.6, color=G2, lw=1.4)
arrow(ax, xs[3] - 1.85, 2.6, xs[3] - BW / 2 - 0.02, 2.6, color=G2, lw=1.4)
save(fig, 'fig14_2_evolvegcn_overview.png')


def evolvegcn_variant(rnn, top_text, equation, name):
    # Wider columns and labels split over two lines: at 5.5 inches a 2.8-unit
    # box left "Top-k summary of H" hanging over both of its edges.
    top = 6.4 if top_text is not None else 4.3
    fig, ax = canvas(15, 7.0 if top_text is not None else 4.9, (0, 15), (-0.5, top))
    xs = [2.5, 7.5, 12.5]
    ts = ['t-1', 't', 't+1']
    for x, t in zip(xs, ts):
        if top_text is not None:
            box(ax, x, 5.4, 4.2, 1.1, top_text.format(t=t), G5, fontsize=8)
            arrow(ax, x, 4.85, x, 4.0)
        box(ax, x, 3.45, 2.2, 0.9, f'{rnn}', G2, fontsize=8)
        box(ax, x, 1.3, 4.2, 1.2,
            f'GCN layer at ${t}$\nuses weights $W^{{(l)}}_{{{t}}}$', G0, fontsize=8)
        arrow(ax, x, 3.0, x, 1.9)
        ax.text(x + 0.15, 2.45, f'$W^{{(l)}}_{{{t}}}$', fontsize=8,
                color=INK_SOFT, va='center')
    arrow(ax, 0.15, 3.45, xs[0] - 1.12, 3.45, color=G3)
    ax.text(0.65, 3.75, '$W^{(l)}_{t-2}$', fontsize=8, color=INK_SOFT, ha='center')
    for x0, x1, t in zip(xs[:-1], xs[1:], ts[:-1]):
        arrow(ax, x0 + 1.12, 3.45, x1 - 1.12, 3.45, color=G3)
        ax.text((x0 + x1) / 2, 3.75, f'$W^{{(l)}}_{{{t}}}$', fontsize=8,
                color=INK_SOFT, ha='center')
    ax.text(7.5, -0.25, equation, ha='center', fontsize=9, color=INK)
    save(fig, name)


evolvegcn_variant('GRU', 'Top-$k$ summary\nof $H^{{(l)}}_{{{t}}}$',
                  r'$W^{(l)}_t = \mathrm{GRU}\left(H^{(l)}_t,\; W^{(l)}_{t-1}\right)$',
                  'fig14_3_evolvegcn_h.png')
evolvegcn_variant('LSTM', None,
                  r'$W^{(l)}_t = \mathrm{LSTM}\left(W^{(l)}_{t-1}\right)$',
                  'fig14_4_evolvegcn_o.png')


# Seven stages in a row needed 17 inches. They run down the page instead,
# with the two skip connections carried on a rail to the left of the stack.
fig, ax = figure(aspect=1.05)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); ax.set_facecolor(BG)

blocks = [
    (0.93, 'Features $x_t$, graph $A_t$', BG6),
    (0.79, 'GCN 1 + BatchNorm + Dropout', BG5),
    (0.65, 'GCN 2 + BatchNorm + Dropout', BG5),
    (0.51, 'Concatenate $[H_1, H_2]$', BG4),
    (0.37, 'LSTM 1 \u2192 LSTM 2', BG5),
    (0.23, 'Concatenate $[h_1, h_2, x_t]$', BG4),
    (0.09, 'Linear \u2192 $\\hat{y}_t$', BG6),
]
BW, BH, BX = 0.56, 0.085, 0.60
for y, text, fc in blocks:
    box(ax, BX, y, BW, BH, text, fc, fontsize=8)
for (y0, _, _), (y1, _, _) in zip(blocks[:-1], blocks[1:]):
    arrow(ax, BX, y0 - BH / 2, BX, y1 + BH / 2)

# H1 skips from GCN 1 to the first concatenation; the input features skip to
# the second. Both run on their own rail, so neither crosses a box.
for x_rail, y_from, y_to, label in [(0.26, 0.79, 0.51, '$H_1$'),
                                    (0.10, 0.93, 0.23, '$x_t$')]:
    ax.plot([BX - BW / 2, x_rail], [y_from, y_from], color=G3, lw=0.9,
            linestyle='--', zorder=1)
    ax.plot([x_rail, x_rail], [y_from, y_to], color=G3, lw=0.9,
            linestyle='--', zorder=1)
    arrow(ax, x_rail, y_to, BX - BW / 2, y_to, ls='--')
    ax.text(x_rail - 0.02, (y_from + y_to) / 2, label, fontsize=8.5,
            color=INK_SOFT, ha='right', va='center', fontfamily=FONT)

ax.text(0.5, 0.995, 'MPNN-LSTM: two skip connections feed the concatenations',
        ha='center', va='top', fontsize=8, color=INK_SOFT, style='italic',
        fontfamily=FONT)
save(fig, 'fig14_10_mpnn_lstm.png')


# Wider boxes and shorter labels: at 5.5 inches the original 2.4-unit boxes
# left "Embedding module (graph attention)" hanging over both edges.
fig, ax = canvas(16, 8.6, (0, 16), (0, 8.6))
# Three short lines a box rather than two long ones: four columns across 5.5
# inches leave each box 1.1 inches, and "(graph attention)" alone is wider.
BW, BH = 3.3, 2.3
TOP, BOT = 6.2, 2.4
box(ax, 1.85, TOP, BW, BH, 'Interaction\nevents\n$(u, v, t, e_{uv})$', G5, fontsize=8)
box(ax, 5.60, TOP, BW, BH, 'Message\nfunction', G1, fontsize=8)
box(ax, 9.35, TOP, BW, BH, 'Message\naggregator', G2, fontsize=8)
box(ax, 13.10, TOP, BW, BH, 'Memory\nupdater\n(GRU)', G2, fontsize=8)
box(ax, 13.10, BOT, BW, BH, 'Memory $s_v$\nfor every\nnode $v$', G0, fontsize=8)
box(ax, 9.35, BOT, BW, BH, 'Embedding\nmodule (graph\nattention)', G3, fontsize=8)
box(ax, 5.60, BOT, BW, BH, 'Temporal\nembeddings\n$z_v(t)$', G5, fontsize=8)
box(ax, 1.85, BOT, BW, BH, 'Link\npredictor', G4, fontsize=8)
for x0, x1 in [(1.85, 5.60), (5.60, 9.35), (9.35, 13.10)]:
    arrow(ax, x0 + BW / 2, TOP, x1 - BW / 2, TOP)
arrow(ax, 13.10, TOP - BH / 2, 13.10, BOT + BH / 2)
for x0, x1 in [(13.10, 9.35), (9.35, 5.60), (5.60, 1.85)]:
    arrow(ax, x0 - BW / 2, BOT, x1 + BW / 2, BOT)
ax.plot([14.75, 15.6, 15.6, 5.60], [BOT, BOT, 8.0, 8.0], color=G3,
        lw=1.0, ls='--', zorder=1)
arrow(ax, 5.60, 8.0, 5.60, TOP + BH / 2, ls='--')
ax.text(10.6, 8.2, 'previous memories $s_u$, $s_v$', fontsize=8,
        color=INK_SOFT, ha='center', style='italic')
save(fig, 'fig14_14_tgn.png')

print("\nDone.")