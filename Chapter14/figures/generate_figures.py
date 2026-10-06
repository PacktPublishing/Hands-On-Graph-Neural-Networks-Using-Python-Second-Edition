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
plt.rcParams['font.family'] = 'DejaVu Sans'
DARK = {G0, G1, G2, G3}


def save(fig, name):
    fig.savefig(name, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved {name}")


def box(ax, cx, cy, w, h, text, fc, fontsize=11, bold=True):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fc, edgecolor=G4, linewidth=1.2))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            color='white' if fc in DARK else G0,
            fontweight='bold' if bold else 'normal', linespacing=1.4)


def arrow(ax, x0, y0, x1, y1, color=G2, style='-|>', ls='-', rad=0.0, lw=1.8):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}", mutation_scale=16))


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis('off')
    return fig, ax


fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
snapshot_times = [1, 2, 3, 4, 5, 6, 7, 8, 9]
event_times = [0.6, 1.1, 1.3, 2.4, 2.6, 2.7, 3.9, 4.5, 4.6, 5.8, 6.1, 6.9, 7.0, 7.2, 8.3, 9.1, 9.4]
for ax, title, note in [
        (axes[0], 'Snapshot-based\n(EvolveGCN, MPNN-LSTM)', 'Graph observed at fixed intervals'),
        (axes[1], 'Event-based\n(TGN)', 'Model updated at every interaction event')]:
    ax.axhline(0, color=G5, linewidth=1.2)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel('Time', fontsize=11)
    ax.spines[['left', 'top', 'right']].set_visible(False)
    ax.set_title(title, fontsize=12, fontweight='bold', color=G0)
    ax.text(5, -0.75, note, ha='center', fontsize=10, color=G3, style='italic')
for t in snapshot_times:
    axes[0].axvline(t, color=G5, linewidth=1, zorder=0)
    axes[0].plot(t, 0, 'o', color=G0, markersize=13)
for t in event_times:
    axes[1].plot(t, 0, '|', color=G1, markersize=18, markeredgewidth=2.5)
fig.tight_layout()
save(fig, 'fig14_1_snapshot_vs_event.png')


fig, ax = canvas(13, 5.2, (0, 13), (0, 5.2))
xs = [1.6, 4.6, 7.6, 11.4]
labels = ['1', '2', '3', 'T']
for x, t in zip(xs, labels):
    box(ax, x, 4.3, 2.2, 0.9, f'Graph $G_{t}$', G2)
    box(ax, x, 2.6, 2.2, 0.9, f'GCN with\nweights $W_{t}$', G0)
    box(ax, x, 0.9, 2.2, 0.9, f'Node embeddings\n$H_{t}$', G5)
    arrow(ax, x, 3.85, x, 3.07)
    arrow(ax, x, 2.15, x, 1.37)
for x0, x1 in zip(xs[:-1], xs[1:]):
    if x1 - x0 > 3.5:
        ax.text((x0 + x1) / 2, 2.6, '...', ha='center', va='center', fontsize=18, color=G2)
        continue
    arrow(ax, x0 + 1.1, 2.6, x1 - 1.12, 2.6, color=G1, lw=2.2)
    ax.text((x0 + x1) / 2, 2.85, 'RNN', ha='center', fontsize=10, color=G1, fontweight='bold')
arrow(ax, xs[2] + 1.1, 2.6, xs[2] + 1.6, 2.6, color=G1, lw=2.2)
arrow(ax, xs[3] - 1.65, 2.6, xs[3] - 1.12, 2.6, color=G1, lw=2.2)
save(fig, 'fig14_2_evolvegcn_overview.png')


def evolvegcn_variant(rnn, top_text, equation, name):
    top = 6.0 if top_text is not None else 4.3
    fig, ax = canvas(13, 6.2 if top_text is not None else 4.6, (0, 13), (-0.4, top))
    xs = [2.2, 6.5, 10.8]
    ts = ['t-1', 't', 't+1']
    for x, t in zip(xs, ts):
        if top_text is not None:
            box(ax, x, 5.1, 2.8, 0.9, top_text.format(t=t), G5, fontsize=10)
            arrow(ax, x, 4.65, x, 3.92)
        box(ax, x, 3.45, 2.0, 0.9, f'{rnn}', G2)
        box(ax, x, 1.4, 3.2, 1.1,
            f'GCN layer at ${t}$\nuses weights $W^{{(l)}}_{{{t}}}$', G0, fontsize=10)
        arrow(ax, x, 3.0, x, 1.97)
        ax.text(x + 0.12, 2.5, f'$W^{{(l)}}_{{{t}}}$', fontsize=11, color=G1, va='center')
    arrow(ax, 0.2, 3.45, xs[0] - 1.02, 3.45, color=G1, lw=2.2)
    ax.text(0.6, 3.65, '$W^{(l)}_{t-2}$', fontsize=11, color=G1, ha='center')
    for x0, x1, t in zip(xs[:-1], xs[1:], ts[:-1]):
        arrow(ax, x0 + 1.02, 3.45, x1 - 1.02, 3.45, color=G1, lw=2.2)
        ax.text((x0 + x1) / 2, 3.65, f'$W^{{(l)}}_{{{t}}}$', fontsize=11, color=G1, ha='center')
    ax.text(6.5, -0.2, equation, ha='center', fontsize=13, color=G0)
    save(fig, name)


evolvegcn_variant('GRU', 'Top-$k$ summary of $H^{{(l)}}_{{{t}}}$',
                  r'$W^{(l)}_t = \mathrm{GRU}\left(H^{(l)}_t,\; W^{(l)}_{t-1}\right)$',
                  'fig14_3_evolvegcn_h.png')
evolvegcn_variant('LSTM', None,
                  r'$W^{(l)}_t = \mathrm{LSTM}\left(W^{(l)}_{t-1}\right)$',
                  'fig14_4_evolvegcn_o.png')


fig, ax = canvas(17, 3.8, (0, 17), (0.9, 4.5))
blocks = [
    (1.2, 'Features $x_t$\nGraph $A_t$', G5),
    (3.6, 'GCN 1\n+ BatchNorm\n+ Dropout', G2),
    (6.0, 'GCN 2\n+ BatchNorm\n+ Dropout', G2),
    (8.4, 'Concatenate\n$[H_1, H_2]$', G4),
    (10.8, 'LSTM 1\n→ LSTM 2', G0),
    (13.2, 'Concatenate\n$[h_1, h_2, x_t]$', G4),
    (15.6, 'Linear\n→ $\\hat{y}_t$', G3),
]
for x, text, fc in blocks:
    box(ax, x, 1.8, 2.0, 1.5, text, fc, fontsize=10)
for (x0, _, _), (x1, _, _) in zip(blocks[:-1], blocks[1:]):
    arrow(ax, x0 + 1.0, 1.8, x1 - 1.02, 1.8)
arrow(ax, 3.6, 2.55, 8.4, 2.57, rad=-0.35, color=G3, ls='--')
arrow(ax, 1.2, 2.55, 13.2, 2.57, rad=-0.25, color=G3, ls='--')
ax.text(5.9, 3.55, '$H_1$', fontsize=11, color=G2, ha='center')
ax.text(7.2, 4.25, 'skip connection of the input features', fontsize=10, color=G2,
        ha='center', style='italic')
save(fig, 'fig14_10_mpnn_lstm.png')


fig, ax = canvas(16, 6.4, (0, 16), (0, 6.4))
box(ax, 1.6, 4.6, 2.4, 1.2, 'Interaction\nevents\n$(u, v, t, e_{uv})$', G5, fontsize=10)
box(ax, 5.0, 4.6, 2.4, 1.2, 'Message\nfunction', G1)
box(ax, 8.4, 4.6, 2.4, 1.2, 'Message\naggregator', G2)
box(ax, 11.8, 4.6, 2.4, 1.2, 'Memory\nupdater\n(GRU)', G2)
box(ax, 11.8, 1.8, 2.4, 1.2, 'Memory\n$s_v$ for\nevery node $v$', G0)
box(ax, 8.4, 1.8, 2.4, 1.2, 'Embedding\nmodule\n(graph attention)', G3)
box(ax, 5.0, 1.8, 2.4, 1.2, 'Temporal\nembeddings\n$z_v(t)$', G5, fontsize=10)
box(ax, 1.6, 1.8, 2.4, 1.2, 'Link\npredictor', G4)
arrow(ax, 2.8, 4.6, 3.78, 4.6)
arrow(ax, 6.2, 4.6, 7.18, 4.6)
arrow(ax, 9.6, 4.6, 10.58, 4.6)
arrow(ax, 11.8, 4.0, 11.8, 2.42)
arrow(ax, 10.6, 1.8, 9.62, 1.8)
arrow(ax, 7.2, 1.8, 6.22, 1.8)
arrow(ax, 3.8, 1.8, 2.82, 1.8)
ax.plot([13.02, 13.9, 13.9, 5.0], [1.8, 1.8, 5.75, 5.75], color=G3, lw=1.8, ls='--')
arrow(ax, 5.0, 5.75, 5.0, 5.22, color=G3, ls='--')
ax.text(9.45, 5.95, 'previous memories $s_u$, $s_v$', fontsize=10, color=G2,
        ha='center', style='italic')
save(fig, 'fig14_14_tgn.png')

print("\nDone.")