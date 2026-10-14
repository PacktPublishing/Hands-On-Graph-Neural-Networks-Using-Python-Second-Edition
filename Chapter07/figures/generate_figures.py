"""
Chapter 7 – Figure generation.

Reads figures/citeseer_stats.json produced by run.py and generates the three
data figures shown in the chapter, in grayscale, matching the visual style
used in Chapter 6:

    fig_07_04.png   CiteSeer sampled subgraph, shaded by class
    fig_07_05.png   CiteSeer node degree distribution
    fig_07_07.png   GAT accuracy per node degree bucket, CiteSeer

All files are saved into the same figures/ directory next to this script.
Run this after run.py (which downloads CiteSeer and produces
citeseer_stats.json):

    python figures/generate_figures.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, DPI, G0, G1, G2, G3, G4, G5, G6)

plt = apply()

# Bars are filled light and labelled dark: the sample sizes are written inside
# them, and white type on a mid grey is 4.5:1 before the press closes it up.
MAIN_BAR = G5    # regular bar fill
ISO_BAR  = G4    # isolated-nodes bar (degree 0), picked out by a dark edge
TEXT     = INK   # main text colour
GRID     = G6    # grid colour

HERE = os.path.dirname(os.path.abspath(__file__))
STATS_PATH = os.path.join(HERE, "citeseer_stats.json")

CITESEER_CLASSES = ["AI", "ML", "IR", "DB", "Agents", "HCI"]


def load_stats() -> dict:
    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(
            f"Missing {STATS_PATH}. Run 'python run.py' first; it writes "
            "citeseer_stats.json alongside its normal output."
        )
    with open(STATS_PATH) as f:
        return json.load(f)


# ── Figure 7.1 — ReLU versus Leaky ReLU ────────────────────────────────
def make_fig_07_01(alpha: float = 0.2) -> str:
    """Both activations on one row, drawn from their definitions."""
    x = np.linspace(-3, 3, 601)
    panels = [("ReLU", np.maximum(0.0, x), G1,
               "flat: gradient 0\nfor every x < 0"),
              (f"Leaky ReLU ($\\alpha$ = {alpha})", np.where(x > 0, x, alpha * x),
               G1, f"slope {alpha},\nnot zero")]

    fig, axes = figure(aspect=0.42, ncols=2, sharey=True)
    for ax, (name, y, colour, note_text) in zip(axes, panels):
        ax.axhline(0, color=G5, lw=0.7, zorder=1)
        ax.axvline(0, color=G5, lw=0.7, zorder=1)
        ax.plot(x, y, color=colour, lw=1.8, zorder=3)
        # The note sits in the empty upper left, clear of both curves, so it
        # needs no leader line across the plot.
        ax.text(-2.85, 2.55, note_text, fontsize=8, color=INK_SOFT,
                fontfamily=FONT, ha="left", va="top", linespacing=1.4)
        ax.set_title(name, fontsize=9.5, fontweight="bold", color=INK,
                     fontfamily=FONT)
        ax.set_xlabel("x", fontsize=9, color=INK, fontfamily=FONT)
        ax.set_xlim(-3, 3); ax.set_ylim(-1, 3)
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        ax.tick_params(axis="both", colors=INK, labelsize=8)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily(FONT)
        ax.spines[["left", "bottom"]].set_color(G3)
    axes[0].set_ylabel("f(x)", fontsize=9, color=INK, fontfamily=FONT)

    out_path = os.path.join(HERE, "fig_07_01.png")
    save_fig(fig, out_path)
    return out_path



# ── Figure 7.4 — CiteSeer sampled subgraph ──────────────────────────────────
def pack_components(G, gap: float = 0.6, row_width: float = 15.0) -> dict:
    import networkx as nx

    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    pos, x0, y0, row_h = {}, 0.0, 0.0, 0.0
    for comp in comps:
        sub = G.subgraph(comp)
        size = max(0.35, np.sqrt(len(comp)) * 0.65)
        if len(comp) == 1:
            local = {next(iter(comp)): np.array([0.0, 0.0])}
        elif len(comp) == 2:
            a, b = list(comp)
            local = {a: np.array([-0.5, 0.0]), b: np.array([0.5, 0.0])}
        else:
            local = nx.kamada_kawai_layout(sub)
        pts = np.array(list(local.values()))
        span = np.ptp(pts, axis=0)
        span[span == 0] = 1.0
        if x0 > 0 and x0 + size > row_width:
            x0, y0, row_h = 0.0, y0 - row_h - gap, 0.0
        for n, p in local.items():
            q = (p - pts.min(axis=0)) / span
            pos[n] = np.array([x0 + q[0] * size, y0 - q[1] * size])
        x0 += size + gap
        row_h = max(row_h, size)
    return pos


def draw_citeseer_subgraph(G, labels, out_path: str) -> str:
    import networkx as nx

    pos = pack_components(G)
    shades  = [G0, G1, G2, G3, G4, G5]
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = figure(aspect=0.80)
    ax.set_facecolor(BG)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=G4, width=0.6)
    for c in range(6):
        idx = [n for n in G.nodes if labels[n] == c]
        nx.draw_networkx_nodes(G, pos, nodelist=idx, ax=ax,
                               node_color=shades[c], node_shape=markers[c],
                               node_size=26, edgecolors="white",
                               linewidths=0.4, label=CITESEER_CLASSES[c])
    # The legend goes under the plot: beside it, it would take a fifth of the
    # 5.5 inches the graph itself needs.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=6,
              frameon=False, handletextpad=0.2, columnspacing=0.9,
              prop={"family": FONT, "size": 8.5})
    ax.set_title(f"CiteSeer: sampled subgraph ({G.number_of_nodes()} nodes, "
                 f"{nx.number_connected_components(G)} components)",
                 fontsize=10.5, fontweight="bold", color=TEXT,
                 fontfamily=FONT)
    ax.set_aspect("equal")
    ax.axis("off")

    save_fig(fig, out_path)
    return out_path


def make_fig_07_04() -> str:
    import torch
    import networkx as nx
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import k_hop_subgraph, subgraph

    data = Planetoid(root=os.path.dirname(HERE), name="CiteSeer")[0]
    rng = np.random.default_rng(0)
    seeds = torch.from_numpy(rng.choice(data.num_nodes, 40, replace=False))
    nodes, _, _, _ = k_hop_subgraph(seeds, 1, data.edge_index,
                                    num_nodes=data.num_nodes)
    edge_index, _ = subgraph(nodes, data.edge_index, relabel_nodes=True,
                             num_nodes=data.num_nodes)
    labels = data.y[nodes].numpy()

    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(edge_index.t().tolist())
    return draw_citeseer_subgraph(G, labels,
                                  os.path.join(HERE, "fig_07_04.png"))


# ── Figure 7.5 — CiteSeer degree distribution ───────────────────────────────
def make_fig_07_05(stats: dict) -> str:
    hist = {int(k): int(v) for k, v in stats["degree_hist"].items()}

    # Keep degrees 0..14 separately, group the rest as a single "15+" tail.
    CUTOFF = 15
    degrees = sorted(hist.keys())
    counts_by_bucket = {str(d): hist.get(d, 0) for d in range(CUTOFF)}
    tail_count       = sum(v for d, v in hist.items() if d >= CUTOFF)
    if tail_count > 0:
        counts_by_bucket[f"{CUTOFF}+"] = tail_count

    labels = list(counts_by_bucket.keys())
    values = list(counts_by_bucket.values())

    fig, ax = figure(aspect=0.52)
    ax.set_facecolor(BG)

    bars = ax.bar(labels, values, color=MAIN_BAR, edgecolor=G3,
                  linewidth=0.5, width=0.78)
    # The degree-0 bar is the point of the figure, so it carries a dark edge
    # rather than a dark fill, which would swallow any label put on it.
    for rect, lbl in zip(bars, labels):
        if lbl == "0":
            rect.set_facecolor(ISO_BAR)
            rect.set_edgecolor(INK)
            rect.set_linewidth(1.1)

    # Sixteen counts of up to four digits do not fit side by side across 5.5
    # inches, so they are set upright instead of across.
    y_max = max(values)
    for rect, v in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + y_max * 0.02,
                str(v), ha="center", va="bottom", rotation=90,
                fontsize=8, color=TEXT, fontfamily=FONT)
    ax.set_ylim(0, y_max * 1.28)

    ax.set_xlabel("Node degree", fontsize=9.5, color=TEXT, fontfamily=FONT)
    ax.set_ylabel("Number of nodes", fontsize=9.5, color=TEXT, fontfamily=FONT)
    ax.set_title("CiteSeer node degree distribution",
                 fontsize=10.5, fontweight="bold", color=TEXT,
                 fontfamily=FONT)

    # Note about the isolated bar
    iso = counts_by_bucket.get("0", 0)
    if iso > 0:
        # No leader line: from the empty upper right it would have to cross
        # every tall bar to reach degree 0. The dark edge marks that bar
        # instead, and the note names it.
        ax.text(5.6, y_max * 1.16,
                f"Degree 0: {iso} isolated nodes,\nno neighbors to attend to",
                fontsize=8.5, color=TEXT, fontfamily=FONT,
                ha="left", va="top", linespacing=1.4)

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(G2)
    ax.tick_params(axis="both", colors=TEXT, labelsize=8.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(FONT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    out_path = os.path.join(HERE, "fig_07_05.png")
    save_fig(fig, out_path)
    return out_path


# ── Figure 7.7 — GAT accuracy per degree bucket ─────────────────────────────
def make_fig_07_07(stats: dict) -> str:
    pdb = stats["per_degree_accuracy"]
    labels     = pdb["labels"]
    accuracies = pdb["accuracies"]
    sizes      = pdb["sizes"]

    fig, ax = figure(aspect=0.54)
    ax.set_facecolor(BG)

    bars = ax.bar(labels, accuracies, color=MAIN_BAR, edgecolor=G3,
                  linewidth=0.5, width=0.7)
    for rect, lbl in zip(bars, labels):
        if lbl == "0":
            rect.set_facecolor(ISO_BAR)
            rect.set_edgecolor(INK)
            rect.set_linewidth(1.1)

    for rect, acc, sz in zip(bars, accuracies, sizes):
        x = rect.get_x() + rect.get_width() / 2
        # Accuracy label above bar
        ax.text(x, rect.get_height() + 0.015,
                f"{acc * 100:.1f}%", ha="center", va="bottom",
                fontsize=8.5, color=TEXT, fontweight="bold", fontfamily=FONT)
        # Sample size inside the bar, in dark type: the fill is light enough
        # to carry it, which white on a mid grey was not.
        if rect.get_height() > 0.12:
            ax.text(x, rect.get_height() / 2, f"n={sz}",
                    ha="center", va="center",
                    fontsize=8, color=INK, fontfamily=FONT)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Node degree", fontsize=9.5, color=TEXT, fontfamily=FONT)
    ax.set_ylabel("Accuracy", fontsize=9.5, color=TEXT, fontfamily=FONT)
    ax.set_title("GAT accuracy by node degree, CiteSeer\n"
                 "(nodes not used for training)",
                 fontsize=10.5, fontweight="bold", color=TEXT,
                 fontfamily=FONT)

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(G2)
    ax.tick_params(axis="both", colors=TEXT, labelsize=8.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(FONT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    out_path = os.path.join(HERE, "fig_07_07.png")
    save_fig(fig, out_path)
    return out_path


def main() -> None:
    p1 = make_fig_07_01()
    p4 = make_fig_07_04()
    stats = load_stats()
    p5 = make_fig_07_05(stats)
    p7 = make_fig_07_07(stats)
    print(f"Wrote {p1}")
    print(f"Wrote {p4}")
    print(f"Wrote {p5}")
    print(f"Wrote {p7}")


if __name__ == "__main__":
    main()