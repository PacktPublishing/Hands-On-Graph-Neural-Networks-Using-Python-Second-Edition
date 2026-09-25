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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Style, matched to Chapter 6 ──────────────────────────────────────────────
FONT = "DejaVu Sans"
BG   = "white"

# Grayscale palette G0..G6
G0 = "#111111"   # near black (darkest, for isolated / problem bars)
G1 = "#333333"
G2 = "#555555"   # dark grey
G3 = "#777777"   # mid grey (main bars)
G4 = "#999999"   # lighter mid grey
G5 = "#BBBBBB"   # light grey (grid / labels)
G6 = "#DDDDDD"   # very light grey

MAIN_BAR = G3    # regular bar fill
ISO_BAR  = G0    # isolated-nodes bar (degree 0) — highlighted
TEXT     = G0    # main text colour
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

    fig, ax = plt.subplots(figsize=(11, 7), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=G4, width=0.9)
    for c in range(6):
        idx = [n for n in G.nodes if labels[n] == c]
        nx.draw_networkx_nodes(G, pos, nodelist=idx, ax=ax,
                               node_color=shades[c], node_shape=markers[c],
                               node_size=55, edgecolors="white",
                               linewidths=0.6, label=CITESEER_CLASSES[c])
    ax.legend(title="Category", loc="upper left", bbox_to_anchor=(1.0, 1.0),
              frameon=False, prop={"family": FONT, "size": 10},
              title_fontproperties={"family": FONT, "size": 10,
                                    "weight": "bold"})
    ax.set_title(f"CiteSeer: sampled subgraph ({G.number_of_nodes()} nodes, "
                 f"{nx.number_connected_components(G)} connected components)",
                 fontsize=13, fontweight="bold", color=TEXT,
                 fontfamily=FONT, pad=12)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
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

    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bar_colors = [ISO_BAR if lbl == "0" else MAIN_BAR for lbl in labels]
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="white",
                  linewidth=0.8, width=0.75)

    # Value labels on top of each bar
    y_max = max(values)
    for rect, v in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + y_max * 0.015,
                str(v), ha="center", va="bottom",
                fontsize=9, color=TEXT, fontfamily=FONT)

    ax.set_xlabel("Node degree", fontsize=11, color=TEXT, fontfamily=FONT)
    ax.set_ylabel("Number of nodes", fontsize=11, color=TEXT, fontfamily=FONT)
    ax.set_title("CiteSeer node degree distribution",
                 fontsize=13, fontweight="bold", color=TEXT,
                 fontfamily=FONT, pad=12)

    # Note about the isolated bar
    iso = counts_by_bucket.get("0", 0)
    if iso > 0:
        ax.annotate(f"{iso} isolated nodes\n(no neighbors to attend to)",
                    xy=(0, iso), xytext=(1.6, y_max * 0.85),
                    fontsize=9.5, color=TEXT, fontfamily=FONT,
                    ha="left", va="top",
                    arrowprops=dict(arrowstyle="->", color=G2, lw=1.0))

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(G2)
    ax.tick_params(axis="both", colors=TEXT, labelsize=10)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(FONT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    out_path = os.path.join(HERE, "fig_07_05.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path


# ── Figure 7.7 — GAT accuracy per degree bucket ─────────────────────────────
def make_fig_07_07(stats: dict) -> str:
    pdb = stats["per_degree_accuracy"]
    labels     = pdb["labels"]
    accuracies = pdb["accuracies"]
    sizes      = pdb["sizes"]

    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bar_colors = [ISO_BAR if lbl == "0" else MAIN_BAR for lbl in labels]
    bars = ax.bar(labels, accuracies, color=bar_colors, edgecolor="white",
                  linewidth=0.8, width=0.7)

    for rect, acc, sz in zip(bars, accuracies, sizes):
        x = rect.get_x() + rect.get_width() / 2
        # Accuracy label above bar
        ax.text(x, rect.get_height() + 0.015,
                f"{acc * 100:.1f}%", ha="center", va="bottom",
                fontsize=10, color=TEXT, fontweight="bold", fontfamily=FONT)
        # Sample-size label inside bar (only where it fits)
        if rect.get_height() > 0.12:
            ax.text(x, rect.get_height() / 2, f"n={sz}",
                    ha="center", va="center",
                    fontsize=9, color="white", fontfamily=FONT)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Node degree", fontsize=11, color=TEXT, fontfamily=FONT)
    ax.set_ylabel("Test accuracy", fontsize=11, color=TEXT, fontfamily=FONT)
    ax.set_title("GAT accuracy by node degree, CiteSeer",
                 fontsize=13, fontweight="bold", color=TEXT,
                 fontfamily=FONT, pad=12)

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(G2)
    ax.tick_params(axis="both", colors=TEXT, labelsize=10)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(FONT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    out_path = os.path.join(HERE, "fig_07_07.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path


def main() -> None:
    p4 = make_fig_07_04()
    stats = load_stats()
    p5 = make_fig_07_05(stats)
    p7 = make_fig_07_07(stats)
    print(f"Wrote {p4}")
    print(f"Wrote {p5}")
    print(f"Wrote {p7}")


if __name__ == "__main__":
    main()