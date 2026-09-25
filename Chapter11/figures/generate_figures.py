"""
Chapter 11 – All figures in grayscale.
Figs 11.1–11.4: pure matplotlib diagrams.
Fig 11.5: synthetic protein graph (networkx).
Figs 11.6–11.7: GIN vs GCN classification grids (trained on PROTEINS).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
import torch
import torch.nn.functional as F
import os, random

random.seed(0); np.random.seed(0); torch.manual_seed(0)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures11")
os.makedirs(OUT, exist_ok=True)

FONT = "DejaVu Sans"
BG   = "white"
G0   = "#111111"   # very dark
G1   = "#333333"
G2   = "#555555"
G3   = "#777777"
G4   = "#999999"
G5   = "#BBBBBB"
G6   = "#DDDDDD"

def save(fig, name, dpi=200):
    fig.savefig(f"{OUT}/{name}", dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"  saved {name}")


# ── Fig 11.1: Two isomorphic graphs ───────────────────────────────────────────
print("Figure 11.1 …")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor(BG)

# Both graphs have the same structure: a 5-cycle
edges = [(0,1),(1,2),(2,3),(3,4),(4,0)]
G_a   = nx.Graph(); G_a.add_edges_from(edges)
G_b   = nx.Graph(); G_b.add_edges_from(edges)

# Different visual layout (different permutation of nodes)
pos_a = nx.circular_layout(G_a)
pos_b = {i: pos_a[(i+2) % 5] for i in range(5)}  # rotate labels

for ax, G, pos, title in zip(
    axes, [G_a, G_b], [pos_a, pos_b],
    ["Graph A", "Graph B  (same structure, relabeled nodes)"]
):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold',
                 color=G0, fontfamily=FONT, pad=10)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=G2, width=2.0)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=G1,
                           node_size=700, edgecolors='white', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=13,
                            font_color='white', font_weight='bold',
                            font_family=FONT)

fig.suptitle("Two isomorphic graphs — same structure, different node labels",
             fontsize=11, color=G3, fontstyle='italic', fontfamily=FONT, y=0.04)

# Annotation: bijection arrow between the two graphs
fig.text(0.50, 0.50, "\u2245", ha='center', va='center',
         fontsize=36, color=G2, fontfamily=FONT)
fig.tight_layout()
save(fig, "fig11_1_isomorphic.png")


# ── Fig 11.3: WL test step-by-step ────────────────────────────────────────────
print("Figure 11.3 …")

# A small graph for demonstration
G_wl = nx.Graph()
G_wl.add_edges_from([(0,1),(0,2),(1,2),(2,3),(3,4)])
pos_wl = {0:(0,1), 1:(1,2), 2:(1,0), 3:(2,1), 4:(3,1)}

# Colors are computed by running 1-WL on G_wl, not assigned by hand.
# Step 0: every node shares the same color.
# Step 1: colors separate nodes by degree.
# Step 2: colors separate nodes by the multiset of their neighbor colors.
def wl_steps(G, n_steps):
    col   = {n: 0 for n in G.nodes()}
    steps = [dict(col)]
    for _ in range(n_steps):
        sig  = {n: (col[n], tuple(sorted(col[m] for m in G[n]))) for n in G.nodes()}
        uniq = {s: i for i, s in enumerate(sorted(set(sig.values())))}
        col  = {n: uniq[sig[n]] for n in G.nodes()}
        steps.append(dict(col))
    return steps

step_colors = wl_steps(G_wl, 2)
SHADES      = [G6, G4, G2, G0]       # one shade per color index
LABELCOL    = [G0, "white", "white", "white"]
step_titles = [
    "Step 0\n(initialize — every node gets the same color)",
    "Step 1\n(aggregate + hash — nodes split by degree)",
    "Step 2\n(aggregate + hash — stable coloring reached)",
]

def partition_label(colors):
    groups = {}
    for n, c in colors.items():
        groups.setdefault(c, []).append(n)
    return "   ".join(f"c{c}: {{{', '.join(str(n) for n in sorted(v))}}}"
                      for c, v in sorted(groups.items()))

fig, axes = plt.subplots(1, 3, figsize=(13, 5.0))
fig.patch.set_facecolor(BG)

for ax, colors, title in zip(axes, step_colors, step_titles):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_title(title, fontsize=10.5, fontweight='bold',
                 color=G0, fontfamily=FONT, pad=8)
    nx.draw_networkx_edges(G_wl, pos_wl, ax=ax, edge_color=G3, width=1.8)
    nc = [SHADES[colors[n]] for n in G_wl.nodes()]
    nx.draw_networkx_nodes(G_wl, pos_wl, ax=ax, node_color=nc,
                           node_size=700, edgecolors=G0, linewidths=1.2)
    for n, (x, y) in pos_wl.items():
        ax.text(x, y, str(n), ha='center', va='center', fontsize=12,
                fontweight='bold', color=LABELCOL[colors[n]],
                fontfamily=FONT, zorder=5)
    # Color identifier printed next to each node: the shade is never the
    # only channel carrying information.
    for n, (x, y) in pos_wl.items():
        ax.text(x + 0.16, y + 0.20, f"c{colors[n]}", fontsize=10,
                fontweight='bold', color=G0, fontfamily=FONT,
                ha='left', va='bottom',
                bbox=dict(boxstyle="round,pad=0.15", fc='white',
                          ec=G5, lw=0.6))
    ax.set_xlim(-0.5, 3.8); ax.set_ylim(-0.35, 2.65)
    ax.text(0.5, 0.01, partition_label(colors), transform=ax.transAxes,
            ha='center', va='bottom', fontsize=9.5, color=G1, fontfamily=FONT)

handles = [mpatches.Patch(facecolor=SHADES[i], edgecolor=G0, label=f"color c{i}")
           for i in range(4)]
fig.tight_layout()
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
           fontsize=9.5, bbox_to_anchor=(0.5, -0.04))
save(fig, "fig11_3_wl_test.png")


# ── Fig 11.2: 1-WL failure case (C6 vs 2×C3) ─────────────────────────────────
print("Figure 11.2 …")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor(BG)

# Left: cycle of 6 nodes (C6)
C6 = nx.cycle_graph(6)
pos_c6 = nx.circular_layout(C6)

# Right: two disconnected triangles (2 x C3)
C3a = nx.cycle_graph(3)
C3b = nx.cycle_graph(range(3, 6))
two_C3 = nx.compose(C3a, C3b)
# Position: two separate triangles side by side
pos_2c3 = {
    0: (-0.6,  0.3),
    1: (-1.0, -0.3),
    2: (-0.2, -0.3),
    3: ( 0.6,  0.3),
    4: ( 0.2, -0.3),
    5: ( 1.0, -0.3),
}

graphs = [C6, two_C3]
positions = [pos_c6, pos_2c3]
titles = [
    "Cycle of 6 nodes (C6)\nAll nodes: degree 2",
    "Two disconnected triangles (2 x C3)\nAll nodes: degree 2"
]

for ax, G, pos, title in zip(axes, graphs, positions, titles):
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold',
                 color=G0, fontfamily=FONT, pad=10)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=G2, width=2.0)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=G1,
                           node_size=700, edgecolors='white', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=13,
                            font_color='white', font_weight='bold',
                            font_family=FONT)

fig.suptitle("1-WL failure case — both graphs are 2-regular,\n"
             "so the WL test assigns identical labels and cannot distinguish them",
             fontsize=11, color=G3, fontstyle='italic', fontfamily=FONT, y=0.02)

# "not equal" symbol between the two
fig.text(0.50, 0.50, "\u2262", ha='center', va='center',
         fontsize=36, color=G2, fontfamily=FONT)
fig.tight_layout()
save(fig, "fig11_2_wl_failure.png")


# ── Fig 11.4: Injective function mapping diagram ───────────────────────────────
print("Figure 11.4 …")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor(BG)

def draw_mapping(ax, title, inputs, outputs, mapping, injective, top):
    ax.set_facecolor(BG); ax.axis('off')
    # ylim must cover the taller of the two columns, plus room for the
    # verdict line below: with 3 inputs and 4 outputs the top output
    # circle used to fall outside the axes and was clipped.
    ax.set_xlim(0, 4); ax.set_ylim(-0.7, top + 0.9)
    ax.set_title(title, fontsize=12, fontweight='bold',
                 color=G0, fontfamily=FONT, pad=10)

    # Draw input nodes
    for i, lbl in enumerate(inputs):
        y = len(inputs) - i
        circ = plt.Circle((0.8, y), 0.28, color=G1, zorder=3)
        ax.add_patch(circ)
        ax.text(0.8, y, lbl, ha='center', va='center', fontsize=11,
                color='white', fontweight='bold', fontfamily=FONT, zorder=4)

    # Draw output nodes
    for i, lbl in enumerate(outputs):
        y = len(outputs) - i
        circ = plt.Circle((3.2, y), 0.28, color=G3, zorder=3)
        ax.add_patch(circ)
        ax.text(3.2, y, lbl, ha='center', va='center', fontsize=11,
                color='white', fontweight='bold', fontfamily=FONT, zorder=4)

    # Draw arrows for the mapping
    for (src_i, dst_i) in mapping:
        src_y = len(inputs)  - src_i
        dst_y = len(outputs) - dst_i
        ax.annotate("", xy=(2.92, dst_y), xytext=(1.08, src_y),
                    arrowprops=dict(arrowstyle="-|>", color=G0,
                                    lw=1.5, mutation_scale=14))

    # Labels
    ax.text(0.8, 0.3, "Input", ha='center', fontsize=9,
            color=G3, fontfamily=FONT)
    ax.text(3.2, 0.3, "Output", ha='center', fontsize=9,
            color=G3, fontfamily=FONT)
    verdict = "\u2713 Injective" if injective else "\u2717 Non-injective"
    clr     = G0 if injective else G3
    ax.text(2.0, -0.1, verdict, ha='center', fontsize=11,
            color=clr, fontweight='bold', fontfamily=FONT)

draw_mapping(axes[0], "Injective function\n(distinct inputs \u2192 distinct outputs)",
             ["A","B","C"], ["1","2","3","4"],
             [(0,0),(1,1),(2,3)], injective=True, top=4)

draw_mapping(axes[1], "Non-injective function\n(two inputs share same output)",
             ["A","B","C"], ["1","2","3"],
             [(0,0),(1,1),(2,1)], injective=False, top=4)

fig.tight_layout()
save(fig, "fig11_4_injective.png")


# ── Fig 11.5: Protein graph visualization ─────────────────────────────────────
print("Figure 11.5 …")

# Synthetic protein-like graph: nodes=amino acids, edges=spatial proximity
# Match PROTEINS dataset stats: avg ~39 nodes, ~145 edges per graph
rng = np.random.default_rng(42)
G_prot = nx.Graph()
n_aa   = 38   # amino acids
G_prot.add_nodes_from(range(n_aa))

# Spatial proximity graph: connect nodes within a threshold
coords = rng.uniform(0, 1, (n_aa, 2))
for i in range(n_aa):
    for j in range(i+1, n_aa):
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist < 0.28:
            G_prot.add_edge(i, j)

pos_prot = {i: tuple(coords[i]) for i in range(n_aa)}
deg_prot = dict(G_prot.degree())
nc_prot_rgb = [(0.15 + 0.60 * (deg_prot.get(n,0) /
                max(deg_prot.values(), default=1)),)*3
               for n in G_prot.nodes()]
ns_prot = [120 + deg_prot.get(n,0)*30 for n in G_prot.nodes()]

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
nx.draw_networkx_edges(G_prot, pos_prot, ax=ax,
                       edge_color=G4, width=0.9, alpha=0.6)
nx.draw_networkx_nodes(G_prot, pos_prot, ax=ax,
                       node_color=nc_prot_rgb, node_size=ns_prot,
                       edgecolors='white', linewidths=0.8)

sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys_r,
                            norm=plt.Normalize(0, max(deg_prot.values(), default=1)))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label('Node degree (amino acid connectivity)', fontsize=9, fontfamily=FONT)

ax.set_title("Schematic protein graph: nodes are amino acids,\n"
             "edges connect residues that are spatially close",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
# The caption in the chapter already states that this layout is illustrative,
# so no in-figure note is needed. Real dataset graphs appear in figs 11.6-11.7.
fig.tight_layout()
save(fig, "fig11_5_protein.png")


# ── Figs 11.6–11.7: GIN vs GCN classification grids ────────────────────────────
print("Training GIN and GCN on PROTEINS for classification grids …")

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.utils import to_networkx

# Load PROTEINS without Constant transform — use native 3-dim features
dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS').shuffle()

torch.manual_seed(0)

# Splits: 80/10/10
n      = len(dataset)
train_d = dataset[:int(n*0.8)]
val_d   = dataset[int(n*0.8):int(n*0.9)]
test_d  = dataset[int(n*0.9):]

train_loader = DataLoader(train_d, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_d,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_d,  batch_size=64, shuffle=False)

# num_node_features = 3 with the default PROTEINS attributes
IN_FEAT = dataset.num_node_features

def accuracy(pred_y, y):
    return (pred_y == y).float().mean().item()


class GIN(torch.nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.conv1 = GINConv(Sequential(
            Linear(IN_FEAT, dim_h), BatchNorm1d(dim_h), ReLU(),
            Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(Sequential(
            Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
            Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(Sequential(
            Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
            Linear(dim_h, dim_h), ReLU()))
        self.lin1  = Linear(dim_h*3, dim_h*3)
        self.lin2  = Linear(dim_h*3, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        h  = torch.cat((h1, h2, h3), dim=1)
        h  = self.lin1(h).relu()
        h  = F.dropout(h, p=0.5, training=self.training)
        return F.log_softmax(self.lin2(h), dim=1)


class GCN(torch.nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.conv1 = GCNConv(IN_FEAT, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin   = Linear(dim_h, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        h = global_mean_pool(h, batch)
        h = F.dropout(h, p=0.5, training=self.training)
        return F.log_softmax(self.lin(h), dim=1)


def train_model(model, epochs=100):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(epochs + 1):
        total_loss, acc_sum = 0.0, 0.0
        for data in train_loader:
            optimizer.zero_grad()
            out  = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            acc_sum    += accuracy(out.argmax(dim=1), data.y)
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            val_loss, val_acc = eval_model(model, val_loader)
            print(f"  Epoch {epoch:>3} | "
                  f"Train Loss: {total_loss/len(train_loader):.2f} | "
                  f"Train Acc: {acc_sum/len(train_loader)*100:>5.2f}% | "
                  f"Val Loss: {val_loss:.2f} | "
                  f"Val Acc: {val_acc*100:.2f}%")
    return model


@torch.no_grad()
def eval_model(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss_sum, acc_sum = 0.0, 0.0
    for data in loader:
        out      = model(data.x, data.edge_index, data.batch)
        loss_sum += criterion(out, data.y).item()
        acc_sum  += accuracy(out.argmax(dim=1), data.y)
    return loss_sum / len(loader), acc_sum / len(loader)


print("Training GIN …")
gin = GIN(dim_h=32)
gin = train_model(gin, epochs=100)
_, test_acc_gin = eval_model(gin, test_loader)
print(f"GIN  test accuracy: {test_acc_gin*100:.2f}%")

print("Training GCN …")
torch.manual_seed(0)
gcn = GCN(dim_h=32)
gcn = train_model(gcn, epochs=100)
_, test_acc_gcn = eval_model(gcn, test_loader)
print(f"GCN  test accuracy: {test_acc_gcn*100:.2f}%")

# Ensemble
gin.eval(); gcn.eval()
acc_gcn = acc_gin = acc_ens = 0.0
for data in test_loader:
    out_gcn = gcn(data.x, data.edge_index, data.batch)
    out_gin = gin(data.x, data.edge_index, data.batch)
    out_ens = (out_gcn + out_gin) / 2
    acc_gcn += accuracy(out_gcn.argmax(dim=1), data.y) / len(test_loader)
    acc_gin += accuracy(out_gin.argmax(dim=1), data.y) / len(test_loader)
    acc_ens += accuracy(out_ens.argmax(dim=1), data.y) / len(test_loader)
print(f"GCN accuracy:     {acc_gcn*100:.2f}%")
print(f"GIN accuracy:     {acc_gin*100:.2f}%")
print(f"Ensemble accuracy:{acc_ens*100:.2f}%")


# ── Classification grid helper ────────────────────────────────────────────────
def plot_classification_grid(model, title, filename):
    """4x4 grid of protein graphs. Dark=correct, light=wrong."""
    model.eval()
    samples = list(test_d[-16:])

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(title, fontsize=13, fontweight='bold',
                 color=G0, fontfamily=FONT, y=1.01)

    correct_color = G0      # dark gray = correct prediction
    wrong_color   = G5      # light gray = wrong prediction

    for i, data in enumerate(samples):
        with torch.no_grad():
            # Add batch dimension (single graph)
            batch = torch.zeros(data.num_nodes, dtype=torch.long)
            out   = model(data.x, data.edge_index, batch)
        pred    = out.argmax(dim=1).item()
        correct = (pred == data.y.item())
        color   = correct_color if correct else wrong_color

        ix = np.unravel_index(i, (4, 4))
        ax = axes[ix]
        ax.set_facecolor(BG); ax.axis('off')

        G = to_networkx(data, to_undirected=True)
        nx.draw_networkx(G,
                         pos=nx.spring_layout(G, seed=0),
                         with_labels=False,
                         node_color=color,
                         node_size=25,
                         edge_color=G3,
                         width=0.7,
                         ax=ax)

    # Legend
    handles = [
        mpatches.Patch(color=correct_color, label='Correct classification'),
        mpatches.Patch(color=wrong_color,   label='Wrong classification'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=11, framealpha=0.95, edgecolor=G5,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout()
    save(fig, filename)


print("Figure 11.6 – GIN classification grid …")
plot_classification_grid(gin, "Graph classifications produced by the GIN model",
                          "fig11_6_gin_classifications.png")

print("Figure 11.7 – GCN classification grid …")
plot_classification_grid(gcn, "Graph classifications produced by the GCN model",
                          "fig11_7_gcn_classifications.png")

print(f"\nAll figures saved to {OUT}")