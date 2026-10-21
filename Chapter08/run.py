"""
Chapter 8 - Scaling Up Graph Neural Networks with GraphSAGE
Hands-On Graph Neural Networks Using Python, Second Edition

Part 1 trains a two-layer GraphSAGE on PubMed with NeighborLoader mini-batches
([10, 10] neighbors, batch size 16) and reports the test accuracy. Part 2
trains a three-layer GraphSAGE on the 20 training graphs of PPI with
[20, 10, 10] neighbors and reports the micro-F1 score on the test graphs.
Both parts print the dataset statistics they rely on and their training time.

Figures saved in figures/, drawn at print width with tools/bookstyle.py:
    fig8_4_pubmed.png      2-hop neighborhood of six PubMed training nodes
    fig8_5_subgraphs.png   the four NeighborLoader subgraphs printed above
    fig8_6_ppi.png         1-hop neighborhood of a protein in a PPI graph

Install the packages listed in requirements.txt before running.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tools'))
from bookstyle import apply, figure, save, FONT, INK, INK_SOFT, G4, G5

plt = apply()
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

GRAY_MAP = {0: '#111111', 1: '#666666', 2: '#BBBBBB'}


def accuracy(pred_y, y):
    return (pred_y == y).float().mean().item()


# =============================================================================
# PART 1 - GraphSAGE on PubMed (transductive)
# =============================================================================

print("\n" + "=" * 60)
print("PART 1 - GraphSAGE on PubMed (transductive)")
print("=" * 60)

from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import k_hop_subgraph, to_networkx, degree

dataset = Planetoid(root='.', name='Pubmed')
data = dataset[0]

print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Training nodes: {sum(data.train_mask).item()}')
print(f'Evaluation nodes: {sum(data.val_mask).item()}')
print(f'Test nodes: {sum(data.test_mask).item()}')
print(f'Edges (directed entries): {data.edge_index.shape[1]}')
print(f'Average degree: {data.edge_index.shape[1] / data.num_nodes:.2f}')

# Figure 8.4: 2-hop neighborhood of two training nodes per class
train_idx = data.train_mask.nonzero().view(-1)
seeds = torch.cat([train_idx[data.y[train_idx] == c][:2]
                   for c in range(dataset.num_classes)])
subset, sub_ei, _, _ = k_hop_subgraph(seeds, 2, data.edge_index,
                                      relabel_nodes=True,
                                      num_nodes=data.num_nodes)
G84 = nx.Graph()
G84.add_nodes_from(range(len(subset)))
G84.add_edges_from(sub_ei.t().tolist())
print(f"\nFigure 8.4: 2-hop neighborhood of training nodes {seeds.tolist()}: "
      f"{G84.number_of_nodes()} nodes, {G84.number_of_edges()} edges")

pos84 = nx.spring_layout(G84, seed=SEED)
colors84 = [GRAY_MAP[int(data.y[subset[n]])] for n in G84.nodes()]
seed_local = {int((subset == s).nonzero()) for s in seeds}
sizes84 = [40 if n in seed_local else 8 for n in G84.nodes()]
fig, ax = figure(aspect=0.75)
ax.axis('off')
nx.draw_networkx_edges(G84, pos84, ax=ax, edge_color=G5, width=0.4)
nx.draw_networkx_nodes(G84, pos84, ax=ax, node_color=colors84,
                       node_size=sizes84, linewidths=0)
handles = [plt.Line2D([0], [0], marker='o', color='white',
                      markerfacecolor=GRAY_MAP[c], markersize=6,
                      label=f'Class {c}') for c in range(3)]
ax.legend(handles=handles, loc='lower right', fontsize=8.5, frameon=False)
save(fig, os.path.join(FIG_DIR, 'fig8_4_pubmed.png'))

# Neighbor sampling
train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=16,
    input_nodes=data.train_mask,
    num_workers=0,
)

subgraphs = list(train_loader)
print()
for i, subgraph in enumerate(subgraphs):
    print(f'Subgraph {i}: {subgraph.num_nodes} nodes, '
          f'{subgraph.edge_index.shape[1]} edges, '
          f'batch_size={subgraph.batch_size}')

# Figure 8.5: the four subgraphs printed above
fig, axes = figure(aspect=1.0, nrows=2, ncols=2)
for idx, (subdata, ax) in enumerate(zip(subgraphs, axes.ravel())):
    G = to_networkx(subdata, to_undirected=True)
    ax.set_title(f'Subgraph {idx}', fontsize=9, color=INK, fontfamily=FONT)
    ax.axis('off')
    nc = [GRAY_MAP[int(y)] for y in subdata.y]
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=0), ax=ax,
                     with_labels=False, node_color=nc, node_size=6,
                     edge_color=G4, width=0.4)
save(fig, os.path.join(FIG_DIR, 'fig8_5_subgraphs.png'))


class GraphSAGE(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.sage1 = SAGEConv(dim_in, dim_h)
        self.sage2 = SAGEConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage2(h, edge_index)
        return F.log_softmax(h, dim=1)

    def fit(self, data, epochs):
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.train()
        for epoch in range(epochs + 1):
            total_loss, val_loss, acc, val_acc = 0, 0, 0, 0
            for batch in train_loader:
                optimizer.zero_grad()
                out = self(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask],
                                 batch.y[batch.train_mask])
                total_loss += loss.item()
                acc += accuracy(out[batch.train_mask].argmax(dim=1),
                                batch.y[batch.train_mask])
                loss.backward()
                optimizer.step()
                val_loss += criterion(out[batch.val_mask],
                                      batch.y[batch.val_mask]).item()
                val_acc += accuracy(out[batch.val_mask].argmax(dim=1),
                                    batch.y[batch.val_mask])
            if epoch % 20 == 0:
                n = len(train_loader)
                print(f'Epoch {epoch:>3} | Train Loss: {total_loss/n:.3f} |'
                      f' Train Acc: {acc/n*100:>6.2f}% |'
                      f' Val Loss: {val_loss/n:.2f} |'
                      f' Val Acc: {val_acc/n*100:.2f}%')

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        return accuracy(out.argmax(dim=1)[data.test_mask],
                        data.y[data.test_mask])


torch.manual_seed(SEED)
graphsage = GraphSAGE(dataset.num_features, 64, dataset.num_classes)
print()
print(graphsage)
start = time.time()
graphsage.fit(data, 200)
print(f"Training time: {time.time() - start:.0f} s")

acc = graphsage.test(data)
print(f'GraphSAGE test accuracy: {acc*100:.2f}%')


# =============================================================================
# PART 2 - GraphSAGE on PPI (inductive, multi-label)
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 - GraphSAGE on PPI (inductive, multi-label)")
print("=" * 60)

from torch_geometric.datasets import PPI
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

train_dataset = PPI(root='.', split='train')
val_dataset = PPI(root='.', split='val')
test_dataset = PPI(root='.', split='test')

all_graphs = list(train_dataset) + list(val_dataset) + list(test_dataset)
n_nodes = sum(g.num_nodes for g in all_graphs)
n_edges = sum(g.edge_index.shape[1] for g in all_graphs)
print(f"Graphs: {len(train_dataset)} train, {len(val_dataset)} val, "
      f"{len(test_dataset)} test ({len(all_graphs)} in total)")
print(f"Nodes in all graphs: {n_nodes}")
print(f"Edges in all graphs: {n_edges} directed entries "
      f"({n_edges // 2} undirected)")
print(f"Average degree: {n_edges / n_nodes:.2f}")
print(f"Node features: {train_dataset.num_features}")
print(f"Labels per node: {train_dataset.num_classes}")
labels_per_protein = torch.cat([g.y for g in all_graphs]).sum(dim=1)
print(f"Active labels per protein: min {int(labels_per_protein.min())}, "
      f"max {int(labels_per_protein.max())}")

# Figure 8.6: 1-hop neighborhood of a protein in the first training graph
g0 = train_dataset[0]
deg0 = degree(g0.edge_index[0], g0.num_nodes)
center = int(torch.argmin((deg0 - 60).abs()))
subset, sub_ei, _, _ = k_hop_subgraph(center, 1, g0.edge_index,
                                      relabel_nodes=True,
                                      num_nodes=g0.num_nodes)
G86 = nx.Graph()
G86.add_nodes_from(range(len(subset)))
G86.add_edges_from(sub_ei.t().tolist())
shade = deg0[subset].numpy()
print(f"\nFigure 8.6: 1-hop neighborhood of protein {center} in training "
      f"graph 0 ({g0.num_nodes} proteins): {G86.number_of_nodes()} nodes, "
      f"{G86.number_of_edges()} edges; degree range "
      f"{int(shade.min())}-{int(shade.max())} in the full graph")

pos86 = nx.spring_layout(G86, seed=SEED)
fig, ax = figure(aspect=0.8)
ax.axis('off')
nx.draw_networkx_edges(G86, pos86, ax=ax, edge_color=G5, width=0.3)
cmap = plt.cm.Greys
norm = plt.Normalize(vmin=shade.min(), vmax=shade.max())
nx.draw_networkx_nodes(G86, pos86, ax=ax, node_color=shade, cmap=cmap,
                       vmin=shade.min(), vmax=shade.max(), node_size=28,
                       edgecolors=G4, linewidths=0.4)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label('Degree in the full graph', fontsize=8.5, color=INK)
cbar.ax.tick_params(labelsize=8.5, colors=INK_SOFT)
save(fig, os.path.join(FIG_DIR, 'fig8_6_ppi.png'))

train_data = Batch.from_data_list(train_dataset)
train_loader = NeighborLoader(
    train_data,
    batch_size=2048,
    shuffle=True,
    num_neighbors=[20, 10, 10],
    num_workers=0,
)
val_loader = DataLoader(val_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=2)


class GraphSAGEPPI(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.sage3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = F.relu(self.sage1(x, edge_index))
        h = F.relu(self.sage2(h, edge_index))
        return self.sage3(h, edge_index)


model = GraphSAGEPPI(
    in_channels=train_dataset.num_features,
    hidden_channels=512,
    out_channels=train_dataset.num_classes,
).to(device)
print()
print(model)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def fit():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader)


@torch.no_grad()
def test(loader):
    model.eval()
    data = next(iter(loader))
    out = model(data.x.to(device), data.edge_index.to(device))
    preds = (out > 0).float().cpu()
    y, pred = data.y.numpy(), preds.numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0.0


start = time.time()
for epoch in range(301):
    loss = fit()
    val_f1 = test(val_loader)
    if epoch % 50 == 0:
        print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Val F1: {val_f1:.4f}')
print(f"Training time: {time.time() - start:.0f} s")

print(f'Test F1 score: {test(test_loader):.4f}')
print("\nDone.")
