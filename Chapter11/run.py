"""
Chapter 11: Defining Expressiveness for Graph Classification

Loads PROTEINS, trains a GIN and a GCN baseline on the same split, evaluates
both models and their ensemble on the test set, and saves into figures/:
    fig11_5_protein_graph.png
    fig11_6_gin_classifications.png
    fig11_7_gcn_classifications.png

Run: python run.py
"""

import os
import platform
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

import torch
torch.manual_seed(0)
np.random.seed(0)
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import to_networkx

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(HERE, '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, FONT, BG, INK,
                       INK_SOFT, G4, G5 as BG5, G6 as BG6)

apply()
G0 = "#111111"
G1 = "#333333"
G3 = "#777777"
G5 = "#BBBBBB"


def save(fig, name):
    save_fig(fig, os.path.join(FIG_DIR, name))
    print(f"Saved figures/{name}")


# =============================================================================
# PART 1: Dataset
# =============================================================================

dataset = TUDataset(root='.', name='PROTEINS').shuffle()

print(f'Number of graphs:   {len(dataset)}')
print(f'Number of features: {dataset.num_node_features}')
print(f'Number of classes:  {dataset.num_classes}')

print('\nNode features of the first five nodes of the first graph:')
print(dataset[0].x[:5])
print(f'Distinct feature rows in the dataset: '
      f'{torch.unique(torch.cat([d.x for d in dataset]), dim=0).tolist()}')

n_nodes = np.array([d.num_nodes for d in dataset])
n_edges = np.array([d.num_edges // 2 for d in dataset])
print(f'Average nodes per graph: {n_nodes.mean():.2f}')
print(f'Average edges per graph: {n_edges.mean():.2f}')


# =============================================================================
# PART 2: Splits and loaders
# =============================================================================

n = len(dataset)
train_dataset = dataset[:int(n*0.8)]
val_dataset   = dataset[int(n*0.8):int(n*0.9)]
test_dataset  = dataset[int(n*0.9):]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

print(f'\nTraining set:   {len(train_dataset)} graphs')
print(f'Validation set: {len(val_dataset)} graphs')
print(f'Test set:       {len(test_dataset)} graphs')


# =============================================================================
# PART 3: Accuracy, evaluation, and training
# =============================================================================

def accuracy(pred_y, y):
    return (pred_y == y).float().mean().item()


@torch.no_grad()
def test(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss_sum, acc_sum = 0.0, 0.0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss_sum += criterion(out, data.y).item() * data.num_graphs
        acc_sum  += accuracy(out.argmax(dim=1), data.y) * data.num_graphs
    n = len(loader.dataset)
    return loss_sum / n, acc_sum / n


def train(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, 101):
        model.train()
        loss_sum, acc_sum = 0.0, 0.0
        for data in loader:
            optimizer.zero_grad()
            out  = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * data.num_graphs
            acc_sum  += accuracy(out.argmax(dim=1), data.y) * data.num_graphs
        if epoch % 20 == 0:
            n = len(loader.dataset)
            val_loss, val_acc = test(model, val_loader)
            print(f'Epoch {epoch:>3} | Train Loss: {loss_sum/n:.2f} |'
                  f' Train Acc: {acc_sum/n*100:>5.2f}% |'
                  f' Val Loss: {val_loss:.2f} |'
                  f' Val Acc: {val_acc*100:.2f}%')
    return model


# =============================================================================
# PART 4: Models
# =============================================================================

IN_FEAT = dataset.num_node_features


class GIN(torch.nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.conv1 = GINConv(Sequential(
            Linear(IN_FEAT, dim_h), BatchNorm1d(dim_h), ReLU(),
            Linear(dim_h, dim_h),   ReLU()))
        self.conv2 = GINConv(Sequential(
            Linear(dim_h, dim_h),   BatchNorm1d(dim_h), ReLU(),
            Linear(dim_h, dim_h),   ReLU()))
        self.conv3 = GINConv(Sequential(
            Linear(dim_h, dim_h),   BatchNorm1d(dim_h), ReLU(),
            Linear(dim_h, dim_h),   ReLU()))
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


# =============================================================================
# PART 5: Train and test the GIN
# =============================================================================

print('\nTraining GIN …')
start = time.perf_counter()
gin = GIN(dim_h=32)
gin = train(gin, train_loader)
gin_time = time.perf_counter() - start

_, test_acc_gin = test(gin, test_loader)
print(f'GIN test accuracy: {test_acc_gin*100:.2f}%')


# =============================================================================
# PART 6: Train and test the GCN baseline
# =============================================================================

print('\nTraining GCN …')
torch.manual_seed(0)
start = time.perf_counter()
gcn = GCN(dim_h=32)
gcn = train(gcn, train_loader)
gcn_time = time.perf_counter() - start

_, test_acc_gcn = test(gcn, test_loader)
print(f'GCN test accuracy: {test_acc_gcn*100:.2f}%')


# =============================================================================
# PART 7: Ensemble
# =============================================================================

gin.eval(); gcn.eval()
acc_gcn = acc_gin = acc_ens = 0.0

with torch.no_grad():
    for data in test_loader:
        out_gcn = gcn(data.x, data.edge_index, data.batch)
        out_gin = gin(data.x, data.edge_index, data.batch)
        out_ens = (out_gcn + out_gin) / 2
        acc_gcn += accuracy(out_gcn.argmax(dim=1), data.y) * data.num_graphs
        acc_gin += accuracy(out_gin.argmax(dim=1), data.y) * data.num_graphs
        acc_ens += accuracy(out_ens.argmax(dim=1), data.y) * data.num_graphs

n_test = len(test_loader.dataset)
print(f'\nGCN accuracy:      {acc_gcn/n_test*100:.2f}%')
print(f'GIN accuracy:      {acc_gin/n_test*100:.2f}%')
print(f'GCN+GIN accuracy:  {acc_ens/n_test*100:.2f}%')


# =============================================================================
# PART 8: Agreement between the two models on the test set
# =============================================================================

with torch.no_grad():
    ok_gin, ok_gcn = [], []
    for data in test_loader:
        ok_gin.append(gin(data.x, data.edge_index, data.batch).argmax(dim=1) == data.y)
        ok_gcn.append(gcn(data.x, data.edge_index, data.batch).argmax(dim=1) == data.y)
    ok_gin = torch.cat(ok_gin)
    ok_gcn = torch.cat(ok_gcn)

print(f'\nTest graphs: {n_test}')
print(f'  Both correct:     {int((ok_gin & ok_gcn).sum())}')
print(f'  Only GIN correct: {int((ok_gin & ~ok_gcn).sum())}')
print(f'  Only GCN correct: {int((~ok_gin & ok_gcn).sum())}')
print(f'  Both wrong:       {int((~ok_gin & ~ok_gcn).sum())}')


# =============================================================================
# PART 9: Training time and environment
# =============================================================================

print(f'\nGIN training time (100 epochs): {gin_time:.1f} s')
print(f'GCN training time (100 epochs): {gcn_time:.1f} s')
print(f'Platform:  {platform.platform()}')
print(f'Processor: {platform.processor() or platform.machine()}')
print(f'Python {platform.python_version()}, PyTorch {torch.__version__}, '
      f'threads {torch.get_num_threads()}')


# =============================================================================
# PART 10: Figure 11.5, a protein graph from the dataset
# =============================================================================

median_nodes = int(np.median(n_nodes))
idx = int(np.argmin(np.abs(n_nodes - median_nodes)))
sample = dataset[idx]
G_prot = to_networkx(sample, to_undirected=True)
print(f'\nFigure 11.5 graph: dataset index {idx}, '
      f'{G_prot.number_of_nodes()} nodes, {G_prot.number_of_edges()} edges, '
      f'label {int(sample.y)}')

fig, ax = figure(aspect=0.80)
ax.set_facecolor(BG); ax.axis('off')
pos = nx.spring_layout(G_prot, seed=0)
nx.draw_networkx_edges(G_prot, pos, ax=ax, edge_color=G3, width=0.7)
nx.draw_networkx_nodes(G_prot, pos, ax=ax, node_color=BG5,
                       node_size=55, edgecolors=G3, linewidths=0.5)
ax.set_title('A protein graph from the PROTEINS dataset',
             fontsize=10.5, fontweight='bold', color=INK, fontfamily=FONT)
save(fig, 'fig11_5_protein_graph.png')


# =============================================================================
# PART 11: Figures 11.6 and 11.7, classification grids
# =============================================================================

def plot_classification_grid(model, title, filename):
    model.eval()
    samples = list(test_dataset[-16:])
    fig, axes = figure(aspect=1.06, nrows=4, ncols=4)
    fig.suptitle(title, fontsize=10.5, fontweight='bold',
                 color=INK, fontfamily=FONT)
    wrong = []
    for i, data in enumerate(samples):
        with torch.no_grad():
            batch = torch.zeros(data.num_nodes, dtype=torch.long)
            pred = model(data.x, data.edge_index, batch).argmax(dim=1).item()
        correct = pred == data.y.item()
        if not correct:
            wrong.append(i + 1)
        ax = axes[np.unravel_index(i, (4, 4))]
        ax.set_facecolor(BG); ax.axis('off')
        G = to_networkx(data, to_undirected=True)
        nx.draw_networkx(G,
                         pos=nx.spring_layout(G, seed=0),
                         with_labels=False,
                         node_color=G4 if correct else BG6,
                         node_size=10,
                         edge_color=G3,
                         width=0.4,
                         ax=ax)
    handles = [
        mpatches.Patch(facecolor=G4, edgecolor=G3, label='Correct'),
        mpatches.Patch(facecolor=BG6, edgecolor=G3, label='Wrong'),
    ]
    # Under the grid the legend sat on the bottom row of graphs; above it,
    # the band beside the title is empty.
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=8,
               frameon=False, handlelength=1.2, handletextpad=0.4,
               bbox_to_anchor=(0.5, 0.975))
    save(fig, filename)
    print(f'  Wrong predictions (grid positions 1-16): {wrong} ({len(wrong)} of 16)')


print()
plot_classification_grid(gin, 'Graph classifications produced by the GIN model',
                         'fig11_6_gin_classifications.png')
plot_classification_grid(gcn, 'Graph classifications produced by the GCN model',
                         'fig11_7_gcn_classifications.png')

print('\nDone.')