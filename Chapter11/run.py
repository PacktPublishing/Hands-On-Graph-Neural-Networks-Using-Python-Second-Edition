"""
Chapter 11 – Defining Expressiveness for Graph Classification
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch-geometric scikit-learn networkx matplotlib numpy

Fixes versus the first edition:
  - accuracy() uses .float().mean() instead of len() division
  - val_loss computed once per epoch (outside the batch loop)
  - GCN baseline for graph classification implemented explicitly
  - node_color in visualisation grid converted to grayscale
  - accuracy() defined before train() to avoid ordering confusion
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

torch.manual_seed(0); np.random.seed(0)

from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import to_networkx


# =============================================================================
# DATASET
# =============================================================================

print("Loading PROTEINS dataset …")

# PROTEINS provides 3-dimensional node attributes (atomic and secondary
# structure features). Earlier rewrites applied Constant(value=1, cat=False)
# as a defensive measure, but cat=False replaces the original features
# with a constant, collapsing GCN training to the majority baseline.
# The default loader gives the correct 3-D features.
dataset = TUDataset(root='.', name='PROTEINS').shuffle()

print(f"Number of graphs:   {len(dataset)}")
print(f"Number of features: {dataset.num_node_features}")
print(f"Number of classes:  {dataset.num_classes}")

n = len(dataset)
train_dataset = dataset[:int(n*0.8)]
val_dataset   = dataset[int(n*0.8):int(n*0.9)]
test_dataset  = dataset[int(n*0.9):]

print(f"\nTraining set:   {len(train_dataset)} graphs")
print(f"Validation set: {len(val_dataset)} graphs")
print(f"Test set:       {len(test_dataset)} graphs")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

IN_FEAT = dataset.num_node_features   # 3 with the default PROTEINS attributes


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def accuracy(pred_y: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of correct predictions. Fixed: .float().mean()."""
    return (pred_y == y).float().mean().item()


@torch.no_grad()
def test(model, loader):
    """Evaluate model on a DataLoader. Returns (mean_loss, mean_accuracy)."""
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss_sum = acc_sum = 0.0
    for data in loader:
        out      = model(data.x, data.edge_index, data.batch)
        loss_sum += criterion(out, data.y).item()
        acc_sum  += accuracy(out.argmax(dim=1), data.y)
    return loss_sum / len(loader), acc_sum / len(loader)


def train(model, loader, epochs=100):
    """
    Train model for `epochs` epochs with mini-batching.
    Validation is computed once per epoch (outside the batch loop)
    to avoid the first-edition bug of re-computing it per batch.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(epochs + 1):
        total_loss = acc_sum = 0.0
        for data in loader:
            optimizer.zero_grad()
            out        = model(data.x, data.edge_index, data.batch)
            loss       = criterion(out, data.y)
            total_loss += loss.item()
            acc_sum    += accuracy(out.argmax(dim=1), data.y)
            loss.backward()
            optimizer.step()
        # Validation: computed once after all training batches
        val_loss, val_acc = test(model, val_loader)
        if epoch % 20 == 0:
            n_b = len(loader)
            print(f"  Epoch {epoch:>3} | "
                  f"Train Loss: {total_loss/n_b:.2f} | "
                  f"Train Acc: {acc_sum/n_b*100:>5.2f}% | "
                  f"Val Loss: {val_loss:.2f} | "
                  f"Val Acc: {val_acc*100:.2f}%")
    return model


# =============================================================================
# MODELS
# =============================================================================

class GIN(torch.nn.Module):
    """
    Graph Isomorphism Network with 3 GINConv layers.
    Uses sum global pooling at each layer and concatenates the results —
    the most expressive pooling strategy according to Xu et al. (2018).
    """

    def __init__(self, dim_h: int):
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
        self.lin1  = Linear(dim_h * 3, dim_h * 3)
        self.lin2  = Linear(dim_h * 3, dataset.num_classes)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        # Sum pooling at each layer — captures multi-resolution structure
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        h  = torch.cat((h1, h2, h3), dim=1)
        h  = self.lin1(h).relu()
        h  = F.dropout(h, p=0.5, training=self.training)
        return F.log_softmax(self.lin2(h), dim=1)


class GCN(torch.nn.Module):
    """
    GCN baseline with global mean pooling.
    Strictly less expressive than GIN according to the WL test framework.
    """

    def __init__(self, dim_h: int):
        super().__init__()
        self.conv1 = GCNConv(IN_FEAT, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin   = Linear(dim_h, dataset.num_classes)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        h = global_mean_pool(h, batch)
        h = F.dropout(h, p=0.5, training=self.training)
        return F.log_softmax(self.lin(h), dim=1)


# =============================================================================
# TRAIN AND EVALUATE
# =============================================================================

print("\n" + "=" * 60)
print("Training GIN (dim_h=32, 100 epochs) …")
print("=" * 60)
gin = GIN(dim_h=32)
gin = train(gin, train_loader, epochs=100)
_, test_acc_gin = test(gin, test_loader)
print(f"\nGIN  test accuracy: {test_acc_gin*100:.2f}%")

print("\n" + "=" * 60)
print("Training GCN baseline (dim_h=32, 100 epochs) …")
print("=" * 60)
torch.manual_seed(0)
gcn = GCN(dim_h=32)
gcn = train(gcn, train_loader, epochs=100)
_, test_acc_gcn = test(gcn, test_loader)
print(f"\nGCN  test accuracy: {test_acc_gcn*100:.2f}%")

# Ensemble
gin.eval(); gcn.eval()
acc_gcn_ens = acc_gin_ens = acc_ens = 0.0
for data in test_loader:
    with torch.no_grad():
        out_gcn = gcn(data.x, data.edge_index, data.batch)
        out_gin = gin(data.x, data.edge_index, data.batch)
        out_ens = (out_gcn + out_gin) / 2
    acc_gcn_ens += accuracy(out_gcn.argmax(dim=1), data.y) / len(test_loader)
    acc_gin_ens += accuracy(out_gin.argmax(dim=1), data.y) / len(test_loader)
    acc_ens     += accuracy(out_ens.argmax(dim=1), data.y) / len(test_loader)

print("\n" + "-" * 40)
print(f"GCN accuracy:      {acc_gcn_ens*100:.2f}%")
print(f"GIN accuracy:      {acc_gin_ens*100:.2f}%")
print(f"GCN+GIN accuracy:  {acc_ens*100:.2f}%")
print("-" * 40)
print("Note: ensemble results vary with the random seed.")
print("Report mean ± std across multiple seeds for a reliable comparison.")


# =============================================================================
# VISUALISE CLASSIFICATION GRIDS (grayscale)
# =============================================================================

def plot_classification_grid(model, title: str, filename: str):
    """
    4x4 grid of protein graphs from the test set.
    Dark grey = correct classification, light grey = wrong classification.
    """
    model.eval()
    samples = list(test_dataset[-16:])
    correct_color = "#111111"   # dark grey
    wrong_color   = "#BBBBBB"   # light grey

    fig, axes = plt.subplots(4, 4, figsize=(12, 12), dpi=150)
    fig.patch.set_facecolor('white')
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    for i, data in enumerate(samples):
        with torch.no_grad():
            batch = torch.zeros(data.num_nodes, dtype=torch.long)
            out   = model(data.x, data.edge_index, batch)
        pred    = out.argmax(dim=1).item()
        correct = (pred == data.y.item())
        color   = correct_color if correct else wrong_color

        ix = np.unravel_index(i, (4, 4))
        ax = axes[ix]
        ax.set_facecolor('white'); ax.axis('off')

        G = to_networkx(data, to_undirected=True)
        nx.draw_networkx(G,
                         pos=nx.spring_layout(G, seed=0),
                         with_labels=False,
                         node_color=color,
                         node_size=20,
                         edge_color="#777777",
                         width=0.7,
                         ax=ax)

    handles = [
        mpatches.Patch(color=correct_color, label='Correct classification'),
        mpatches.Patch(color=wrong_color,   label='Wrong classification'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=11, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"Saved {filename}")


plot_classification_grid(gin, "Graph classifications — GIN",
                          "gin_classifications.png")
plot_classification_grid(gcn, "Graph classifications — GCN",
                          "gcn_classifications.png")

print("\nDone.")