"""
Chapter 13 – Learning from Heterogeneous Graphs
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch-geometric numpy

Fixes versus the first edition:
  - accuracy() uses .float().mean() instead of .sum() / mask.sum()
  - Reference [5] corrected: Wang et al. (2019) WWW, not Liu et al. (2021)
  - GAT import aliased as PyGGAT to avoid name clash with Chapter 7 custom class
  - Comparison table added across all three approaches
  - Opening paragraph updated to reflect new chapter position (Ch 13)
  - torch.manual_seed(0) reset before each PART for cell-by-cell determinism
"""

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import nn
from torch_geometric.datasets import DBLP
from torch_geometric.nn import (
    GAT as PyGGAT,
    GATConv, Linear, to_hetero,
    HANConv, MessagePassing
)
from torch_geometric.utils import add_self_loops, degree

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# UTILITY: shared accuracy function
# Fixed: .float().mean() instead of .sum() / mask.sum()
# =============================================================================

def accuracy(pred: torch.Tensor, mask: torch.Tensor,
             labels: torch.Tensor) -> float:
    return (pred[mask] == labels[mask]).float().mean().item()


# =============================================================================
# PART 0 – MPNN framework: GCNConv from scratch via MessagePassing
# =============================================================================

print("\n" + "=" * 60)
print("PART 0 – GCNConv via MessagePassing (illustration)")
print("=" * 60)


class GCNConv(MessagePassing):
    """
    GCN layer implemented using PyG's MessagePassing base class.
    Demonstrates the three MPNN steps explicitly:
      message()    -> normalise neighbour features
      aggregate()  -> sum (specified via aggr='add')
      update()     -> identity (self-loops handle target node)
    """

    def __init__(self, dim_in: int, dim_h: int):
        super().__init__(aggr='add')
        self.linear = torch.nn.Linear(dim_in, dim_h, bias=False)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.linear(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


conv = GCNConv(16, 32)
print(f"GCNConv via MessagePassing: {conv}")
x_dummy = torch.randn(10, 16)
ei_dummy = torch.randint(0, 10, (2, 20))
out_dummy = conv(x_dummy, ei_dummy)
print(f"Input: {x_dummy.shape}  Output: {out_dummy.shape}  ok")


# =============================================================================
# PART 1 – Homogeneous GAT baseline with meta-path
# =============================================================================

torch.manual_seed(0)
print("\n" + "=" * 60)
print("PART 1 – Homogeneous GAT + Author-Paper-Author meta-path")
print("=" * 60)

# Author-Paper-Author: connects authors who have written the same paper
metapaths = [[('author', 'paper'), ('paper', 'author')]]
transform  = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True)

dataset1 = DBLP('.', transform=transform)
data1    = dataset1[0].to(device)
print(data1)

# PyGGAT alias avoids name clash with any custom GAT class from Chapter 7
model1     = PyGGAT(in_channels=-1, hidden_channels=64,
                    out_channels=4, num_layers=1).to(device)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001, weight_decay=0.001)


@torch.no_grad()
def test1(mask):
    model1.eval()
    pred = model1(
        data1.x_dict['author'],
        data1.edge_index_dict[('author', 'metapath_0', 'author')]
    ).argmax(dim=-1)
    return accuracy(pred, mask, data1['author'].y)


print("\nTraining homogeneous GAT …")
for epoch in range(101):
    model1.train()
    optimizer1.zero_grad()
    out  = model1(data1.x_dict['author'],
                  data1.edge_index_dict[('author', 'metapath_0', 'author')])
    mask = data1['author'].train_mask
    loss = F.cross_entropy(out[mask], data1['author'].y[mask])
    loss.backward()
    optimizer1.step()
    if epoch % 20 == 0:
        print(f"  Epoch {epoch:>3} | Loss: {loss:.4f} | "
              f"Train: {test1(data1['author'].train_mask)*100:.2f}% | "
              f"Val: {test1(data1['author'].val_mask)*100:.2f}%")

acc1 = test1(data1['author'].test_mask)
print(f"\nHomogeneous GAT test accuracy: {acc1*100:.2f}%")


# =============================================================================
# PART 2 – Heterogeneous GAT via to_hetero()
# =============================================================================

torch.manual_seed(0)
print("\n" + "=" * 60)
print("PART 2 – Heterogeneous GAT via to_hetero()")
print("=" * 60)

dataset2 = DBLP(root='.')
data2    = dataset2[0]

# conference nodes have no features in the original dataset
# zero-padding is the standard convention for feature-less node types
data2['conference'].x = torch.zeros(20, 1)
data2 = data2.to(device)


class HeteroGAT(nn.Module):
    def __init__(self, dim_h, dim_out):
        super().__init__()
        # (-1, -1): lazy init, PyG infers input dims from each node type
        self.conv   = GATConv((-1, -1), dim_h, add_self_loops=False)
        self.linear = nn.Linear(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index).relu()
        return self.linear(h)


model2     = HeteroGAT(dim_h=64, dim_out=4)
model2     = to_hetero(model2, data2.metadata(), aggr='sum').to(device)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001, weight_decay=0.001)


@torch.no_grad()
def test2(mask):
    model2.eval()
    pred = model2(data2.x_dict, data2.edge_index_dict)['author'].argmax(dim=-1)
    return accuracy(pred, mask, data2['author'].y)


print("\nTraining heterogeneous GAT …")
for epoch in range(101):
    model2.train()
    optimizer2.zero_grad()
    out  = model2(data2.x_dict, data2.edge_index_dict)['author']
    mask = data2['author'].train_mask
    loss = F.cross_entropy(out[mask], data2['author'].y[mask])
    loss.backward()
    optimizer2.step()
    if epoch % 20 == 0:
        print(f"  Epoch {epoch:>3} | Loss: {loss:.4f} | "
              f"Val: {test2(data2['author'].val_mask)*100:.2f}%")

acc2 = test2(data2['author'].test_mask)
print(f"\nHeterogeneous GAT test accuracy: {acc2*100:.2f}%")


# =============================================================================
# PART 3 – HAN (Hierarchical Attention Network)
# =============================================================================

torch.manual_seed(0)
print("\n" + "=" * 60)
print("PART 3 – HAN (Wang et al., 2019)")
print("=" * 60)

dataset3 = DBLP('.')
data3    = dataset3[0]
data3['conference'].x = torch.zeros(20, 1)
data3 = data3.to(device)


class HAN(nn.Module):
    def __init__(self, dim_in, dim_out, dim_h=128, heads=8):
        super().__init__()
        self.han    = HANConv(dim_in, dim_h, heads=heads,
                              dropout=0.6, metadata=data3.metadata())
        self.linear = nn.Linear(dim_h, dim_out)

    def forward(self, x_dict, edge_index_dict):
        out = self.han(x_dict, edge_index_dict)
        return self.linear(out['author'])


# dim_in=-1: lazy initialisation infers each node type's input dimension
model3     = HAN(dim_in=-1, dim_out=4).to(device)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001, weight_decay=0.001)


@torch.no_grad()
def test3(mask):
    model3.eval()
    pred = model3(data3.x_dict, data3.edge_index_dict).argmax(dim=-1)
    return accuracy(pred, mask, data3['author'].y)


print("\nTraining HAN …")
for epoch in range(101):
    model3.train()
    optimizer3.zero_grad()
    out  = model3(data3.x_dict, data3.edge_index_dict)
    mask = data3['author'].train_mask
    loss = F.cross_entropy(out[mask], data3['author'].y[mask])
    loss.backward()
    optimizer3.step()
    if epoch % 20 == 0:
        print(f"  Epoch {epoch:>3} | Loss: {loss:.4f} | "
              f"Train: {test3(data3['author'].train_mask)*100:.2f}% | "
              f"Val: {test3(data3['author'].val_mask)*100:.2f}%")

acc3 = test3(data3['author'].test_mask)
print(f"\nHAN test accuracy: {acc3*100:.2f}%")


# =============================================================================
# COMPARISON
# =============================================================================

print("\n" + "=" * 60)
print("COMPARISON — Author classification on DBLP")
print("=" * 60)
print(f"{'Model':<30} {'Test accuracy':>15}")
print("-" * 47)
print(f"{'GAT (homogeneous + meta-path)':<30} {acc1*100:>14.2f}%")
print(f"{'Het-GAT (to_hetero())':<30} {acc2*100:>14.2f}%  (+{(acc2-acc1)*100:.2f}%)")
print(f"{'HAN':<30} {acc3*100:>14.2f}%  (+{(acc3-acc1)*100:.2f}%)")
print("-" * 47)
print("\nDone.")