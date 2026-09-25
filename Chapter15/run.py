"""
Chapter 15 – Explaining Graph Neural Networks
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch-geometric captum

Fixes versus the first edition (ex Chapter 14):
  - GNNExplainer updated to new PyG >= 2.3 API (Explainer + GNNExplainer)
  - to_captum_input signature corrected (edge_index, not edge_mask)
  - accuracy() uses .float().mean() instead of len() division
  - GitHub URL updated to Chapter15
  - Chapter reference in summary updated ("Chapter 16" for traffic)
  - Fidelity and sparsity metrics added after GNNExplainer
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU

torch.manual_seed(0); np.random.seed(0)


# =============================================================================
# SHARED UTILITY
# =============================================================================

def accuracy(pred_y: torch.Tensor, y: torch.Tensor) -> float:
    """Fixed: .float().mean() instead of len() division."""
    return (pred_y == y).float().mean().item()


# =============================================================================
# PART 1 – GNNExplainer on MUTAG
# =============================================================================

print("=" * 60)
print("PART 1 – GNNExplainer on MUTAG")
print("=" * 60)

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
# New API: Explainer and GNNExplainer from torch_geometric.explain
from torch_geometric.explain import Explainer, GNNExplainer

dataset = TUDataset(root='.', name='MUTAG').shuffle()
train_d  = dataset[:int(len(dataset)*0.8)]
val_d    = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
test_d   = dataset[int(len(dataset)*0.9):]

train_loader = DataLoader(train_d,  batch_size=64, shuffle=True)
val_loader   = DataLoader(val_d,    batch_size=64, shuffle=False)
test_loader  = DataLoader(test_d,   batch_size=64, shuffle=False)

print(f"MUTAG: {len(dataset)} graphs, "
      f"{dataset.num_features} features, "
      f"{dataset.num_classes} classes")


class GIN(torch.nn.Module):
    """GIN model from Chapter 9 — reused unchanged."""

    def __init__(self, dim_h):
        super().__init__()
        self.conv1 = GINConv(Sequential(
            Linear(dataset.num_features, dim_h),
            BatchNorm1d(dim_h), ReLU(),
            Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(Sequential(
            Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
            Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(Sequential(
            Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
            Linear(dim_h, dim_h), ReLU()))
        self.lin1  = Linear(dim_h * 3, dim_h * 3)
        self.lin2  = Linear(dim_h * 3, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        h  = torch.cat([global_add_pool(h, batch)
                        for h in [h1, h2, h3]], dim=1)
        h  = self.lin1(h).relu()
        h  = F.dropout(h, p=0.5, training=self.training)
        return F.log_softmax(self.lin2(h), dim=1)


def train_gin(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(101):
        total_loss = acc_sum = 0.0
        for data in loader:
            optimizer.zero_grad()
            out        = model(data.x, data.edge_index, data.batch)
            loss       = criterion(out, data.y)
            total_loss += loss.item(); acc_sum += accuracy(out.argmax(1), data.y)
            loss.backward(); optimizer.step()
        if epoch % 20 == 0:
            vl, va = test_gin(model, val_loader)
            print(f"  Epoch {epoch:>3} | "
                  f"Train Acc: {acc_sum/len(loader)*100:.2f}% | "
                  f"Val Acc: {va*100:.2f}%")
    return model


@torch.no_grad()
def test_gin(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    ls = ac = 0.0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        ls += criterion(out, data.y).item()
        ac += accuracy(out.argmax(1), data.y)
    return ls / len(loader), ac / len(loader)


print("\nTraining GIN on MUTAG …")
model_gin = GIN(dim_h=32)
model_gin = train_gin(model_gin, train_loader)
_, test_acc = test_gin(model_gin, test_loader)
print(f"\nGIN test accuracy: {test_acc*100:.2f}%")

# ── GNNExplainer — new API ─────────────────────────────────────────────────────
# Old API (broken in PyG >= 2.3):
#   from torch_geometric.nn import GNNExplainer
#   explainer = GNNExplainer(model, epochs=100, num_hops=1)
#   feature_mask, edge_mask = explainer.explain_graph(...)
#
# New API (PyG >= 2.3):
explainer = Explainer(
    model=model_gin,
    # edge_size and node_feat_size are the sparsity regularisation coefficients.
    # PyG defaults (0.005) are too soft: GNNExplainer keeps almost every edge.
    # Values around 0.1 push toward genuinely sparse, chirurgical explanations
    # that match the sparsity levels reported in the GNNExplainer paper.
    algorithm=GNNExplainer(epochs=200, edge_size=0.1, node_feat_size=0.1),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='graph',
        return_type='log_probs',
    ),
)

# Illustrative single-graph explanation (for showing node/edge masks in the book)
data_explain = dataset[-1]
# For a single graph, batch is all zeros (all nodes belong to graph 0).
# We pass it as a kwarg so PyG's Explainer propagates it to the model,
# which needs batch for global_add_pool.
batch_single = torch.zeros(data_explain.x.size(0), dtype=torch.long)
explanation  = explainer(
    x=data_explain.x,
    edge_index=data_explain.edge_index,
    batch=batch_single,
    index=None,
)

print("\nNode mask (feature importance per atom type):")
print(explanation.node_mask.squeeze())
print("\nEdge mask shape:", explanation.edge_mask.shape)
print("Top edges by importance:", explanation.edge_mask.topk(3).indices.tolist())

# =============================================================================
# Aggregated fidelity and sparsity over multiple test graphs.
# Single-graph fidelity saturates at 0 or 1 because it measures a discrete
# prediction change; averaging over many graphs gives statistically meaningful
# values comparable to those reported in the GNNExplainer literature.
# =============================================================================
from torch_geometric.explain import fidelity

print("\nComputing aggregated metrics over the last 30 graphs of MUTAG …")
fid_plus_list, fid_minus_list = [], []
edge_sparsity_list, node_sparsity_list = [], []
n_samples = min(30, len(dataset))
for idx in range(len(dataset) - n_samples, len(dataset)):
    g            = dataset[idx]
    batch_single = torch.zeros(g.x.size(0), dtype=torch.long)
    expl_i       = explainer(x=g.x, edge_index=g.edge_index,
                              batch=batch_single, index=None)
    fp, fm       = fidelity(explainer, expl_i)
    fid_plus_list.append(float(fp))
    fid_minus_list.append(float(fm))
    # Two sparsity metrics: edge-based and node-feature based.
    # GNNExplainer can be selective on features but not on edges, or vice versa,
    # so reporting both gives the reader a complete picture.
    edge_sparsity_list.append(
        1 - (expl_i.edge_mask > 0.5).float().mean().item()
    )
    node_sparsity_list.append(
        1 - (expl_i.node_mask > 0.5).float().mean().item()
    )

import statistics as _s
print(f"\nFidelity+:       {_s.mean(fid_plus_list):.4f}  "
      f"(std {_s.stdev(fid_plus_list):.4f}, n={n_samples})")
print(f"Fidelity-:       {_s.mean(fid_minus_list):.4f}  "
      f"(std {_s.stdev(fid_minus_list):.4f})")
print(f"Edge sparsity:   {_s.mean(edge_sparsity_list):.4f}  "
      f"(std {_s.stdev(edge_sparsity_list):.4f})")
print(f"Node sparsity:   {_s.mean(node_sparsity_list):.4f}  "
      f"(std {_s.stdev(node_sparsity_list):.4f})")


# =============================================================================
# PART 2 – Integrated gradients on Amazon Photo via Captum
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 – Integrated gradients on Amazon Photo (Captum)")
print("=" * 60)

from captum.attr import IntegratedGradients
from torch_geometric.datasets import Amazon
from torch_geometric.nn import GCNConv, to_captum_model, to_captum_input

# Amazon Photo: 7650 nodes, ~238k edges, 745 features, 8 classes.
# Nodes represent photography-related Amazon products; edges connect products
# that are frequently co-purchased. Task: classify each product's category.
# Substituted for Twitch EN, whose upstream host (graphmining.ai) went offline.
dataset_tw = Amazon('.', name='Photo')
data_tw    = dataset_tw[0]
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_tw    = data_tw.to(device)

print(f"\nAmazon Photo: {data_tw.num_nodes} nodes, "
      f"{data_tw.num_edges} edges, "
      f"{dataset_tw.num_features} features, "
      f"{dataset_tw.num_classes} classes")


class GCN(torch.nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.conv1 = GCNConv(dataset_tw.num_features, dim_h)
        self.conv2 = GCNConv(dim_h, dataset_tw.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        return F.log_softmax(self.conv2(h, edge_index), dim=1)


model_gcn = GCN(64).to(device)
optimizer = torch.optim.Adam(model_gcn.parameters(),
                              lr=0.01, weight_decay=5e-4)

print("\nTraining GCN on Amazon Photo (200 epochs) …")
for epoch in range(200):
    model_gcn.train()
    optimizer.zero_grad()
    out  = model_gcn(data_tw.x, data_tw.edge_index)
    loss = F.nll_loss(out, data_tw.y)
    loss.backward(); optimizer.step()

model_gcn.eval()
with torch.no_grad():
    out = model_gcn(data_tw.x, data_tw.edge_index)
acc = accuracy(out.argmax(dim=1), data_tw.y)
print(f"Accuracy: {acc*100:.2f}%")


def explain_node(node_idx):
    """Compute integrated gradients for a specific node."""
    # to_captum_model: wraps the PyG model for Captum
    captum_model = to_captum_model(model_gcn,
                                    mask_type='node_and_edge',
                                    output_idx=node_idx)
    ig = IntegratedGradients(captum_model)

    # to_captum_input signature (PyG >= 2.3):
    #   to_captum_input(x, edge_index, mask_type)
    # NOT: to_captum_input(x, edge_mask, mask_type)  ← old signature
    inputs, add_args = to_captum_input(data_tw.x, data_tw.edge_index,
                                        mask_type='node_and_edge')

    attr_node, attr_edge = ig.attribute(
        inputs,
        target=int(data_tw.y[node_idx]),
        additional_forward_args=add_args,
        internal_batch_size=1,
    )

    # Normalise to [0, 1]
    attr_node = attr_node.squeeze(0).abs().sum(dim=1)
    attr_node = (attr_node / attr_node.max()).detach().cpu()

    attr_edge = attr_edge.squeeze(0).abs()
    attr_edge = (attr_edge / attr_edge.max()).detach().cpu()

    return attr_node, attr_edge


print("\nExplaining node 0 …")
attr_n0, attr_e0 = explain_node(0)
top5_nodes = attr_n0.topk(5).indices.tolist()
top5_edges = attr_e0.topk(5).indices.tolist()
top5_classes = [int(data_tw.y[n]) for n in top5_nodes]
print(f"  Node 0 class: {int(data_tw.y[0])}")
print(f"  Top-5 nodes by attribution: {top5_nodes}")
print(f"  Top-5 classes:              {top5_classes}")
same_class_n0 = sum(1 for c in top5_classes[1:] if c == int(data_tw.y[0]))
print(f"  Same-class neighbours (top 4 excl. self): {same_class_n0}/4")
print(f"  Top-5 edges by attribution: {top5_edges}")

print("\nExplaining node 101 …")
attr_n101, attr_e101 = explain_node(101)
top5_nodes = attr_n101.topk(5).indices.tolist()
top5_edges = attr_e101.topk(5).indices.tolist()
top5_classes = [int(data_tw.y[n]) for n in top5_nodes]
print(f"  Node 101 class: {int(data_tw.y[101])}")
print(f"  Top-5 nodes by attribution: {top5_nodes}")
print(f"  Top-5 classes:              {top5_classes}")
same_class_n101 = sum(1 for c in top5_classes[1:] if c == int(data_tw.y[101]))
print(f"  Same-class neighbours (top 4 excl. self): {same_class_n101}/4")
print(f"  Top-5 edges by attribution: {top5_edges}")

print("\nDone.")