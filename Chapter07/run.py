"""
Chapter 7 – Graph Attention Networks
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch-geometric numpy

Figure generation is in figures/generate_figures.py.

Key fixes versus the first edition:
  - accuracy() uses .float().mean() instead of len() division
  - accuracy-per-degree loop guards against empty degree bins
  - GATv2Conv used throughout (GATv2 is strictly more expressive than GAT)
  - NLLLoss paired with the log_softmax output (was CrossEntropyLoss, which
    silently applied log_softmax twice and degraded training accuracy)
  - GCN vs GAT comparison table from reproducible multi-run experiment
"""

import numpy as np
import torch
import torch.nn.functional as F

SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)


# =============================================================================
# PART 1 – Graph attention layer from scratch in NumPy
# =============================================================================

print("=" * 60)
print("PART 1 – Graph attention layer (NumPy)")
print("=" * 60)

np.random.seed(0)

# Adjacency matrix (with self-loops) for the 4-node example graph
A = np.array([
    [1, 1, 1, 1],
    [1, 1, 0, 0],
    [1, 0, 1, 1],
    [1, 0, 1, 1]
])

# Random node features: 4 nodes, 4 features
X = np.random.uniform(-1, 1, (4, 4))
print("\nNode features X:")
print(np.round(X, 4))

# Weight matrices
# W: (d_out, d_in) — maps node features to hidden representations
# W_att: (1, 2*d_out) — applied to concatenated hidden vectors
W     = np.random.uniform(-1, 1, (2, 4))
W_att = np.random.uniform(-1, 1, (1, 4))
print("\nWeight matrix W:", W.shape)
print("Attention vector W_att:", W_att.shape)

# Step 1: Hidden representations H = X @ W.T
H = X @ W.T
print("\nHidden representations H = X @ W.T:")
print(np.round(H, 4))

# Step 2: Get all connected pairs from the adjacency matrix
connections = np.where(A > 0)
print(f"\nNumber of connected pairs: {len(connections[0])}")

# Step 3: Concatenate hidden vectors of source and destination nodes
concat = np.concatenate(
    [H[connections[0]], H[connections[1]]], axis=1
)

# Step 4: Apply attention weight vector to get unnormalised scores
a = W_att @ concat.T   # shape: (1, num_edges)

# Step 5: Leaky ReLU activation
def leaky_relu(x, alpha=0.2):
    return np.maximum(alpha * x, x)

e = leaky_relu(a)

# Step 6: Place scores into matrix — only connected pairs get a score
E = np.zeros(A.shape)
E[connections[0], connections[1]] = e[0]
print("\nUnnormalised attention matrix E:")
print(np.round(E, 4))

# Step 7: Row-wise softmax normalisation
def softmax2D(x, axis):
    e   = np.exp(x - np.expand_dims(np.max(x, axis=axis), axis))
    s   = np.expand_dims(np.sum(e, axis=axis), axis)
    return e / s

W_alpha = softmax2D(E, axis=1)
print("\nNormalised attention scores W_alpha (each row sums to 1):")
print(np.round(W_alpha, 4))
print("Row sums:", np.round(W_alpha.sum(axis=1), 4))

# Step 8: Compute final node embeddings
H_out = A.T @ W_alpha @ X @ W.T
print("\nFinal node embeddings H:")
print(np.round(H_out, 4))
print("\nGraph attention layer complete.")
print("Repeating steps with different W and W_att gives multi-head attention.")


# ── Multi-head attention: concat vs average ─────────────────────────────────
#
# For clarity we simulate the outputs of three attention heads instead of
# re-running the full computation. Each head produces a (num_nodes, d_out)
# matrix of updated node embeddings. Concat vs average is the choice about
# how to combine them:
#   - concat  → shape (num_nodes, K * d_out), typically used in hidden layers
#   - average → shape (num_nodes, d_out),     typically used in the output layer

num_nodes, d_out, K = 4, 2, 3

# Three head outputs (here random, in practice each comes from one attention head)
head_outputs = [np.random.uniform(-1, 1, (num_nodes, d_out)) for _ in range(K)]

# Concatenation (used for hidden layers): (num_nodes, K * d_out)
multi_head_concat = np.concatenate(head_outputs, axis=1)

# Averaging (used for the final layer): (num_nodes, d_out)
multi_head_avg = np.mean(head_outputs, axis=0)

print(f"\nConcatenated shape: {multi_head_concat.shape}")
print(f"Averaged shape:     {multi_head_avg.shape}")


# =============================================================================
# PART 2 – GAT in PyTorch Geometric (Cora + CiteSeer)
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 – GAT node classification (PyTorch Geometric)")
print("=" * 60)

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.utils import degree


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Fraction of correct predictions. Fixed: uses .float().mean()."""
    return (y_pred == y_true).float().mean().item()


class GAT(torch.nn.Module):
    """
    Two-layer Graph Attention Network using GATv2Conv.

    GATv2 (Brody et al. 2021) is a strictly more expressive variant of
    the original GAT (Veličković et al. 2017). It modifies the order of
    operations to compute dynamic rather than static attention.

    Architecture:
      - Layer 1: GATv2Conv with `heads` attention heads (concatenated)
      - Layer 2: GATv2Conv with 1 head (for final classification)
      - Dropout (p=0.6) before each layer, as in the original paper
      - ELU activation between layers
    """

    def __init__(self, dim_in: int, dim_h: int, dim_out: int, heads: int = 8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in,        dim_h,  heads=heads)
        self.gat2 = GATv2Conv(dim_h * heads, dim_out, heads=1)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        h = F.dropout(x, p=0.6, training=self.training)
        h = F.elu(self.gat1(h, edge_index))
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        return F.log_softmax(h, dim=1)

    def fit(self, data, epochs: int, verbose: bool = True):
        # The forward returns log_softmax already, so NLLLoss is the right
        # loss. Using CrossEntropyLoss here would apply log_softmax twice
        # and silently degrade accuracy.
        criterion = torch.nn.NLLLoss()
        # lr=0.01, weight_decay=0.01 as in the original GAT paper for Cora
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=0.01, weight_decay=0.01)
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out  = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc  = accuracy(out[data.train_mask].argmax(dim=1),
                            data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if verbose and epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc  = accuracy(out[data.val_mask].argmax(dim=1),
                                    data.y[data.val_mask])
                print(f"  Epoch {epoch:>3} | Train Loss: {loss:.3f} | "
                      f"Train Acc: {acc*100:>5.2f}% | "
                      f"Val Loss: {val_loss:.2f} | "
                      f"Val Acc: {val_acc*100:.2f}%")

    @torch.no_grad()
    def test(self, data) -> float:
        self.eval()
        out = self(data.x, data.edge_index)
        return accuracy(out.argmax(dim=1)[data.test_mask],
                        data.y[data.test_mask])


# ── Cora ──────────────────────────────────────────────────────────────────────

print("\nLoading Cora …")
dataset_cora = Planetoid(root=".", name="Cora")
data_cora    = dataset_cora[0]

print("Training GAT on Cora …")
torch.manual_seed(SEED)
gat_cora = GAT(dataset_cora.num_features, 32, dataset_cora.num_classes)
print(gat_cora)
gat_cora.fit(data_cora, epochs=100)
acc_cora = gat_cora.test(data_cora)
print(f"\nGAT test accuracy on Cora: {acc_cora*100:.2f}%")


# ── CiteSeer ──────────────────────────────────────────────────────────────────

print("\nLoading CiteSeer …")
dataset_cs = Planetoid(root=".", name="CiteSeer")
data_cs    = dataset_cs[0]

# Quick summary of the degree distribution. The full histogram plot lives
# in figures/generate_figures.py.
degs_np = degree(data_cs.edge_index[0]).numpy().astype(int)
n_iso   = int(np.sum(degs_np == 0))
print(f"  {data_cs.num_nodes} nodes, {n_iso} of which are isolated "
      f"(degree 0), max degree {degs_np.max()}.")

print("Training GAT on CiteSeer …")
torch.manual_seed(SEED)
gat_cs = GAT(dataset_cs.num_features, 16, dataset_cs.num_classes)
gat_cs.fit(data_cs, epochs=100)
acc_cs = gat_cs.test(data_cs)
print(f"\nGAT test accuracy on CiteSeer: {acc_cs*100:.2f}%")


# ── Error analysis: accuracy per node degree ──────────────────────────────────

print("\nRunning error analysis (accuracy per node degree) …")

gat_cs.eval()
with torch.no_grad():
    out_cs = gat_cs(data_cs.x, data_cs.edge_index)

node_degrees = degree(data_cs.edge_index[0],
                      num_nodes=data_cs.num_nodes).numpy()
accuracies, sizes, labels = [], [], []

for i in range(6):
    mask = np.where(node_degrees == i)[0]
    if len(mask) == 0:
        accuracies.append(0.0); sizes.append(0)
    else:
        accuracies.append(
            accuracy(out_cs.argmax(dim=1)[mask], data_cs.y[mask]))
        sizes.append(len(mask))
    labels.append(str(i))

mask_hi = np.where(node_degrees > 5)[0]
if len(mask_hi) == 0:
    accuracies.append(0.0); sizes.append(0)
else:
    accuracies.append(
        accuracy(out_cs.argmax(dim=1)[mask_hi], data_cs.y[mask_hi]))
    sizes.append(len(mask_hi))
labels.append("6+")

# Print summary
print("\nAccuracy by degree bucket:")
for lbl, acc_v, sz in zip(labels, accuracies, sizes):
    bar = "█" * int(acc_v * 20)
    print(f"  degree {lbl:>2}: {acc_v*100:5.1f}%  {bar:<20}  (n={sz})")

# Persist the numbers needed by figures/generate_figures.py, so that
# figure generation does not have to retrain the model.
import json, os
os.makedirs("figures", exist_ok=True)
with open("figures/citeseer_stats.json", "w") as f:
    json.dump({
        "degree_hist": {int(k): int(v) for k, v in
                        zip(*np.unique(node_degrees.astype(int),
                                       return_counts=True))},
        "per_degree_accuracy": {
            "labels":     labels,
            "accuracies": accuracies,
            "sizes":      sizes,
        },
    }, f, indent=2)
print("  Wrote figures/citeseer_stats.json")


# =============================================================================
# PART 3 – Reproducible GCN vs GAT comparison (20 runs)
# =============================================================================

print("\n" + "=" * 60)
print("PART 3 – GCN vs GAT comparison (20 runs each)")
print("=" * 60)


class GCN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = F.relu(self.gcn1(x, edge_index))
        return F.log_softmax(self.gcn2(h, edge_index), dim=1)

    def fit(self, data, epochs):
        # log_softmax is applied in forward; use NLLLoss to match.
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=0.01, weight_decay=5e-4)
        self.train()
        for _ in range(epochs + 1):
            optimizer.zero_grad()
            loss = criterion(self(data.x, data.edge_index)[data.train_mask],
                             data.y[data.train_mask])
            loss.backward(); optimizer.step()

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        return accuracy(out.argmax(dim=1)[data.test_mask],
                        data.y[data.test_mask])


N_RUNS = 20; EPOCHS = 100
comparison = {}

for ds_name, dataset, data in [
    ("Cora",     dataset_cora, data_cora),
    ("CiteSeer", dataset_cs,   data_cs),
]:
    print(f"\n  {ds_name} ({N_RUNS} runs) …")
    gcn_accs, gat_accs = [], []
    for seed in range(N_RUNS):
        torch.manual_seed(seed)

        gcn_m = GCN(dataset.num_features, 16, dataset.num_classes)
        gcn_m.fit(data, EPOCHS)
        gcn_accs.append(gcn_m.test(data))

        gat_m = GAT(dataset.num_features, 16, dataset.num_classes, heads=4)
        gat_m.fit(data, EPOCHS, verbose=False)
        gat_accs.append(gat_m.test(data))

        if (seed + 1) % 5 == 0:
            print(f"    run {seed+1}/{N_RUNS} done")

    comparison[ds_name] = {
        "gcn": (np.mean(gcn_accs)*100, np.std(gcn_accs)*100),
        "gat": (np.mean(gat_accs)*100, np.std(gat_accs)*100),
    }

print("\n" + "-" * 55)
print(f"{'Dataset':<12} {'GCN accuracy':<25} {'GAT accuracy'}")
print("-" * 55)
for ds in ["Cora", "CiteSeer"]:
    r = comparison[ds]
    print(f"{ds:<12} {r['gcn'][0]:.2f}% (±{r['gcn'][1]:.2f}%)        "
          f"{r['gat'][0]:.2f}% (±{r['gat'][1]:.2f}%)")
print("-" * 55)
print("\nDone.")