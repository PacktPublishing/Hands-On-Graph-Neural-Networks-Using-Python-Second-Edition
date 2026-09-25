"""
Chapter 5 – Including Node Features with Vanilla Neural Networks
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch-geometric scikit-learn pandas matplotlib numpy

Key fixes versus the first edition:
  - Dense adjacency matrix replaced with sparse (crash-safe for large graphs)
  - Facebook Page-Page masks use proper boolean tensors (PyG >= 2.0 compatible)
  - accuracy() uses .float().mean() instead of len() division
  - Results table generated from a real 20-run reproducible experiment
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Fraction of correct predictions."""
    return (y_pred == y_true).float().mean().item()


def build_sparse_adj(data) -> torch.Tensor:
    """
    Build a sparse adjacency matrix A_tilde = A + I (with self-loops).

    Uses torch.sparse_coo_tensor so it scales to graphs with tens of
    thousands of nodes without materialising a dense (N x N) matrix.
    The first edition used to_dense_adj(), which produces a ~2 GB tensor
    for Facebook Page-Page and crashes on most machines.
    """
    from torch_geometric.utils import add_self_loops
    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    n    = data.num_nodes
    vals = torch.ones(edge_index.size(1))
    adj  = torch.sparse_coo_tensor(edge_index, vals, (n, n))
    return adj.coalesce()


# =============================================================================
# PART 1 – Load and inspect datasets
# =============================================================================

print("=" * 60)
print("PART 1 – Datasets")
print("=" * 60)

from torch_geometric.datasets import Planetoid

# ── Cora ─────────────────────────────────────────────────────────────────────

dataset_cora = Planetoid(root=".", name="Cora")
data_cora    = dataset_cora[0]

print("\nCora:")
print(f"  Nodes:    {data_cora.x.shape[0]}")
print(f"  Features: {dataset_cora.num_features}")
print(f"  Classes:  {dataset_cora.num_classes}")
print(f"  Directed: {data_cora.is_directed()}")
print(f"  Isolated: {data_cora.has_isolated_nodes()}")
print(f"  Loops:    {data_cora.has_self_loops()}")
print(f"  Train/Val/Test: {data_cora.train_mask.sum().item()} / "
      f"{data_cora.val_mask.sum().item()} / "
      f"{data_cora.test_mask.sum().item()}")

# ── Facebook Page-Page (rebuilt from the original MUSAE files on GitHub) ─────
#
# PyG's built-in FacebookPagePage downloader points at graphmining.ai, which
# has been offline for a while. Instead we fetch the three raw MUSAE files
# straight from the author's public repository (long-term stable) and
# assemble the PyG Data object ourselves. This makes the chapter
# self-contained: we depend only on the original source, not on a mirror.

import json, os, ssl
import certifi
from urllib.request import urlopen
from torch_geometric.data import Data

MUSAE_BASE = ("https://raw.githubusercontent.com/benedekrozemberczki/"
              "MUSAE/master/input")
MUSAE_FILES = {
    "musae_facebook_edges.csv":   f"{MUSAE_BASE}/edges/facebook_edges.csv",
    "musae_facebook_features.json": f"{MUSAE_BASE}/features/facebook.json",
    "musae_facebook_target.csv":  f"{MUSAE_BASE}/target/facebook_target.csv",
}

class FacebookPagePageSNAP:
    """A drop-in replacement for torch_geometric.datasets.FacebookPagePage
    that rebuilds the dataset from the raw MUSAE files on GitHub.

    Exposes the two attributes the rest of this chapter reads
    (num_features, num_classes) plus a [0] Data object, so the surrounding
    code does not change.
    """

    def __init__(self, root: str = "."):
        raw_dir = os.path.join(root, "facebook_large")
        os.makedirs(raw_dir, exist_ok=True)

        # Explicit certifi SSL context: macOS Python installs don't always
        # find the system CA bundle, so we hand one in ourselves.
        ctx = ssl.create_default_context(cafile=certifi.where())
        for local_name, url in MUSAE_FILES.items():
            local_path = os.path.join(raw_dir, local_name)
            if not os.path.exists(local_path):
                print(f"Downloading {local_name} …")
                with urlopen(url, context=ctx) as resp:
                    with open(local_path, "wb") as f:
                        f.write(resp.read())

        edges_p  = os.path.join(raw_dir, "musae_facebook_edges.csv")
        feats_p  = os.path.join(raw_dir, "musae_facebook_features.json")
        target_p = os.path.join(raw_dir, "musae_facebook_target.csv")

        # Edges: the CSV lists each undirected edge once. We stack both
        # directions into a [2, 2*E] tensor.
        edges_df   = pd.read_csv(edges_p)
        src        = torch.tensor(edges_df["id_1"].values, dtype=torch.long)
        dst        = torch.tensor(edges_df["id_2"].values, dtype=torch.long)
        edge_index = torch.stack([torch.cat([src, dst]),
                                  torch.cat([dst, src])], dim=0)

        # Features: multi-hot over the union of all feature IDs. The MUSAE
        # archive stores each node's features as a list of integer IDs,
        # so we widen them into a fixed-length binary vector.
        with open(feats_p) as f:
            feats_raw = {int(k): v for k, v in json.load(f).items()}
        num_nodes = max(feats_raw) + 1
        max_feat  = max(fid for lst in feats_raw.values() for fid in lst) + 1
        x_full = np.zeros((num_nodes, max_feat), dtype=np.float32)
        for node_id, feat_ids in feats_raw.items():
            x_full[node_id, feat_ids] = 1.0

        # Reduce the multi-hot matrix to 128 dense dimensions with
        # TruncatedSVD. The .npz that PyG's built-in downloader used to serve
        # shipped 128-dim precomputed embeddings; reducing here keeps the
        # dimensionality (and the downstream MLP/GNN sizes) consistent with
        # the classical setup for this dataset.
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=128, random_state=0)
        x   = torch.from_numpy(svd.fit_transform(x_full)).float()

        # Labels: 4 page types mapped to consecutive integers.
        target_df    = pd.read_csv(target_p)
        target_df    = target_df.sort_values("id").reset_index(drop=True)
        label_names  = sorted(target_df["page_type"].unique())
        label_to_int = {name: i for i, name in enumerate(label_names)}
        y = torch.tensor(target_df["page_type"].map(label_to_int).values,
                         dtype=torch.long)

        self._data        = Data(x=x, edge_index=edge_index, y=y)
        self.num_features = x.shape[1]
        self.num_classes  = len(label_names)

    def __getitem__(self, idx: int) -> Data:
        assert idx == 0
        return self._data


dataset_fb = FacebookPagePageSNAP(root=".")
data_fb    = dataset_fb[0]

print("\nFacebook Page-Page:")
print(f"  Nodes:    {data_fb.x.shape[0]}")
print(f"  Features: {dataset_fb.num_features}")
print(f"  Classes:  {dataset_fb.num_classes}")
print(f"  Directed: {data_fb.is_directed()}")
print(f"  Isolated: {data_fb.has_isolated_nodes()}")
print(f"  Loops:    {data_fb.has_self_loops()}")

# Create proper boolean masks (range objects are NOT compatible with PyG >= 2.0)
n = data_fb.num_nodes
train_mask = torch.zeros(n, dtype=torch.bool)
val_mask   = torch.zeros(n, dtype=torch.bool)
test_mask  = torch.zeros(n, dtype=torch.bool)

train_mask[:18000]    = True
val_mask[18000:20000] = True
test_mask[20000:]     = True

data_fb.train_mask = train_mask
data_fb.val_mask   = val_mask
data_fb.test_mask  = test_mask

print(f"  Train/Val/Test: {train_mask.sum().item()} / "
      f"{val_mask.sum().item()} / "
      f"{test_mask.sum().item()}")

# Optional: tabular view with pandas
df = pd.DataFrame(data_cora.x.numpy())
df['label'] = data_cora.y.numpy()
print(f"\nCora as tabular dataset: {df.shape}")
print(df.head(3).to_string())


# =============================================================================
# PART 2 – Multilayer Perceptron (topology-agnostic)
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 – MLP (topology-agnostic)")
print("=" * 60)


class MLP(nn.Module):
    """
    Two-layer Multilayer Perceptron for node classification.
    Treats node features as an independent tabular dataset —
    completely ignores graph topology.
    """

    def __init__(self, dim_in: int, dim_h: int, dim_out: int):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        return F.log_softmax(self.linear2(x), dim=1)

    def fit(self, data, epochs: int, verbose: bool = True):
        # forward() returns log_softmax, so we pair it with NLLLoss.
        # CrossEntropyLoss would apply log_softmax internally, double-counting.
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=0.01, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out  = self(data.x)
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

    def test(self, data) -> float:
        self.eval()
        with torch.no_grad():
            out = self(data.x)
        return accuracy(out.argmax(dim=1)[data.test_mask],
                        data.y[data.test_mask])


# ── Train on Cora ─────────────────────────────────────────────────────────────

print("\nTraining MLP on Cora …")
mlp_cora = MLP(dataset_cora.num_features, 16, dataset_cora.num_classes)
print(mlp_cora)
mlp_cora.fit(data_cora, epochs=100)
acc_mlp_cora = mlp_cora.test(data_cora)
print(f"\nMLP test accuracy on Cora: {acc_mlp_cora*100:.2f}%")

# ── Train on Facebook ─────────────────────────────────────────────────────────

print("\nTraining MLP on Facebook Page-Page …")
mlp_fb = MLP(dataset_fb.num_features, 16, dataset_fb.num_classes)
mlp_fb.fit(data_fb, epochs=100)
acc_mlp_fb = mlp_fb.test(data_fb)
print(f"\nMLP test accuracy on Facebook: {acc_mlp_fb*100:.2f}%")


# =============================================================================
# PART 3 – Vanilla GNN (topology-aware)
# =============================================================================

print("\n" + "=" * 60)
print("PART 3 – Vanilla GNN (topology-aware)")
print("=" * 60)


class VanillaGNNLayer(nn.Module):
    """
    Single graph neural network layer: H = A_tilde · X · W

    A_tilde is the adjacency matrix with added self-loops.
    No bias: the neighbourhood aggregation already shifts the representation.
    Equivalent to PyG's MessagePassing with aggr='add' and no normalisation.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)           # X · W
        x = torch.sparse.mm(adj, x)  # A_tilde · (X · W)
        return x


class VanillaGNN(nn.Module):
    """
    Two-layer vanilla GNN for node classification.
    Each layer aggregates neighbourhood features via sparse matrix multiply.
    """

    def __init__(self, dim_in: int, dim_h: int, dim_out: int):
        super().__init__()
        self.gnn1 = VanillaGNNLayer(dim_in, dim_h)
        self.gnn2 = VanillaGNNLayer(dim_h, dim_out)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gnn1(x, adj))
        return F.log_softmax(self.gnn2(h, adj), dim=1)

    def fit(self, data, adj: torch.Tensor,
            epochs: int, verbose: bool = True):
        # forward() returns log_softmax, so we pair it with NLLLoss.
        # CrossEntropyLoss would apply log_softmax internally, double-counting.
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=0.01, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out  = self(data.x, adj)
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

    def test(self, data, adj: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            out = self(data.x, adj)
        return accuracy(out.argmax(dim=1)[data.test_mask],
                        data.y[data.test_mask])


# ── Build sparse adjacency matrices ───────────────────────────────────────────

adj_cora = build_sparse_adj(data_cora)
adj_fb   = build_sparse_adj(data_fb)

# ── Train on Cora ─────────────────────────────────────────────────────────────

print("\nTraining Vanilla GNN on Cora …")
gnn_cora = VanillaGNN(dataset_cora.num_features, 16, dataset_cora.num_classes)
print(gnn_cora)
gnn_cora.fit(data_cora, adj_cora, epochs=100)
acc_gnn_cora = gnn_cora.test(data_cora, adj_cora)
print(f"\nGNN test accuracy on Cora: {acc_gnn_cora*100:.2f}%")

# ── Train on Facebook ─────────────────────────────────────────────────────────

print("\nTraining Vanilla GNN on Facebook Page-Page …")
gnn_fb = VanillaGNN(dataset_fb.num_features, 16, dataset_fb.num_classes)
gnn_fb.fit(data_fb, adj_fb, epochs=100)
acc_gnn_fb = gnn_fb.test(data_fb, adj_fb)
print(f"\nGNN test accuracy on Facebook: {acc_gnn_fb*100:.2f}%")


# =============================================================================
# PART 4 – Reproducible comparison over 20 runs
# =============================================================================

print("\n" + "=" * 60)
print("PART 4 – Comparison: MLP vs GNN (20 runs each)")
print("=" * 60)

N_RUNS  = 20
EPOCHS  = 100
results = {
    "Cora":     {"mlp": [], "gnn": []},
    "Facebook": {"mlp": [], "gnn": []},
}

for ds_name, dataset, data, adj in [
    ("Cora",     dataset_cora, data_cora, adj_cora),
    ("Facebook", dataset_fb,   data_fb,   adj_fb),
]:
    print(f"\n  {ds_name} …")
    for seed in range(N_RUNS):
        torch.manual_seed(seed)

        mlp = MLP(dataset.num_features, 16, dataset.num_classes)
        mlp.fit(data, EPOCHS, verbose=False)
        results[ds_name]["mlp"].append(mlp.test(data))

        gnn = VanillaGNN(dataset.num_features, 16, dataset.num_classes)
        gnn.fit(data, adj, EPOCHS, verbose=False)
        results[ds_name]["gnn"].append(gnn.test(data, adj))

        if (seed + 1) % 5 == 0:
            print(f"    run {seed+1}/{N_RUNS} done")

print("\n" + "-" * 60)
print(f"{'Dataset':<12} {'MLP accuracy':<26} {'GNN accuracy':<26} {'Improvement'}")
print("-" * 60)
for ds in ["Cora", "Facebook"]:
    m_mlp = np.mean(results[ds]["mlp"]) * 100
    s_mlp = np.std(results[ds]["mlp"])  * 100
    m_gnn = np.mean(results[ds]["gnn"]) * 100
    s_gnn = np.std(results[ds]["gnn"])  * 100
    imp   = m_gnn - m_mlp
    print(f"{ds:<12} {m_mlp:.2f}% (±{s_mlp:.2f}%)        "
          f"{m_gnn:.2f}% (±{s_gnn:.2f}%)        +{imp:.2f}%")
print("-" * 60)

# ── Plot results ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
fig.suptitle("MLP vs Vanilla GNN — test accuracy distribution",
             fontsize=12, fontweight="bold")

for ax, ds in zip(axes, ["Cora", "Facebook"]):
    mlp_vals = [v * 100 for v in results[ds]["mlp"]]
    gnn_vals = [v * 100 for v in results[ds]["gnn"]]
    ax.boxplot([mlp_vals, gnn_vals], tick_labels=["MLP", "Vanilla GNN"],
               patch_artist=True,
               boxprops=dict(facecolor="#BDD7EE"),
               medianprops=dict(color="#C55A11", linewidth=2))
    ax.set_title(ds, fontsize=11, fontweight="bold")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(40, 100)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("mlp_vs_gnn_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved mlp_vs_gnn_comparison.png")
print("\nDone.")