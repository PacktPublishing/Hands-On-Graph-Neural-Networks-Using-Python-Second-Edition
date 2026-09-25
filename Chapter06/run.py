"""
Chapter 6 – Introducing Graph Convolutional Networks
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch-geometric scikit-learn matplotlib pandas numpy scipy

Key fixes versus the first edition:
  - sns.distplot() removed (Seaborn 0.12+); replaced with matplotlib + scipy
  - val_loss dtype mismatch fixed (added .float() consistently)
  - accuracy() uses .float().mean() instead of len() division
  - WikipediaNetwork loaded with geom_gcn_preprocess=False for raw targets
  - Seaborn dependency removed entirely from this chapter
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# =============================================================================
# SHARED UTILITY
# =============================================================================

def accuracy(pred_y: torch.Tensor, y: torch.Tensor) -> float:
    return (pred_y == y).float().mean().item()


# =============================================================================
# CUSTOM DATASET LOADERS
# =============================================================================
#
# PyG's built-in FacebookPagePage and WikipediaNetwork downloaders target the
# graphmining.ai server, which has been offline for a while. We replace them
# with two small loaders that fetch the raw MUSAE files (edges, features,
# targets) straight from the author's GitHub repository. Each loader rebuilds
# the PyG Data object with a 128-dim feature layout via TruncatedSVD, keeping
# the downstream code unchanged.

import json, os, ssl
import certifi
import pandas as pd
from urllib.request import urlopen
from torch_geometric.data import Data

MUSAE_BASE = ("https://raw.githubusercontent.com/benedekrozemberczki/"
              "MUSAE/master/input")


def _download(url: str, local: str) -> None:
    """Download `url` to `local` using an explicit certifi-backed SSL context
    so macOS Python installs without system CA certificates still work."""
    if os.path.exists(local):
        return
    print(f"  Downloading {os.path.basename(local)} …")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, context=ctx) as resp:
        with open(local, "wb") as f:
            f.write(resp.read())


def _reduce_multi_hot(feats_raw: dict, n_components: int = 128) -> torch.Tensor:
    """Turn a {node_id: [feature_id, ...]} dict into a dense (N, 128) tensor
    via multi-hot encoding followed by TruncatedSVD. The .npz files that PyG
    used to serve shipped 128-dim precomputed embeddings; this reproduces
    the same layout from the raw data."""
    from sklearn.decomposition import TruncatedSVD
    num_nodes = max(feats_raw) + 1
    max_feat  = max(fid for lst in feats_raw.values() for fid in lst) + 1
    x_full = np.zeros((num_nodes, max_feat), dtype=np.float32)
    for node_id, feat_ids in feats_raw.items():
        x_full[node_id, feat_ids] = 1.0
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    return torch.from_numpy(svd.fit_transform(x_full)).float()


class FacebookPagePageSNAP:
    """Drop-in replacement for torch_geometric.datasets.FacebookPagePage."""

    def __init__(self, root: str = "."):
        raw_dir = os.path.join(root, "facebook_large")
        os.makedirs(raw_dir, exist_ok=True)

        files = {
            "musae_facebook_edges.csv":    f"{MUSAE_BASE}/edges/facebook_edges.csv",
            "musae_facebook_features.json":f"{MUSAE_BASE}/features/facebook.json",
            "musae_facebook_target.csv":   f"{MUSAE_BASE}/target/facebook_target.csv",
        }
        for local_name, url in files.items():
            _download(url, os.path.join(raw_dir, local_name))

        edges_df = pd.read_csv(os.path.join(raw_dir, "musae_facebook_edges.csv"))
        src = torch.tensor(edges_df["id_1"].values, dtype=torch.long)
        dst = torch.tensor(edges_df["id_2"].values, dtype=torch.long)
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

        with open(os.path.join(raw_dir, "musae_facebook_features.json")) as f:
            feats_raw = {int(k): v for k, v in json.load(f).items()}
        x = _reduce_multi_hot(feats_raw, n_components=128)

        target_df   = pd.read_csv(
            os.path.join(raw_dir, "musae_facebook_target.csv"))
        target_df   = target_df.sort_values("id").reset_index(drop=True)
        label_names = sorted(target_df["page_type"].unique())
        label_to_i  = {name: i for i, name in enumerate(label_names)}
        y = torch.tensor(target_df["page_type"].map(label_to_i).values,
                         dtype=torch.long)

        self._data        = Data(x=x, edge_index=edge_index, y=y)
        self.num_features = x.shape[1]
        self.num_classes  = len(label_names)

    def __getitem__(self, idx: int) -> Data:
        assert idx == 0
        return self._data


class WikipediaChameleonSNAP:
    """Drop-in replacement for WikipediaNetwork(name='chameleon',
    geom_gcn_preprocess=False), with continuous target values already
    attached to data.y as log10(monthly traffic)."""

    def __init__(self, root: str = ".", num_val: int = 200, num_test: int = 500):
        raw_dir = os.path.join(root, "wikipedia", "chameleon")
        os.makedirs(raw_dir, exist_ok=True)

        files = {
            "musae_chameleon_edges.csv":     f"{MUSAE_BASE}/edges/chameleon_edges.csv",
            "musae_chameleon_features.json": f"{MUSAE_BASE}/features/chameleon.json",
            "musae_chameleon_target.csv":    f"{MUSAE_BASE}/target/chameleon_target.csv",
        }
        for local_name, url in files.items():
            _download(url, os.path.join(raw_dir, local_name))

        # Edges — note MUSAE uses id1/id2 for chameleon, not id_1/id_2 as
        # for facebook.
        edges_df = pd.read_csv(os.path.join(raw_dir, "musae_chameleon_edges.csv"))
        src = torch.tensor(edges_df["id1"].values, dtype=torch.long)
        dst = torch.tensor(edges_df["id2"].values, dtype=torch.long)
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

        # Features — multi-hot then TruncatedSVD to 128 dims.
        with open(os.path.join(raw_dir, "musae_chameleon_features.json")) as f:
            feats_raw = {int(k): v for k, v in json.load(f).items()}
        x = _reduce_multi_hot(feats_raw, n_components=128)

        # Targets — continuous, use log10 as in the original chapter.
        target_df = pd.read_csv(
            os.path.join(raw_dir, "musae_chameleon_target.csv"))
        target_df = target_df.sort_values("id").reset_index(drop=True)
        y = torch.tensor(np.log10(target_df["target"].values),
                         dtype=torch.float32)

        # Random node split (train/val/test) with fixed seed.
        n = x.shape[0]
        perm = torch.tensor(
            np.random.default_rng(0).permutation(n), dtype=torch.long)
        test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[perm[:num_test]] = True
        val_mask   = torch.zeros(n, dtype=torch.bool); val_mask[perm[num_test:num_test+num_val]] = True
        train_mask = torch.zeros(n, dtype=torch.bool); train_mask[perm[num_test+num_val:]] = True

        self._data        = Data(x=x, edge_index=edge_index, y=y,
                                 train_mask=train_mask, val_mask=val_mask,
                                 test_mask=test_mask)
        self.num_features = x.shape[1]

    def __getitem__(self, idx: int) -> Data:
        assert idx == 0
        return self._data


# =============================================================================
# PART 1 – Deriving the GCN layer (NumPy illustration)
# =============================================================================

print("=" * 60)
print("PART 1 – GCN normalisation derivation")
print("=" * 60)

import numpy as np

# Degree matrix for the 4-node example graph (A has 3 neighbours after self-loop)
D = np.array([
    [3, 0, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 2, 0],
    [0, 0, 0, 1]
])

print("\nDegree matrix D:")
print(D)

print("\nD_tilde_inv = inv(D + I):")
D_tilde_inv = np.linalg.inv(D + np.identity(4))
print(D_tilde_inv)

# Adjacency matrix with self-loops
A = np.array([
    [1, 1, 1, 1],
    [1, 1, 0, 0],
    [1, 0, 1, 1],
    [1, 0, 0, 1]
])

print("\nRow normalisation D_tilde_inv @ A:")
print(D_tilde_inv @ A)

print("\nColumn normalisation A @ D_tilde_inv:")
print(A @ D_tilde_inv)

# Symmetric normalisation D^(-1/2) @ A @ D^(-1/2)
D_tilde     = D + np.identity(4)
D_half_inv  = np.diag(1.0 / np.sqrt(np.diag(D_tilde)))
sym_norm    = D_half_inv @ A @ D_half_inv
print("\nSymmetric normalisation D^(-1/2) @ A_tilde @ D^(-1/2):")
print(np.round(sym_norm, 4))


# =============================================================================
# PART 2 – GCN for node classification on Cora + Facebook Page-Page
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 – GCN node classification")
print("=" * 60)

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree, add_self_loops
import torch_geometric.transforms as T


# ── GCN model ─────────────────────────────────────────────────────────────────

class GCN(nn.Module):
    """
    Two-layer Graph Convolutional Network for node classification.
    Uses PyG's GCNConv which applies symmetric degree normalisation
    (D_tilde^-0.5 @ A_tilde @ D_tilde^-0.5) automatically.
    """

    def __init__(self, dim_in: int, dim_h: int, dim_out: int):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gcn1(x, edge_index))
        h = self.gcn2(h, edge_index)
        return F.log_softmax(h, dim=1)

    def fit(self, data, epochs: int, verbose: bool = True):
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=0.01, weight_decay=5e-4)
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


def plot_degree_distribution(edge_index, title: str, filename: str,
                              color: str = "#2E75B6"):
    """Plot and save the node degree distribution."""
    degs    = degree(edge_index[0]).numpy()
    numbers = Counter(degs)
    keys    = np.array(sorted(numbers.keys()))
    vals    = [numbers[k] for k in keys]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.bar(keys, vals, color=color, alpha=0.85, width=0.8)
    ax.set_xlabel("Node degree", fontsize=11)
    ax.set_ylabel("Number of nodes", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.97, 0.95,
            f"Min: {int(keys.min())}  Max: {int(keys.max())}\n"
            f"Mean: {np.mean(degs):.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


# ── Cora ──────────────────────────────────────────────────────────────────────

print("\nLoading Cora …")
dataset_cora = Planetoid(root=".", name="Cora")
data_cora    = dataset_cora[0]

plot_degree_distribution(
    data_cora.edge_index,
    "Node degree distribution — Cora dataset",
    "cora_degrees.png", color="#2E75B6"
)

print("Training GCN on Cora …")
gcn_cora = GCN(dataset_cora.num_features, 16, dataset_cora.num_classes)
print(gcn_cora)
gcn_cora.fit(data_cora, epochs=100)
acc_cora = gcn_cora.test(data_cora)
print(f"GCN test accuracy on Cora: {acc_cora*100:.2f}%")

# ── Facebook ──────────────────────────────────────────────────────────────────

print("\nLoading Facebook Page-Page …")
dataset_fb = FacebookPagePageSNAP(root=".")
data_fb    = dataset_fb[0]

# Proper boolean masks (range objects break in PyG >= 2.0)
n = data_fb.num_nodes
tm = torch.zeros(n, dtype=torch.bool); tm[:18000]     = True
vm = torch.zeros(n, dtype=torch.bool); vm[18000:20000]= True
em = torch.zeros(n, dtype=torch.bool); em[20000:]     = True
data_fb.train_mask, data_fb.val_mask, data_fb.test_mask = tm, vm, em

plot_degree_distribution(
    data_fb.edge_index,
    "Node degree distribution — Facebook Page-Page dataset",
    "facebook_degrees.png", color="#C55A11"
)

print("Training GCN on Facebook Page-Page …")
gcn_fb = GCN(dataset_fb.num_features, 16, dataset_fb.num_classes)
gcn_fb.fit(data_fb, epochs=100)
acc_fb = gcn_fb.test(data_fb)
print(f"GCN test accuracy on Facebook: {acc_fb*100:.2f}%")

# ── Reproducible 20-run comparison ───────────────────────────────────────────

print("\nRunning 20-run comparison (MLP / Vanilla GNN / GCN) …")

# Import models from previous chapter utilities (inline here for self-containment)
class MLP(nn.Module):
    def __init__(self, d, h, o):
        super().__init__()
        self.l1 = nn.Linear(d, h); self.l2 = nn.Linear(h, o)
    def forward(self, x):
        return F.log_softmax(self.l2(F.relu(self.l1(x))), dim=1)
    def fit(self, data, epochs):
        opt = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        crit = nn.NLLLoss()
        self.train()
        for _ in range(epochs + 1):
            opt.zero_grad()
            loss = crit(self(data.x)[data.train_mask], data.y[data.train_mask])
            loss.backward(); opt.step()
    @torch.no_grad()
    def test(self, data):
        self.eval()
        return accuracy(self(data.x).argmax(dim=1)[data.test_mask],
                        data.y[data.test_mask])

class VanillaGNNLayer(nn.Module):
    def __init__(self, d, o):
        super().__init__()
        self.linear = nn.Linear(d, o, bias=False)
    def forward(self, x, adj):
        return torch.sparse.mm(adj, self.linear(x))

class VanillaGNN(nn.Module):
    def __init__(self, d, h, o):
        super().__init__()
        self.g1 = VanillaGNNLayer(d, h); self.g2 = VanillaGNNLayer(h, o)
    def forward(self, x, adj):
        return F.log_softmax(self.g2(F.relu(self.g1(x, adj)), adj), dim=1)
    def fit(self, data, adj, epochs):
        opt = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        crit = nn.NLLLoss()
        self.train()
        for _ in range(epochs + 1):
            opt.zero_grad()
            loss = crit(self(data.x, adj)[data.train_mask],
                        data.y[data.train_mask])
            loss.backward(); opt.step()
    @torch.no_grad()
    def test(self, data, adj):
        self.eval()
        return accuracy(self(data.x, adj).argmax(dim=1)[data.test_mask],
                        data.y[data.test_mask])

def build_sparse_adj(data):
    ei, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    v = torch.ones(ei.size(1))
    return torch.sparse_coo_tensor(ei, v, (data.num_nodes, data.num_nodes)).coalesce()

N_RUNS = 20; EPOCHS = 100
comparison = {}

for ds_name, dataset, data in [
    ("Cora",     dataset_cora, data_cora),
    ("Facebook", dataset_fb,   data_fb),
]:
    print(f"\n  {ds_name} …")
    adj = build_sparse_adj(data)
    mlp_a, gnn_a, gcn_a = [], [], []
    for seed in range(N_RUNS):
        torch.manual_seed(seed)
        m = MLP(dataset.num_features, 16, dataset.num_classes)
        m.fit(data, EPOCHS); mlp_a.append(m.test(data))
        g = VanillaGNN(dataset.num_features, 16, dataset.num_classes)
        g.fit(data, adj, EPOCHS); gnn_a.append(g.test(data, adj))
        c = GCN(dataset.num_features, 16, dataset.num_classes)
        c.fit(data, EPOCHS, verbose=False); gcn_a.append(c.test(data))
        if (seed + 1) % 5 == 0:
            print(f"    run {seed+1}/{N_RUNS} done")
    comparison[ds_name] = {
        "mlp": (np.mean(mlp_a)*100, np.std(mlp_a)*100),
        "gnn": (np.mean(gnn_a)*100, np.std(gnn_a)*100),
        "gcn": (np.mean(gcn_a)*100, np.std(gcn_a)*100),
    }

print("\n" + "-" * 65)
print(f"{'Dataset':<12} {'MLP':<22} {'Vanilla GNN':<22} {'GCN'}")
print("-" * 65)
for ds in ["Cora", "Facebook"]:
    r = comparison[ds]
    print(f"{ds:<12} "
          f"{r['mlp'][0]:.2f}% (±{r['mlp'][1]:.2f}%)   "
          f"{r['gnn'][0]:.2f}% (±{r['gnn'][1]:.2f}%)   "
          f"{r['gcn'][0]:.2f}% (±{r['gcn'][1]:.2f}%)")
print("-" * 65)


# =============================================================================
# PART 3 – GCN node regression on Wikipedia Network (chameleon)
# =============================================================================

print("\n" + "=" * 60)
print("PART 3 – Node regression (Wikipedia Network / chameleon)")
print("=" * 60)

import pandas as pd

print("\nLoading Wikipedia Network (chameleon) …")
dataset_wiki = WikipediaChameleonSNAP(root=".", num_val=200, num_test=500)
data_wiki    = dataset_wiki[0]

print(f"  Nodes:    {data_wiki.x.shape[0]}")
print(f"  Features: {data_wiki.x.shape[1]}")
print(f"  Edges:    {data_wiki.edge_index.shape[1] // 2}")
print(f"  Loaded continuous targets (log10 monthly traffic)")
print(f"  Target range: {data_wiki.y.min():.2f} – {data_wiki.y.max():.2f}")

# ── Degree distribution ───────────────────────────────────────────────────────

plot_degree_distribution(
    data_wiki.edge_index,
    "Node degree distribution — Wikipedia Network (chameleon)",
    "wiki_degrees.png", color="#1E8449"
)

# ── Target distribution ───────────────────────────────────────────────────────

vals_np = data_wiki.y.numpy()
mu, sigma = norm.fit(vals_np)
x_range   = np.linspace(vals_np.min(), vals_np.max(), 200)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
ax.hist(vals_np, bins=40, density=True, alpha=0.65,
        color="#2E75B6", label="Target distribution")
ax.plot(x_range, norm.pdf(x_range, mu, sigma),
        color="#C55A11", linewidth=2.5,
        label=f"Normal fit  (μ={mu:.2f}, σ={sigma:.2f})")
ax.set_xlabel("Log₁₀ average monthly traffic", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Distribution of target values — Wikipedia Network (chameleon)",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("wiki_target_density.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved wiki_target_density.png")

# ── Regression GCN ────────────────────────────────────────────────────────────

class GCNRegressor(nn.Module):
    """
    Three-layer GCN regressor for continuous target prediction.
    Uses MSE loss and outputs a single continuous value (no softmax).
    """

    def __init__(self, dim_in: int, dim_h: int, dim_out: int):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.gcn1   = GCNConv(dim_in, dim_h * 4)
        self.gcn2   = GCNConv(dim_h * 4, dim_h * 2)
        self.gcn3   = GCNConv(dim_h * 2, dim_h)
        self.linear = nn.Linear(dim_h, dim_out)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        h = F.dropout(F.relu(self.gcn1(x, edge_index)),
                      p=0.5, training=self.training)
        h = F.dropout(F.relu(self.gcn2(h, edge_index)),
                      p=0.5, training=self.training)
        h = F.relu(self.gcn3(h, edge_index))
        return self.linear(h)   # no softmax: continuous output

    def fit(self, data, epochs: int, verbose: bool = True):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=0.02, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out  = self(data.x, data.edge_index)
            # .float() on both sides prevents float32/float64 dtype mismatch
            loss = F.mse_loss(out.squeeze()[data.train_mask],
                              data.y[data.train_mask].float())
            loss.backward()
            optimizer.step()
            if verbose and epoch % 20 == 0:
                val_loss = F.mse_loss(
                    out.squeeze()[data.val_mask],
                    data.y[data.val_mask].float()   # fix: was missing .float()
                )
                print(f"  Epoch {epoch:>3} | Train Loss: {loss:.5f} "
                      f"| Val Loss: {val_loss:.5f}")

    @torch.no_grad()
    def test(self, data) -> float:
        self.eval()
        out = self(data.x, data.edge_index)
        return F.mse_loss(out.squeeze()[data.test_mask],
                          data.y[data.test_mask].float()).item()


print("\nTraining GCN regressor (200 epochs) …")
torch.manual_seed(SEED)
gcn_reg = GCNRegressor(dataset_wiki.num_features, 128, 1)
print(gcn_reg)
gcn_reg.fit(data_wiki, epochs=200)

test_mse = gcn_reg.test(data_wiki)
print(f"\nGCN test MSE: {test_mse:.5f}")

# ── Detailed metrics ──────────────────────────────────────────────────────────

gcn_reg.eval()
with torch.no_grad():
    out_reg = gcn_reg(data_wiki.x, data_wiki.edge_index)

y_pred = out_reg.squeeze()[data_wiki.test_mask].detach().numpy()
y_true = data_wiki.y[data_wiki.test_mask].numpy()

mse  = mean_squared_error(y_true, y_pred)
mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)

print("=" * 43)
print(f"MSE = {mse:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}")
print("=" * 43)

# ── Scatter plot ──────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
ax.scatter(y_true, y_pred, alpha=0.45, s=30, color="#2E75B6",
           edgecolors="none", label="Test nodes")
m_fit  = np.polyfit(y_true, y_pred, 1)
x_line = np.linspace(y_true.min(), y_true.max(), 100)
ax.plot(x_line, np.polyval(m_fit, x_line), color="#C55A11",
        linewidth=2.2, label="Regression line")
ax.plot([y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        color="#C0392B", linewidth=1.4, linestyle="--", alpha=0.6,
        label="Perfect prediction")
ax.set_xlabel("Ground truth (log₁₀ monthly traffic)", fontsize=11)
ax.set_ylabel("Predicted value", fontsize=11)
ax.set_title(f"GCN regression — predicted vs ground truth\n"
             f"MSE={mse:.4f}  |  RMSE={rmse:.4f}  |  MAE={mae:.4f}",
             fontsize=11, fontweight="bold", pad=10)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("gcn_regression_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved gcn_regression_scatter.png")
print("\nDone.")