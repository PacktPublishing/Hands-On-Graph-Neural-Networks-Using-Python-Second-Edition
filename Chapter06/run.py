"""
Chapter 6 - Introducing Graph Convolutional Networks
Hands-On Graph Neural Networks Using Python, Second Edition

Runs the code of the chapter in order and prints the outputs shown in the book.
Saves Figures 6.3 to 6.8 to figures/. Downloads Cora, Facebook Page-Page and
Wikipedia chameleon on first run.

Usage:
    python run.py
"""

import contextlib
import io
import json
import os
import ssl
import sys
from collections import Counter
from urllib.request import urlopen

import certifi
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import (add_self_loops, degree, remove_self_loops,
                                   to_undirected)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'tools'))
from bookstyle import (apply, figure, FONT, BG, DPI, INK, INK_SOFT,
                       G1, G3, G4, G5, G6)

plt = apply()

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def save(fig, name):
    """Like bookstyle.save, but shows the figure inline before closing it."""
    fig.savefig(os.path.join(FIG_DIR, name), dpi=DPI, facecolor=BG,
                edgecolor="none")
    print(f"[saved figures/{name}]")
    if "ipykernel" in sys.modules:
        plt.show()
    else:
        plt.close(fig)


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def degree_figure(edge_index, name):
    degrees = degree(edge_index[0]).numpy()
    numbers = Counter(degrees)
    fig, ax = figure(aspect=0.48)
    ax.bar(list(numbers.keys()), list(numbers.values()), color=G4, width=0.8)
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Number of nodes")
    ax.grid(axis="y", color=G6, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.95, f"Min: {int(degrees.min())}   Max: {int(degrees.max())}"
            f"\nMean: {degrees.mean():.1f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=INK)
    save(fig, name)


# =============================================================================
# Custom loaders for the MUSAE datasets
# =============================================================================
MUSAE_BASE = ("https://raw.githubusercontent.com/benedekrozemberczki/"
              "MUSAE/master/input")


def _download(url, local):
    if os.path.exists(local):
        return
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, context=ctx) as resp, open(local, "wb") as f:
        f.write(resp.read())


def _reduce_multi_hot(feats_raw, n_components=128):
    num_nodes = max(feats_raw) + 1
    max_feat = max(fid for lst in feats_raw.values() for fid in lst) + 1
    x_full = np.zeros((num_nodes, max_feat), dtype=np.float32)
    for node_id, feat_ids in feats_raw.items():
        x_full[node_id, feat_ids] = 1.0
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    return torch.from_numpy(svd.fit_transform(x_full)).float()


class FacebookPagePageSNAP:
    def __init__(self, root="."):
        raw_dir = os.path.join(root, "facebook_large")
        os.makedirs(raw_dir, exist_ok=True)
        files = {
            "musae_facebook_edges.csv": f"{MUSAE_BASE}/edges/facebook_edges.csv",
            "musae_facebook_features.json": f"{MUSAE_BASE}/features/facebook.json",
            "musae_facebook_target.csv": f"{MUSAE_BASE}/target/facebook_target.csv",
        }
        for local_name, url in files.items():
            _download(url, os.path.join(raw_dir, local_name))
        edges_df = pd.read_csv(os.path.join(raw_dir, "musae_facebook_edges.csv"))
        src = torch.tensor(edges_df["id_1"].values, dtype=torch.long)
        dst = torch.tensor(edges_df["id_2"].values, dtype=torch.long)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        with open(os.path.join(raw_dir, "musae_facebook_features.json")) as f:
            feats_raw = {int(k): v for k, v in json.load(f).items()}
        x = _reduce_multi_hot(feats_raw)
        target_df = pd.read_csv(os.path.join(raw_dir, "musae_facebook_target.csv"))
        target_df = target_df.sort_values("id").reset_index(drop=True)
        label_names = sorted(target_df["page_type"].unique())
        label_to_i = {name: i for i, name in enumerate(label_names)}
        y = torch.tensor(target_df["page_type"].map(label_to_i).values,
                         dtype=torch.long)
        n = x.shape[0]
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[:18000] = True
        val_mask[18000:20000] = True
        test_mask[20000:] = True
        self._data = Data(x=x, edge_index=edge_index, y=y,
                          train_mask=train_mask, val_mask=val_mask,
                          test_mask=test_mask)
        self.num_features = x.shape[1]
        self.num_classes = len(label_names)

    def __getitem__(self, idx):
        return self._data


class WikipediaChameleonSNAP:
    def __init__(self, root=".", num_val=200, num_test=500):
        raw_dir = os.path.join(root, "wikipedia", "chameleon")
        os.makedirs(raw_dir, exist_ok=True)
        files = {
            "musae_chameleon_edges.csv": f"{MUSAE_BASE}/edges/chameleon_edges.csv",
            "musae_chameleon_features.json": f"{MUSAE_BASE}/features/chameleon.json",
            "musae_chameleon_target.csv": f"{MUSAE_BASE}/target/chameleon_target.csv",
        }
        for local_name, url in files.items():
            _download(url, os.path.join(raw_dir, local_name))
        edges_df = pd.read_csv(os.path.join(raw_dir, "musae_chameleon_edges.csv"))
        edge_index = torch.tensor(edges_df[["id1", "id2"]].values.T,
                                  dtype=torch.long)
        edge_index = to_undirected(edge_index)
        edge_index, _ = remove_self_loops(edge_index)
        with open(os.path.join(raw_dir, "musae_chameleon_features.json")) as f:
            feats_raw = {int(k): v for k, v in json.load(f).items()}
        x = _reduce_multi_hot(feats_raw)
        target_df = pd.read_csv(os.path.join(raw_dir, "musae_chameleon_target.csv"))
        target_df = target_df.sort_values("id").reset_index(drop=True)
        y = torch.tensor(np.log10(target_df["target"].values),
                         dtype=torch.float32)
        n = x.shape[0]
        perm = torch.tensor(np.random.default_rng(0).permutation(n))
        test_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        train_mask = torch.zeros(n, dtype=torch.bool)
        test_mask[perm[:num_test]] = True
        val_mask[perm[num_test:num_test + num_val]] = True
        train_mask[perm[num_test + num_val:]] = True
        self._data = Data(x=x, edge_index=edge_index, y=y,
                          train_mask=train_mask, val_mask=val_mask,
                          test_mask=test_mask)
        self.num_features = x.shape[1]

    def __getitem__(self, idx):
        return self._data


# =============================================================================
# PART 1 - From formula to matrix multiplication
# =============================================================================
banner("PART 1 - From formula to matrix multiplication")

D = np.array([
    [3, 0, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 2, 0],
    [0, 0, 0, 1]
])

print(np.linalg.inv(D + np.identity(4)).__repr__())

A = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 0],
    [1, 0, 0, 1]
])

D_tilde_inv = np.linalg.inv(D + np.identity(4))
print(D_tilde_inv @ A)


# =============================================================================
# PART 2 - Node degree analysis
# =============================================================================
banner("PART 2 - Node degree analysis")

dataset = Planetoid(root='.', name='Cora')
data = dataset[0]

degrees = degree(data.edge_index[0]).numpy()
numbers = Counter(degrees)
print(f"Cora: degrees {int(degrees.min())}-{int(degrees.max())}, "
      f"{numbers[1.0]} nodes with 1 neighbor, "
      f"{numbers[degrees.max()]} with {int(degrees.max())}")
degree_figure(data.edge_index, "fig6_3_cora_degrees.png")

dataset_fb = FacebookPagePageSNAP(root='.')
data_fb = dataset_fb[0]
degrees_fb = degree(data_fb.edge_index[0]).numpy()
print(f"Facebook Page-Page: degrees {int(degrees_fb.min())}-"
      f"{int(degrees_fb.max())}")
degree_figure(data_fb.edge_index, "fig6_4_facebook_degrees.png")


# =============================================================================
# PART 3 - Implementing the GCN
# =============================================================================
banner("PART 3 - Implementing the GCN")


def accuracy(pred_y, y):
    return (pred_y == y).float().mean().item()


class GCN(torch.nn.Module):
    """Graph Convolutional Network for node classification."""

    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = F.relu(self.gcn1(x, edge_index))
        h = self.gcn2(h, edge_index)
        return F.log_softmax(h, dim=1)

    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=0.01, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask],
                             data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1),
                           data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask],
                                     data.y[data.val_mask])
                val_acc = accuracy(out[data.val_mask].argmax(dim=1),
                                   data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} |'
                      f' Train Acc: {acc*100:>5.2f}% |'
                      f' Val Loss: {val_loss:.2f} |'
                      f' Val Acc: {val_acc*100:.2f}%')

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        return accuracy(out.argmax(dim=1)[data.test_mask],
                        data.y[data.test_mask])


torch.manual_seed(0)
gcn = GCN(dataset.num_features, 16, dataset.num_classes)
print(gcn)
gcn.fit(data, epochs=100)


# =============================================================================
# PART 4 - Comparing all three architectures
# =============================================================================
banner("PART 4 - Comparing all three architectures")


class MLP(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_out)

    def forward(self, x):
        return F.log_softmax(self.linear2(F.relu(self.linear1(x))), dim=1)

    def fit(self, data, epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01,
                                     weight_decay=5e-4)
        criterion = nn.NLLLoss()
        self.train()
        for _ in range(epochs + 1):
            optimizer.zero_grad()
            loss = criterion(self(data.x)[data.train_mask],
                             data.y[data.train_mask])
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def test(self, data):
        self.eval()
        return accuracy(self(data.x).argmax(dim=1)[data.test_mask],
                        data.y[data.test_mask])


class VanillaGNNLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x, adjacency):
        return torch.sparse.mm(adjacency, self.linear(x))


class VanillaGNN(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gnn1 = VanillaGNNLayer(dim_in, dim_h)
        self.gnn2 = VanillaGNNLayer(dim_h, dim_out)

    def forward(self, x, adjacency):
        h = F.relu(self.gnn1(x, adjacency))
        return F.log_softmax(self.gnn2(h, adjacency), dim=1)

    def fit(self, data, adjacency, epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01,
                                     weight_decay=5e-4)
        criterion = nn.NLLLoss()
        self.train()
        for _ in range(epochs + 1):
            optimizer.zero_grad()
            loss = criterion(self(data.x, adjacency)[data.train_mask],
                             data.y[data.train_mask])
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def test(self, data, adjacency):
        self.eval()
        return accuracy(self(data.x, adjacency).argmax(dim=1)[data.test_mask],
                        data.y[data.test_mask])


def sparse_adjacency(data):
    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    values = torch.ones(edge_index.size(1))
    return torch.sparse_coo_tensor(edge_index, values,
                                   (data.num_nodes, data.num_nodes)).coalesce()


results = {}
for name, ds, d in [("Cora", dataset, data), ("Facebook", dataset_fb, data_fb)]:
    adjacency = sparse_adjacency(d)
    accs = {"MLP": [], "Vanilla GNN": [], "GCN": []}
    for seed in range(20):
        torch.manual_seed(seed)
        mlp = MLP(ds.num_features, 16, ds.num_classes)
        mlp.fit(d, 100)
        accs["MLP"].append(mlp.test(d))
        gnn = VanillaGNN(ds.num_features, 16, ds.num_classes)
        gnn.fit(d, adjacency, 100)
        accs["Vanilla GNN"].append(gnn.test(d, adjacency))
        model = GCN(ds.num_features, 16, ds.num_classes)
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(d, 100)
        accs["GCN"].append(model.test(d))
    results[name] = {k: (np.mean(v) * 100, np.std(v) * 100)
                     for k, v in accs.items()}
    print(f"{name:<9} " + "   ".join(
        f"{k}: {m:.2f}% ± {s:.2f}%" for k, (m, s) in results[name].items()))

fig, ax = figure(aspect=0.62)
models = ["MLP", "Vanilla GNN", "GCN"]
fills = [G6, G4, G1]
width = 0.26
for k, (model_name, fill) in enumerate(zip(models, fills)):
    xs = np.arange(2) + (k - 1) * width
    means = [results[n][model_name][0] for n in ("Cora", "Facebook")]
    stds = [results[n][model_name][1] for n in ("Cora", "Facebook")]
    ax.bar(xs, means, width, yerr=stds, capsize=3, color=fill,
           edgecolor=INK, linewidth=0.6, label=model_name,
           error_kw=dict(ecolor=INK, lw=0.8))
    for x, m, s in zip(xs, means, stds):
        ax.text(x, m + s + 0.8, f"{m:.1f}", ha="center", va="bottom",
                fontsize=8, color=INK, fontweight="bold")
ax.set_xticks(range(2))
ax.set_xticklabels(["Cora", "Facebook Page-Page"])
ax.set_ylabel("Mean test accuracy (%)")
ax.set_ylim(40, 100)
ax.grid(axis="y", color=G6, lw=0.6)
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=8, framealpha=1.0, edgecolor=G6)
save(fig, "fig6_5_comparison.png")


# =============================================================================
# PART 5 - The Wikipedia Network dataset
# =============================================================================
banner("PART 5 - The Wikipedia Network dataset")

dataset = WikipediaChameleonSNAP(root='.', num_val=200, num_test=500)
data = dataset[0]

print(f'Nodes:    {data.x.shape[0]}')
print(f'Features: {data.x.shape[1]}')
print(f'Edges:    {data.edge_index.shape[1] // 2}')

print(f'Target range: {data.y.min():.2f} – {data.y.max():.2f}')

degrees_wiki = degree(data.edge_index[0]).numpy()
print(f"Wikipedia chameleon: degrees {int(degrees_wiki.min())}-"
      f"{int(degrees_wiki.max())}, mean {degrees_wiki.mean():.1f}")
degree_figure(data.edge_index, "fig6_6_wiki_degrees.png")

values = data.y.numpy()
mu, sigma = norm.fit(values)
x_range = np.linspace(values.min(), values.max(), 200)
print(f"Normal fit: mu={mu:.2f}, sigma={sigma:.2f}")

fig, ax = figure(aspect=0.55)
ax.hist(values, bins=40, density=True, color=G5, edgecolor="white",
        linewidth=0.4, label="Target")
ax.plot(x_range, norm.pdf(x_range, mu, sigma), color=INK, linewidth=1.6,
        label=f"Normal fit (mu={mu:.2f}, sigma={sigma:.2f})")
ax.set_xlabel("Log10 average monthly traffic")
ax.set_ylabel("Density")
ax.legend(loc="upper left", fontsize=8, framealpha=1.0, edgecolor=G6)
save(fig, "fig6_7_wiki_target_density.png")


# =============================================================================
# PART 6 - Implementing the GCN regressor
# =============================================================================
banner("PART 6 - Implementing the GCN regressor")


class GCNRegressor(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h * 4)
        self.gcn2 = GCNConv(dim_h * 4, dim_h * 2)
        self.gcn3 = GCNConv(dim_h * 2, dim_h)
        self.linear = torch.nn.Linear(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = F.dropout(F.relu(self.gcn1(x, edge_index)),
                      p=0.5, training=self.training)
        h = F.dropout(F.relu(self.gcn2(h, edge_index)),
                      p=0.5, training=self.training)
        h = F.relu(self.gcn3(h, edge_index))
        return self.linear(h)

    def fit(self, data, epochs):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=0.02, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = F.mse_loss(out.squeeze()[data.train_mask],
                              data.y[data.train_mask].float())
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                val_loss = F.mse_loss(
                    out.squeeze()[data.val_mask],
                    data.y[data.val_mask].float())
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.5f}'
                      f' | Val Loss: {val_loss:.5f}')

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        return F.mse_loss(out.squeeze()[data.test_mask],
                          data.y[data.test_mask].float())


torch.manual_seed(0)
gcn = GCNRegressor(dataset.num_features, 128, 1)
print(gcn)
gcn.fit(data, epochs=200)


# =============================================================================
# PART 7 - Evaluating the regression results
# =============================================================================
banner("PART 7 - Evaluating the regression results")

gcn.eval()
out = gcn(data.x, data.edge_index)
y_pred = out.squeeze()[data.test_mask].detach().numpy()
y_true = data.y[data.test_mask].numpy()

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)

print('=' * 43)
print(f'MSE = {mse:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}')
print('=' * 43)
print(f"10**MAE = {10**mae:.2f}, 10**RMSE = {10**rmse:.2f}")

fig, ax = figure(aspect=0.80)
ax.scatter(y_true, y_pred, s=10, color=G4, alpha=0.6, edgecolors="none",
           label="Test nodes")
m = np.polyfit(y_true, y_pred, 1)
x_line = np.linspace(y_true.min(), y_true.max(), 100)
ax.plot(x_line, np.polyval(m, x_line), color=INK, linewidth=1.6,
        label="Regression line")
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
        color=G3, linewidth=1.2, linestyle="--", label="Perfect prediction")
ax.set_xlabel("Ground truth (log10 monthly traffic)")
ax.set_ylabel("Predicted value")
ax.legend(loc="upper left", fontsize=8, framealpha=1.0, edgecolor=G6)
save(fig, "fig6_8_regression_scatter.png")