"""
Chapter 6 – All figures in grayscale.
Loads the real datasets (Cora via Planetoid; Facebook Page-Page and
Wikipedia chameleon via the custom MUSAE loaders) and regenerates every
figure used in the chapter. Results-only numbers (the 20-run comparison)
are hardcoded from the run; everything else is drawn from real data.
"""

import json
import os
import ssl
from collections import Counter

import certifi
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import norm
from urllib.request import urlopen

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv
from sklearn.decomposition import TruncatedSVD

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT  = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

FONT = "DejaVu Sans"
BG   = "white"
G0 = "#111111"; G1 = "#333333"; G2 = "#555555"
G3 = "#777777"; G4 = "#999999"; G5 = "#BBBBBB"; G6 = "#DDDDDD"


def save(fig, name, dpi=200):
    fig.savefig(f"{OUT}/{name}", dpi=dpi, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  saved {name}")


# =============================================================================
# Custom MUSAE loaders (kept in sync with run.py)
# =============================================================================

MUSAE_BASE = ("https://raw.githubusercontent.com/benedekrozemberczki/"
              "MUSAE/master/input")


def _download(url, local):
    if os.path.exists(local):
        return
    print(f"  Downloading {os.path.basename(local)} …")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, context=ctx) as resp:
        with open(local, "wb") as f:
            f.write(resp.read())


def _reduce_multi_hot(feats_raw, n_components=128):
    num_nodes = max(feats_raw) + 1
    max_feat  = max(fid for lst in feats_raw.values() for fid in lst) + 1
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
            "musae_facebook_edges.csv":    f"{MUSAE_BASE}/edges/facebook_edges.csv",
            "musae_facebook_features.json":f"{MUSAE_BASE}/features/facebook.json",
            "musae_facebook_target.csv":   f"{MUSAE_BASE}/target/facebook_target.csv",
        }
        for local_name, url in files.items():
            _download(url, os.path.join(raw_dir, local_name))
        edges_df = pd.read_csv(os.path.join(raw_dir, "musae_facebook_edges.csv"))
        src = torch.tensor(edges_df["id_1"].values, dtype=torch.long)
        dst = torch.tensor(edges_df["id_2"].values, dtype=torch.long)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        with open(os.path.join(raw_dir, "musae_facebook_features.json")) as f:
            feats_raw = {int(k): v for k, v in json.load(f).items()}
        x = _reduce_multi_hot(feats_raw, 128)
        target_df   = pd.read_csv(os.path.join(raw_dir, "musae_facebook_target.csv"))
        target_df   = target_df.sort_values("id").reset_index(drop=True)
        label_names = sorted(target_df["page_type"].unique())
        label_to_i  = {name: i for i, name in enumerate(label_names)}
        y = torch.tensor(target_df["page_type"].map(label_to_i).values, dtype=torch.long)
        self._data = Data(x=x, edge_index=edge_index, y=y)
        self.num_features = x.shape[1]; self.num_classes = len(label_names)
    def __getitem__(self, idx):
        assert idx == 0
        return self._data


class WikipediaChameleonSNAP:
    def __init__(self, root=".", num_val=200, num_test=500):
        raw_dir = os.path.join(root, "wikipedia", "chameleon")
        os.makedirs(raw_dir, exist_ok=True)
        files = {
            "musae_chameleon_edges.csv":     f"{MUSAE_BASE}/edges/chameleon_edges.csv",
            "musae_chameleon_features.json": f"{MUSAE_BASE}/features/chameleon.json",
            "musae_chameleon_target.csv":    f"{MUSAE_BASE}/target/chameleon_target.csv",
        }
        for local_name, url in files.items():
            _download(url, os.path.join(raw_dir, local_name))
        edges_df = pd.read_csv(os.path.join(raw_dir, "musae_chameleon_edges.csv"))
        src = torch.tensor(edges_df["id1"].values, dtype=torch.long)
        dst = torch.tensor(edges_df["id2"].values, dtype=torch.long)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        with open(os.path.join(raw_dir, "musae_chameleon_features.json")) as f:
            feats_raw = {int(k): v for k, v in json.load(f).items()}
        x = _reduce_multi_hot(feats_raw, 128)
        target_df = pd.read_csv(os.path.join(raw_dir, "musae_chameleon_target.csv"))
        target_df = target_df.sort_values("id").reset_index(drop=True)
        y = torch.tensor(np.log10(target_df["target"].values), dtype=torch.float32)
        n = x.shape[0]
        perm = torch.tensor(np.random.default_rng(0).permutation(n), dtype=torch.long)
        test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[perm[:num_test]] = True
        val_mask   = torch.zeros(n, dtype=torch.bool); val_mask[perm[num_test:num_test+num_val]] = True
        train_mask = torch.zeros(n, dtype=torch.bool); train_mask[perm[num_test+num_val:]] = True
        self._data = Data(x=x, edge_index=edge_index, y=y,
                          train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        self.num_features = x.shape[1]
    def __getitem__(self, idx):
        assert idx == 0
        return self._data


# =============================================================================
# Degree distribution (grayscale)
# =============================================================================

def plot_degree_distribution(edge_index, title, filename):
    degs    = degree(edge_index[0]).numpy()
    numbers = Counter(degs)
    keys    = np.array(sorted(numbers.keys()))
    vals    = [numbers[k] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.bar(keys, vals, color=G3, width=0.8)
    ax.set_xlabel("Node degree", fontsize=11, fontfamily=FONT)
    ax.set_ylabel("Number of nodes", fontsize=11, fontfamily=FONT)
    ax.set_title(title, fontsize=12, fontweight="bold", color=G0, pad=10, fontfamily=FONT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.97, 0.95,
            f"Min: {int(keys.min())}  Max: {int(keys.max())}\nMean: {np.mean(degs):.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, fontfamily=FONT)
    save(fig, filename)


# =============================================================================
# GCN regressor (kept in sync with run.py) — for the scatter figure
# =============================================================================

class GCNRegressor(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1   = GCNConv(dim_in, dim_h * 4)
        self.gcn2   = GCNConv(dim_h * 4, dim_h * 2)
        self.gcn3   = GCNConv(dim_h * 2, dim_h)
        self.linear = nn.Linear(dim_h, dim_out)
    def forward(self, x, edge_index):
        h = F.dropout(F.relu(self.gcn1(x, edge_index)), p=0.5, training=self.training)
        h = F.dropout(F.relu(self.gcn2(h, edge_index)), p=0.5, training=self.training)
        h = F.relu(self.gcn3(h, edge_index))
        return self.linear(h)
    def fit(self, data, epochs):
        opt = torch.optim.Adam(self.parameters(), lr=0.02, weight_decay=5e-4)
        self.train()
        for _ in range(epochs + 1):
            opt.zero_grad()
            out = self(data.x, data.edge_index)
            loss = F.mse_loss(out.squeeze()[data.train_mask], data.y[data.train_mask].float())
            loss.backward(); opt.step()


def plot_regression_scatter(data_wiki, filename):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = GCNRegressor(data_wiki.x.shape[1], 128, 1)
    model.fit(data_wiki, epochs=200)
    model.eval()
    with torch.no_grad():
        out = model(data_wiki.x, data_wiki.edge_index)
    y_pred = out.squeeze()[data_wiki.test_mask].numpy()
    y_true = data_wiki.y[data_wiki.test_mask].numpy()
    mse = np.mean((y_true - y_pred) ** 2); rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.scatter(y_true, y_pred, alpha=0.45, s=30, color=G4, edgecolors="none", label="Test nodes")
    m_fit = np.polyfit(y_true, y_pred, 1)
    xs = np.linspace(y_true.min(), y_true.max(), 100)
    ax.plot(xs, np.polyval(m_fit, xs), color=G0, linewidth=2.2, label="Regression line")
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
            color=G2, linewidth=1.4, linestyle="--", alpha=0.7, label="Perfect prediction")
    ax.set_xlabel("Ground truth (log10 monthly traffic)", fontsize=11, fontfamily=FONT)
    ax.set_ylabel("Predicted value", fontsize=11, fontfamily=FONT)
    ax.set_title(f"GCN regression — predicted vs ground truth\n"
                 f"MSE={mse:.4f}  |  RMSE={rmse:.4f}  |  MAE={mae:.4f}",
                 fontsize=11, fontweight="bold", color=G0, pad=10, fontfamily=FONT)
    ax.legend(fontsize=10); ax.spines[["top", "right"]].set_visible(False)
    save(fig, filename)


def plot_target_density(data_wiki, filename):
    vals = data_wiki.y.numpy()
    mu, sigma = norm.fit(vals)
    xs = np.linspace(vals.min(), vals.max(), 200)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.hist(vals, bins=40, density=True, alpha=0.7, color=G4, label="Target distribution")
    ax.plot(xs, norm.pdf(xs, mu, sigma), color=G0, linewidth=2.5,
            label=f"Normal fit  (mu={mu:.2f}, sigma={sigma:.2f})")
    ax.set_xlabel("Log10 average monthly traffic", fontsize=11, fontfamily=FONT)
    ax.set_ylabel("Density", fontsize=11, fontfamily=FONT)
    ax.set_title("Distribution of target values — Wikipedia Network (chameleon)",
                 fontsize=12, fontweight="bold", color=G0, pad=10, fontfamily=FONT)
    ax.legend(fontsize=10); ax.spines[["top", "right"]].set_visible(False)
    save(fig, filename)


# =============================================================================
# Figure 6.5 — 20-run comparison (numbers hardcoded from the run)
# =============================================================================

def plot_comparison(filename):
    datasets = ["Cora", "Facebook"]
    models   = ["MLP", "Vanilla GNN", "GCN"]
    means = {"MLP": [53.77, 77.45], "Vanilla GNN": [75.09, 86.58], "GCN": [80.41, 90.00]}
    stds  = {"MLP": [1.41, 0.25],  "Vanilla GNN": [1.33, 1.50],  "GCN": [0.53, 0.13]}
    colors = {"MLP": G5, "Vanilla GNN": G3, "GCN": G0}
    x = np.arange(len(datasets)); w = 0.26
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    for i, m in enumerate(models):
        bars = ax.bar(x + (i - 1) * w, means[m], w, yerr=stds[m], capsize=4,
                      color=colors[m], edgecolor="white", linewidth=0.8, label=m,
                      error_kw=dict(ecolor=G0, lw=1.2))
        for b, mu in zip(bars, means[m]):
            ax.text(b.get_x() + b.get_width() / 2, mu + 2.0, f"{mu:.1f}",
                    ha="center", va="bottom", fontsize=9, color=G0,
                    fontweight="bold", fontfamily=FONT)
    ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=11, fontfamily=FONT)
    ax.set_ylabel("Mean test accuracy (%)", fontsize=11, fontfamily=FONT)
    ax.set_ylim(40, 100)
    ax.set_title("Mean test accuracy over 20 runs — MLP vs Vanilla GNN vs GCN",
                 fontsize=12, fontweight="bold", color=G0, pad=12, fontfamily=FONT)
    ax.legend(fontsize=10); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    save(fig, filename)


# =============================================================================
# Row vs symmetric normalisation diagram (schematic)
# =============================================================================

def plot_normalisation_diagram(filename):
    RECV, SEND, EDGE, FAINT = G0, G3, G0, G6
    d_i, d_j = 2, 6
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_facecolor(BG)
    def panel(ax, title, weight_text, lw, receiver_only):
        ax.set_facecolor(BG); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 8)
        ax.set_title(title, fontsize=12, fontweight="bold", color=G0, fontfamily=FONT, pad=12)
        pi, pj = (2.4, 4.0), (7.6, 4.0)
        for k in range(d_i - 1):
            a = np.pi * (0.6 + 0.5 * k)
            p = (pi[0] + 1.35 * np.cos(a), pi[1] + 1.35 * np.sin(a))
            ax.plot([pi[0], p[0]], [pi[1], p[1]], color=FAINT, lw=1.5, zorder=1)
            ax.add_patch(Circle(p, 0.22, color=FAINT, zorder=2))
        for k in range(d_j - 1):
            a = np.pi * (-0.5 + 0.28 * k)
            p = (pj[0] + 1.5 * np.cos(a), pj[1] + 1.5 * np.sin(a))
            ax.plot([pj[0], p[0]], [pj[1], p[1]], color=FAINT, lw=1.5, zorder=1)
            ax.add_patch(Circle(p, 0.22, color=FAINT, zorder=2))
        ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color=EDGE, lw=lw, zorder=3, solid_capstyle="round")
        ax.text(5.0, 4.5, weight_text, ha="center", va="bottom", fontsize=13, color=G0,
                fontweight="bold", fontfamily=FONT,
                bbox=dict(facecolor="white", edgecolor=G2, boxstyle="round,pad=0.25", lw=1.3))
        ax.add_patch(Circle(pi, 0.42, color=RECV, zorder=4, ec="white", lw=1.5))
        ax.text(*pi, "i", ha="center", va="center", color="white", fontsize=14,
                fontweight="bold", fontfamily=FONT, zorder=5)
        ax.add_patch(Circle(pj, 0.42, color=SEND, zorder=4, ec="white", lw=1.5))
        ax.text(*pj, "j", ha="center", va="center", color="white", fontsize=14,
                fontweight="bold", fontfamily=FONT, zorder=5)
        ax.text(pi[0], pi[1] - 0.95, f"receiver\ndegree $d_i$={d_i}", ha="center",
                va="top", fontsize=9.5, color=G0, fontfamily=FONT)
        ax.text(pj[0], pj[1] - 0.95, f"sender\ndegree $d_j$={d_j}", ha="center",
                va="top", fontsize=9.5, color=G0, fontfamily=FONT)
        note = "scaled by the receiver's degree only" if receiver_only else "scaled by both endpoints' degrees"
        ax.text(5.0, 0.7, note, ha="center", fontsize=10, color=G2,
                fontstyle="italic", fontfamily=FONT)
    panel(axes[0], "Row normalisation  (asymmetric)",
          r"$\dfrac{1}{d_i}=\dfrac{1}{2}=0.50$", 6.0, True)
    panel(axes[1], "Symmetric normalisation",
          r"$\dfrac{1}{\sqrt{d_i\,d_j}}=\dfrac{1}{\sqrt{12}}\approx0.29$", 3.4, False)
    fig.tight_layout()
    save(fig, filename)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Data-independent figures:")
    plot_comparison("mlp_gnn_gcn_comparison.png")
    plot_normalisation_diagram("fig_row_vs_symmetric_norm.png")

    print("Data-dependent figures (require dataset download):")
    data_cora = Planetoid(root=".", name="Cora")[0]
    plot_degree_distribution(data_cora.edge_index,
                             "Node degree distribution — Cora dataset", "cora_degrees.png")

    data_fb = FacebookPagePageSNAP(root=".")[0]
    plot_degree_distribution(data_fb.edge_index,
                             "Node degree distribution — Facebook Page-Page dataset",
                             "facebook_degrees.png")

    data_wiki = WikipediaChameleonSNAP(root=".", num_val=200, num_test=500)[0]
    plot_degree_distribution(data_wiki.edge_index,
                             "Node degree distribution — Wikipedia Network (chameleon)",
                             "wiki_degrees.png")
    plot_target_density(data_wiki, "wiki_target_density.png")
    plot_regression_scatter(data_wiki, "gcn_regression_scatter.png")

    print("Done.")