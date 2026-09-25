"""
Chapter 15 – Forecasting Traffic Using A3T-GCN
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch-geometric torch-geometric-temporal==0.54.0
    pip install pandas numpy matplotlib networkx

Fixes versus the first edition:
  - 'PeMSD7_W_228.csv.csv' corrected to 'PeMSD7_W_228.csv'
  - snapshot.edge_weight corrected to snapshot.edge_attr in evaluation loop
  - All plot colors converted to grayscale
  - edge_index computed with np.stack(np.where(...)) for clarity
  - Note added explaining why CPU is preferred for A3TGCN training
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

# =============================================================================
# PART 1 – Load and explore the PeMS-M dataset
# =============================================================================

print("=" * 60)
print("PART 1 – Loading and exploring PeMS-M")
print("=" * 60)

url = ('https://github.com/VeritasYin/STGCN_IJCAI-18/raw/master/'
       'dataset/PeMSD7_Full.zip')
print("Downloading PeMSD7_Full.zip …")
with urlopen(url) as zurl:
    with ZipFile(BytesIO(zurl.read())) as zfile:
        zfile.extractall('.')

# Fix from first edition: 'PeMSD7_W_228.csv.csv' → 'PeMSD7_W_228.csv'
speeds    = pd.read_csv('PeMSD7_V_228.csv', names=range(0, 228))
distances = pd.read_csv('PeMSD7_W_228.csv', names=range(0, 228))

print(f"\nSpeeds shape:    {speeds.shape}    (time steps × stations)")
print(f"Distances shape: {distances.shape}")

# ── All-station speed plot ────────────────────────────────────────────────────
plt.figure(figsize=(10, 5), dpi=150)
plt.plot(speeds, color='#555555', linewidth=0.3, alpha=0.4)
plt.grid(linestyle=':')
plt.xlabel('Time (5 min)'); plt.ylabel('Traffic speed (mph)')
plt.title('Traffic speed — all 228 sensor stations')
plt.tight_layout()
plt.savefig('speeds_all.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved speeds_all.png")

# ── Mean + std plot ───────────────────────────────────────────────────────────
mean = speeds.mean(axis=1)
std  = speeds.std(axis=1)

plt.figure(figsize=(10, 5), dpi=150)
plt.plot(mean, color='#111111', linewidth=1.2, label='Mean')
plt.fill_between(mean.index, mean-std, mean+std,
                 color='#AAAAAA', alpha=0.3, label='+/- 1 std')
plt.grid(linestyle=':')
plt.xlabel('Time (5 min)'); plt.ylabel('Traffic speed (mph)')
plt.title('Mean traffic speed with standard deviation')
plt.legend(); plt.tight_layout()
plt.savefig('speeds_mean.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved speeds_mean.png")

# ── Distance vs correlation ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
fig.tight_layout(pad=3.0)
ax1.matshow(distances, cmap='Greys')
ax1.set_xlabel('Sensor'); ax1.set_ylabel('Sensor')
ax1.set_title('Distance matrix')
ax2.matshow(-np.corrcoef(speeds.T), cmap='Greys')
ax2.set_xlabel('Sensor'); ax2.set_ylabel('Sensor')
ax2.set_title('Negated correlation')
plt.savefig('distance_corr.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved distance_corr.png")


# =============================================================================
# PART 2 – Process: adjacency matrix + temporal graph
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 – Processing: adjacency matrix and temporal graph")
print("=" * 60)


def compute_adj(distances, sigma2=0.1, epsilon=0.5):
    """
    Gaussian kernel weighted adjacency matrix.
    Source: Li et al. (2018), Diffusion Convolutional RNN.
    sigma^2=0.1 and epsilon=0.5 match the original implementation.
    """
    d      = distances.to_numpy() / 10000.
    d2     = d * d
    n      = distances.shape[0]
    w_mask = np.ones([n, n]) - np.identity(n)   # zero out diagonal
    return np.exp(-d2/sigma2) * (np.exp(-d2/sigma2) >= epsilon) * w_mask


adj = compute_adj(distances)
print(f"Adjacency matrix: {adj.shape}, non-zero entries: {(adj > 0).sum()}")
print(f"Sample row 0: {adj[0, :5]}")

# Adjacency matrix plot
plt.figure(figsize=(7, 6), dpi=150)
cax = plt.matshow(adj, cmap='Greys_r', fignum=False)
plt.colorbar(cax); plt.xlabel('Sensor'); plt.ylabel('Sensor')
plt.title('PeMS-M weighted adjacency matrix')
plt.tight_layout()
plt.savefig('adj_matrix.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved adj_matrix.png")

# Graph plot
rows_g, cols_g = np.where(adj > 0)
G = nx.Graph()
G.add_edges_from(zip(rows_g.tolist(), cols_g.tolist()))
plt.figure(figsize=(10, 5), dpi=150)
nx.draw(G, with_labels=False, node_size=20, node_color='#333333',
        edge_color='#AAAAAA', width=0.5)
plt.title('PeMS-M sensor network as a graph')
plt.tight_layout()
plt.savefig('sensor_graph.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved sensor_graph.png")

# Z-score normalisation
def zscore(x, mean, std):
    return (x - mean) / std

def inverse_zscore(x, mean, std):
    return x * std + mean

speeds_norm = zscore(speeds, speeds.mean(axis=0), speeds.std(axis=0))
print(f"\nNormalised speeds (first row, first 5 cols):")
print(speeds_norm.head(1).iloc[0, :5].values)

# Build temporal graph dataset
# horizon=48 (4 hours ahead) is the task A3T-GCN was designed for.
# PART 4's LSTM and STAEformer use horizon=1 as a separate short-range
# forecasting experiment. The two experiments are presented separately.
lags    = 12
horizon = 48
# Materialise the DataFrame as a numpy array once to avoid copying inside the loop.
speeds_arr = speeds_norm.to_numpy()
xs, ys  = [], []
for i in range(lags, speeds_arr.shape[0] - horizon):
    xs.append(speeds_arr[i-lags:i].T)
    ys.append(speeds_arr[i+horizon-1])

from torch_geometric_temporal.signal import (
    StaticGraphTemporalSignal, temporal_signal_split
)

# Fix: np.stack for unambiguous COO format
edge_index  = np.stack(np.where(adj > 0))
edge_weight = adj[adj > 0]

dataset = StaticGraphTemporalSignal(edge_index, edge_weight, xs, ys)
print(f"\nFirst graph snapshot: {dataset[0]}")

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
print(f"Train snapshots: {sum(1 for _ in train_dataset)}")
print(f"Test snapshots:  {sum(1 for _ in test_dataset)}")

# reset iterators
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)


# =============================================================================
# PART 3 – A3T-GCN model
# =============================================================================

print("\n" + "=" * 60)
print("PART 3 – A3T-GCN training and evaluation")
print("=" * 60)

from torch_geometric_temporal.nn.recurrent import A3TGCN


class TemporalGNN(torch.nn.Module):
    def __init__(self, dim_in, periods):
        super().__init__()
        self.tgnn   = A3TGCN(in_channels=dim_in,
                              out_channels=32,
                              periods=periods)
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index, edge_attr):
        h = self.tgnn(x, edge_index, edge_attr).relu()
        return self.linear(h)


# A3TGCN runs more efficiently on CPU for this dataset:
# the attention mechanism is sequential and benefits from CPU cache locality.
model     = TemporalGNN(lags, 1).to('cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
print(f"\nModel: {model}")

print("\nTraining A3T-GCN (30 epochs) …")
model.train()
for epoch in range(30):
    optimizer.zero_grad()
    epoch_losses = []
    for snapshot in train_dataset:
        y_pred = model(snapshot.x.unsqueeze(2),
                       snapshot.edge_index,
                       snapshot.edge_attr)
        loss = torch.mean((y_pred - snapshot.y)**2)
        # backward per snapshot to free the autograd graph, but do NOT step:
        # gradients accumulate in each parameter's .grad tensor across the epoch.
        loss.backward()
        epoch_losses.append(loss.item())
    # One optimizer step per epoch using the accumulated gradients.
    # Semantically equivalent to the first edition's full-batch update, but
    # memory-safe because the autograd graph of each snapshot is freed at once.
    optimizer.step()
    if (epoch+1) % 10 == 0:
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  Epoch {epoch+1:>2} | Train MSE: {avg_loss:.4f}")

# reset iterator for evaluation
_, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

# ── Evaluation ────────────────────────────────────────────────────────────────

def MAE(real, pred):  return np.mean(np.abs(pred - real))
def RMSE(real, pred): return np.sqrt(np.mean((pred - real)**2))
def MAPE(real, pred): return np.mean(np.abs(pred - real) / (real + 1e-5))

# Ground truth (inverse z-score)
y_test = []
for snapshot in test_dataset:
    y_hat  = inverse_zscore(snapshot.y.numpy(),
                            speeds.mean(axis=0), speeds.std(axis=0))
    y_test = np.append(y_test, y_hat)

_, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

# A3T-GCN predictions
# Fix from first edition: snapshot.edge_attr (not snapshot.edge_weight)
model.eval()
gnn_pred = []
for snapshot in test_dataset:
    y_hat = model(snapshot.x.unsqueeze(2),
                  snapshot.edge_index,
                  snapshot.edge_attr).squeeze().detach().numpy()
    y_hat = inverse_zscore(y_hat, speeds.mean(axis=0), speeds.std(axis=0))
    gnn_pred = np.append(gnn_pred, y_hat)

_, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

# Random Walk baseline
rw_pred = []
for snapshot in test_dataset:
    y_hat = inverse_zscore(snapshot.x[:, -1].numpy(),
                           speeds.mean(axis=0), speeds.std(axis=0))
    rw_pred = np.append(rw_pred, y_hat)

# Historical Average baseline
ha_pred = []
for i in range(lags, speeds_arr.shape[0] - horizon):
    y_hat = speeds_arr[i-lags:i].T.mean(axis=1)
    y_hat = inverse_zscore(y_hat, speeds.mean(axis=0), speeds.std(axis=0))
    ha_pred.append(y_hat)
ha_pred = np.array(ha_pred).flatten()[-len(y_test):]

print("\n" + "-" * 50)
print(f"{'Model':<20} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
print("-" * 50)
for name, pred in [('A3T-GCN',    gnn_pred),
                   ('Random Walk', rw_pred),
                   ('Hist. Avg',   ha_pred)]:
    print(f"{name:<20} {RMSE(y_test,pred):>8.4f} "
          f"{MAE(y_test,pred):>8.4f} {MAPE(y_test,pred)*100:>7.2f}%")
print("-" * 50)

# ── Prediction plot ───────────────────────────────────────────────────────────
_, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

model.eval()
y_preds = []
for snapshot in test_dataset:
    y_hat = model(snapshot.x.unsqueeze(2),
                  snapshot.edge_index,
                  snapshot.edge_attr).squeeze().detach().numpy()
    y_hat = inverse_zscore(y_hat, speeds.mean(axis=0), speeds.std(axis=0))
    y_preds.append(y_hat.mean())

split = len(speeds) - len(y_preds)
plt.figure(figsize=(10, 5), dpi=150)
plt.plot(mean, color='#111111', linewidth=0.9, label='Mean')
plt.fill_between(mean.index, mean-std, mean+std,
                 color='#BBBBBB', alpha=0.25)
plt.plot(range(split, len(speeds)), y_preds,
         color='#333333', linestyle='--', linewidth=1.8, label='A3T-GCN')
plt.axvline(x=split, color='#777777', linestyle='--')
plt.xlabel('Time (5 min)'); plt.ylabel('Traffic speed (mph)')
plt.title('A3T-GCN — mean predicted traffic speed on the test set')
plt.legend(); plt.grid(linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
plt.close(); print("\nSaved predictions.png")
print("Done.")


# =============================================================================
# PART 4 – LSTM baseline (no graph) and STAEformer
# =============================================================================

print("\n" + "=" * 60)
print("PART 4 – LSTM baseline and STAEformer")
print("=" * 60)
print("Both models use a batched (B, N, T) format instead of per-snapshot.\n")

from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

# Build batched arrays from the same normalised speeds
lags_b = 12; horizon_b = 1
# Reuse the speeds_arr materialised in PART 2 to avoid the inner-loop copy bug.
xs_b, ys_b = [], []
for i in range(lags_b, speeds_arr.shape[0] - horizon_b):
    xs_b.append(speeds_arr[i-lags_b:i].T)   # (228, 12)
    ys_b.append(speeds_arr[i+horizon_b-1])  # (228,)

xs_b = np.array(xs_b, dtype=np.float32)
ys_b = np.array(ys_b, dtype=np.float32)

split_b = int(0.8 * len(xs_b))
x_tr = torch.tensor(xs_b[:split_b]); y_tr = torch.tensor(ys_b[:split_b])
x_te = torch.tensor(xs_b[split_b:]); y_te = torch.tensor(ys_b[split_b:])

tr_loader = TorchDataLoader(TensorDataset(x_tr, y_tr), batch_size=32, shuffle=True)
te_loader = TorchDataLoader(TensorDataset(x_te, y_te), batch_size=32, shuffle=False)


def eval_metrics_batch(model, loader):
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb).numpy()
            true = yb.numpy()
            pred_mph = pred * speeds.std(axis=0).values + speeds.mean(axis=0).values
            true_mph = true * speeds.std(axis=0).values + speeds.mean(axis=0).values
            all_pred.append(pred_mph); all_true.append(true_mph)
    pred = np.concatenate(all_pred); true = np.concatenate(all_true)
    rmse = np.sqrt(np.mean((pred-true)**2))
    mae  = np.mean(np.abs(pred-true))
    mape = np.mean(np.abs(pred-true)/(true+1e-5))*100
    return rmse, mae, mape


# ── LSTM baseline ─────────────────────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    """
    Independent LSTM per node — no spatial information.
    Ablation study: quantifies the value of the graph in A3T-GCN.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm   = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=0.1)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        B, N, T = x.shape
        x       = x.reshape(B * N, T, 1)
        out, _  = self.lstm(x)
        pred    = self.linear(out[:, -1, :]).squeeze(-1)
        return pred.reshape(B, N)


lstm_m = LSTMBaseline()
opt_l  = torch.optim.Adam(lstm_m.parameters(), lr=1e-3)

print("Training LSTM baseline (30 epochs) …")
for epoch in range(30):
    lstm_m.train()
    for xb, yb in tr_loader:
        opt_l.zero_grad()
        F.mse_loss(lstm_m(xb), yb).backward()
        opt_l.step()
    if (epoch+1) % 10 == 0:
        lstm_m.eval()
        rmse, mae, mape = eval_metrics_batch(lstm_m, te_loader)
        print(f"  Epoch {epoch+1} | RMSE={rmse:.4f} MAE={mae:.4f}")

lstm_m.eval()
rmse_l, mae_l, mape_l = eval_metrics_batch(lstm_m, te_loader)
print(f"\nLSTM baseline: RMSE={rmse_l:.4f}  MAE={mae_l:.4f}  MAPE={mape_l:.2f}%")


# ── STAEformer ────────────────────────────────────────────────────────────────

class STAEformer(nn.Module):
    """
    Spatial-Temporal Adaptive Embedding Transformer (Liu et al. 2023).
    Does not require a predefined adjacency matrix.
    Spatial structure is learned implicitly through self-attention.
    """
    def __init__(self, num_nodes, in_steps, out_steps=1,
                 d_model=64, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.node_emb   = nn.Embedding(num_nodes, d_model)
        self.input_proj = nn.Linear(in_steps, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, out_steps)

    def forward(self, x):
        B, N, T = x.shape
        h   = self.input_proj(x)
        idx = torch.arange(N, device=x.device)
        h   = h + self.node_emb(idx)
        h   = self.transformer(h)
        return self.output_proj(h).squeeze(-1)


stae_m = STAEformer(num_nodes=228, in_steps=lags_b)
opt_s  = torch.optim.Adam(stae_m.parameters(), lr=1e-3, weight_decay=1e-4)
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=30)

print("\nTraining STAEformer (30 epochs) …")
for epoch in range(30):
    stae_m.train()
    for xb, yb in tr_loader:
        opt_s.zero_grad()
        F.mse_loss(stae_m(xb), yb).backward()
        opt_s.step()
    sched.step()
    if (epoch+1) % 10 == 0:
        stae_m.eval()
        rmse, mae, mape = eval_metrics_batch(stae_m, te_loader)
        print(f"  Epoch {epoch+1} | RMSE={rmse:.4f} MAE={mae:.4f}")

stae_m.eval()
rmse_s, mae_s, mape_s = eval_metrics_batch(stae_m, te_loader)
print(f"\nSTAEformer: RMSE={rmse_s:.4f}  MAE={mae_s:.4f}  MAPE={mape_s:.2f}%")


# ── Final comparison ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("FINAL COMPARISON — all models on PeMS-M")
print("=" * 60)
print("\nPART 3 – A3T-GCN long-range forecasting (horizon=48, 4 hours ahead)")
print("-" * 65)
print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
print("-" * 65)
print(f"{'A3T-GCN':<22} {RMSE(y_test, gnn_pred):>8.4f} "
      f"{MAE(y_test, gnn_pred):>8.4f} "
      f"{MAPE(y_test, gnn_pred)*100:>7.2f}%")
print(f"{'Random Walk':<22} {RMSE(y_test, rw_pred):>8.4f} "
      f"{MAE(y_test, rw_pred):>8.4f} "
      f"{MAPE(y_test, rw_pred)*100:>7.2f}%")
print(f"{'Hist. Avg':<22} {RMSE(y_test, ha_pred):>8.4f} "
      f"{MAE(y_test, ha_pred):>8.4f} "
      f"{MAPE(y_test, ha_pred)*100:>7.2f}%")
print("-" * 65)

print("\nPART 4 – short-range forecasting (horizon=1, 5 minutes ahead)")
print("-" * 65)
print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
print("-" * 65)
print(f"{'LSTM (no graph)':<22} {rmse_l:>8.4f} {mae_l:>8.4f} {mape_l:>7.2f}%")
print(f"{'STAEformer':<22} {rmse_s:>8.4f} {mae_s:>8.4f} {mape_s:>7.2f}%")
print("-" * 65)

print("""
The two experiments target different forecasting horizons and are not
directly comparable. PART 3 evaluates A3T-GCN on long-range prediction
(4 hours ahead), which is the task A3T-GCN was designed for. PART 4
compares LSTM and STAEformer on short-range prediction (5 minutes
ahead), where the temporal signal dominates and the value of an
explicit graph structure is limited.
""")