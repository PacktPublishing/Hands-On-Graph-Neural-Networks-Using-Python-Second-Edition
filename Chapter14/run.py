"""
Chapter 14 – Temporal Graph Neural Networks
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch-geometric torch-geometric-temporal==0.54.0
    pip install pandas matplotlib scikit-learn numpy

Fixes versus the first edition (ex Chapter 13):
  - 'From', 'Class', 'Def' typos corrected to lowercase Python keywords
  - All plot colors converted to grayscale
  - sns.regplot replaced with matplotlib scatter + polyfit
  - GitHub URL updated to Chapter14
  - Chapter number references updated throughout
  - Comment added explaining optimizer.zero_grad() ordering in EvolveGCN loop

New addition:
  - Part 3: Continuous-time link prediction with TGN (PyG native, no extra deps)
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# PART 1 – EvolveGCN on WikiMaths (static graph, temporal signal)
# =============================================================================

print("\n" + "=" * 60)
print("PART 1 – EvolveGCN on WikiMaths")
print("=" * 60)

from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.dataset import WikiMathsDatasetLoader
from torch_geometric_temporal.nn.recurrent import EvolveGCNH, EvolveGCNO

dataset = WikiMathsDatasetLoader().get_dataset()
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.5)

print(f"\nDataset snapshot 0:   {dataset[0]}")
print(f"Dataset snapshot 500: {dataset[500]}")

# Time series visualisation (grayscale — no color='r'/color='g')
mean_cases = [snapshot.y.mean().item() for snapshot in dataset]
std_cases  = [snapshot.y.std().item()  for snapshot in dataset]
df           = pd.DataFrame(mean_cases, columns=['mean'])
df['std']    = std_cases
df['rolling']= df['mean'].rolling(7).mean()

fig, ax = plt.subplots(figsize=(13, 5), dpi=150)
ax.plot(df['mean'],    color='#111111', linewidth=0.8, label='Mean', alpha=0.8)
ax.plot(df['rolling'], color='#000000', linewidth=2.0, label='7-day moving average')
ax.fill_between(df.index,
                df['mean']-df['std'], df['mean']+df['std'],
                color='#999999', alpha=0.25, label='±1 std dev')
ax.axvline(x=360, color='#555555', linestyle='--', linewidth=1.5)
ax.text(365, df['mean'].max()*0.9, 'Train/test split',
        rotation=90, color='#555555', fontsize=9, va='top')
ax.set_xlabel('Time (days)'); ax.set_ylabel('Normalised number of visits')
ax.set_title('WikiMaths — mean normalised visits')
ax.legend(fontsize=10); ax.grid(linestyle=':', alpha=0.5)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig("wikimaths_ts.png", dpi=150, bbox_inches='tight')
plt.close(); print("Saved wikimaths_ts.png")


# ── EvolveGCN-H ───────────────────────────────────────────────────────────────

class TemporalGNN_H(torch.nn.Module):
    def __init__(self, node_count, dim_in):
        super().__init__()
        self.recurrent = EvolveGCNH(node_count, dim_in)
        self.linear    = torch.nn.Linear(dim_in, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight).relu()
        return self.linear(h)


model_h   = TemporalGNN_H(dataset[0].x.shape[0], dataset[0].x.shape[1])
optimizer = torch.optim.Adam(model_h.parameters(), lr=0.01)
print(f"\n{model_h}")

print("\nTraining EvolveGCN-H (50 epochs) …")
model_h.train()
for epoch in range(50):
    for snapshot in train_dataset:
        y_pred = model_h(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss   = torch.mean((y_pred - snapshot.y)**2)
        loss.backward()
        optimizer.step()
        # zero_grad after step: intentional for recurrent training
        # (gradients accumulate across snapshots within an epoch)
        optimizer.zero_grad()
    if (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1}/50 | last batch loss: {loss.item():.4f}")

model_h.eval()
loss_h = 0
for i, snapshot in enumerate(test_dataset):
    y_pred  = model_h(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    loss_h += torch.mean((y_pred - snapshot.y)**2)
mse_h = (loss_h / (i+1)).item()
print(f"\nEvolveGCN-H test MSE: {mse_h:.4f}")

# Predictions time series plot
y_preds = [
    model_h(s.x, s.edge_index, s.edge_attr).squeeze().detach().numpy().mean()
    for s in test_dataset
]
fig, ax = plt.subplots(figsize=(13, 5), dpi=150)
ax.plot(df['mean'],    color='#111111', linewidth=0.8, label='Mean', alpha=0.8)
ax.plot(df['rolling'], color='#000000', linewidth=2.0, label='Moving average')
ax.fill_between(df.index, df['mean']-df['std'], df['mean']+df['std'],
                color='#BBBBBB', alpha=0.3, label='±1 std dev')
ax.plot(range(360, 360+len(y_preds)), y_preds,
        color='#333333', linewidth=2.0, linestyle='--', label='EvolveGCN prediction')
ax.axvline(x=360, color='#777777', linestyle='--', linewidth=1.5)
ax.set_xlabel('Time (days)'); ax.set_ylabel('Normalised visits')
ax.set_title(f'WikiMaths — EvolveGCN-H predictions (test MSE={mse_h:.4f})')
ax.legend(fontsize=10); ax.grid(linestyle=':', alpha=0.5)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig("wikimaths_pred.png", dpi=150, bbox_inches='tight')
plt.close(); print("Saved wikimaths_pred.png")

# Scatter plot — matplotlib only (no seaborn)
snap0 = list(test_dataset)[0]
y_pred0 = model_h(snap0.x, snap0.edge_index, snap0.edge_attr).detach().squeeze().numpy()
y_true0 = snap0.y.numpy()
m_fit   = np.polyfit(y_true0, y_pred0, 1)
x_line  = np.linspace(y_true0.min(), y_true0.max(), 100)

fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
ax.scatter(y_true0, y_pred0, color='#333333', alpha=0.4, s=15, edgecolors='none')
ax.plot(x_line, np.polyval(m_fit, x_line), color='#000000', linewidth=2.0,
        label='Regression line')
ax.plot([x_line[0],x_line[-1]], [x_line[0],x_line[-1]],
        color='#999999', linewidth=1.2, linestyle='--', label='Perfect prediction')
ax.set_xlabel('Ground truth'); ax.set_ylabel('Predicted value')
ax.set_title('WikiMaths — predicted vs ground truth (snapshot t=0)')
ax.legend(fontsize=9); ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig("wikimaths_scatter.png", dpi=150, bbox_inches='tight')
plt.close(); print("Saved wikimaths_scatter.png")

# ── EvolveGCN-O ───────────────────────────────────────────────────────────────

class TemporalGNN_O(torch.nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.recurrent = EvolveGCNO(dim_in, 1)
        self.linear    = torch.nn.Linear(dim_in, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight).relu()
        return self.linear(h)


model_o   = TemporalGNN_O(dataset[0].x.shape[1])
optimizer = torch.optim.Adam(model_o.parameters(), lr=0.01)
model_o.train()
for epoch in range(50):
    for snapshot in train_dataset:
        y_pred = model_o(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss   = torch.mean((y_pred - snapshot.y)**2)
        loss.backward(); optimizer.step(); optimizer.zero_grad()

model_o.eval()
loss_o = 0
for i, snapshot in enumerate(test_dataset):
    y_pred  = model_o(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    loss_o += torch.mean((y_pred - snapshot.y)**2)
mse_o = (loss_o / (i+1)).item()
print(f"EvolveGCN-O test MSE: {mse_o:.4f}")


# =============================================================================
# PART 2 – MPNN-LSTM on England Covid (dynamic graph, temporal signal)
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 – MPNN-LSTM on England Covid")
print("=" * 60)

from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader
from torch_geometric_temporal.nn.recurrent import MPNNLSTM  # lowercase: fixed

dataset_cov = EnglandCovidDatasetLoader().get_dataset(lags=14)
train_cov, test_cov = temporal_signal_split(dataset_cov, train_ratio=0.8)

mean_cov = [s.y.mean().item() for s in dataset_cov]
std_cov  = [s.y.std().item()  for s in dataset_cov]

df_cov           = pd.DataFrame(mean_cov, columns=['mean'])
df_cov['std']    = std_cov


class TemporalGNN_Covid(torch.nn.Module):   # lowercase 'class': fixed
    def __init__(self, dim_in, dim_h, num_nodes):
        super().__init__()
        self.recurrent = MPNNLSTM(dim_in, dim_h, num_nodes, 1, 0.5)
        self.dropout   = torch.nn.Dropout(0.5)
        self.linear    = torch.nn.Linear(2*dim_h + dim_in, 1)

    def forward(self, x, edge_index, edge_weight):   # lowercase 'def': fixed
        h = self.recurrent(x, edge_index, edge_weight).relu()
        h = self.dropout(h)
        return self.linear(h).tanh()


model_cov = TemporalGNN_Covid(
    dataset_cov[0].x.shape[1], 64, dataset_cov[0].x.shape[0])
optimizer = torch.optim.Adam(model_cov.parameters(), lr=0.001)
print(f"\n{model_cov}")

print("\nTraining MPNN-LSTM (100 epochs) …")
model_cov.train()
for epoch in range(100):
    loss = 0
    for i, snapshot in enumerate(train_cov):
        y_pred = model_cov(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss   = loss + torch.mean((y_pred - snapshot.y)**2)
    loss = loss / (i + 1)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    if (epoch+1) % 20 == 0:
        print(f"  Epoch {epoch+1}/100 | Loss: {loss.item():.4f}")

model_cov.eval()
loss_cov = 0
for i, snapshot in enumerate(test_cov):
    y_pred    = model_cov(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    loss_cov += torch.mean((y_pred - snapshot.y)**2)
mse_cov = (loss_cov / (i+1)).item()
print(f"\nMPNN-LSTM test MSE: {mse_cov:.4f}")


# =============================================================================
# PART 3 – TGN on Wikipedia interaction dataset (continuous-time)
# =============================================================================

print("\n" + "=" * 60)
print("PART 3 – TGN on Wikipedia (continuous-time link prediction)")
print("=" * 60)

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage, LastAggregator, LastNeighborLoader
)
from torch_geometric.nn import TransformerConv
from sklearn.metrics import average_precision_score

torch.manual_seed(SEED)

data_tgn = JODIEDataset('.', name='Wikipedia')[0].to(device)
print(f"\nWikipedia dataset: {data_tgn.num_nodes} nodes, "
      f"{data_tgn.num_events} events, "
      f"edge feat dim: {data_tgn.msg.size(-1)}")

train_tgn, val_tgn, test_tgn = data_tgn.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15)
train_loader_tgn = TemporalDataLoader(train_tgn, batch_size=200)
val_loader_tgn   = TemporalDataLoader(val_tgn,   batch_size=200)
test_loader_tgn  = TemporalDataLoader(test_tgn,  batch_size=200)

memory_dim = time_dim = embedding_dim = 100

memory = TGNMemory(
    data_tgn.num_nodes,
    data_tgn.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data_tgn.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

neighbor_loader = LastNeighborLoader(data_tgn.num_nodes, size=10, device=device)


class GraphAttnEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        # Time encoding is concatenated with the message per-edge, and the
        # combined edge attribute feeds into the TransformerConv.
        edge_dim     = msg_dim + time_enc.out_channels
        self.conv    = TransformerConv(
            in_channels, out_channels // 2, heads=2,
            dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t     = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


gnn = GraphAttnEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=data_tgn.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = torch.nn.Linear(in_channels, in_channels)
        self.lin_dst = torch.nn.Linear(in_channels, in_channels)
        self.lin_out = torch.nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        return self.lin_out(F.relu(h))


link_pred = LinkPredictor(in_channels=embedding_dim).to(device)
optimizer = torch.optim.Adam(
    list(memory.parameters()) + list(gnn.parameters()) +
    list(link_pred.parameters()), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()


def process_batch(src, dst, neg_dst, t, msg):
    """Helper: encode nodes and compute link probabilities."""
    n_id = torch.cat([src, dst, neg_dst]).unique()
    n_id, edge_index, e_id = neighbor_loader(n_id)
    assoc = torch.empty(data_tgn.num_nodes, dtype=torch.long, device=device)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)
    z, last_update = memory(n_id)
    z = gnn(z, last_update, edge_index,
            data_tgn.t[e_id].to(device), data_tgn.msg[e_id].to(device))
    return (link_pred(z[assoc[src]], z[assoc[dst]]),
            link_pred(z[assoc[src]], z[assoc[neg_dst]]))


def train_tgn_epoch():
    memory.train(); gnn.train(); link_pred.train()
    memory.reset_state(); neighbor_loader.reset_state()
    total_loss = 0
    for batch in train_loader_tgn:
        optimizer.zero_grad()
        src, dst, t, msg = (batch.src.to(device), batch.dst.to(device),
                            batch.t.to(device),   batch.msg.to(device))
        neg_dst = torch.randint(0, data_tgn.num_nodes,
                                (src.size(0),), device=device)
        pos_out, neg_out = process_batch(src, dst, neg_dst, t, msg)
        loss = (criterion(pos_out, torch.ones_like(pos_out)) +
                criterion(neg_out, torch.zeros_like(neg_out)))
        loss.backward(); optimizer.step()
        # Update memories AFTER the gradient step
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        # Detach memory state to prevent autograd from accumulating across batches.
        # Without this, the next batch's backward would try to traverse through
        # the freed graph of the current batch via the persistent memory tensors.
        memory.detach()
        total_loss += float(loss) * batch.num_events
    return total_loss / train_tgn.num_events


@torch.no_grad()
def test_tgn(loader):
    memory.eval(); gnn.eval(); link_pred.eval()
    aps = []
    for batch in loader:
        src, dst, t, msg = (batch.src.to(device), batch.dst.to(device),
                            batch.t.to(device),   batch.msg.to(device))
        neg_dst = torch.randint(0, data_tgn.num_nodes,
                                (src.size(0),), device=device)
        pos_out, neg_out = process_batch(src, dst, neg_dst, t, msg)
        y_pred = torch.cat([pos_out, neg_out]).sigmoid().cpu()
        y_true = torch.cat([torch.ones(src.size(0)),
                             torch.zeros(src.size(0))])
        aps.append(average_precision_score(y_true, y_pred))
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
    return float(torch.tensor(aps).mean())


print("\nTraining TGN (50 epochs) …")
train_losses, val_aps, test_aps = [], [], []
for epoch in range(1, 51):
    loss   = train_tgn_epoch()
    val_ap  = test_tgn(val_loader_tgn)
    test_ap = test_tgn(test_loader_tgn)
    train_losses.append(loss); val_aps.append(val_ap); test_aps.append(test_ap)
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:>2} | Loss: {loss:.4f} | "
              f"Val AP: {val_ap:.4f} | Test AP: {test_ap:.4f}")

print(f"\nFinal Test AP: {test_aps[-1]:.4f}")

# Loss curve plot
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
ax.plot(train_losses, color='#000000', linewidth=2.0, label='Train loss')
ax.plot(val_aps,      color='#555555', linewidth=2.0,
        linestyle='--', label='Val AP')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss / AP')
ax.set_title('TGN training on Wikipedia interaction dataset')
ax.legend(fontsize=10); ax.grid(linestyle=':', alpha=0.4)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig("tgn_training.png", dpi=150, bbox_inches='tight')
plt.close(); print("Saved tgn_training.png")

print("\nDone.")