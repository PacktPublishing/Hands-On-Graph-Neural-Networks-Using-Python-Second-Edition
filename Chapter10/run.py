"""
Chapter 10 – Graph Transformers: Attention Beyond Local Neighborhoods
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch-geometric pandas matplotlib

Prerequisites:
    None beyond the dependencies above. ZINC-subset (12k molecules) is
    downloaded automatically by PyG on first run.


Design notes:
  - Both models output raw logits at the head; the regression loss is
    L1Loss (mean absolute error). No log_softmax or activation on output.
  - Laplacian eigenvectors are sign-arbitrary. The GraphGPS model applies
    a random sign flip to the positional encoding during training only,
    which forces the model to become sign-invariant.
  - We stratify the test error by molecule size for both models. The
    resulting numbers feed into Figure 10.3 (see generate_figures.py).
"""

import time

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.nn import Embedding, Linear, ModuleList, Sequential, ReLU, BatchNorm1d

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# Common utilities
# =============================================================================

from torch_geometric.datasets import ZINC
from torch_geometric.loader   import DataLoader


def stratified_mae(model, loader, forward_fn):
    """Return a DataFrame with per-graph n_atoms and absolute error.

    forward_fn(batch) -> pred tensor of shape [batch.num_graphs]
    """
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred  = forward_fn(batch)
            for i in range(batch.num_graphs):
                n_atoms = int((batch.batch == i).sum())
                error   = float((pred[i] - batch.y[i]).abs())
                records.append({'n_atoms': n_atoms, 'error': error})
    df = pd.DataFrame(records)
    df['bin'] = pd.cut(df['n_atoms'], bins=[0, 15, 20, 25, 30, 100],
                       labels=['<=15', '16-20', '21-25', '26-30', '>30'])
    return df


# =============================================================================
# PART 1 – GINE baseline on ZINC-subset
# =============================================================================

print("\n" + "=" * 60)
print("PART 1 – GINE baseline on ZINC-subset")
print("=" * 60)

from torch_geometric.nn import GINEConv, global_add_pool

train_ds = ZINC(root='./data/ZINC', subset=True, split='train')
val_ds   = ZINC(root='./data/ZINC', subset=True, split='val')
test_ds  = ZINC(root='./data/ZINC', subset=True, split='test')
print(f"\nZINC-subset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
print(f"Sample: {train_ds[0]}")


class GINEBaseline(torch.nn.Module):
    def __init__(self, hidden_channels=64, num_layers=4,
                 num_atom_types=28, num_bond_types=4):
        super().__init__()
        self.atom_embedding = Embedding(num_atom_types, hidden_channels)
        self.bond_embedding = Embedding(num_bond_types, hidden_channels)

        self.convs = ModuleList()
        self.bns   = ModuleList()
        for _ in range(num_layers):
            nn_mlp = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINEConv(nn_mlp, train_eps=True))
            self.bns.append(BatchNorm1d(hidden_channels))

        self.head = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_embedding(x.squeeze(-1))
        e = self.bond_embedding(edge_attr)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index, e)
            h = bn(h)
            h = F.relu(h)
        h = global_add_pool(h, batch)
        return self.head(h).squeeze(-1)


train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=128)
test_loader  = DataLoader(test_ds,  batch_size=128)

gine      = GINEBaseline().to(device)
n_par_gine = sum(p.numel() for p in gine.parameters())
print(f"GINE parameters: {n_par_gine:,}")
# Same optimizer recipe used for GraphGPS below, so the two models train
# with matched budgets. Any accuracy gap comes from the architecture, not
# from a more careful training loop for one of them.
optim_g     = torch.optim.AdamW(gine.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=150)
criterion = torch.nn.L1Loss()


def evaluate_gine(loader):
    gine.eval()
    total_error = 0.0; total_n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred  = gine(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            total_error += (pred - batch.y).abs().sum().item()
            total_n     += batch.y.size(0)
    return total_error / total_n


print("\nTraining GINE baseline (150 epochs) …")
best_val_g, best_state_g, t0 = float('inf'), None, time.time()
for epoch in range(1, 151):
    gine.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optim_g.zero_grad()
        pred = gine(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(pred, batch.y)
        loss.backward(); optim_g.step()
        total_loss += float(loss) * batch.num_graphs
    scheduler_g.step()
    val_mae = evaluate_gine(val_loader)
    # Keep the weights that generalise best, rather than whatever the last
    # epoch happens to produce.
    if val_mae < best_val_g:
        best_val_g = val_mae
        best_state_g = {k: v.detach().clone()
                        for k, v in gine.state_dict().items()}
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:03d}/150 | "
              f"Train loss: {total_loss/len(train_ds):.4f} | Val MAE: {val_mae:.4f}")
sec_per_epoch_gine = (time.time() - t0) / 150
gine.load_state_dict(best_state_g)

gine_test_mae = evaluate_gine(test_loader)
print(f"\nGINE test MAE: {gine_test_mae:.4f} "
      f"(best val {best_val_g:.4f}) | "
      f"{sec_per_epoch_gine:.1f}s per epoch on {device}")

# Stratified error for Figure 10.3 (left bars)
df_gine = stratified_mae(
    gine, test_loader,
    lambda b: gine(b.x, b.edge_index, b.edge_attr, b.batch))
print("\nGINE test MAE stratified by molecule size:")
print(df_gine.groupby('bin', observed=True)['error'].agg(['mean', 'count']))


# =============================================================================
# PART 2 – GraphGPS on ZINC-subset with Laplacian positional encodings
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 – GraphGPS on ZINC-subset with Laplacian PEs")
print("=" * 60)

from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.nn         import GPSConv

# A separate root directory so the pre_transform cache is not confused
# with the plain ZINC used in Part 1.
pe_transform = AddLaplacianEigenvectorPE(k=8, attr_name='pe',
                                        is_undirected=True)

train_pe = ZINC(root='./data/ZINC_PE', subset=True, split='train',
                pre_transform=pe_transform)
val_pe   = ZINC(root='./data/ZINC_PE', subset=True, split='val',
                pre_transform=pe_transform)
test_pe  = ZINC(root='./data/ZINC_PE', subset=True, split='test',
                pre_transform=pe_transform)
print(f"\nZINC-subset with PEs: sample = {train_pe[0]}")


class GraphGPS(torch.nn.Module):
    def __init__(self, hidden_channels=96, num_layers=6, pe_dim=8,
                 heads=8, num_atom_types=28, num_bond_types=4):
        super().__init__()
        self.atom_embedding = Embedding(num_atom_types, hidden_channels)
        self.bond_embedding = Embedding(num_bond_types, hidden_channels)
        self.pe_lin         = Linear(pe_dim, hidden_channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            local_nn = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
            conv = GPSConv(hidden_channels,
                           conv=GINEConv(local_nn, train_eps=True),
                           heads=heads,
                           dropout=0.1,
                           attn_type='multihead')
            self.convs.append(conv)

        self.head = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, 1),
        )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        # Random sign flip on the positional encodings - training only.
        # Laplacian eigenvectors are sign-arbitrary. One sign per graph and
        # per eigenvector: sampling a single sign for the whole mini-batch
        # would give every graph in it the same flip, which is a much weaker
        # invariance signal.
        if self.training:
            num_graphs = int(batch.max().item()) + 1
            sign = torch.randint(0, 2, (num_graphs, pe.size(1)),
                                 device=pe.device) * 2 - 1
            pe = pe * sign[batch]

        h = self.atom_embedding(x.squeeze(-1)) + self.pe_lin(pe)
        e = self.bond_embedding(edge_attr)
        for conv in self.convs:
            h = conv(h, edge_index, batch, edge_attr=e)
        h = global_add_pool(h, batch)
        return self.head(h).squeeze(-1)


train_loader_pe = DataLoader(train_pe, batch_size=128, shuffle=True)
val_loader_pe   = DataLoader(val_pe,   batch_size=128)
test_loader_pe  = DataLoader(test_pe,  batch_size=128)

gps     = GraphGPS().to(device)
n_par_gps = sum(p.numel() for p in gps.parameters())
print(f"GraphGPS parameters: {n_par_gps:,} "
      f"({n_par_gps/n_par_gine:.1f}x the GINE baseline)")
# Weight decay + cosine annealing are what the GraphGPS paper uses for ZINC.
# Adam without a schedule and with zero decay leaves the attention layers
# oscillating for the entire run — the training does not converge cleanly
# and the model ends up worse than the GINE baseline.
optim_p     = torch.optim.AdamW(gps.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler_p = torch.optim.lr_scheduler.CosineAnnealingLR(optim_p, T_max=150)


def evaluate_gps(loader):
    gps.eval()
    total_error = 0.0; total_n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred  = gps(batch.x, batch.pe, batch.edge_index,
                        batch.edge_attr, batch.batch)
            total_error += (pred - batch.y).abs().sum().item()
            total_n     += batch.y.size(0)
    return total_error / total_n


print("\nTraining GraphGPS (150 epochs) …")
best_val_p, best_state_p, t0 = float('inf'), None, time.time()
for epoch in range(1, 151):
    gps.train()
    total_loss = 0.0
    for batch in train_loader_pe:
        batch = batch.to(device)
        optim_p.zero_grad()
        pred = gps(batch.x, batch.pe, batch.edge_index,
                   batch.edge_attr, batch.batch)
        loss = criterion(pred, batch.y)
        loss.backward(); optim_p.step()
        total_loss += float(loss) * batch.num_graphs
    scheduler_p.step()
    val_mae = evaluate_gps(val_loader_pe)
    if val_mae < best_val_p:
        best_val_p = val_mae
        best_state_p = {k: v.detach().clone()
                        for k, v in gps.state_dict().items()}
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:03d}/150 | "
              f"Train loss: {total_loss/len(train_pe):.4f} | Val MAE: {val_mae:.4f}")
sec_per_epoch_gps = (time.time() - t0) / 150
gps.load_state_dict(best_state_p)

gps_test_mae = evaluate_gps(test_loader_pe)
print(f"\nGraphGPS test MAE: {gps_test_mae:.4f} "
      f"(best val {best_val_p:.4f}) | "
      f"{sec_per_epoch_gps:.1f}s per epoch on {device}")

df_gps = stratified_mae(
    gps, test_loader_pe,
    lambda b: gps(b.x, b.pe, b.edge_index, b.edge_attr, b.batch))
print("\nGraphGPS test MAE stratified by molecule size:")
print(df_gps.groupby('bin', observed=True)['error'].agg(['mean', 'count']))


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("Summary — Chapter 10 results")
print("=" * 60)
print(f"  GINE test MAE:     {gine_test_mae:.4f}")
print(f"  GraphGPS test MAE: {gps_test_mae:.4f}")
print(f"  Reduction:         {(1 - gps_test_mae/gine_test_mae)*100:.1f}%")
print(f"  Parameters:        GINE {n_par_gine:,} vs GraphGPS {n_par_gps:,} "
      f"({n_par_gps/n_par_gine:.1f}x)")
print(f"  Seconds/epoch:     GINE {sec_per_epoch_gine:.1f} vs "
      f"GraphGPS {sec_per_epoch_gps:.1f} on {device}")
print("  The two models are not parameter-matched. See ablation.ipynb for a "
      "controlled comparison.")
print("\nTo regenerate Figure 10.3 with these real values, edit "
      "generate_figures.py and replace the `gine` / `gps` arrays with "
      "the stratified means printed above.")
print("\nDone.")