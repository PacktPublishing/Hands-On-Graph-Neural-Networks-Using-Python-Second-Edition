"""
Chapter 8 – Scaling Up Graph Neural Networks with GraphSAGE
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch-geometric scikit-learn numpy

Figure generation is in figures/generate_figures.py.

Fixes versus the first edition:
  - accuracy() uses .float().mean() instead of len() division
  - Typo 'import torchmport' corrected to two separate import lines
  - total_loss (not last-batch loss) used in epoch print
  - num_workers=0 as safe default in NeighborLoader
  - NLLLoss paired with the log_softmax output (was CrossEntropyLoss, which
    silently applied log_softmax twice and degraded training accuracy)
"""

import torch
import torch.nn.functional as F
import numpy as np

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# SHARED UTILITY
# =============================================================================

def accuracy(pred_y: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of correct predictions. Fixed: .float().mean()."""
    return (pred_y == y).float().mean().item()


# =============================================================================
# PART 1 – Transductive node classification on PubMed
# =============================================================================

print("\n" + "=" * 60)
print("PART 1 – GraphSAGE on PubMed (transductive)")
print("=" * 60)

from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

dataset = Planetoid(root='.', name='Pubmed')
data    = dataset[0]

print(f"\nNodes:           {data.x.shape[0]}")
print(f"Features:        {dataset.num_features}")
print(f"Classes:         {dataset.num_classes}")
print(f"Training nodes:  {sum(data.train_mask).item()}")
print(f"Test nodes:      {sum(data.test_mask).item()}")

# ── Neighbor sampling ─────────────────────────────────────────────────────────
# Sample 10 neighbours at each of 2 hops; batch the 60 training nodes in 16s.
# num_workers=0 is the safe default across platforms.
train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=16,
    input_nodes=data.train_mask,
    num_workers=0,
)

print("\nSubgraphs from NeighborLoader:")
for i, subgraph in enumerate(train_loader):
    print(f"  Subgraph {i}: {subgraph.num_nodes} nodes, "
          f"{subgraph.edge_index.shape[1]} edges, "
          f"batch_size={subgraph.batch_size}")

# The four subgraphs above are rendered by figures/generate_figures.py
# (fig8_5_subgraphs.png), so nothing else needs to happen here.


# ── GraphSAGE model ───────────────────────────────────────────────────────────

class GraphSAGE(torch.nn.Module):
    """
    Two-layer GraphSAGE with mean aggregation.
    SAGEConv selects the mean aggregator by default.
    """

    def __init__(self, dim_in: int, dim_h: int, dim_out: int):
        super().__init__()
        self.sage1 = SAGEConv(dim_in, dim_h)
        self.sage2 = SAGEConv(dim_h, dim_out)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        h = self.sage1(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage2(h, edge_index)
        return F.log_softmax(h, dim=1)

    def fit(self, data, epochs: int, verbose: bool = True):
        # The forward returns log_softmax already, so NLLLoss is the right
        # loss. Using CrossEntropyLoss here would apply log_softmax twice
        # and silently degrade training accuracy.
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.train()
        for epoch in range(epochs + 1):
            # Accumulators reset each epoch
            total_loss, val_loss, acc, val_acc = 0.0, 0.0, 0.0, 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                out   = self(batch.x, batch.edge_index)
                loss  = criterion(out[batch.train_mask],
                                  batch.y[batch.train_mask])
                # Accumulate total_loss (not the last batch's loss)
                total_loss += loss.item()
                acc        += accuracy(out[batch.train_mask].argmax(dim=1),
                                       batch.y[batch.train_mask])
                loss.backward()
                optimizer.step()
                val_loss += criterion(out[batch.val_mask],
                                      batch.y[batch.val_mask]).item()
                val_acc  += accuracy(out[batch.val_mask].argmax(dim=1),
                                     batch.y[batch.val_mask])

            if verbose and epoch % 20 == 0:
                n = len(train_loader)
                print(f"  Epoch {epoch:>3} | "
                      f"Train Loss: {total_loss/n:.3f} | "
                      f"Train Acc: {acc/n*100:>6.2f}% | "
                      f"Val Loss: {val_loss/n:.2f} | "
                      f"Val Acc: {val_acc/n*100:.2f}%")

    @torch.no_grad()
    def test(self, data) -> float:
        self.eval()
        out = self(data.x, data.edge_index)
        return accuracy(out.argmax(dim=1)[data.test_mask],
                        data.y[data.test_mask])


print("\nTraining GraphSAGE on PubMed …")
torch.manual_seed(SEED)
graphsage = GraphSAGE(dataset.num_features, 64, dataset.num_classes)
print(graphsage)
graphsage.fit(data, epochs=200)

acc = graphsage.test(data)
print(f"\nGraphSAGE test accuracy: {acc*100:.2f}%")
print("(Expected: ~74-76%  GCN: ~75.7%  GAT: ~76.1%)")


# =============================================================================
# PART 2 – Inductive multi-label classification on PPI
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 – GraphSAGE on PPI (inductive, multi-label)")
print("=" * 60)

from torch_geometric.datasets import PPI
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

# Load the three inductive splits — training/val/test are separate graphs
train_dataset = PPI(root='.', split='train')   # 20 graphs
val_dataset   = PPI(root='.', split='val')     # 2 graphs
test_dataset  = PPI(root='.', split='test')    # 2 graphs

print(f"\nTraining graphs:   {len(train_dataset)}")
print(f"Validation graphs: {len(val_dataset)}")
print(f"Test graphs:       {len(test_dataset)}")
print(f"Node features:     {train_dataset.num_features}")
print(f"Labels per node:   {train_dataset.num_classes}  (multi-label)")

# Merge training graphs into one object for NeighborLoader
train_data = Batch.from_data_list(train_dataset)
train_loader = NeighborLoader(
    train_data,
    batch_size=2048,
    shuffle=True,
    num_neighbors=[20, 10, 10],
    num_workers=0,   # safe default; increase on Linux if desired
)

# batch_size=2 captures all graphs in each val/test set in one batch
val_loader  = DataLoader(val_dataset,  batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=2)

# ── GraphSAGE model (explicit three-layer SAGEConv) ──────────────────────────
# Three layers matched by three-hop neighbour sampling above (num_neighbors
# has three entries). No dropout, no normalisation, raw logits for BCE.

class GraphSAGEPPI(torch.nn.Module):
    """
    Three-layer GraphSAGE for inductive multi-label classification.
    Outputs raw logits suitable for BCEWithLogitsLoss.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.sage3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.sage1(x, edge_index))
        h = F.relu(self.sage2(h, edge_index))
        return self.sage3(h, edge_index)


model = GraphSAGEPPI(
    in_channels=train_dataset.num_features,
    hidden_channels=512,
    out_channels=train_dataset.num_classes,
).to(device)

print(f"\nModel: {model}")

# Multi-label loss: BCEWithLogitsLoss (not CrossEntropy)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def fit() -> float:
    """Train one epoch over the PPI training set."""
    model.train()
    total_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader)


@torch.no_grad()
def test(loader) -> float:
    """
    Evaluate on the full validation or test set.
    Since each loader has exactly 2 graphs and batch_size=2,
    next(iter(loader)) already contains the full set.
    Predictions are thresholded at 0 to convert real outputs to binary labels.
    """
    model.eval()
    data  = next(iter(loader))
    out   = model(data.x.to(device), data.edge_index.to(device))
    preds = (out > 0).float().cpu()
    y, pred = data.y.numpy(), preds.numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0.0


print("\nTraining GraphSAGE on PPI (300 epochs) …")
for epoch in range(301):
    loss   = fit()
    val_f1 = test(val_loader)
    if epoch % 50 == 0:
        print(f"  Epoch {epoch:>3} | Train Loss: {loss:.3f} | Val F1: {val_f1:.4f}")

test_f1 = test(test_loader)
print(f"\nTest F1 score: {test_f1:.4f}")
print("(Expected: ~0.85–0.86 with hidden_channels=512)")
print("\nNote: the hidden channel size significantly affects performance.")
print("Reducing it to 128 or increasing to 1024 produces measurably")
print("different F1 scores — a useful parameter to tune.")
print("\nDone.")