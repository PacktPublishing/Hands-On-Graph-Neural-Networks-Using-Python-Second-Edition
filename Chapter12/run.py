"""
Chapter 12 - Predicting Links with Graph Neural Networks
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch-geometric scikit-learn scipy numpy

Fixes versus the first edition:
  - Comment added explaining dst-1 index arithmetic in seal_processing
  - Warning added about SEAL preprocessing time
  - Final comparison table added

Fixes applied in this revision (technical review):
  - DGCNN.forward returns raw logits. It previously returned
    .sigmoid(), while training used BCEWithLogitsLoss, which applies
    sigmoid internally. The activation was therefore applied twice and
    the gradients were flattened. THIS CHANGES SEAL'S RESULTS.
  - The Linear input size is computed from k instead of being
    hardcoded to 352, so changing k no longer requires manual
    recomputation. With k=30 the computed value is 352, i.e. identical
    to the previous hardcoded one.
  - Total GCN dimension (97) is derived from the layer sizes instead of
    being repeated as a literal.

Note on metrics: AUC and AP are invariant under any monotonic transform,
so computing them on raw logits gives exactly the same values as
computing them on sigmoid probabilities. No sigmoid is needed at test
time.
"""

import numpy as np
import torch
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# PART 1 - Variational Graph Autoencoder (VGAE)
# =============================================================================

print("\n" + "=" * 60)
print("PART 1 - VGAE on Cora (link prediction)")
print("=" * 60)

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, VGAE

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(
        num_val=0.05, num_test=0.1,
        is_undirected=True, split_labels=True,
        # False: VGAE performs its own negative sampling internally
        add_negative_train_samples=False,
    ),
])

dataset = Planetoid('.', name='Cora', transform=transform)
train_data, val_data, test_data = dataset[0]


class Encoder(torch.nn.Module):
    """
    Three-layer GCN encoder for the VGAE.
    conv1 is shared; conv_mu and conv_logstd produce mean and log-std.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.conv1       = GCNConv(dim_in, 2 * dim_out)
        self.conv_mu     = GCNConv(2 * dim_out, dim_out)
        self.conv_logstd = GCNConv(2 * dim_out, dim_out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


model     = VGAE(Encoder(dataset.num_features, 16)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train_vgae() -> float:
    model.train()
    optimizer.zero_grad()
    z    = model.encode(train_data.x, train_data.edge_index)
    loss = (model.recon_loss(z, train_data.pos_edge_label_index)
            + (1 / train_data.num_nodes) * model.kl_loss())
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test_vgae(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


print("\nTraining VGAE (301 epochs) ...")
for epoch in range(301):
    loss = train_vgae()
    val_auc, val_ap = test_vgae(val_data)
    if epoch % 50 == 0:
        print(f"  Epoch {epoch:>3} | Loss: {loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

test_auc_vgae, test_ap_vgae = test_vgae(test_data)
print(f"\nVGAE Test AUC: {test_auc_vgae:.4f} | Test AP: {test_ap_vgae:.4f}")

# Inspect the approximated adjacency matrix
z    = model.encode(test_data.x, test_data.edge_index)
Ahat = torch.sigmoid(z @ z.T)
print(f"\nApproximated adjacency matrix shape: {Ahat.shape}")
print(f"Sample values (first 3x3):")
print(Ahat[:3, :3].detach().cpu())


# =============================================================================
# PART 2 - SEAL framework
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 - SEAL on Cora (link prediction)")
print("=" * 60)
print("NOTE: seal_processing extracts thousands of enclosing subgraphs;\n"
      "      this preprocessing step dominates SEAL's overall runtime.")

from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse.csgraph import shortest_path
from torch.nn import Conv1d, MaxPool1d, Linear, Dropout, BCEWithLogitsLoss
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import aggr
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix

transform_seal = RandomLinkSplit(
    num_val=0.05, num_test=0.1,
    is_undirected=True, split_labels=True,
)
dataset_seal = Planetoid('.', name='Cora', transform=transform_seal)
train_data_s, val_data_s, test_data_s = dataset_seal[0]


def seal_processing(dataset, edge_label_index, y):
    """
    Extract 2-hop enclosing subgraphs and apply DRNL node labeling.
    A fixed DRNL_MAX_CLASSES is used so that all subgraphs produce node
    features of the same width, which the DataLoader needs for batching.
    """
    data_list = []
    for src, dst in edge_label_index.t().tolist():
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src, dst], 2, dataset.edge_index, relabel_nodes=True)
        src, dst = mapping.tolist()

        # Remove the target edge before computing shortest paths
        mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
        mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
        sub_edge_index = sub_edge_index[:, mask1 & mask2]

        # Ensure src < dst for consistent index arithmetic below
        src, dst = (dst, src) if src > dst else (src, dst)

        adj = to_scipy_sparse_matrix(
            sub_edge_index, num_nodes=sub_nodes.size(0)).tocsr()

        # Distance to src (adjacency without dst row/col)
        idx         = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst  = adj[idx, :][:, idx]
        d_src       = shortest_path(adj_wo_dst, directed=False,
                                     unweighted=True, indices=src)
        d_src       = np.insert(d_src, dst, 0, axis=0)
        d_src       = torch.from_numpy(d_src)

        # Distance to dst (adjacency without src row/col)
        # dst-1 is correct: since src < dst, removing src shifts dst's index by -1
        idx         = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src  = adj[idx, :][:, idx]
        d_dst       = shortest_path(adj_wo_src, directed=False,
                                     unweighted=True, indices=dst - 1)
        d_dst       = np.insert(d_dst, src, 0, axis=0)
        d_dst       = torch.from_numpy(d_dst)

        # DRNL labels
        dist        = d_src + d_dst
        z           = 1 + torch.min(d_src, d_dst) + dist//2 * (dist//2 + dist%2 - 1)
        z[src], z[dst], z[torch.isnan(z)] = 1., 1., 0.
        z           = z.to(torch.long)

        # DRNL one-hot encoding: a fixed upper bound is required because
        # subgraphs must have identical feature dimensions for the DataLoader
        # to batch them. 200 is a generous bound: on Cora the actual max
        # DRNL label rarely exceeds double digits, but using z.max() per
        # subgraph would produce variable-width x and break torch.cat at
        # batching time.
        DRNL_MAX_CLASSES = 200
        node_labels = F.one_hot(z, num_classes=DRNL_MAX_CLASSES).to(torch.float)
        node_emb    = dataset.x[sub_nodes]
        node_x      = torch.cat([node_emb, node_labels], dim=1)

        data_list.append(Data(x=node_x, z=z, edge_index=sub_edge_index, y=y))
    return data_list


print("\nExtracting enclosing subgraphs (this may take a while) ...")
train_dataset = (seal_processing(train_data_s, train_data_s.pos_edge_label_index, 1) +
                 seal_processing(train_data_s, train_data_s.neg_edge_label_index, 0))
val_dataset   = (seal_processing(val_data_s,   val_data_s.pos_edge_label_index,   1) +
                 seal_processing(val_data_s,   val_data_s.neg_edge_label_index,   0))
test_dataset  = (seal_processing(test_data_s,  test_data_s.pos_edge_label_index,  1) +
                 seal_processing(test_data_s,  test_data_s.neg_edge_label_index,  0))

print(f"Subgraphs - train: {len(train_dataset)}, "
      f"val: {len(val_dataset)}, test: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32)
test_loader  = DataLoader(test_dataset,  batch_size=32)


class DGCNN(torch.nn.Module):
    """
    Deep Graph Convolutional Neural Network for SEAL.

    Architecture notes:
    - Four GCN layers produce embeddings of dim 32+32+32+1 = 97 per node.
    - SortAggregation (k) selects and orders the k most informative nodes.
    - Conv1d kernel size equals the total GCN dimension, with the same
      stride, so each node embedding is read exactly once.
    - The Linear input size is derived from k, so changing k needs no
      manual recomputation. With k=30 it evaluates to 352.
    - forward returns raw logits. BCEWithLogitsLoss applies the sigmoid
      internally; returning probabilities here would apply it twice.
    """

    def __init__(self, dim_in: int, k: int = 30):
        super().__init__()
        gcn_dims = (32, 32, 32, 1)
        self.gcn1 = GCNConv(dim_in, gcn_dims[0])
        self.gcn2 = GCNConv(gcn_dims[0], gcn_dims[1])
        self.gcn3 = GCNConv(gcn_dims[1], gcn_dims[2])
        self.gcn4 = GCNConv(gcn_dims[2], gcn_dims[3])

        total_dim = sum(gcn_dims)                 # 97
        self.global_pool = aggr.SortAggregation(k=k)
        self.conv1   = Conv1d(1, 16, total_dim, total_dim)
        conv2_out, conv2_kernel = 32, 5
        self.conv2   = Conv1d(16, conv2_out, conv2_kernel, 1)
        self.maxpool = MaxPool1d(2, 2)

        # Length of the sequence as it flows through the 1D stack:
        #   after conv1  -> k          (one position per pooled node)
        #   after maxpool-> (k-2)//2+1
        #   after conv2  -> that minus (kernel-1)
        length = k
        length = (length - 2) // 2 + 1
        length = length - conv2_kernel + 1
        dense_dim = conv2_out * length            # 352 when k=30

        self.linear1 = Linear(dense_dim, 128)
        self.dropout = Dropout(0.5)
        self.linear2 = Linear(128, 1)

    def forward(self, x, edge_index, batch):
        h1 = self.gcn1(x, edge_index).tanh()
        h2 = self.gcn2(h1, edge_index).tanh()
        h3 = self.gcn3(h2, edge_index).tanh()
        h4 = self.gcn4(h3, edge_index).tanh()
        h  = torch.cat([h1, h2, h3, h4], dim=-1)
        h  = self.global_pool(h, batch)
        h  = h.view(h.size(0), 1, h.size(-1))
        h  = self.conv1(h).relu()
        h  = self.maxpool(h)
        h  = self.conv2(h).relu()
        h  = h.view(h.size(0), -1)
        h  = self.linear1(h).relu()
        h  = self.dropout(h)
        # Raw logits: BCEWithLogitsLoss applies the sigmoid internally
        return self.linear2(h)


model_seal = DGCNN(train_dataset[0].num_features).to(device)
optimizer  = torch.optim.Adam(model_seal.parameters(), lr=0.0001)
criterion  = BCEWithLogitsLoss()


def train_seal() -> float:
    model_seal.train()
    total_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out  = model_seal(data.x, data.edge_index, data.batch)
        loss = criterion(out.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_dataset)


@torch.no_grad()
def test_seal(loader):
    """
    AUC and AP are rank-based, so raw logits give the same values as
    sigmoid probabilities. No activation is applied here.
    """
    model_seal.eval()
    y_pred, y_true = [], []
    for data in loader:
        data = data.to(device)
        out  = model_seal(data.x, data.edge_index, data.batch)
        y_pred.append(out.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    return (roc_auc_score(torch.cat(y_true), torch.cat(y_pred)),
            average_precision_score(torch.cat(y_true), torch.cat(y_pred)))


print("\nTraining DGCNN (31 epochs) ...")
for epoch in range(31):
    loss = train_seal()
    val_auc, val_ap = test_seal(val_loader)
    print(f"  Epoch {epoch:>2} | Loss: {loss:.4f} | "
          f"Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

test_auc_seal, test_ap_seal = test_seal(test_loader)
print(f"\nSEAL Test AUC: {test_auc_seal:.4f} | Test AP: {test_ap_seal:.4f}")


# =============================================================================
# COMPARISON TABLE
# =============================================================================

print("\n" + "=" * 60)
print("COMPARISON: VGAE vs SEAL")
print("=" * 60)
print(f"{'Model':<10} {'Test AUC':<12} {'Test AP'}")
print("-" * 35)
print(f"{'VGAE':<10} {test_auc_vgae:.4f}       {test_ap_vgae:.4f}")
print(f"{'SEAL':<10} {test_auc_seal:.4f}       {test_ap_seal:.4f}")
print("-" * 35)
print("\nDone.")