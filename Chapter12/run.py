"""
Chapter 12: Predicting Links with Graph Neural Networks

Splits the Cora edges once into training, validation, and test sets, then
trains a VGAE and a SEAL model (DGCNN) on that same split, evaluates both
on the same test edges, and saves:
    fig12_4_comparison.png

Run: python run.py
"""

import platform
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, Linear, Dropout, BCEWithLogitsLoss
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse.csgraph import shortest_path
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, VGAE, aggr
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix

np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

FONT = "DejaVu Sans"
G0 = "#111111"
G1 = "#333333"
G6 = "#DDDDDD"


# =============================================================================
# PART 1: Dataset and shared link split
# =============================================================================

transform = T.Compose([
    T.NormalizeFeatures(),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1,
                      is_undirected=True, split_labels=True),
])
dataset = Planetoid('.', name='Cora', transform=transform)
train_data, val_data, test_data = dataset[0]

for name, d in (('Train', train_data), ('Val', val_data), ('Test', test_data)):
    print(f'{name:<5} positive edges: {d.pos_edge_label_index.size(1):>5} | '
          f'negative edges: {d.neg_edge_label_index.size(1):>5}')


# =============================================================================
# PART 2: VGAE
# =============================================================================

class Encoder(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv1       = GCNConv(dim_in, 2 * dim_out)
        self.conv_mu     = GCNConv(2 * dim_out, dim_out)
        self.conv_logstd = GCNConv(2 * dim_out, dim_out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


train_data, val_data, test_data = [d.to(device) for d in (train_data, val_data, test_data)]

model     = VGAE(Encoder(dataset.num_features, 16)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z    = model.encode(train_data.x, train_data.edge_index)
    loss = (model.recon_loss(z, train_data.pos_edge_label_index)
            + (1 / train_data.num_nodes) * model.kl_loss())
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


print('\nTraining VGAE …')
start = time.perf_counter()
for epoch in range(301):
    loss = train()
    val_auc, val_ap = test(val_data)
    if epoch % 50 == 0:
        print(f'Epoch {epoch:>3} | Loss: {loss:.4f} |'
              f' Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}')
vgae_time = time.perf_counter() - start

test_auc_vgae, test_ap_vgae = test(test_data)
print(f'Test AUC: {test_auc_vgae:.4f} | Test AP: {test_ap_vgae:.4f}')

with torch.no_grad():
    z    = model.encode(test_data.x, test_data.edge_index)
    Ahat = torch.sigmoid(z @ z.T)
print(Ahat)


# =============================================================================
# PART 3: SEAL preprocessing (enclosing subgraphs and DRNL labels)
# =============================================================================

train_data, val_data, test_data = [d.cpu() for d in (train_data, val_data, test_data)]


def seal_processing(dataset, edge_label_index, y):
    data_list = []
    for src, dst in edge_label_index.t().tolist():
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src, dst], 2, dataset.edge_index, relabel_nodes=True)
        src, dst = mapping.tolist()

        mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
        mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
        sub_edge_index = sub_edge_index[:, mask1 & mask2]

        src, dst = (dst, src) if src > dst else (src, dst)
        adj = to_scipy_sparse_matrix(
            sub_edge_index, num_nodes=sub_nodes.size(0)).tocsr()

        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]
        d_src = shortest_path(adj_wo_dst, directed=False,
                              unweighted=True, indices=src)
        d_src = np.insert(d_src, dst, 0, axis=0)
        d_src = torch.from_numpy(d_src)

        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]
        d_dst = shortest_path(adj_wo_src, directed=False,
                              unweighted=True, indices=dst - 1)
        d_dst = np.insert(d_dst, src, 0, axis=0)
        d_dst = torch.from_numpy(d_dst)

        dist = d_src + d_dst
        z = 1 + torch.min(d_src, d_dst) + dist // 2 * (dist // 2 + dist % 2 - 1)
        z[src], z[dst], z[torch.isnan(z)] = 1., 1., 0.
        z = z.to(torch.long)

        node_labels = F.one_hot(z, num_classes=200).to(torch.float)
        node_emb    = dataset.x[sub_nodes]
        node_x      = torch.cat([node_emb, node_labels], dim=1)

        data_list.append(Data(x=node_x, z=z,
                              edge_index=sub_edge_index, y=y))
    return data_list


print('\nExtracting enclosing subgraphs …')
start = time.perf_counter()
train_pos = seal_processing(train_data, train_data.pos_edge_label_index, 1)
train_neg = seal_processing(train_data, train_data.neg_edge_label_index, 0)
val_pos   = seal_processing(val_data,   val_data.pos_edge_label_index,   1)
val_neg   = seal_processing(val_data,   val_data.neg_edge_label_index,   0)
test_pos  = seal_processing(test_data,  test_data.pos_edge_label_index,  1)
test_neg  = seal_processing(test_data,  test_data.neg_edge_label_index,  0)
extraction_time = time.perf_counter() - start

train_dataset = train_pos + train_neg
val_dataset   = val_pos   + val_neg
test_dataset  = test_pos  + test_neg

all_graphs = train_dataset + val_dataset + test_dataset
print(f'Subgraphs: train {len(train_dataset)}, val {len(val_dataset)}, '
      f'test {len(test_dataset)}, total {len(all_graphs)}')
print(f'Largest DRNL label: {max(int(g.z.max()) for g in all_graphs)}')
print(f'Average subgraph size: {np.mean([g.num_nodes for g in all_graphs]):.1f} nodes')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32)
test_loader  = DataLoader(test_dataset,  batch_size=32)


# =============================================================================
# PART 4: DGCNN
# =============================================================================

class DGCNN(torch.nn.Module):
    def __init__(self, dim_in, k=30):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, 32)
        self.gcn2 = GCNConv(32, 32)
        self.gcn3 = GCNConv(32, 32)
        self.gcn4 = GCNConv(32, 1)
        self.global_pool = aggr.SortAggregation(k=k)
        self.conv1   = Conv1d(1, 16, 97, 97)
        self.conv2   = Conv1d(16, 32, 5, 1)
        self.maxpool = MaxPool1d(2, 2)
        length = (k - 2) // 2 + 1 - 5 + 1
        self.linear1 = Linear(32 * length, 128)
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
        return self.linear2(h)


model     = DGCNN(train_dataset[0].num_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = BCEWithLogitsLoss()
print(f'\nInput size of linear1: {model.linear1.in_features}')


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index, data.batch)
        loss = criterion(out.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    y_pred, y_true = [], []
    for data in loader:
        data = data.to(device)
        out  = model(data.x, data.edge_index, data.batch)
        y_pred.append(out.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    return (roc_auc_score(torch.cat(y_true), torch.cat(y_pred)),
            average_precision_score(torch.cat(y_true), torch.cat(y_pred)))


# =============================================================================
# PART 5: Train and test SEAL
# =============================================================================

print('\nTraining DGCNN …')
history = []
start = time.perf_counter()
for epoch in range(31):
    loss = train()
    val_auc, val_ap = test(val_loader)
    history.append(val_auc)
    print(f'Epoch {epoch:>2} | Loss: {loss:.4f} |'
          f' Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}')
seal_time = time.perf_counter() - start

test_auc_seal, test_ap_seal = test(test_loader)
print(f'Test AUC: {test_auc_seal:.4f} | Test AP: {test_ap_seal:.4f}')

best = int(np.argmax(history))
print(f'\nBest SEAL validation AUC: {history[best]:.4f} at epoch {best} '
      f'(last epoch: {history[-1]:.4f})')


# =============================================================================
# PART 6: Comparison, timing, and environment
# =============================================================================

print(f'\n{"Model":<6} {"Test AUC":>9} {"Test AP":>9}')
print(f'{"VGAE":<6} {test_auc_vgae:>9.4f} {test_ap_vgae:>9.4f}')
print(f'{"SEAL":<6} {test_auc_seal:>9.4f} {test_ap_seal:>9.4f}')

print(f'\nVGAE training (301 epochs):  {vgae_time:.1f} s')
print(f'SEAL subgraph extraction:    {extraction_time:.1f} s')
print(f'SEAL training (31 epochs):   {seal_time:.1f} s')
print(f'Platform:  {platform.platform()}')
print(f'Processor: {platform.processor() or platform.machine()}')
print(f'Python {platform.python_version()}, PyTorch {torch.__version__}, '
      f'threads {torch.get_num_threads()}')


# =============================================================================
# PART 7: Figure 12.4, results table
# =============================================================================

def fmt_time(seconds):
    return f'{seconds:.0f} s' if seconds < 120 else f'{seconds / 60:.1f} min'


rows = [
    ['VGAE', 'Node embeddings',
     f'{test_auc_vgae:.4f}', f'{test_ap_vgae:.4f}',
     fmt_time(vgae_time), '–'],
    ['SEAL', 'Enclosing subgraphs',
     f'{test_auc_seal:.4f}', f'{test_ap_seal:.4f}',
     fmt_time(seal_time), fmt_time(extraction_time)],
]
cols = ['Model', 'Approach', 'Test AUC', 'Test AP',
        'Training time', 'Preprocessing time']

fig, ax = plt.subplots(figsize=(12, 1.9))
fig.patch.set_facecolor('white'); ax.axis('off')
table = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor('white')
    cell.get_text().set_fontfamily(FONT)
    if r == 0:
        cell.set_facecolor(G1)
        cell.get_text().set_color('white')
        cell.get_text().set_fontweight('bold')
    elif c == 0:
        cell.set_facecolor('white')
        cell.get_text().set_color(G0)
        cell.get_text().set_fontweight('bold')
    else:
        cell.set_facecolor(G6)
        cell.get_text().set_color(G0)
fig.tight_layout()
fig.savefig('fig12_4_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('\nSaved fig12_4_comparison.png')
print('\nDone.')