"""Chapter 14 – Temporal Graph Neural Networks.

Part 1  EvolveGCN-H and EvolveGCN-O on WikiMaths (torch-geometric-temporal)
Part 2  MPNN-LSTM on England Covid (torch-geometric-temporal)
Part 3  TGN on the JODIE Wikipedia dataset (PyTorch Geometric)

Figures produced, all from the real datasets and trained models:
    fig14_5_wikimaths_graph.png    fig14_6_wikimaths_ts.png
    fig14_7_wikimaths_pred.png     fig14_8_wikimaths_scatter.png
    fig14_9_england_graph.png      fig14_11_covid_ts.png
    fig14_12_covid_pred.png        fig14_13_covid_scatter.png
    fig14_15_tgn_loss.png

Conceptual diagrams are produced by generate_figures.py.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

G0 = "#111111"; G1 = "#333333"; G2 = "#555555"; G3 = "#777777"
G4 = "#999999"; G5 = "#BBBBBB"; G6 = "#DDDDDD"
plt.rcParams['font.family'] = 'DejaVu Sans'
GREYS = LinearSegmentedColormap.from_list('greys_book', [G6, G0])


def save(fig, name):
    fig.savefig(name, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved {name}")


def plot_graph(edge_index, num_nodes, title, cbar_label, name, top_k=None):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from((int(u), int(v)) for u, v in edge_index.t().tolist() if u != v)
    if top_k is not None:
        keep = sorted(G.degree, key=lambda d: d[1], reverse=True)[:top_k]
        G = G.subgraph([n for n, _ in keep]).copy()
    deg = dict(G.degree())
    nodes = list(G.nodes())
    values = np.array([deg[n] for n in nodes])
    isolates = list(nx.isolates(G))
    pos = nx.spring_layout(G.subgraph([n for n in G if n not in isolates]), seed=SEED)
    radius = 1.1 * max(np.linalg.norm(p) for p in pos.values())
    for k, n in enumerate(isolates):
        angle = np.pi / 4 + 2 * np.pi * k / len(isolates)
        pos[n] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axis('off')
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color=G4, width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes, node_color=values,
                           cmap=GREYS, vmin=0, vmax=values.max(),
                           node_size=20 + 200 * values / values.max(),
                           edgecolors=G5, linewidths=0.4)
    sm = plt.cm.ScalarMappable(cmap=GREYS, norm=plt.Normalize(0, values.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, label=cbar_label)
    ax.set_title(title, fontsize=12, fontweight='bold', color=G0)
    save(fig, name)


def plot_series(mean, std, split, xlabel, ylabel, title, name,
                rolling=None, pred=None, pred_label=None):
    x = np.arange(len(mean))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, mean, color=G2, linewidth=0.8, label='Mean')
    ax.fill_between(x, mean - std, mean + std, color=G5, alpha=0.4, label='±1 std dev')
    if rolling is not None:
        ax.plot(x, rolling, color=G0, linewidth=2.0, label='7-day moving average')
    if pred is not None:
        ax.plot(np.arange(split, split + len(pred)), pred, color=G1,
                linewidth=2.0, linestyle='--', label=pred_label)
    ax.axvline(x=split, color=G3, linestyle=':', linewidth=1.5, label='Train/test split')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', color=G0)
    ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.13),
              ncol=5, frameon=False)
    ax.grid(linestyle=':', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, name)


def plot_scatter(y_true, y_pred, xlabel, title, name):
    fit = np.polyfit(y_true, y_pred, 1)
    x_line = np.linspace(y_true.min(), y_true.max(), 100)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, color=G1, alpha=0.5, s=18, edgecolors='none')
    ax.plot(x_line, np.polyval(fit, x_line), color=G0, linewidth=2.0, label='Regression line')
    ax.plot(x_line, x_line, color=G4, linewidth=1.2, linestyle='--', label='Perfect prediction')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Predicted value', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold', color=G0)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, name)


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
split = train_dataset.snapshot_count

print(f"\nSnapshots: {dataset.snapshot_count} "
      f"(train {train_dataset.snapshot_count}, test {test_dataset.snapshot_count})")
print(dataset[0])
print(dataset[500])

plot_graph(dataset[0].edge_index, dataset[0].x.shape[0],
           'WikiMaths: the 150 most connected articles',
           'Node degree (links within the subgraph)',
           'fig14_5_wikimaths_graph.png', top_k=150)

mean_visits = [snapshot.y.mean().item() for snapshot in dataset]
std_visits = [snapshot.y.std().item() for snapshot in dataset]

df = pd.DataFrame({'mean': mean_visits, 'std': std_visits})
df['rolling'] = df['mean'].rolling(7).mean()

plot_series(df['mean'].values, df['std'].values, split,
            'Snapshot (day)', 'Normalized number of visits',
            'WikiMaths: mean normalized number of visits',
            'fig14_6_wikimaths_ts.png', rolling=df['rolling'].values)


class TemporalGNN_H(torch.nn.Module):
    def __init__(self, node_count, dim_in):
        super().__init__()
        self.recurrent = EvolveGCNH(node_count, dim_in)
        self.linear = torch.nn.Linear(dim_in, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight).relu()
        return self.linear(h)


class TemporalGNN_O(torch.nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.recurrent = EvolveGCNO(dim_in)
        self.linear = torch.nn.Linear(dim_in, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight).relu()
        return self.linear(h)


def train_evolvegcn(model, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(epochs):
        model.recurrent.reinitialize_weight()
        for snapshot in train_dataset:
            optimizer.zero_grad()
            y_pred = model(snapshot.x, snapshot.edge_index,
                           snapshot.edge_attr).squeeze()
            loss = torch.mean((y_pred - snapshot.y) ** 2)
            loss.backward()
            optimizer.step()
            model.recurrent.weight = model.recurrent.weight.detach()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"last snapshot loss: {loss.item():.4f}")
    return model.recurrent.weight


@torch.no_grad()
def evaluate_evolvegcn(model, weight_after_training):
    model.eval()
    model.recurrent.weight = weight_after_training
    loss, mean_preds, first_pred = 0, [], None
    for i, snapshot in enumerate(test_dataset):
        y_pred = model(snapshot.x, snapshot.edge_index,
                       snapshot.edge_attr).squeeze()
        loss += torch.mean((y_pred - snapshot.y) ** 2)
        mean_preds.append(y_pred.mean().item())
        if first_pred is None:
            first_pred = (snapshot.y.numpy(), y_pred.numpy())
    return (loss / (i + 1)).item(), np.array(mean_preds), first_pred


torch.manual_seed(0)
model_h = TemporalGNN_H(dataset[0].x.shape[0], dataset[0].x.shape[1])
print(model_h)

weight_h = train_evolvegcn(model_h)
mse_h, preds_h, (y_true_h, y_pred_h) = evaluate_evolvegcn(model_h, weight_h)
print(f"EvolveGCN-H test MSE: {mse_h:.4f}")

plot_series(df['mean'].values, df['std'].values, split,
            'Snapshot (day)', 'Normalized number of visits',
            'WikiMaths: EvolveGCN-H predicted mean normalized visits',
            'fig14_7_wikimaths_pred.png', rolling=df['rolling'].values,
            pred=preds_h, pred_label='EvolveGCN-H prediction')

plot_scatter(y_true_h, y_pred_h, 'Ground truth (normalized visits)',
             'WikiMaths: predicted vs ground truth (first test snapshot)',
             'fig14_8_wikimaths_scatter.png')

torch.manual_seed(0)
model_o = TemporalGNN_O(dataset[0].x.shape[1])
print(model_o)
weight_o = train_evolvegcn(model_o)
mse_o, _, _ = evaluate_evolvegcn(model_o, weight_o)
print(f"EvolveGCN-O test MSE: {mse_o:.4f}")


# =============================================================================
# PART 2 – MPNN-LSTM on England Covid (dynamic graph, temporal signal)
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 – MPNN-LSTM on England Covid")
print("=" * 60)

from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader
from torch_geometric_temporal.nn.recurrent import MPNNLSTM

dataset_cov = EnglandCovidDatasetLoader().get_dataset(lags=14)
train_cov, test_cov = temporal_signal_split(dataset_cov, train_ratio=0.8)
split_cov = train_cov.snapshot_count

print(f"\nSnapshots: {dataset_cov.snapshot_count} "
      f"(train {train_cov.snapshot_count}, test {test_cov.snapshot_count})")
print(dataset_cov[0])

plot_graph(dataset_cov[0].edge_index, dataset_cov[0].x.shape[0],
           'England Covid: 129 NUTS 3 regions and mobility edges (first snapshot)',
           'Node degree (connected regions)',
           'fig14_9_england_graph.png')

mean_cov = np.array([s.y.mean().item() for s in dataset_cov])
std_cov = np.array([s.y.std().item() for s in dataset_cov])

plot_series(mean_cov, std_cov, split_cov,
            'Snapshot (day)', 'Normalized number of cases',
            'England Covid: mean normalized number of reported cases',
            'fig14_11_covid_ts.png')


class TemporalGNN_Covid(torch.nn.Module):
    def __init__(self, dim_in, dim_h, num_nodes):
        super().__init__()
        self.recurrent = MPNNLSTM(dim_in, dim_h, num_nodes, 1, 0.5)
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(2 * dim_h + dim_in, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight).relu()
        h = self.dropout(h)
        return self.linear(h)


torch.manual_seed(0)
model_cov = TemporalGNN_Covid(dataset_cov[0].x.shape[1], 64,
                              dataset_cov[0].x.shape[0])
print(model_cov)

optimizer = torch.optim.Adam(model_cov.parameters(), lr=0.001)
model_cov.train()
for epoch in range(100):
    optimizer.zero_grad()
    loss = 0
    for i, snapshot in enumerate(train_cov):
        y_pred = model_cov(snapshot.x, snapshot.edge_index,
                           snapshot.edge_attr).squeeze()
        loss = loss + torch.mean((y_pred - snapshot.y) ** 2)
    loss = loss / (i + 1)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/100 | Train MSE: {loss.item():.4f}")

model_cov.eval()
loss_cov, preds_cov, first_cov = 0, [], None
with torch.no_grad():
    for i, snapshot in enumerate(test_cov):
        y_pred = model_cov(snapshot.x, snapshot.edge_index,
                           snapshot.edge_attr).squeeze()
        loss_cov += torch.mean((y_pred - snapshot.y) ** 2)
        preds_cov.append(y_pred.mean().item())
        if first_cov is None:
            first_cov = (snapshot.y.numpy(), y_pred.numpy())
mse_cov = (loss_cov / (i + 1)).item()
print(f"MPNN-LSTM test MSE: {mse_cov:.4f}")

plot_series(mean_cov, std_cov, split_cov,
            'Snapshot (day)', 'Normalized number of cases',
            'England Covid: MPNN-LSTM predicted mean normalized cases',
            'fig14_12_covid_pred.png',
            pred=np.array(preds_cov), pred_label='MPNN-LSTM prediction')

plot_scatter(first_cov[0], first_cov[1], 'Ground truth (normalized cases)',
             'England Covid: predicted vs ground truth (first test snapshot)',
             'fig14_13_covid_scatter.png')


# =============================================================================
# PART 3 – TGN on the JODIE Wikipedia dataset (continuous time)
# =============================================================================

print("\n" + "=" * 60)
print("PART 3 – TGN on Wikipedia (continuous-time link prediction)")
print("=" * 60)

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage, LastAggregator, LastNeighborLoader)

torch.manual_seed(0)

data = JODIEDataset('data/JODIE', name='wikipedia')[0].to(device)
print(f"Wikipedia: {data.num_nodes} nodes, {data.num_events} events, "
      f"edge feature dim {data.msg.size(-1)}")

train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15)
train_loader = TemporalDataLoader(train_data, batch_size=200,
                                  neg_sampling_ratio=1.0)
val_loader = TemporalDataLoader(val_data, batch_size=200,
                                neg_sampling_ratio=1.0)
test_loader = TemporalDataLoader(test_data, batch_size=200,
                                 neg_sampling_ratio=1.0)
neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)

memory_dim = time_dim = embedding_dim = 100

memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = torch.nn.Linear(in_channels, in_channels)
        self.lin_dst = torch.nn.Linear(in_channels, in_channels)
        self.lin_out = torch.nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        return self.lin_out(h.relu())


gnn = GraphAttentionEmbedding(memory_dim, embedding_dim, data.msg.size(-1),
                              memory.time_enc).to(device)
link_pred = LinkPredictor(embedding_dim).to(device)

params = dict.fromkeys(
    list(memory.parameters()) + list(gnn.parameters())
    + list(link_pred.parameters()))
optimizer = torch.optim.Adam(list(params), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def score_batch(batch):
    n_id, edge_index, e_id = neighbor_loader(batch.n_id)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)
    z, last_update = memory(n_id)
    z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])
    pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
    neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])
    loss = (criterion(pos_out, torch.ones_like(pos_out)) +
            criterion(neg_out, torch.zeros_like(neg_out)))
    return pos_out, neg_out, loss


def train_tgn():
    memory.train(); gnn.train(); link_pred.train()
    memory.reset_state()
    neighbor_loader.reset_state()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        _, _, loss = score_batch(batch)
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += loss.item() * batch.num_events
    return total_loss / train_data.num_events


@torch.no_grad()
def test_tgn(loader, num_events):
    memory.eval(); gnn.eval(); link_pred.eval()
    torch.manual_seed(12345)
    aps, total_loss = [], 0
    for batch in loader:
        batch = batch.to(device)
        pos_out, neg_out, loss = score_batch(batch)
        y_pred = torch.cat([pos_out, neg_out]).sigmoid().cpu()
        y_true = torch.cat([torch.ones(pos_out.size(0)),
                            torch.zeros(neg_out.size(0))])
        aps.append(average_precision_score(y_true, y_pred))
        total_loss += loss.item() * batch.num_events
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
    return float(torch.tensor(aps).mean()), total_loss / num_events


train_losses, val_losses = [], []
for epoch in range(1, 51):
    loss = train_tgn()
    val_ap, val_loss = test_tgn(val_loader, val_data.num_events)
    train_losses.append(loss)
    val_losses.append(val_loss)
    if epoch % 10 == 0:
        test_ap, _ = test_tgn(test_loader, test_data.num_events)
        print(f"Epoch {epoch:>2} | Loss: {loss:.4f} | "
              f"Val AP: {val_ap:.4f} | Test AP: {test_ap:.4f}")

print(f"Final test AP: {test_ap:.4f}")

epochs = np.arange(1, len(train_losses) + 1)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, train_losses, color=G0, linewidth=2.0, label='Train loss')
ax.plot(epochs, val_losses, color=G3, linewidth=2.0, linestyle='--', label='Validation loss')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Binary cross-entropy loss', fontsize=11)
ax.set_title('TGN training on the Wikipedia interaction dataset',
             fontsize=12, fontweight='bold', color=G0)
ax.legend(fontsize=10)
ax.grid(linestyle=':', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
save(fig, 'fig14_15_tgn_loss.png')

print("\nDone.")