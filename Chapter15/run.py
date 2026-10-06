"""
Chapter 15 - Explaining Graph Neural Networks
Hands-On Graph Neural Networks Using Python, Second Edition

PART 1  GIN on MUTAG, explained with GNNExplainer (edge mask only) at two
        values of the edge-size regularization coefficient (0.005, the PyG
        default, and 0.1). For each value: the strongest bonds of the first
        mutagenic test graph classified correctly, a per-graph table, the
        number of empty explanations, and fidelity+, fidelity- and edge
        sparsity on edge masks thresholded at 0.5, for all test graphs and by
        predicted class, over five explainer seeds.
PART 2  GCN on Amazon Photo (random 70/10/20 split), explained with integrated
        gradients through Captum for nodes 0 and 101.
PART 3  Figures derived from the runs above, saved to figures/:
        fig15_3_mutag_explanation.png  (edge_size = 0.005)
        fig15_4_amazon_node0.png
        fig15_5_amazon_node101.png

Figures 15.1 and 15.2 are conceptual diagrams produced by
figures/generate_figures.py.
"""

import os
import random
import statistics

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, Amazon
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (GINConv, GCNConv, global_add_pool,
                                to_captum_model, to_captum_input)
from torch_geometric.explain import Explainer, GNNExplainer, fidelity
from torch_geometric.utils import k_hop_subgraph

import captum
from captum.attr import IntegratedGradients

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


print(f'torch {torch.__version__} | '
      f'torch_geometric {torch_geometric.__version__} | '
      f'captum {captum.__version__}')


# =============================================================================
# PART 1 - GNNExplainer on MUTAG
# =============================================================================

print('\n' + '=' * 70)
print('PART 1 - GNNExplainer on MUTAG')
print('=' * 70)

set_seed(SEED)

dataset = TUDataset(root='.', name='MUTAG').shuffle()
train_dataset = dataset[:int(len(dataset)*0.8)]
val_dataset   = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
test_dataset  = dataset[int(len(dataset)*0.9):]

print(f'MUTAG: {len(dataset)} graphs | train {len(train_dataset)} | '
      f'val {len(val_dataset)} | test {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)


def accuracy(pred_y, y):
    return (pred_y == y).float().mean().item()


class GIN(torch.nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.conv1 = GINConv(Sequential(
            Linear(dataset.num_features, dim_h),
            BatchNorm1d(dim_h), ReLU(),
            Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(Sequential(
            Linear(dim_h, dim_h),
            BatchNorm1d(dim_h), ReLU(),
            Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(Sequential(
            Linear(dim_h, dim_h),
            BatchNorm1d(dim_h), ReLU(),
            Linear(dim_h, dim_h), ReLU()))
        self.lin1  = Linear(dim_h*3, dim_h*3)
        self.lin2  = Linear(dim_h*3, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        h  = torch.cat([global_add_pool(h, batch)
                        for h in [h1, h2, h3]], dim=1)
        h  = self.lin1(h).relu()
        h  = F.dropout(h, p=0.5, training=self.training)
        return F.log_softmax(self.lin2(h), dim=1)


@torch.no_grad()
def test(model, loader):
    criterion = torch.nn.NLLLoss()
    model.eval()
    loss = acc = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y).item() / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
    return loss, acc


def train(model, loader):
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 100
    for epoch in range(epochs + 1):
        model.train()
        total_loss = acc = 0
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item() / len(loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            val_loss, val_acc = test(model, val_loader)
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} | '
                  f'Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | '
                  f'Val Acc: {val_acc*100:.2f}%')
    return model


model = GIN(dim_h=32)
model = train(model, train_loader)
test_loss, test_acc = test(model, test_loader)
print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')

ATOMS = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
EDGE_SIZES = [0.005, 0.1]


def make_explainer(edge_size):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200, edge_size=edge_size),
        explanation_type='model',
        node_mask_type=None,
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='log_probs',
        ),
    )


explainers = {c: make_explainer(c) for c in EDGE_SIZES}


@torch.no_grad()
def predict(model, g):
    model.eval()
    b = torch.zeros(g.num_nodes, dtype=torch.long)
    return int(model(g.x, g.edge_index, b).argmax(dim=1))


def bond_scores(edge_index, edge_mask):
    scores = {}
    for (u, v), m in zip(edge_index.t().tolist(), edge_mask.tolist()):
        scores.setdefault((min(u, v), max(u, v)), []).append(m)
    return {bond: sum(m) / len(m) for bond, m in scores.items()}


idx = next(i for i, g in enumerate(test_dataset)
           if int(g.y) == 1 and predict(model, g) == 1)
graph = test_dataset[idx]
batch = torch.zeros(graph.num_nodes, dtype=torch.long)
atom_of = [ATOMS[i] for i in graph.x.argmax(dim=1).tolist()]

print(f'\nExplained graph: test graph {idx}, first mutagenic graph classified '
      f'correctly | {graph.num_nodes} atoms, {graph.num_edges // 2} bonds | '
      f'true class {int(graph.y)} | predicted class {predict(model, graph)}')

mutag_bonds = {}
for c, explainer in explainers.items():
    torch.manual_seed(SEED)
    explanation = explainer(x=graph.x, edge_index=graph.edge_index,
                            batch=batch)
    mutag_bonds[c] = bond_scores(graph.edge_index, explanation.edge_mask)
    kept = sum(m > 0.5 for m in mutag_bonds[c].values())
    print(f'\nedge_size={c}: {kept} of {len(mutag_bonds[c])} bonds above 0.5')
    print('Five strongest bonds (edge mask averaged over both directions):')
    for (u, v), score in sorted(mutag_bonds[c].items(),
                                key=lambda item: -item[1])[:5]:
        print(f'  {atom_of[u]}{u}-{atom_of[v]}{v}  {score:.4f}')

SEEDS = [0, 1, 2, 3, 4]
GROUPS = ['All', 'Predicted 1', 'Predicted 0']
metrics = {c: {group: {'Fidelity+': [], 'Fidelity-': [], 'Edge sparsity': []}
               for group in GROUPS} for c in EDGE_SIZES}
empty_counts = {c: [] for c in EDGE_SIZES}
rows = {c: [] for c in EDGE_SIZES}

for c, explainer in explainers.items():
    for seed in SEEDS:
        torch.manual_seed(seed)
        per_graph = []
        for i, g in enumerate(test_dataset):
            b = torch.zeros(g.num_nodes, dtype=torch.long)
            expl = explainer(x=g.x, edge_index=g.edge_index, batch=b)
            hard = expl.threshold(threshold_type='hard', value=0.5)
            fp, fm = fidelity(explainer, hard)
            pred = int(expl.target)
            kept = int(hard.edge_mask.sum())
            is_empty = bool((expl.edge_mask == 0).all())
            per_graph.append((pred, fp, fm, 1 - kept / g.num_edges, is_empty))
            if seed == SEEDS[0]:
                rows[c].append((i, int(g.y), pred, kept, g.num_edges, fp, fm,
                                is_empty))
        for group in GROUPS:
            selected = [r for r in per_graph
                        if group == 'All' or r[0] == int(group[-1])]
            metrics[c][group]['Fidelity+'].append(
                statistics.mean(r[1] for r in selected))
            metrics[c][group]['Fidelity-'].append(
                statistics.mean(r[2] for r in selected))
            metrics[c][group]['Edge sparsity'].append(
                statistics.mean(r[3] for r in selected))
        empty_counts[c].append(sum(r[4] for r in per_graph))

c0, c1 = EDGE_SIZES
print(f'\nPer-graph results, explainer seed {SEEDS[0]}, '
      f'edge mask thresholded at 0.5:')
print(f'                  edge_size={c0:<16} edge_size={c1}')
print('graph true pred    kept    fid+ fid-     kept    fid+ fid-  empty')
for r0, r1 in zip(rows[c0], rows[c1]):
    i, y, p = r0[:3]
    empty = ','.join(str(c) for c, r in ((c0, r0), (c1, r1)) if r[7])
    print(f'{i:>5} {y:>4} {p:>4}  {r0[3]:>3}/{r0[4]:<4} {r0[5]:>4.0f} '
          f'{r0[6]:>4.0f}  {r1[3]:>3}/{r1[4]:<4} {r1[5]:>4.0f} {r1[6]:>4.0f}  '
          f'{empty}')

for c in EDGE_SIZES:
    print(f'\nedge_size={c} | empty explanations per seed: {empty_counts[c]}')
    print(f'Metrics on {len(test_dataset)} test graphs, {len(SEEDS)} explainer '
          f'seeds, edge mask thresholded at 0.5 (mean, std across seeds):')
    for group in GROUPS:
        n = len([r for r in rows[c] if group == 'All' or r[2] == int(group[-1])])
        print(f'  {group} ({n} graphs)')
        for name, values in metrics[c][group].items():
            print(f'    {name + ":":<15}{statistics.mean(values):.4f}  '
                  f'({statistics.stdev(values):.4f})')


# =============================================================================
# PART 2 - Integrated gradients on Amazon Photo (Captum)
# =============================================================================

print('\n' + '=' * 70)
print('PART 2 - Integrated gradients on Amazon Photo')
print('=' * 70)

set_seed(SEED)

dataset = Amazon(root='.', name='Photo')
data = T.RandomNodeSplit(num_val=0.1, num_test=0.2)(dataset[0])

print(f'Amazon Photo: {data.num_nodes} nodes | {data.num_edges} edges | '
      f'{dataset.num_features} features | {dataset.num_classes} classes')
print(f'Split: train {int(data.train_mask.sum())} | '
      f'val {int(data.val_mask.sum())} | test {int(data.test_mask.sum())}')
values = torch.unique(data.x)
print(f'Feature values: {values.numel()} distinct, '
      f'min {values.min().item():g}, max {values.max().item():g}')


class GCN(torch.nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, dim_h)
        self.conv2 = GCNConv(dim_h, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        return F.log_softmax(self.conv2(h, edge_index), dim=1)


device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = GCN(64).to(device)
data      = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    return accuracy(out[mask].argmax(dim=1), data.y[mask])


for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0 or epoch == 199:
        print(f'Epoch {epoch:>3} | Train Loss: {loss.item():.3f} | '
              f'Val Acc: {evaluate(model, data, data.val_mask)*100:.2f}%')

print(f'Test accuracy: {evaluate(model, data, data.test_mask)*100:.2f}%')

model.eval()
with torch.no_grad():
    predictions = model(data.x, data.edge_index).argmax(dim=1).cpu()

labels     = data.y.cpu()
edge_index = data.edge_index.cpu()


def split_of(node_idx):
    for name, mask in [('train', data.train_mask), ('val', data.val_mask),
                       ('test', data.test_mask)]:
        if bool(mask[node_idx]):
            return name


def hop_distance(node_idx, other):
    if other == node_idx:
        return 0
    for hops in (1, 2):
        subset = k_hop_subgraph(node_idx, hops, edge_index)[0]
        if other in subset.tolist():
            return hops
    return '>2'


def explain_node(node_idx):
    captum_model = to_captum_model(model,
                                   mask_type='node_and_edge',
                                   output_idx=node_idx)
    ig = IntegratedGradients(captum_model)
    inputs, add_args = to_captum_input(data.x, data.edge_index,
                                       mask_type='node_and_edge')
    attr_node, attr_edge = ig.attribute(
        inputs,
        target=int(data.y[node_idx]),
        additional_forward_args=add_args,
        internal_batch_size=1,
    )
    attr_node = attr_node.squeeze(0).abs().sum(dim=1)
    attr_node = (attr_node / attr_node.max()).detach().cpu()
    attr_edge = attr_edge.squeeze(0).abs()
    attr_edge = (attr_edge / attr_edge.max()).detach().cpu()
    return attr_node, attr_edge


amazon_explanations = {}
for node_idx in (0, 101):
    attr_node, attr_edge = explain_node(node_idx)
    amazon_explanations[node_idx] = (attr_node, attr_edge)
    print(f'\nNode {node_idx}: class {int(labels[node_idx])} | '
          f'predicted {int(predictions[node_idx])} | {split_of(node_idx)} split')
    print('  Top 5 nodes by attribution:')
    print('  node    score   class   hops')
    scores, nodes = attr_node.topk(5)
    for n, s in zip(nodes.tolist(), scores.tolist()):
        print(f'  {n:<7} {s:.4f}  {int(labels[n]):<7} {hop_distance(node_idx, n)}')
    print('  Top 5 edges by attribution:')
    scores, edges = attr_edge.topk(5)
    for e, s in zip(edges.tolist(), scores.tolist()):
        u, v = edge_index[:, e].tolist()
        print(f'  {u:>5} -> {v:<5} {s:.4f}')


# =============================================================================
# PART 3 - Figures derived from the runs
# =============================================================================

print('\n' + '=' * 70)
print('PART 3 - Figures')
print('=' * 70)

FONT = 'DejaVu Sans'
G0 = '#111111'; G1 = '#333333'; G2 = '#555555'
G3 = '#777777'; G4 = '#999999'; G5 = '#BBBBBB'; G6 = '#DDDDDD'
SHADES = [G5, G4, G3, G2, G1]
plt.rcParams['font.family'] = FONT


def shade(value):
    return SHADES[min(int(value * len(SHADES)), len(SHADES) - 1)]


def save(fig, name, dpi=200):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  saved figures/{name}')


def plot_molecule(atoms, bonds, name):
    G = nx.Graph()
    G.add_nodes_from(range(len(atoms)))
    G.add_edges_from(bonds.keys())
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    for (u, v), m in bonds.items():
        (x0, y0), (x1, y1) = pos[u], pos[v]
        ax.plot([x0, x1], [y0, y1], color=G0 if m > 0.5 else G4,
                linewidth=0.8 + 4.0 * m,
                linestyle='-' if m > 0.5 else '--', zorder=1)
    for n, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), 0.06, color=G2, zorder=3))
        ax.text(x, y, atoms[n], ha='center', va='center', fontsize=10,
                color='white', fontweight='bold', zorder=4)
        bond_angles = [np.arctan2(pos[m][1] - y, pos[m][0] - x)
                       for m in G.neighbors(n)]
        candidates = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        angle = max(candidates, key=lambda t: min(
            abs((t - b + np.pi) % (2 * np.pi) - np.pi) for b in bond_angles))
        ax.text(x + 0.1 * np.cos(angle), y + 0.1 * np.sin(angle), str(n),
                ha='center', va='center', fontsize=7, color=G3, zorder=4)
    ax.legend(handles=[
        Line2D([0], [0], color=G0, linewidth=4.0, label='Edge mask > 0.5'),
        Line2D([0], [0], color=G4, linewidth=1.2, linestyle='--',
               label='Edge mask \u2264 0.5'),
    ], loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9,
       framealpha=0.95, edgecolor=G5)
    ax.set_aspect('equal')
    ax.autoscale_view()
    save(fig, name)


def plot_attribution(target, attr_node, attr_edge, labels, edge_index, name,
                     top_k=10):
    order = [n for n in attr_node.argsort(descending=True).tolist()
             if n != target][:top_k]
    one_hop = set(k_hop_subgraph(target, 1, edge_index)[0].tolist())
    nodes = {target, *order}
    for n in order:
        if n not in one_hop:
            neighbors = set(k_hop_subgraph(n, 1, edge_index)[0].tolist())
            bridges = (one_hop & neighbors) - {target, n}
            if bridges:
                nodes.add(max(bridges, key=lambda m: attr_node[m].item()))
    node_list = sorted(nodes)
    node_tensor = torch.tensor(node_list)
    inside = (torch.isin(edge_index[0], node_tensor)
              & torch.isin(edge_index[1], node_tensor))
    edges = {}
    for e in inside.nonzero().view(-1).tolist():
        u, v = edge_index[:, e].tolist()
        if u != v:
            key = (min(u, v), max(u, v))
            edges[key] = max(edges.get(key, 0.0), attr_edge[e].item())
    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edges.keys())
    pos = nx.spring_layout(G, seed=SEED, k=0.9)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    for (u, v), a in edges.items():
        (x0, y0), (x1, y1) = pos[u], pos[v]
        ax.plot([x0, x1], [y0, y1], color=shade(a),
                linewidth=0.5 + 3.5 * a, zorder=1)
    for n in node_list:
        x, y = pos[n]
        a = attr_node[n].item()
        ax.add_patch(plt.Circle((x, y), 0.045 + 0.04 * a,
                                color=G0 if n == target else shade(a),
                                zorder=3))
        if n == target:
            ax.add_patch(plt.Circle((x, y), 0.13, fill=False, edgecolor=G0,
                                    linewidth=2.0, zorder=4))
        ax.text(x, y + 0.11 + 0.04 * a, f'{n} ({int(labels[n])})',
                ha='center', fontsize=7.5, color=G1, zorder=5,
                bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
    ax.legend(handles=[
        Patch(color=G0, label=f'Target node {target} '
                              f'(class {int(labels[target])})'),
        Patch(color=G1, label='High attribution'),
        Patch(color=G5, label='Low attribution'),
        Line2D([0], [0], color=G1, linewidth=3.5,
               label='High edge attribution'),
    ], loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8.5,
       framealpha=0.95, edgecolor=G5)
    ax.set_aspect('equal')
    ax.autoscale_view()
    save(fig, name)


plot_molecule(atom_of, mutag_bonds[EDGE_SIZES[0]],
              'fig15_3_mutag_explanation.png')
plot_attribution(0, *amazon_explanations[0], labels, edge_index,
                 'fig15_4_amazon_node0.png')
plot_attribution(101, *amazon_explanations[101], labels, edge_index,
                 'fig15_5_amazon_node101.png')

print('\nDone.')