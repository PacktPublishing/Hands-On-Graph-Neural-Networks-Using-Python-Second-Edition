"""
Chapter 17: Detecting Anomalies Using Heterogeneous GNNs

Downloads CIDDS-001, preprocesses the OpenStack week-1 traffic, builds
heterogeneous host/flow subgraphs, trains a HeteroConv GNN for five-class
flow classification, compares it with an MLP trained on the same flow
features, and saves:
    fig17_2_class_distribution.png
    fig17_3_distributions_before.png
    fig17_4_distributions_after.png
    fig17_6_misclassified.png
    fig17_7_confusion_matrix.png

Run: python run.py
"""

import os
import platform
import resource
import ssl
import sys
import time
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.nn import Linear, HeteroConv, SAGEConv

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

FONT = "DejaVu Sans"
plt.rcParams['font.family'] = FONT
G0, G1, G2, G3, G4, G5 = "#111111", "#333333", "#555555", "#777777", "#999999", "#BBBBBB"


def save(fig, name, dpi=200):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=dpi, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f'Saved figures/{name}')


def peak_memory_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == 'darwin' else rss / 2**10


# =============================================================================
# PART 1: Download and explore CIDDS-001
# =============================================================================

url = 'https://www.hs-coburg.de/wp-content/uploads/2024/11/CIDDS-001.zip'
if not os.path.isdir('CIDDS-001'):
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    print(f'Downloading {url} …')
    with urlopen(url, context=context) as zurl:
        with ZipFile(BytesIO(zurl.read())) as zfile:
            zfile.extractall('.')

df = pd.read_csv('CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv')
print(f'\nRows in CIDDS-001-internal-week1.csv: {len(df):,}')
print(f'Columns: {list(df.columns)}')
print(f"Time span: {df['Date first seen'].min()} to {df['Date first seen'].max()}")

addresses = pd.concat([df['Src IP Addr'], df['Dst IP Addr']]).astype(str)
internal = addresses.str.startswith('192.168.')
print(f'\nAddress fields starting with 192.168.: {internal.mean() * 100:.2f}%')
print('Most frequent addresses that do not start with 192.168.:')
print(addresses[~internal].value_counts().head(15).to_string())
print(f'Distinct addresses that do not start with 192.168.: '
      f'{addresses[~internal].nunique():,}')

df = df.drop(columns=['Src Pt', 'Dst Pt', 'Flows', 'Tos',
                      'class', 'attackID', 'attackDescription'])
df['attackType']      = df['attackType'].replace('---', 'benign')
df['Date first seen'] = pd.to_datetime(df['Date first seen'])

print('\nClass distribution:')
counts = df['attackType'].value_counts()
for name, n in counts.items():
    print(f'  {name:<11} {n:>10,}  {n / len(df) * 100:8.4f}%')

count_labels = counts / len(df) * 100
fig, ax = plt.subplots(figsize=(7, 6))
_, _, autotexts = ax.pie(count_labels[:3].values, labels=count_labels.index[:3],
                         colors=[G5, G2, G4], autopct='%.0f%%',
                         wedgeprops=dict(edgecolor='white', linewidth=1.5))
for at in autotexts:
    at.set_color('white')
ax.set_title('Class distribution (three largest classes)', color=G0)
save(fig, 'fig17_2_class_distribution.png')

bytes_raw = pd.to_numeric(df['Bytes'], errors='coerce')
print('\nRaw numerical features:')
print(pd.DataFrame({'Duration': df['Duration'], 'Packets': df['Packets'],
                    'Bytes': bytes_raw}).describe(
                        percentiles=[0.5, 0.9, 0.99]).to_string())
print(f"Flows with Duration == 0: {(df['Duration'] == 0).mean() * 100:.2f}%")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
df['Duration'].hist(ax=ax1, bins=50, range=(0, 2), color=G2)
ax1.set_xlabel('Duration'); ax1.set_ylabel('Number of flows')
df['Packets'].hist(ax=ax2, bins=20, range=(0, 20), color=G2)
ax2.set_xlabel('Packets per flow')
bytes_raw.hist(ax=ax3, bins=50, range=(0, 5000), color=G2)
ax3.set_xlabel('Bytes per flow')
fig.suptitle('Distributions before PowerTransformer', color=G0)
fig.tight_layout()
save(fig, 'fig17_3_distributions_before.png')


# =============================================================================
# PART 2: Preprocessing
# =============================================================================

df['weekday'] = df['Date first seen'].dt.weekday
df = pd.get_dummies(df, columns=['weekday']).rename(columns={
    'weekday_0': 'Monday',    'weekday_1': 'Tuesday',
    'weekday_2': 'Wednesday', 'weekday_3': 'Thursday',
    'weekday_4': 'Friday',    'weekday_5': 'Saturday',
    'weekday_6': 'Sunday'})

df['daytime'] = (
    df['Date first seen'].dt.second
    + df['Date first seen'].dt.minute * 60
    + df['Date first seen'].dt.hour   * 3600
) / 86400


def one_hot_flags(flags):
    flag_names = ['ACK', 'PSH', 'RST', 'SYN', 'FIN']
    return [1 if f in str(flags) else 0 for f in flag_names]


df = df.reset_index(drop=True)
df[['ACK', 'PSH', 'RST', 'SYN', 'FIN']] = pd.DataFrame(
    df['Flags'].apply(one_hot_flags).to_list(),
    columns=['ACK', 'PSH', 'RST', 'SYN', 'FIN'])


def binary_encode_ip(df, col_name, prefix):
    temp = df[col_name].astype(str).copy()
    temp[~temp.str.contains(r'\d{1,3}\.', regex=True)] = '0.0.0.0'
    parts = temp.str.split('.', expand=True)
    parts = parts.rename(columns={2: f'{prefix}3', 3: f'{prefix}4'})
    parts = parts[[f'{prefix}3', f'{prefix}4']].astype(int)
    parts[prefix] = (
        parts[f'{prefix}3'].apply(lambda x: format(x, 'b').zfill(8))
        + parts[f'{prefix}4'].apply(lambda x: format(x, 'b').zfill(8)))
    encoded = parts[prefix].str.split('', expand=True).drop(columns=[0, 17])
    encoded.columns = [f'{prefix}_{i}' for i in range(1, 17)]
    return encoded.astype('int32')


df = df.join(binary_encode_ip(df, 'Src IP Addr', 'ipsrc'))
df = df.join(binary_encode_ip(df, 'Dst IP Addr', 'ipdst'))

m_index = df[pd.to_numeric(df['Bytes'], errors='coerce').isnull()].index
print(f'\nBytes values written with the m suffix: {len(m_index):,}')
df.loc[m_index, 'Bytes'] = df.loc[m_index, 'Bytes'].apply(
    lambda x: 1e6 * float(x.strip().split()[0]))
df['Bytes'] = pd.to_numeric(df['Bytes'], errors='coerce', downcast='integer')

df = pd.get_dummies(df, prefix='', prefix_sep='',
                    columns=['Proto', 'attackType'])
labels = ['benign', 'bruteForce', 'dos', 'pingScan', 'portScan']

df_train, df_test = train_test_split(df, random_state=0, test_size=0.2,
                                     stratify=df[labels])
df_val, df_test   = train_test_split(df_test, random_state=0, test_size=0.5,
                                     stratify=df_test[labels])
print(f'Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}')

scaler = PowerTransformer()
cols = ['Duration', 'Packets', 'Bytes']
df_train[cols] = scaler.fit_transform(df_train[cols])
df_val[cols]   = scaler.transform(df_val[cols])
df_test[cols]  = scaler.transform(df_test[cols])
print('\nTransformed features on the training set:')
print(df_train[cols].describe().to_string())

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
df_train['Duration'].hist(ax=ax1, bins=50, color=G2)
ax1.set_xlabel('Duration (transformed)'); ax1.set_ylabel('Number of flows')
df_train['Packets'].hist(ax=ax2, bins=50, color=G2)
ax2.set_xlabel('Packets (transformed)')
df_train['Bytes'].hist(ax=ax3, bins=50, color=G2)
ax3.set_xlabel('Bytes (transformed)')
fig.suptitle('Distributions after PowerTransformer', color=G0)
fig.tight_layout()
save(fig, 'fig17_4_distributions_after.png')


# =============================================================================
# PART 3: Heterogeneous graph dataset
# =============================================================================

BATCH_SIZE = 16

features_flow = [f for f in ['daytime', 'Monday', 'Tuesday', 'Wednesday',
                             'Thursday', 'Friday', 'Duration', 'Packets',
                             'Bytes', 'ACK', 'PSH', 'RST', 'SYN', 'FIN',
                             'ICMP ', 'IGMP ', 'TCP  ', 'UDP  ']
                 if f in df.columns]
print(f'\nFlow features ({len(features_flow)}): {features_flow}')


def ip_to_bits(ip):
    parts = str(ip).split('.')
    try:
        third, fourth = int(parts[2]), int(parts[3])
    except (IndexError, ValueError):
        third, fourth = 0, 0
    bits = format(third, '08b') + format(fourth, '08b')
    return [float(b) for b in bits]


def get_connections(ip_map, src_ip, dst_ip):
    flow_idx = np.arange(len(src_ip))
    s = np.array([ip_map[ip] for ip in src_ip])
    d = np.array([ip_map[ip] for ip in dst_ip])
    sends    = torch.tensor(np.stack([s, flow_idx]), dtype=torch.long)
    receives = torch.tensor(np.stack([d, flow_idx]), dtype=torch.long)
    return sends, receives


def create_dataloader(df, subgraph_size=1024, shuffle=False):
    data = []
    n_subgraphs = len(df) // subgraph_size
    for i in range(1, n_subgraphs + 1):
        subgraph = df[(i - 1) * subgraph_size: i * subgraph_size]
        src_ip   = subgraph['Src IP Addr'].to_numpy()
        dst_ip   = subgraph['Dst IP Addr'].to_numpy()

        hosts  = np.unique(np.append(src_ip, dst_ip))
        ip_map = {ip: idx for idx, ip in enumerate(hosts)}
        sends, receives = get_connections(ip_map, src_ip, dst_ip)

        batch = HeteroData()
        batch['host'].x = torch.tensor(
            [ip_to_bits(ip) for ip in hosts], dtype=torch.float)
        batch['flow'].x = torch.tensor(
            subgraph[features_flow].to_numpy().astype(np.float32))
        batch['flow'].y = torch.tensor(
            subgraph[labels].to_numpy().astype(np.float32))

        batch['host', 'sends',       'flow'].edge_index = sends
        batch['host', 'receives',    'flow'].edge_index = receives
        batch['flow', 'sent_by',     'host'].edge_index = sends.flip(0)
        batch['flow', 'received_by', 'host'].edge_index = receives.flip(0)

        data.append(batch)
    return DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle)


start = time.perf_counter()
train_loader = create_dataloader(df_train, shuffle=True)
val_loader   = create_dataloader(df_val)
test_loader  = create_dataloader(df_test)
graph_time = time.perf_counter() - start

for name, loader, split in (('Train', train_loader, df_train),
                            ('Val', val_loader, df_val),
                            ('Test', test_loader, df_test)):
    graphs = loader.dataset
    hosts = np.array([g['host'].num_nodes for g in graphs])
    print(f'{name:<5} subgraphs: {len(graphs):>5,} | flows used: '
          f'{len(graphs) * 1024:>9,} of {len(split):,} | hosts per subgraph: '
          f'mean {hosts.mean():.1f}, min {hosts.min()}, max {hosts.max()}')
print(f'Graph construction time: {graph_time:.1f} s')


# =============================================================================
# PART 4: HeteroGNN
# =============================================================================

class HeteroGNN(torch.nn.Module):
    def __init__(self, dim_h, dim_out, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('host', 'sends',       'flow'): SAGEConv((-1, -1), dim_h),
                ('host', 'receives',    'flow'): SAGEConv((-1, -1), dim_h),
                ('flow', 'sent_by',     'host'): SAGEConv((-1, -1), dim_h),
                ('flow', 'received_by', 'host'): SAGEConv((-1, -1), dim_h),
            }, aggr='sum')
            self.convs.append(conv)
        self.lin = Linear(dim_h, dim_out)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict['flow'])


model     = HeteroGNN(dim_h=64, dim_out=5, num_layers=3).to(device)
optimizer = Adam(model.parameters(), lr=0.001)


@torch.no_grad()
def test(loader):
    model.eval()
    y_pred, y_true = [], []
    n_flows = total_loss = 0
    for batch in loader:
        batch.to(device)
        out  = model(batch.x_dict, batch.edge_index_dict)
        y    = batch['flow'].y.argmax(dim=1)
        loss = F.cross_entropy(out, y)
        y_pred.append(out.argmax(dim=1))
        y_true.append(y)
        n_flows    += y.size(0)
        total_loss += float(loss) * y.size(0)
    y_pred  = torch.cat(y_pred).cpu()
    y_true  = torch.cat(y_true).cpu()
    f1score = f1_score(y_true, y_pred, average='macro')
    return total_loss / n_flows, f1score, y_pred, y_true


print('\nTraining HeteroGNN …')
history = {}
start = time.perf_counter()
for epoch in range(101):
    model.train()
    n_flows = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch.to(device)
        out  = model(batch.x_dict, batch.edge_index_dict)
        y    = batch['flow'].y.argmax(dim=1)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        n_flows    += y.size(0)
        total_loss += float(loss) * y.size(0)
    if epoch % 10 == 0:
        val_loss, f1score, _, _ = test(val_loader)
        history[epoch] = f1score
        print(f'Epoch {epoch:>3} | Loss: {total_loss/n_flows:.4f} |'
              f' Val loss: {val_loss:.4f} |'
              f' Val F1: {f1score:.4f}')
gnn_time = time.perf_counter() - start
best = max(history, key=history.get)
print(f'Best validation F1: {history[best]:.4f} at epoch {best} '
      f'(last: {history[100]:.4f})')


# =============================================================================
# PART 5: Test evaluation
# =============================================================================

_, f1_gnn, y_pred, y_true = test(test_loader)
print('\nHeteroGNN on the test set:')
print(classification_report(y_true, y_pred, target_names=labels, digits=4))

cm = confusion_matrix(y_true, y_pred)
print('Confusion matrix (rows = true, columns = predicted):')
print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

df_pred = pd.DataFrame({'pred': y_pred.numpy(), 'true': y_true.numpy()})
miss = df_pred['true'][df_pred['pred'] != df_pred['true']].value_counts()
print('\nMisclassified flows per true class:')
for i, n in miss.items():
    print(f'  {labels[i]:<11} {n:>6}')

fig, ax = plt.subplots(figsize=(7, 6))
ax.pie(miss.values, labels=[labels[i] for i in miss.index],
       colors=[G5, G2, G4, G0, G3][:len(miss)], autopct='%.0f%%',
       wedgeprops=dict(edgecolor='white', linewidth=1.5))
ax.set_title('Proportion of each misclassified class', color=G0)
save(fig, 'fig17_6_misclassified.png')

cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_norm, cmap='Greys', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label='Normalized count')
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = cm_norm[i, j]
        ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                fontsize=9, color='white' if v > 0.5 else G0)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Heterogeneous GNN, CIDDS-001 test set (row-normalized)', color=G0)
fig.tight_layout()
save(fig, 'fig17_7_confusion_matrix.png')


# =============================================================================
# PART 6: MLP on flow features only, same flows, same epochs
# =============================================================================

def to_tensors(split, n_rows):
    part = split.iloc[:n_rows]
    x = torch.tensor(part[features_flow].to_numpy().astype(np.float32))
    y = torch.tensor(part[labels].to_numpy().astype(np.float32)).argmax(dim=1)
    return x.to(device), y.to(device)


x_tr, y_tr = to_tensors(df_train, len(train_loader.dataset) * 1024)
x_te, y_te = to_tensors(df_test,  len(test_loader.dataset) * 1024)

torch.manual_seed(0)
mlp = nn.Sequential(
    nn.Linear(len(features_flow), 64), nn.LeakyReLU(),
    nn.Linear(64, 64),                 nn.LeakyReLU(),
    nn.Linear(64, 5),
).to(device)
mlp_opt = Adam(mlp.parameters(), lr=0.001)

print('\nTraining MLP …')
step_size = BATCH_SIZE * 1024
start = time.perf_counter()
for epoch in range(101):
    mlp.train()
    perm = torch.randperm(x_tr.size(0), device=device)
    for i in range(0, x_tr.size(0), step_size):
        idx = perm[i:i + step_size]
        mlp_opt.zero_grad()
        loss = F.cross_entropy(mlp(x_tr[idx]), y_tr[idx])
        loss.backward()
        mlp_opt.step()
    if epoch % 20 == 0:
        print(f'Epoch {epoch:>3} | Loss: {float(loss):.4f}')
mlp_time = time.perf_counter() - start

mlp.eval()
with torch.no_grad():
    mlp_pred = mlp(x_te).argmax(dim=1).cpu()
y_te_cpu = y_te.cpu()
f1_mlp = f1_score(y_te_cpu, mlp_pred, average='macro')
print('\nMLP on the same test flows:')
print(classification_report(y_te_cpu, mlp_pred, target_names=labels, digits=4))
print('Confusion matrix (rows = true, columns = predicted):')
print(pd.DataFrame(confusion_matrix(y_te_cpu, mlp_pred),
                   index=labels, columns=labels).to_string())
print(f'Same test flows as the GNN: {bool((y_te_cpu == y_true).all())}')


# =============================================================================
# PART 7: Summary, timing, and environment
# =============================================================================

print(f'\nHeteroGNN macro F1: {f1_gnn:.4f}')
print(f'MLP       macro F1: {f1_mlp:.4f}')
print(f'Difference:         {f1_gnn - f1_mlp:+.4f}')

print(f'\nGraph construction:            {graph_time:.1f} s')
print(f'HeteroGNN training (101 epochs): {gnn_time:.1f} s')
print(f'MLP training (101 epochs):       {mlp_time:.1f} s')
print(f'Peak process memory:             {peak_memory_mb():,.0f} MB')
if device.type == 'cuda':
    print(f'Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:,.0f} MB')
print(f'Platform:  {platform.platform()}')
print(f'Processor: {platform.processor() or platform.machine()}')
print(f'Python {platform.python_version()}, PyTorch {torch.__version__}, '
      f'threads {torch.get_num_threads()}')
print('\nDone.')