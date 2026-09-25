"""
Chapter 17 - Detecting Anomalies Using Heterogeneous GNNs
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch-geometric pandas scikit-learn numpy matplotlib

Fixes versus the first edition (ex Chapter 16):
  - n_batches -> n_subgraphs in create_dataloader (NameError fix)
  - df['Bytes'].loc[m_index] -> df.loc[m_index, 'Bytes'] (pandas >= 2.0)
  - one_hot_flags takes a single flag string instead of an array
  - GitHub URL updated to Chapter17
  - Confusion matrix code included (was GitHub-only in first edition)
  - All pie chart colors converted to grayscale

Graph construction fixes (technical review, Mursel):
  1. HOST FEATURES. Both editions assigned batch['host'].x =
     subgraph[features_host], i.e. one row per FLOW (1024 rows), while host
     node indices come from ip_map over UNIQUE IPs (~150 on CIDDS-001).
     The two numberings are unrelated, so host node i received the features
     of flow i, and roughly 870 feature rows were never referenced by any
     edge. Host features are now built per unique host, from that host's own
     address, and are 16-dimensional rather than 32: ipsrc_* + ipdst_*
     describes an ordered pair of endpoints, which is a property of a flow,
     not of a host.
  2. DIRECTION. Source and destination host indices were interleaved into a
     single ('host', 'to', 'flow') relation, so the model could not tell who
     initiated a connection. They are now separate relations, which is the
     signal that distinguishes portScan and dos.
  3. edge_index tensors are int64, as PyG expects.

Two additions, both controlled by the flags below:
  - RUN_MLP_BASELINE trains an MLP on flow features alone, on the same
    splits for the same number of epochs. It answers the question the
    chapter currently cannot: how much does the graph actually contribute?
  - USE_CLASS_WEIGHTS enables class-weighted cross-entropy, which the
    chapter mentions without implementing.

NOTE: this file has not been executed. Run it before trusting any number.
"""

import numpy as np
import pandas as pd
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

torch.manual_seed(0); np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

RUN_MLP_BASELINE  = True
USE_CLASS_WEIGHTS = False


# =============================================================================
# PART 1 - Download and explore CIDDS-001
# =============================================================================

print("\n" + "=" * 60)
print("PART 1 - Exploring CIDDS-001")
print("=" * 60)

import os
import ssl
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

url = 'https://www.hs-coburg.de/wp-content/uploads/2024/11/CIDDS-001.zip'

# The dataset is large, so skip the download when it is already on disk.
if os.path.isdir('CIDDS-001'):
    print("CIDDS-001 already extracted, skipping download.")
else:
    # Python on macOS does not use the system trust store, so urlopen fails
    # with CERTIFICATE_VERIFY_FAILED unless it is given a CA bundle. certifi
    # ships one; falling back to the default context covers the platforms
    # where that is unnecessary.
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()

    print(f"Downloading {url} ...")
    with urlopen(url, context=context) as zurl:
        with ZipFile(BytesIO(zurl.read())) as zfile:
            zfile.extractall('.')

df = pd.read_csv(
    'CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv')
print(f"Dataset: {len(df):,} rows, {len(df.columns)} columns")

df = df.drop(columns=['Src Pt', 'Dst Pt', 'Flows', 'Tos',
                       'class', 'attackID', 'attackDescription'])
df['attackType']      = df['attackType'].replace('---', 'benign')
df['Date first seen'] = pd.to_datetime(df['Date first seen'])

count_labels = df['attackType'].value_counts() / len(df) * 100
fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
wedges, texts, autotexts = ax.pie(
    count_labels[:3].values,
    labels=count_labels.index[:3],
    colors=['#BBBBBB', '#555555', '#999999'],
    autopct='%.0f%%',
    wedgeprops=dict(edgecolor='white', linewidth=1.5))
for at in autotexts:
    at.set_color('white')
ax.set_title('Class distribution (top 3)')
plt.tight_layout()
plt.savefig('class_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved class_dist.png")

# Distributions of the three numerical features, before any scaling.
# The ranges are clipped on purpose: without them the long tails stretch
# each axis so far that every value collapses into the first bar.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5), dpi=150)
df['Duration'].hist(ax=ax1, bins=50, range=(0, 2), color='#555555')
ax1.set_xlabel('Duration'); ax1.set_ylabel('Count')
df['Packets'].hist(ax=ax2, bins=20, range=(0, 20), color='#555555')
ax2.set_xlabel('Packets')
pd.to_numeric(df['Bytes'], errors='coerce').hist(ax=ax3, bins=50,
                                                  range=(0, 5000),
                                                  color='#555555')
ax3.set_xlabel('Bytes')
fig.suptitle('Distributions before PowerTransformer')
plt.tight_layout()
plt.savefig('dist_before.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved dist_before.png")


# =============================================================================
# PART 2 - Preprocessing
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 - Preprocessing")
print("=" * 60)

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
    """One TCP flag string in, five binary features out."""
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
print(f"Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

scaler = PowerTransformer()
for split in [df_train, df_val, df_test]:
    cols = ['Duration', 'Packets', 'Bytes']
    if split is df_train:
        split[cols] = scaler.fit_transform(split[cols])
    else:
        split[cols] = scaler.transform(split[cols])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5), dpi=150)
df_train['Duration'].hist(ax=ax1, bins=50, color='#555555')
ax1.set_xlabel('Duration (transformed)'); ax1.set_ylabel('Count')
df_train['Packets'].hist(ax=ax2, bins=50, color='#555555')
ax2.set_xlabel('Packets (transformed)')
df_train['Bytes'].hist(ax=ax3, bins=50, color='#555555')
ax3.set_xlabel('Bytes (transformed)')
fig.suptitle('Distributions after PowerTransformer')
plt.tight_layout()
plt.savefig('dist_after.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved dist_after.png")


# =============================================================================
# PART 3 - Build heterogeneous graph dataset
# =============================================================================

print("\n" + "=" * 60)
print("PART 3 - Building heterogeneous graph")
print("=" * 60)

BATCH_SIZE = 16

features_flow_candidates = [
    'daytime', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
    'Duration', 'Packets', 'Bytes', 'ACK', 'PSH', 'RST', 'SYN', 'FIN',
    'ICMP ', 'IGMP ', 'TCP  ', 'UDP  '
]
features_flow = [f for f in features_flow_candidates if f in df.columns]


def ip_to_bits(ip):
    """
    Sixteen binary features from the last two octets of one address.
    This is a property of a single host, unlike the ipsrc_*/ipdst_* columns,
    which together describe the two endpoints of a flow.
    """
    parts = str(ip).split('.')
    try:
        third, fourth = int(parts[2]), int(parts[3])
    except (IndexError, ValueError):
        third, fourth = 0, 0
    bits = format(third, '08b') + format(fourth, '08b')
    return [float(b) for b in bits]


def get_connections(ip_map, src_ip, dst_ip):
    """
    Two directed relations instead of one. Keeping source and destination
    apart is what lets the model see which host initiated a flow.
    """
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
        # One row per unique host, built from that host's own address.
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


train_loader = create_dataloader(df_train, shuffle=True)
val_loader   = create_dataloader(df_val)
test_loader  = create_dataloader(df_test)

sample = next(iter(train_loader))
print("DataLoaders created.")
print(f"  host nodes in first batch: {sample['host'].num_nodes}")
print(f"  flow nodes in first batch: {sample['flow'].num_nodes}")
print(f"  host feature dim:          {sample['host'].x.size(1)}")
print(f"  flow feature dim:          {sample['flow'].x.size(1)}")


# =============================================================================
# PART 4 - HeteroGNN model
# =============================================================================

print("\n" + "=" * 60)
print("PART 4 - Training HeteroGNN")
print("=" * 60)


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

# Inverse-frequency class weights, computed on the training split.
if USE_CLASS_WEIGHTS:
    counts  = df_train[labels].sum().to_numpy().astype(np.float32)
    weights = torch.tensor(counts.sum() / (len(labels) * counts),
                           dtype=torch.float, device=device)
    print(f"Class weights: {[round(w, 2) for w in weights.tolist()]}")
else:
    weights = None


def loss_fn(out, y_onehot):
    """y is one-hot; argmax gives the class indices that weighting needs."""
    return F.cross_entropy(out, y_onehot.argmax(dim=1), weight=weights)


@torch.no_grad()
def test(loader):
    model.eval()
    y_pred, y_true = [], []
    n_flows = total_loss = 0
    for batch in loader:
        batch.to(device)
        out  = model(batch.x_dict, batch.edge_index_dict)
        y    = batch['flow'].y
        loss = loss_fn(out, y)
        y_pred.append(out.argmax(dim=1))
        y_true.append(y.argmax(dim=1))
        n_flows    += y.size(0)
        total_loss += float(loss) * y.size(0)
    y_pred  = torch.cat(y_pred).cpu()
    y_true  = torch.cat(y_true).cpu()
    f1score = f1_score(y_true, y_pred, average='macro')
    return total_loss / n_flows, f1score, y_pred, y_true


print("Training (101 epochs) ...")
for epoch in range(101):
    model.train()
    n_flows = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch.to(device)
        out  = model(batch.x_dict, batch.edge_index_dict)
        y    = batch['flow'].y
        loss = loss_fn(out, y)
        loss.backward(); optimizer.step()
        n_flows    += y.size(0)
        total_loss += float(loss) * y.size(0)
    if epoch % 10 == 0:
        val_loss, f1score, _, _ = test(val_loader)
        print(f"  Epoch {epoch:>3} | Loss: {total_loss/n_flows:.4f} | "
              f"Val loss: {val_loss:.4f} | Val F1: {f1score:.4f}")


# =============================================================================
# PART 5 - Evaluation
# =============================================================================

print("\n" + "=" * 60)
print("PART 5 - Test evaluation")
print("=" * 60)

_, f1_gnn, y_pred, y_true = test(test_loader)
print(classification_report(y_true, y_pred, target_names=labels, digits=4))

df_pred = pd.DataFrame({'pred': y_pred.numpy(), 'true': y_true.numpy()})
miss    = df_pred['true'][df_pred['pred'] != df_pred['true']].value_counts()

if len(miss) > 0:
    miss_labels = [labels[i] for i in miss.index]
    gray_colors = ['#BBBBBB', '#555555', '#999999', '#111111', '#777777']
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.pie(miss.values, labels=miss_labels,
           colors=gray_colors[:len(miss)],
           autopct='%.0f%%',
           wedgeprops=dict(edgecolor='white', linewidth=1.5))
    ax.set_title('Proportion of each misclassified class')
    plt.tight_layout()
    plt.savefig('misclassified.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved misclassified.png")

cm      = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
im = ax.imshow(cm_norm, cmap='Greys', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label='Normalised count')
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = cm_norm[i, j]
        ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                fontsize=9, color='white' if v > 0.5 else '#111111')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Heterogeneous GNN - CIDDS-001 test set (row-normalised)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved confusion_matrix.png")


# =============================================================================
# PART 6 - MLP baseline on flow features only
# =============================================================================

if RUN_MLP_BASELINE:
    print("\n" + "=" * 60)
    print("PART 6 - MLP baseline (flow features only, no graph)")
    print("=" * 60)

    def to_tensors(split):
        x = torch.tensor(split[features_flow].to_numpy().astype(np.float32))
        y = torch.tensor(
            split[labels].to_numpy().astype(np.float32)).argmax(dim=1)
        return x.to(device), y.to(device)

    x_tr, y_tr = to_tensors(df_train)
    x_te, y_te = to_tensors(df_test)

    mlp = nn.Sequential(
        nn.Linear(len(features_flow), 64), nn.LeakyReLU(),
        nn.Linear(64, 64),                 nn.LeakyReLU(),
        nn.Linear(64, 5),
    ).to(device)
    mlp_opt = Adam(mlp.parameters(), lr=0.001)

    n_rows   = x_tr.size(0)
    step_size = BATCH_SIZE * 1024      # same flows per step as the GNN
    print("Training MLP (101 epochs) ...")
    for epoch in range(101):
        mlp.train()
        perm = torch.randperm(n_rows, device=device)
        for i in range(0, n_rows, step_size):
            idx = perm[i:i + step_size]
            mlp_opt.zero_grad()
            loss = F.cross_entropy(mlp(x_tr[idx]), y_tr[idx], weight=weights)
            loss.backward(); mlp_opt.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:>3} | Loss: {float(loss):.4f}")

    mlp.eval()
    with torch.no_grad():
        mlp_pred = mlp(x_te).argmax(dim=1).cpu()
    y_te_cpu = y_te.cpu()
    f1_mlp   = f1_score(y_te_cpu, mlp_pred, average='macro')
    print(classification_report(y_te_cpu, mlp_pred,
                                target_names=labels, digits=4))

    print("-" * 52)
    print(f"HeteroGNN macro F1: {f1_gnn:.4f}")
    print(f"MLP       macro F1: {f1_mlp:.4f}")
    print(f"Difference:         {f1_gnn - f1_mlp:+.4f}")
    print("-" * 52)
    print("A difference near zero means the graph structure is not")
    print("contributing, and the framing of the chapter has to change.")

print("\nDone.")