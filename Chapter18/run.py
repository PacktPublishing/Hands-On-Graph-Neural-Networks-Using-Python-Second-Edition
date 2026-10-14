"""
Chapter 18 - Building a Recommender System Using LightGCN
Hands-On Graph Neural Networks Using Python, Second Edition

Downloads the Book-Crossing dataset and explores it, builds the user-book
bipartite graph from ratings >= 8 (random 100,000-row subset), trains a
four-layer LightGCN with BPR loss, evaluates Recall@20 and NDCG@20 on the
validation and test edges, and prints recommendations for user 277427.
It also prints the size of the graph, the peak memory of the run, the most
rated books, and how popular and how varied the recommendations are.

Figures saved in figures/ (200 DPI):
    fig18_1_rating_distribution.png
    fig18_2_book_rating_counts.png
    fig18_3_user_rating_counts.png
    fig18_4_bookcrossing_subgraph.png
    fig18_7_training_metrics.png

Install the packages listed in requirements.txt before running.
"""

import html
import os
import ssl
import sys
import time
from collections import Counter
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn, optim
from sklearn.model_selection import train_test_split
from torch_geometric.nn import LGConv

try:
    import resource
except ImportError:
    resource = None

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

plt.rcParams['font.family'] = 'DejaVu Sans'
G0 = "#111111"; G1 = "#333333"; G2 = "#555555"
G3 = "#777777"; G4 = "#999999"; G5 = "#BBBBBB"; G6 = "#DDDDDD"


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"Saved figures/{name}")


def peak_rss_gb():
    if resource is None:
        return float('nan')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == 'darwin' else rss / 1e6


# =============================================================================
# PART 1 - Download and explore
# =============================================================================

print("=" * 60)
print("PART 1 - Loading the Book-Crossing dataset")
print("=" * 60)

mirror = ('https://raw.githubusercontent.com/XuefengHuang/'
          'RecommendationSystem/master/datasets/BX-CSV-Dump')
csv_files = ['BX-Book-Ratings.csv', 'BX-Users.csv', 'BX-Books.csv']

if all(os.path.exists(name) for name in csv_files):
    print("CSV files already present, skipping download.")
else:
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    try:
        for name in csv_files:
            with urlopen(f"{mirror}/{name}", timeout=30, context=ctx) as resp:
                with open(name, 'wb') as f:
                    f.write(resp.read())
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        raise SystemExit(1)

ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1')
users = pd.read_csv('BX-Users.csv', sep=';', encoding='latin-1')
books = pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1',
                    low_memory=False, on_bad_lines='skip')

print(f"\nRatings: {len(ratings):,} rows, columns {list(ratings.columns)}")
print(f"Users:   {len(users):,} rows")
print(f"Books:   {len(books):,} rows")
print(f"Unique User-IDs in ratings: {ratings['User-ID'].nunique():,}")
print(f"Unique ISBNs in ratings:    {ratings['ISBN'].nunique():,}")

n_zero = int((ratings['Book-Rating'] == 0).sum())
print(f"Rating 0 (implicit): {n_zero:,} "
      f"({100 * n_zero / len(ratings):.1f}% of all ratings)")
print(f"Ratings 1-10 (explicit): {len(ratings) - n_zero:,}")

rating_counts = ratings['Book-Rating'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(rating_counts.index, rating_counts.values,
       color=[G0 if k == 0 else G2 for k in rating_counts.index],
       alpha=0.88, width=0.7)
ax.set_xlabel('Book rating')
ax.set_ylabel('Count')
ax.set_title('Rating distribution\n'
             '(0 = implicit interaction; 1-10 = explicit ratings)')
ax.set_xticks(list(rating_counts.index))
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.4)
fig.tight_layout()
save(fig, 'fig18_1_rating_distribution.png')

isbn_counts = ratings.groupby('ISBN').size()
count_occ_isbn = isbn_counts.value_counts().sort_index()
count_occ_isbn = count_occ_isbn.reindex(range(1, 16), fill_value=0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(1, 16), count_occ_isbn.values, color=G2, alpha=0.88, width=0.7)
ax.set_xlabel('Number of times an ISBN appears in ratings')
ax.set_ylabel('Count')
ax.set_xticks(range(1, 16))
ax.set_title('Distribution of book rating counts (first 15 values)')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.4)
fig.tight_layout()
save(fig, 'fig18_2_book_rating_counts.png')

userid_counts = ratings.groupby('User-ID').size()
count_occ_user = userid_counts.value_counts().sort_index()
count_occ_user = count_occ_user.reindex(range(1, 16), fill_value=0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(1, 16), count_occ_user.values, color=G3, alpha=0.88, width=0.7)
ax.set_xlabel('Number of times a User-ID appears in ratings')
ax.set_ylabel('Count')
ax.set_xticks(range(1, 16))
ax.set_title('Distribution of user rating counts (first 15 values)')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.4)
fig.tight_layout()
save(fig, 'fig18_3_user_rating_counts.png')

print(f"\nBooks rated once: {int((isbn_counts == 1).sum()):,} "
      f"of {len(isbn_counts):,}")
print(f"Users with one rating: {int((userid_counts == 1).sum()):,} "
      f"of {len(userid_counts):,}")

bookid_title = dict(zip(books['ISBN'], books['Book-Title']))
bookid_author = dict(zip(books['ISBN'], books['Book-Author']))

print("\nMost rated books, all ratings (0-10):")
for isbn, n in isbn_counts.sort_values(ascending=False).head(10).items():
    print(f"  {n:>5}  {bookid_title.get(isbn, isbn)}")

high = ratings[ratings['Book-Rating'] >= 8].groupby('ISBN').size()
print("\nMost rated books, ratings >= 8:")
for isbn, n in high.sort_values(ascending=False).head(10).items():
    print(f"  {n:>5}  {bookid_title.get(isbn, isbn)}")

rated = ratings[ratings['ISBN'].isin(books['ISBN'])]
book_degree = rated.groupby('ISBN').size()
user_degree = rated.groupby('User-ID').size()
HUB_ISBN = '0385504209'
readers = rated.loc[rated['ISBN'] == HUB_ISBN, 'User-ID'].unique()
candidates = [u for u in readers if 3 <= user_degree[u] <= 10]
rng = np.random.default_rng(SEED)
sampled_users = rng.choice(np.array(candidates), size=20, replace=False)
sub = rated[rated['User-ID'].isin(sampled_users)]

G_sub = nx.Graph()
user_nodes = [f"u{u}" for u in sampled_users]
G_sub.add_nodes_from(user_nodes)
G_sub.add_edges_from((f"u{u}", isbn)
                     for u, isbn in zip(sub['User-ID'], sub['ISBN']))
book_nodes = [n for n in G_sub.nodes if not str(n).startswith('u')]
book_sizes = [30 + 40 * G_sub.degree(b) for b in book_nodes]
shared_books = [b for b in book_nodes if G_sub.degree(b) >= 2]

print(f"\nFigure 18.4 subgraph: {len(user_nodes)} users "
      f"(of {len(candidates)} candidates), {len(book_nodes)} books, "
      f"{G_sub.number_of_edges()} edges")
print("Books in the subgraph, by ratings in the full dataset:")
for b in sorted(book_nodes, key=lambda b: -book_degree[b])[:10]:
    print(f"  {book_degree[b]:>5}  {bookid_title.get(b, b)}  "
          f"(sampled users connected: {G_sub.degree(b)})")
print("Books shared by at least two sampled users:")
for b in sorted(book_nodes, key=lambda b: -G_sub.degree(b)):
    if G_sub.degree(b) >= 2:
        print(f"  {G_sub.degree(b):>3}  {bookid_title.get(b, b)}")

pos = nx.spring_layout(G_sub, seed=SEED, k=0.35, iterations=200)
fig, ax = plt.subplots(figsize=(10, 9))
ax.axis('off')
nx.draw_networkx_edges(G_sub, pos, ax=ax, edge_color=G4, width=0.6,
                       alpha=0.7)
nx.draw_networkx_nodes(G_sub, pos, ax=ax, nodelist=book_nodes,
                       node_color=G5, node_size=book_sizes,
                       edgecolors=G2, linewidths=0.4)
nx.draw_networkx_nodes(G_sub, pos, ax=ax, nodelist=user_nodes,
                       node_color=G1, node_shape='s', node_size=60)
for b in shared_books:
    title = html.unescape(str(bookid_title.get(b, b)))
    title = title if len(title) <= 36 else title[:35] + '...'
    x, y = pos[b]
    ax.annotate(title, (x, y), xytext=(8, 8), textcoords='offset points',
                fontsize=9, color=G0,
                bbox=dict(facecolor='white', edgecolor=G5,
                          boxstyle='round,pad=0.2'))
handles = [
    Line2D([0], [0], marker='s', color='white', markerfacecolor=G1,
           markersize=8, label='User'),
    Line2D([0], [0], marker='o', color='white', markerfacecolor=G5,
           markeredgecolor=G2, markersize=10,
           label='Book (size = number of connections shown)'),
]
ax.legend(handles=handles, loc='lower left', fontsize=9, framealpha=0.95,
          edgecolor=G5)
ax.set_title('Book-Crossing: 20 readers of The Da Vinci Code '
             'and every book they rated')
fig.tight_layout()
save(fig, 'fig18_4_bookcrossing_subgraph.png')


# =============================================================================
# PART 2 - Preprocessing
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 - Preprocessing")
print("=" * 60)

df = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1')

df = df.loc[df['ISBN'].isin(books['ISBN'].unique()) &
            df['User-ID'].isin(users['User-ID'].unique())]
print(f"Rows with known ISBN and User-ID: {len(df):,}")

df = df[df['Book-Rating'] >= 8]
print(f"Rows with rating >= 8: {len(df):,}")

if len(df) > 100_000:
    df = df.sample(n=100_000, random_state=0)
print(f"Rows used: {len(df):,}")

user_mapping = {userid: i for i, userid in
                enumerate(df['User-ID'].unique())}
item_mapping = {isbn: i for i, isbn in
                enumerate(df['ISBN'].unique())}

num_users = len(user_mapping)
num_items = len(item_mapping)
num_total = num_users + num_items

print(f"Users: {num_users:,}  Items: {num_items:,}  Nodes: {num_total:,}")
print(f"Density: {len(df) / (num_users * num_items):.2e}")

user_ids = torch.LongTensor([user_mapping[i] for i in df['User-ID']])
item_ids = torch.LongTensor([item_mapping[i] + num_users for i in df['ISBN']])
edge_index = torch.stack((user_ids, item_ids))

train_index, test_index = train_test_split(range(len(df)),
                                           test_size=0.2, random_state=0)
val_index, test_index = train_test_split(test_index,
                                         test_size=0.5, random_state=0)

train_edge_index = edge_index[:, train_index]
val_edge_index = edge_index[:, val_index]
test_edge_index = edge_index[:, test_index]

print(f"Edges - train: {train_edge_index.shape[1]:,}  "
      f"val: {val_edge_index.shape[1]:,}  "
      f"test: {test_edge_index.shape[1]:,}")

subset_degree = df.groupby('ISBN').size().sort_values(ascending=False)
print("\nMost connected books in the LightGCN graph (100k subset):")
for isbn, n in subset_degree.head(10).items():
    print(f"  {n:>4}  {bookid_title.get(isbn, isbn)}")

per_user = df.groupby('User-ID').size()
print(f"\nItems per user - mean: {per_user.mean():.2f}  "
      f"median: {per_user.median():.0f}  max: {per_user.max()}")
print(f"Mean chance that a uniform negative is a positive of the user: "
      f"{(per_user / num_items).mean():.2e}")

BATCH_SIZE = 1024


def sample_mini_batch(edge_index):
    """
    Sample a mini-batch of (user, pos_item, neg_item) triplets, with item
    indices in item space. structured_negative_sampling is not used here:
    it works on a single node index space, so with offset item indices it
    would return user nodes as negative items.
    """
    index = np.random.choice(edge_index.shape[1], size=BATCH_SIZE)
    users = edge_index[0, index]
    pos_items = edge_index[1, index] - num_users
    neg_items = torch.randint(0, num_items, (BATCH_SIZE,),
                              device=edge_index.device)
    return users, pos_items, neg_items


# =============================================================================
# PART 3 - LightGCN, BPR loss, evaluation and training
# =============================================================================

print("\n" + "=" * 60)
print("PART 3 - LightGCN")
print("=" * 60)


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, num_layers=4, dim_h=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_users = nn.Embedding(num_embeddings=num_users,
                                      embedding_dim=dim_h)
        self.emb_items = nn.Embedding(num_embeddings=num_items,
                                      embedding_dim=dim_h)
        self.convs = nn.ModuleList(LGConv() for _ in range(num_layers))
        nn.init.normal_(self.emb_users.weight, std=0.01)
        nn.init.normal_(self.emb_items.weight, std=0.01)

    def forward(self, edge_index):
        emb = torch.cat([self.emb_users.weight, self.emb_items.weight])
        embs = [emb]
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)
        emb_final = torch.mean(torch.stack(embs, dim=1), dim=1)
        emb_users_final, emb_items_final = torch.split(
            emb_final, [self.num_users, self.num_items])
        return (emb_users_final, self.emb_users.weight,
                emb_items_final, self.emb_items.weight)


K = 20
LAMBDA = 1e-6


def bpr_loss(emb_users_final, emb_users,
             emb_pos_items_final, emb_pos_items,
             emb_neg_items_final, emb_neg_items):
    reg_loss = LAMBDA * (emb_users.norm().pow(2) +
                         emb_pos_items.norm().pow(2) +
                         emb_neg_items.norm().pow(2))
    pos_ratings = torch.mul(emb_users_final, emb_pos_items_final).sum(dim=-1)
    neg_ratings = torch.mul(emb_users_final, emb_neg_items_final).sum(dim=-1)
    bpr = -torch.mean(F.logsigmoid(pos_ratings - neg_ratings))
    return bpr + reg_loss


def get_user_positive_items(edge_index):
    """Build a dict: user_id -> set of positive item indices."""
    user_pos = {}
    for u, i in edge_index.T.tolist():
        user_pos.setdefault(u, set()).add(i - num_users)
    return user_pos


def RecallPrecision_ATk(groundTruth, r, k):
    num_correct_pred = torch.sum(r, dim=-1)
    user_num_liked = torch.Tensor([len(groundTruth[i])
                                   for i in range(len(groundTruth))])
    valid = user_num_liked > 0
    recall = torch.mean(num_correct_pred[valid] / user_num_liked[valid])
    precision = torch.mean(num_correct_pred[valid]) / k
    return recall.item(), precision.item()


def NDCGatK_r(groundTruth, r, k):
    test_matrix = torch.zeros((len(r), k))
    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    discount = 1.0 / torch.log2(torch.arange(2, k + 2).float())
    idcg = torch.sum(test_matrix * discount, axis=1)
    dcg = torch.sum(r * discount, axis=1)
    valid = idcg > 0
    return torch.mean(dcg[valid] / idcg[valid]).item()


@torch.no_grad()
def test(model, edge_index, exclude_edge_indices, k=K):
    """
    Recall@K and NDCG@K on edge_index. Propagation always happens on
    train_edge_index: the model ranks items using what it learned in
    training, not using the edges it is being evaluated on.
    """
    emb_users, _, emb_items, _ = model.forward(train_edge_index)
    rating = torch.matmul(emb_users, emb_items.T)
    for excl in exclude_edge_indices:
        rating[excl[0], excl[1] - num_users] = -(1 << 10)
    _, top_K_items = torch.topk(rating, k=k)
    users_pos = get_user_positive_items(edge_index)
    groundTruth = [list(users_pos.get(u, set()))
                   for u in range(model.num_users)]
    r = torch.Tensor([[1 if i in set(groundTruth[u]) else 0
                       for i in top_K_items[u].tolist()]
                      for u in range(model.num_users)]).float()
    recall, _ = RecallPrecision_ATk(groundTruth, r, k)
    ndcg = NDCGatK_r(groundTruth, r, k)
    sample = torch.randperm(edge_index.shape[1])[:BATCH_SIZE]
    u_s = edge_index[0, sample]
    i_s = edge_index[1, sample] - num_users
    neg = torch.randint(0, model.num_items, (BATCH_SIZE,),
                        device=edge_index.device)
    val_loss = bpr_loss(emb_users[u_s], model.emb_users.weight[u_s],
                        emb_items[i_s], model.emb_items.weight[i_s],
                        emb_items[neg], model.emb_items.weight[neg])
    return val_loss.item(), recall, ndcg


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Dense score matrix in test(): {num_users:,} x {num_items:,} "
      f"float32 = {num_users * num_items * 4 / 1e9:.2f} GB")

model = LightGCN(num_users, num_items).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
n_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {n_params:,}")

train_edge_index = train_edge_index.to(device)
val_edge_index = val_edge_index.to(device)

num_batch = int(len(train_index) / BATCH_SIZE)
history = {'epoch': [], 'train_loss': [], 'val_loss': [],
           'recall': [], 'ndcg': []}

start = time.time()
for epoch in range(31):
    model.train()
    for _ in range(num_batch):
        optimizer.zero_grad()
        embs = model.forward(train_edge_index)
        emb_users_final, emb_users, emb_items_final, emb_items = embs
        user_idx, pos_idx, neg_idx = sample_mini_batch(train_edge_index)
        emb_users_final = emb_users_final[user_idx]
        emb_users = emb_users[user_idx]
        emb_pos_items_final = emb_items_final[pos_idx]
        emb_pos_items = emb_items[pos_idx]
        emb_neg_items_final = emb_items_final[neg_idx]
        emb_neg_items = emb_items[neg_idx]
        train_loss = bpr_loss(
            emb_users_final, emb_users,
            emb_pos_items_final, emb_pos_items,
            emb_neg_items_final, emb_neg_items)
        train_loss.backward()
        optimizer.step()

    if epoch % 5 == 0:
        model.eval()
        val_loss, recall, ndcg = test(
            model, val_edge_index, [train_edge_index])
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss)
        history['recall'].append(recall)
        history['ndcg'].append(ndcg)
        print(f'Epoch {epoch:>2} | Train loss: {train_loss.item():.5f} |'
              f' Val loss: {val_loss:.5f} |'
              f' Val recall@{K}: {recall:.5f} |'
              f' Val ndcg@{K}: {ndcg:.5f}')

print(f"\nTraining time: {time.time() - start:.0f} s")

test_loss, test_recall, test_ndcg = test(
    model, test_edge_index.to(device),
    [train_edge_index, val_edge_index])
print(f'Test loss: {test_loss:.5f} | Test recall@{K}: {test_recall:.5f}'
      f' | Test ndcg@{K}: {test_ndcg:.5f}')

if device.type == 'cuda':
    print(f"Peak GPU memory allocated: "
          f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"Peak process memory (RSS): {peak_rss_gb():.2f} GB")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(history['epoch'], history['train_loss'], color=G0, linewidth=2.0,
         marker='o', markersize=5, label='Train')
ax1.plot(history['epoch'], history['val_loss'], color=G3, linewidth=2.0,
         marker='s', markersize=5, linestyle='--', label='Val')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('BPR loss')
ax1.set_title('Training and validation loss')
ax1.legend()
ax1.spines[['top', 'right']].set_visible(False)
ax1.grid(linestyle=':', alpha=0.4)
ax2.plot(history['epoch'], history['recall'], color=G0, linewidth=2.0,
         marker='o', markersize=5, label=f'Recall@{K}')
ax2.plot(history['epoch'], history['ndcg'], color=G3, linewidth=2.0,
         marker='s', markersize=5, linestyle='--', label=f'NDCG@{K}')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Metric value')
ax2.set_title(f'Validation Recall@{K} and NDCG@{K}')
ax2.legend()
ax2.spines[['top', 'right']].set_visible(False)
ax2.grid(linestyle=':', alpha=0.4)
fig.tight_layout()
save(fig, 'fig18_7_training_metrics.png')


# =============================================================================
# PART 4 - Recommendations
# =============================================================================

print("\n" + "=" * 60)
print("PART 4 - Generating recommendations")
print("=" * 60)

user_pos_items = get_user_positive_items(train_edge_index.cpu())
item_reverse = {idx: isbn for isbn, idx in item_mapping.items()}

with torch.no_grad():
    emb_users_star, _, emb_items_star, _ = model.forward(train_edge_index)


def recommend(user_id, num_recs):
    user = user_mapping[user_id]
    emb_user = emb_users_star[user]
    ratings = emb_items_star @ emb_user
    _, indices = torch.topk(ratings, k=100)
    seen = user_pos_items.get(user, set())
    liked_ids = list(seen)[:num_recs]
    liked_isbns = [item_reverse[b] for b in liked_ids]
    print(f'Favorite books from user {user_id}:')
    for isbn in liked_isbns:
        print(f' - {bookid_title.get(isbn, isbn)},'
              f' by {bookid_author.get(isbn, "Unknown")}')
    rec_ids = [int(idx) for idx in indices
               if int(idx) not in seen][:num_recs]
    rec_isbns = [item_reverse[b] for b in rec_ids]
    print(f'\nRecommended books for user {user_id}:')
    for isbn in rec_isbns:
        print(f' - {bookid_title.get(isbn, isbn)},'
              f' by {bookid_author.get(isbn, "Unknown")}')


TARGET_USER = 277427
if TARGET_USER in user_mapping:
    recommend(TARGET_USER, 5)
else:
    print(f"User {TARGET_USER} is not in the sampled graph.")


# =============================================================================
# PART 5 - Inspecting the recommendations
# =============================================================================

print("\n" + "=" * 60)
print("PART 5 - Inspecting the recommendations")
print("=" * 60)

item_degree = torch.bincount(train_edge_index[1].cpu() - num_users,
                             minlength=num_items)
order = torch.argsort(item_degree, descending=True)
popularity_rank = torch.empty_like(order)
popularity_rank[order] = torch.arange(1, num_items + 1)
top_1pct = max(1, num_items // 100)


def top_recs(user, n):
    scores = emb_items_star @ emb_users_star[user]
    _, indices = torch.topk(scores, k=100)
    seen = user_pos_items.get(user, set())
    return [int(i) for i in indices if int(i) not in seen][:n]


def describe(item):
    isbn = item_reverse[item]
    return (f"{bookid_title.get(isbn, isbn)} | train degree "
            f"{int(item_degree[item])} | popularity rank "
            f"{int(popularity_rank[item])} of {num_items:,}")


if TARGET_USER in user_mapping:
    u = user_mapping[TARGET_USER]
    print(f"User {TARGET_USER} - training items:")
    for item in user_pos_items.get(u, set()):
        print(f"  {describe(item)}")
    print(f"User {TARGET_USER} - top 5 recommendations:")
    for item in top_recs(u, 5):
        print(f"  {describe(item)}")

active_users = [u for u in range(num_users) if u in user_pos_items]
sample_users = np.random.default_rng(SEED).choice(
    active_users, size=min(100, len(active_users)), replace=False)
recs = [item for u in sample_users for item in top_recs(int(u), 5)]
counts = Counter(recs)
in_top = sum(1 for item in recs if int(popularity_rank[item]) <= top_1pct)

print(f"\nTop-5 lists for {len(sample_users)} random users: "
      f"{len(recs)} slots, {len(counts)} distinct books")
print(f"Slots filled by the {top_1pct} most popular books (top 1%): "
      f"{in_top} ({100 * in_top / len(recs):.1f}%)")
print("Most frequently recommended books:")
for item, n in counts.most_common(10):
    print(f"  {n:>3}x  {describe(item)}")

print("\nDone.")