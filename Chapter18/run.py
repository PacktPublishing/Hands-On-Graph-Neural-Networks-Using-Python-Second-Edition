"""
Chapter 18 – Building a Recommender System Using LightGCN
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch-geometric pandas scikit-learn numpy

Fixes versus the first edition:
  - error_bad_lines=False -> on_bad_lines='skip' (pandas >= 2.0)
  - test() function (Recall@K, NDCG@K) included in full
  - Indentation of mini-batch embedding selection corrected
  - Further Reading references reordered to match in-text citations
  - Seaborn removed; all plots use matplotlib only

Fixes applied after the technical review:
  1. ITEM INDEX OFFSET. Users and items both started at 0, so in the unified
     node matrix [emb_users; emb_items] the item indices collided with user
     indices and LGConv propagated over a corrupted topology. Item indices are
     now offset by num_users, and every place that consumes them converts back
     to item space. THIS CHANGES EVERY NUMBER IN THE CHAPTER.
  2. BPR LOSS SIGN. The loss returned -mean(softplus(pos - neg)), which is
     unbounded below: the model could drive it to minus infinity by scaling
     the embeddings, which is what the first-edition loss curve showed. It is
     now the standard -mean(logsigmoid(pos - neg)), bounded below by zero.
  3. LAYER COMBINATION. forward() multiplied the mean over layers by a further
     1/(L+1), shrinking the final embeddings 5x. torch.mean already averages,
     so the extra factor is removed. It had been added to mask the diverging
     loss of point 2.
  4. EVALUATION USED THE WRONG EMBEDDINGS. test() and recommend() scored items
     with emb_users.weight @ emb_items.weight.T, i.e. the layer-0 embeddings,
     so the metrics described matrix factorization and not LightGCN. Both now
     propagate through the training graph first.
  5. NEGATIVE SAMPLING. structured_negative_sampling works on a single node
     index space and would return user nodes as negative items once the offset
     is applied. Negatives are now drawn uniformly in item space.
  6. SAMPLING BIAS. The 100k subset was taken with .iloc[:100000], i.e. the
     first rows in file order. It is now a random sample.

NOTE: not executed. Run before trusting any number.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
from sklearn.model_selection import train_test_split
from torch_geometric.nn import LGConv

torch.manual_seed(0); np.random.seed(0)

# =============================================================================
# PART 1 – Download and explore
# =============================================================================

print("=" * 60)
print("PART 1 – Loading Book-Crossing dataset")
print("=" * 60)

import os
import ssl
from urllib.request import urlopen

# The original Book-Crossing dataset was hosted at
# http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip
# which went offline. We now pull the three CSVs from a stable GitHub mirror
# (XuefengHuang/RecommendationSystem) that has hosted them unchanged for years.
mirror = ('https://raw.githubusercontent.com/XuefengHuang/'
          'RecommendationSystem/master/datasets/BX-CSV-Dump')
csv_files = ['BX-Book-Ratings.csv', 'BX-Users.csv', 'BX-Books.csv']

if all(os.path.exists(name) for name in csv_files):
    print("CSV files already present, skipping download.")
else:
    # Python on macOS does not use the system trust store, so urlopen fails
    # with CERTIFICATE_VERIFY_FAILED unless given a CA bundle.
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    print(f"Downloading from {mirror} ...")
    try:
        for name in csv_files:
            with urlopen(f"{mirror}/{name}", timeout=30, context=ctx) as resp:
                with open(name, 'wb') as f:
                    f.write(resp.read())
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Alternative: kagglehub.dataset_download("
              "'ruchi798/bookcrossing-dataset')")
        raise SystemExit(1)

ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1')
users   = pd.read_csv('BX-Users.csv',         sep=';', encoding='latin-1')
# Fix: on_bad_lines='skip' replaces deprecated error_bad_lines=False (pandas >= 2.0)
books   = pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1',
                      low_memory=False,
                       on_bad_lines='skip')

print(f"\nRatings: {len(ratings):,} rows")
print(f"Users:   {len(users):,} rows")
print(f"Books:   {len(books):,} rows")
print(f"Unique users in ratings: {ratings['User-ID'].nunique():,}")
print(f"Unique ISBNs in ratings: {ratings['ISBN'].nunique():,}")

# Rating distribution — matplotlib only (no seaborn)
rating_counts = ratings['Book-Rating'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.bar(rating_counts.index, rating_counts.values,
       color=['#111111' if k == 0 else '#555555'
              for k in rating_counts.index],
       alpha=0.88, width=0.7)
ax.set_xlabel('Book rating'); ax.set_ylabel('Count')
ax.set_title('Rating distribution\n'
             '(0 = implicit interaction; 1-10 = explicit ratings)')
ax.set_xticks(list(rating_counts.index))
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig('rating_dist.png', dpi=150, bbox_inches='tight')
plt.close(); print("\nSaved rating_dist.png")

# ISBN occurrence distribution
isbn_counts = ratings.groupby('ISBN').size()
count_occ_isbn = isbn_counts.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.bar(range(1, 16), count_occ_isbn.iloc[:15].values,
       color='#555555', alpha=0.88, width=0.7)
ax.set_xlabel('Number of times an ISBN appears in ratings')
ax.set_ylabel('Count')
ax.set_xticks(range(1, 16))
ax.set_title('Distribution of book rating counts (first 15 values)')
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig('isbn_dist.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved isbn_dist.png")

# User occurrence distribution
userid_counts = ratings.groupby('User-ID').size()
count_occ_user = userid_counts.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.bar(range(1, 16), count_occ_user.iloc[:15].values,
       color='#777777', alpha=0.88, width=0.7)
ax.set_xlabel('Number of times a User-ID appears in ratings')
ax.set_ylabel('Count')
ax.set_xticks(range(1, 16))
ax.set_title('Distribution of user rating counts (first 15 values)')
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig('user_dist.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved user_dist.png")


# =============================================================================
# PART 2 – Preprocessing
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 – Preprocessing")
print("=" * 60)

df = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1')

# Keep only rows where both ISBN and User-ID are in their tables
df = df.loc[df['ISBN'].isin(books['ISBN'].unique()) &
            df['User-ID'].isin(users['User-ID'].unique())]

# Keep high ratings and limit size for speed. Sampling at random rather than
# taking the first rows avoids the ordering bias of the raw file.
df = df[df['Book-Rating'] >= 8]
if len(df) > 100000:
    df = df.sample(n=100000, random_state=0)
print(f"Filtered df: {len(df):,} rows")

user_mapping = {uid: i for i, uid in enumerate(df['User-ID'].unique())}
item_mapping = {isbn: i for i, isbn in enumerate(df['ISBN'].unique())}

num_users = len(user_mapping)
num_items = len(item_mapping)
num_total = num_users + num_items

print(f"Users: {num_users:,}  Items: {num_items:,}")

# Item indices are offset by num_users so that, in the unified node matrix
# [emb_users; emb_items], a user and an item never share the same index.
user_ids   = torch.LongTensor([user_mapping[i] for i in df['User-ID']])
item_ids   = torch.LongTensor([item_mapping[i] + num_users for i in df['ISBN']])
edge_index = torch.stack((user_ids, item_ids))

train_index, test_index = train_test_split(range(len(df)),
                                            test_size=0.2, random_state=0)
val_index, test_index   = train_test_split(test_index,
                                            test_size=0.5, random_state=0)

train_edge_index = edge_index[:, train_index]
val_edge_index   = edge_index[:, val_index]
test_edge_index  = edge_index[:, test_index]

K          = 20
LAMBDA     = 1e-6
BATCH_SIZE = 1024


def sample_mini_batch(edge_index):
    """
    Returns (user, positive item, negative item) indices in *item space*.
    structured_negative_sampling is not used here: it works on a single node
    index space, so with offset item indices it would return user nodes as
    negative items. Uniform sampling over items is the standard BPR choice.
    """
    index     = np.random.choice(edge_index.shape[1], size=BATCH_SIZE)
    users     = edge_index[0, index]
    pos_items = edge_index[1, index] - num_users
    neg_items = torch.randint(0, num_items, (BATCH_SIZE,),
                              device=edge_index.device)
    return users, pos_items, neg_items


# =============================================================================
# PART 3 – LightGCN model
# =============================================================================

print("\n" + "=" * 60)
print("PART 3 – LightGCN")
print("=" * 60)


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, num_layers=4, dim_h=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_users = nn.Embedding(num_users, dim_h)
        self.emb_items = nn.Embedding(num_items, dim_h)
        self.convs     = nn.ModuleList(LGConv() for _ in range(num_layers))
        nn.init.normal_(self.emb_users.weight, std=0.01)
        nn.init.normal_(self.emb_items.weight, std=0.01)

    def forward(self, edge_index):
        emb  = torch.cat([self.emb_users.weight, self.emb_items.weight])
        embs = [emb]
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)
        # Layer combination with equal alpha_k, exactly as in the paper:
        # torch.mean already divides by the number of layers, so no further
        # scaling factor is applied.
        emb_final = torch.mean(torch.stack(embs, dim=1), dim=1)
        emb_users_final, emb_items_final = torch.split(
            emb_final, [self.num_users, self.num_items])
        return (emb_users_final, self.emb_users.weight,
                emb_items_final, self.emb_items.weight)


def bpr_loss(emb_uf, emb_u, emb_pf, emb_p, emb_nf, emb_n):
    reg_loss    = LAMBDA * (emb_u.norm()**2 + emb_p.norm()**2 + emb_n.norm()**2)
    pos_ratings = torch.mul(emb_uf, emb_pf).sum(dim=-1)
    neg_ratings = torch.mul(emb_uf, emb_nf).sum(dim=-1)
    # Standard BPR: maximise log sigma(pos - neg), i.e. minimise its negative.
    # The previous form, -mean(softplus(pos - neg)), was unbounded below and
    # could be driven to minus infinity by scaling the embeddings.
    bpr         = -torch.mean(F.logsigmoid(pos_ratings - neg_ratings))
    return bpr + reg_loss


def get_user_positive_items(edge_index):
    """Maps each user to the items it interacted with, in item space."""
    user_pos = {}
    for u, i in edge_index.T.tolist():
        user_pos.setdefault(u, set()).add(i - num_users)
    return user_pos


def RecallPrecision_ATk(groundTruth, r, k):
    num_correct  = torch.sum(r, dim=-1)
    user_n_liked = torch.tensor([len(groundTruth[i])
                                  for i in range(len(groundTruth))],
                                 dtype=torch.float)
    # Restrict the metric to users with at least one positive item in the
    # evaluated split. Otherwise the division by zero produces NaN.
    valid = user_n_liked > 0
    if valid.sum() == 0:
        return 0.0, 0.0
    recall    = torch.mean(num_correct[valid] / user_n_liked[valid])
    precision = torch.mean(num_correct[valid]) / k
    return recall.item(), precision.item()


def NDCGatK_r(groundTruth, r, k):
    test_matrix = torch.zeros((len(r), k))
    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    idcg = torch.sum(test_matrix / torch.log2(torch.arange(2, k+2).float()),
                     axis=1)
    dcg  = r / torch.log2(torch.arange(2, k+2).float())
    dcg  = torch.sum(dcg, axis=1)
    # Same guard as recall: only average over users with at least one item.
    valid = idcg > 0.0
    if valid.sum() == 0:
        return 0.0
    return torch.mean(dcg[valid] / idcg[valid]).item()


@torch.no_grad()
def test(model, edge_index, exclude_indices, graph_edge_index, k=K):
    """
    Recall@k and NDCG@k on edge_index. Propagation always happens on the
    training graph: the model must score items using what it learned there,
    not using the edges it is being evaluated on. The first edition scored
    with emb_users.weight @ emb_items.weight.T, i.e. the layer-0 embeddings,
    which measured matrix factorization rather than LightGCN.
    """
    model.eval()
    emb_users_final, _, emb_items_final, _ = model.forward(graph_edge_index)
    rating = torch.matmul(emb_users_final, emb_items_final.T)
    for excl in exclude_indices:
        rating[excl[0], excl[1] - num_users] = -(1 << 10)
    _, top_K = torch.topk(rating, k=k)
    users_pos   = get_user_positive_items(edge_index)
    groundTruth = [list(users_pos.get(u, set()))
                   for u in range(model.num_users)]
    r = torch.tensor([[1 if i in set(groundTruth[u]) else 0
                        for i in top_K[u].tolist()]
                       for u in range(model.num_users)], dtype=torch.float)
    recall, _ = RecallPrecision_ATk(groundTruth, r, k)
    ndcg      = NDCGatK_r(groundTruth, r, k)
    # Approximate val loss using a sample of the edge index
    sample_idx = torch.randperm(edge_index.shape[1])[:BATCH_SIZE]
    u_s = edge_index[0, sample_idx]
    i_s = edge_index[1, sample_idx] - num_users
    neg = torch.randint(0, model.num_items, (BATCH_SIZE,),
                        device=edge_index.device)
    val_loss = bpr_loss(emb_users_final[u_s], model.emb_users.weight[u_s],
                        emb_items_final[i_s], model.emb_items.weight[i_s],
                        emb_items_final[neg], model.emb_items.weight[neg])
    return val_loss.item(), recall, ndcg


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model     = LightGCN(num_users, num_items).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_edge_index = train_edge_index.to(device)
val_edge_index   = val_edge_index.to(device)

num_batch = int(len(train_index) / BATCH_SIZE)

print("\nTraining LightGCN (31 epochs) …")
history = {'epoch':[], 'train_loss':[], 'val_loss':[], 'recall':[], 'ndcg':[]}

for epoch in range(31):
    model.train()
    for _ in range(num_batch):
        optimizer.zero_grad()
        emb_uf, emb_u, emb_if, emb_i = model.forward(train_edge_index)
        u_idx, pos_idx, neg_idx       = sample_mini_batch(train_edge_index)
        # All embedding selections inside the batch loop
        loss = bpr_loss(
            emb_uf[u_idx],    emb_u[u_idx],
            emb_if[pos_idx],  emb_i[pos_idx],
            emb_if[neg_idx],  emb_i[neg_idx])
        loss.backward(); optimizer.step()

    if epoch % 5 == 0:
        model.eval()
        vl, recall, ndcg = test(model, val_edge_index, [train_edge_index],
                                train_edge_index)
        tl = loss.item()
        history['epoch'].append(epoch); history['train_loss'].append(tl)
        history['val_loss'].append(vl); history['recall'].append(recall)
        history['ndcg'].append(ndcg)
        print(f"  Epoch {epoch:>2} | Train: {tl:.5f} | Val: {vl:.5f} | "
              f"Recall@{K}: {recall:.5f} | NDCG@{K}: {ndcg:.5f}")

# Test
test_loss, test_recall, test_ndcg = test(
    model, test_edge_index.to(device), [train_edge_index, val_edge_index],
    train_edge_index)
print(f"\nTest loss: {test_loss:.5f} | "
      f"Test recall@{K}: {test_recall:.5f} | "
      f"Test ndcg@{K}: {test_ndcg:.5f}")

# Training metrics plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
fig.patch.set_facecolor('white')
ax1.plot(history['epoch'], history['train_loss'],
         color='#111111', linewidth=2.0, marker='o', markersize=5, label='Train')
ax1.plot(history['epoch'], history['val_loss'],
         color='#777777', linewidth=2.0, marker='s', markersize=5,
         linestyle='--', label='Val')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('BPR loss')
ax1.set_title('Training and validation loss')
ax1.legend(); ax1.spines[['top','right']].set_visible(False)
ax1.grid(linestyle=':', alpha=0.4)

ax2.plot(history['epoch'], history['recall'],
         color='#111111', linewidth=2.0, marker='o', markersize=5,
         label=f'Recall@{K}')
ax2.plot(history['epoch'], history['ndcg'],
         color='#777777', linewidth=2.0, marker='s', markersize=5,
         linestyle='--', label=f'NDCG@{K}')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Metric value')
ax2.set_title(f'Validation Recall@{K} and NDCG@{K}')
ax2.legend(); ax2.spines[['top','right']].set_visible(False)
ax2.grid(linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
plt.close(); print("\nSaved training_metrics.png")


# =============================================================================
# PART 4 – Recommendations
# =============================================================================

print("\n" + "=" * 60)
print("PART 4 – Generating recommendations")
print("=" * 60)

bookid_title  = dict(zip(books['ISBN'], books['Book-Title']))
bookid_author = dict(zip(books['ISBN'], books['Book-Author']))
user_pos_items = get_user_positive_items(train_edge_index.cpu())
# Reverse lookup built once: the first edition searched the mapping linearly
# for every recommended item.
item_reverse  = {idx: isbn for isbn, idx in item_mapping.items()}

with torch.no_grad():
    emb_users_star, _, emb_items_star, _ = model.forward(train_edge_index)


def recommend(user_id, num_recs):
    if user_id not in user_mapping:
        print(f"User {user_id} not found."); return
    user     = user_mapping[user_id]
    # Scored with the propagated embeddings, consistently with test()
    emb_user = emb_users_star[user]
    ratings  = emb_items_star @ emb_user
    _, indices = torch.topk(ratings, k=100)

    # Favourites come straight from the interaction data. The first edition
    # took them from the top-100 ranking, which returns an empty list whenever
    # none of the user's own books happens to rank that high.
    seen        = user_pos_items.get(user, set())
    liked_ids   = list(seen)[:num_recs]
    liked_isbns = [item_reverse[b] for b in liked_ids]
    print(f"Favorite books from user {user_id}:")
    for isbn in liked_isbns:
        print(f"  - {bookid_title.get(isbn, isbn)}, "
              f"by {bookid_author.get(isbn, 'Unknown')}")

    # int(i) matters: `i` is a 0-d tensor, and `tensor not in set_of_ints` is
    # always True because the hash never matches, so the filter silently let
    # already-read books through.
    rec_ids = [int(i) for i in indices if int(i) not in seen][:num_recs]
    rec_isbns = [item_reverse[b] for b in rec_ids]
    print(f"\nRecommended books for user {user_id}:")
    for isbn in rec_isbns:
        print(f"  - {bookid_title.get(isbn, isbn)}, "
              f"by {bookid_author.get(isbn, 'Unknown')}")


recommend(277427, 5)
print("\nDone.")