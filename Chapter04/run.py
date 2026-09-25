"""
Chapter 4 – Improving Embeddings with Biased Random Walks in Node2Vec
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch-geometric torch-cluster scikit-learn networkx matplotlib pandas
    (torch-cluster provides torch.ops.torch_cluster.random_walk, required by
     PyG's Node2Vec for the biased walks in Part 2; install the wheel that
     matches your torch/CUDA build, e.g. from https://data.pyg.org/whl/)
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from collections import defaultdict
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_networkx, from_networkx

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# =============================================================================
# PART 1 – Biased random walk functions
# =============================================================================

print("\n" + "=" * 60)
print("PART 1 – Biased random walks")
print("=" * 60)

def next_node(G: nx.Graph, previous, current, p: float, q: float):
    """
    Select the next node in a Node2Vec random walk.

    Edge weights (if present as ``G[u][v]['weight']``) are multiplied into
    the transition probability, so biased random walks on a weighted graph
    prefer stronger edges — something PyG's native ``Node2Vec`` does not do.

    Parameters
    ----------
    G        : networkx graph
    previous : previously visited node (None at the start)
    current  : current node
    p        : return parameter  — higher → less backtracking
    q        : in-out parameter  — higher → stays local (BFS-like)

    Returns
    -------
    The index of the next node.
    """
    neighbors = list(G.neighbors(current))
    alphas    = []

    for neighbor in neighbors:
        weight = G[current][neighbor].get("weight", 1.0)
        if neighbor == previous:
            # Going back to the previous node
            alpha = 1 / p
        elif previous is not None and G.has_edge(neighbor, previous):
            # Neighbor is also connected to the previous node (d=1)
            alpha = 1.0
        else:
            # Neighbor is farther from the previous node (d=2)
            alpha = 1 / q
        alphas.append(alpha * weight)

    total        = sum(alphas)
    probs        = [a / total for a in alphas]
    next_node_id = np.random.choice(neighbors, size=1, p=probs)[0]
    return next_node_id


def random_walk(G: nx.Graph, start, length: int, p: float, q: float):
    """
    Generate a single biased random walk.

    Parameters
    ----------
    G      : networkx graph
    start  : starting node
    length : number of steps
    p, q   : Node2Vec bias parameters

    Returns
    -------
    List of node ids as strings (for Word2Vec compatibility).
    """
    walk = [start]
    for _ in range(length):
        current   = walk[-1]
        previous  = walk[-2] if len(walk) > 1 else None
        neighbors = list(G.neighbors(current))
        if not neighbors:          # isolated node — stop early
            break
        walk.append(next_node(G, previous, current, p, q))
    return walk


# ── Demo on a small random graph ──────────────────────────────────────────────

G_demo = nx.erdos_renyi_graph(10, 0.3, seed=1, directed=False)

print("DeepWalk  (p=1, q=1):")
print(f"  {random_walk(G_demo, start=0, length=8, p=1, q=1)}")

print("High q    (p=1, q=10) — stays local (BFS-like):")
print(f"  {random_walk(G_demo, start=0, length=8, p=1, q=10)}")

print("High p    (p=10, q=1) — less backtracking:")
print(f"  {random_walk(G_demo, start=0, length=8, p=10, q=1)}")


# =============================================================================
# PART 2 – Node2Vec vs DeepWalk on Zachary's Karate Club
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 – Node2Vec vs DeepWalk on Zachary's Karate Club")
print("=" * 60)

# ── Dataset ───────────────────────────────────────────────────────────────────

dataset = KarateClub()
data    = dataset[0]

# Use the historical 2-class split (Mr. Hi vs Officer) instead of PyG's
# 4-class modularity labels. The 2-class problem is balanced (16 vs 18),
# matches the original Node2Vec paper, and reduces the granularity floor:
# with a 14-node test set, 2 classes ≈ random baseline 50% (vs 25% for 4),
# so the p/q signal stops being drowned by 7% per-error noise.
_G_zachary = nx.karate_club_graph()
labels = np.array([1 if _G_zachary.nodes[n]["club"] == "Officer" else 0
                   for n in range(data.num_nodes)])

# ── Plot ──────────────────────────────────────────────────────────────────────

G_karate = to_networkx(data, to_undirected=True)
pos_k    = nx.spring_layout(G_karate, seed=SEED)
colors_k = ["#2E75B6" if l == 0 else "#C0392B" for l in labels]

plt.figure(figsize=(8, 7), dpi=150)
plt.axis("off")
nx.draw_networkx(G_karate, pos=pos_k, node_color=colors_k, node_size=600,
                 font_size=9, font_color="white", font_weight="bold",
                 edge_color="#EDEDED")
plt.title("Zachary's Karate Club", fontsize=13)
plt.tight_layout()
plt.savefig("karate_club_ch4.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved karate_club_ch4.png")

# ── Helper: train Node2Vec with given p and q ─────────────────────────────────

def train_node2vec(data, p: float, q: float,
                   epochs: int = 100, seed: int = 0) -> np.ndarray:
    """
    Train a Node2Vec model and return node embeddings as a numpy array.
    Setting p=q=1 gives standard DeepWalk behaviour.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    EMB_DIM = 64
    LR      = 0.025

    model = Node2Vec(
        data.edge_index,
        embedding_dim        = EMB_DIM,
        walk_length          = 10,
        context_size         = 5,
        walks_per_node       = 40,
        num_negative_samples = 5,
        p                    = p,
        q                    = q,
        sparse               = True,
    ).to(device)

    # gensim-style small uniform init: without this, the model gets stuck
    # and the embeddings fail to separate (see Chapter03 for the same fix).
    with torch.no_grad():
        model.embedding.weight.data.uniform_(-0.5 / EMB_DIM, 0.5 / EMB_DIM)

    # batch_size=1: PyG's loader iterates over nodes (range(num_nodes)),
    # not over walk pairs. With 34 nodes and bs=1 we get 34 gradient
    # steps/epoch instead of just 1.
    loader    = model.loader(batch_size=1, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=LR * 0.02)

    model.train()
    for _ in range(epochs):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        return model().cpu().numpy()


def evaluate(embeddings: np.ndarray, labels: np.ndarray,
             n_runs: int = 20, n_estimators: int = 100) -> tuple:
    """
    Evaluate embeddings using a Random Forest over multiple train/test splits.
    Returns (mean_accuracy, std_accuracy).
    """
    accs = []
    for seed in range(n_runs):
        X_tr, X_te, y_tr, y_te = train_test_split(
            embeddings, labels,
            test_size=0.4, stratify=labels, random_state=seed
        )
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     random_state=seed)
        clf.fit(X_tr, y_tr)
        accs.append(accuracy_score(y_te, clf.predict(X_te)))
    return float(np.mean(accs)), float(np.std(accs))


# ── p/q parameter grid (run first to find the best Node2Vec config) ──────────
#
# We follow the original Node2Vec paper and explore q in {0.25, 0.5, 1, 2, 4}.
# Low q gives DFS-like walks (community/homophily), high q gives BFS-like
# walks (local/structural roles). On a graph this small the differences
# turn out to be within run-to-run variance.

print("\nRunning p/q parameter grid (this may take a few minutes) …")

p_values  = [0.25, 0.5, 1, 2, 4]
q_values  = [0.25, 0.5, 1, 2, 4]
GRID_RUNS = 10
results   = np.zeros((len(p_values), len(q_values)))

for pi, p in enumerate(p_values):
    for qi, q in enumerate(q_values):
        accs = []
        for run in range(GRID_RUNS):
            emb = train_node2vec(data, p=p, q=q, epochs=80, seed=run)
            X_tr, X_te, y_tr, y_te = train_test_split(
                emb, labels, test_size=0.4, stratify=labels, random_state=run
            )
            clf = RandomForestClassifier(n_estimators=100, random_state=run)
            clf.fit(X_tr, y_tr)
            accs.append(accuracy_score(y_te, clf.predict(X_te)))
        results[pi, qi] = np.mean(accs)
        print(f"  p={p:>5}, q={q:>5} → {results[pi, qi]:.3f}")

# Pick the best (p, q) from the grid. With only 10 runs per cell several
# cells often tie near the max (grid resolution ≈ 1/14 per correct vote),
# and np.argmax becomes a coin flip. Break the tie by re-evaluating every
# cell within 1 pp of the max on 20 runs and picking the real winner.
grid_max = results.max()
tied = [(pi, qi) for pi in range(len(p_values)) for qi in range(len(q_values))
        if results[pi, qi] >= grid_max - 0.01]
if len(tied) > 1:
    print(f"\n{len(tied)} cells tied within 1pp of {grid_max:.3f}; "
          f"resolving with a 20-run re-evaluation …")
    best_i, best_j, best_acc = tied[0][0], tied[0][1], -1.0
    for pi, qi in tied:
        p, q = p_values[pi], q_values[qi]
        emb  = train_node2vec(data, p=p, q=q, epochs=80, seed=0)
        acc, _ = evaluate(emb, labels, n_runs=20)
        print(f"  p={p:>5}, q={q:>5} → {acc:.3f}")
        if acc > best_acc:
            best_i, best_j, best_acc = pi, qi, acc
else:
    best_i, best_j = tied[0]
p_best, q_best = p_values[best_i], q_values[best_j]
print(f"\nBest grid cell: p={p_best}, q={q_best} → "
      f"{results[best_i, best_j]:.3f}")

# ── DeepWalk vs best Node2Vec — final comparison ─────────────────────────────

print(f"\nFinal comparison (n_runs=20 each) …")
print("Training DeepWalk  (p=1, q=1) …")
emb_deepwalk = train_node2vec(data, p=1, q=1)

print(f"Training Node2Vec  (p={p_best}, q={q_best}) …")
emb_node2vec = train_node2vec(data, p=p_best, q=q_best)

mean_dw,  std_dw  = evaluate(emb_deepwalk, labels, n_runs=20)
mean_n2v, std_n2v = evaluate(emb_node2vec, labels, n_runs=20)

print(f"\nDeepWalk  (p=1, q=1)         : {mean_dw*100:.2f}% ± {std_dw*100:.2f}%")
print(f"Node2Vec  (p={p_best}, q={q_best}): "
      f"{mean_n2v*100:.2f}% ± {std_n2v*100:.2f}%")
print(f"Improvement                 : +{(mean_n2v - mean_dw)*100:.2f} pp")

# Plot heatmap
fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
vmin, vmax = float(results.min()), float(results.max())
im = ax.imshow(results, cmap="Blues", vmin=vmin, vmax=vmax,
               aspect="auto", origin="upper")

threshold = vmin + 0.6 * (vmax - vmin)
for i in range(len(p_values)):
    for j in range(len(q_values)):
        val = results[i, j]
        tc  = "white" if val > threshold else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=12, color=tc, fontweight="bold")

ax.set_xticks(range(len(q_values)))
ax.set_yticks(range(len(p_values)))
ax.set_xticklabels([f"q={q}" for q in q_values], fontsize=11)
ax.set_yticklabels([f"p={p}" for p in p_values], fontsize=11)
ax.set_xlabel("In-out parameter q  (lower → DFS / community structure)",
              fontsize=11, labelpad=8)
ax.set_ylabel("Return parameter p  (higher → less backtracking)",
              fontsize=11, labelpad=8)
ax.set_title("Mean accuracy for different p and q values\n"
             "Zachary's Karate Club",
             fontsize=11, fontweight="bold", pad=10)

from matplotlib.patches import Rectangle
# Highlight DeepWalk cell (p=q=1)
dw_i, dw_j = p_values.index(1), q_values.index(1)
ax.add_patch(Rectangle((dw_j - 0.5, dw_i - 0.5), 1, 1, linewidth=2.5,
                       edgecolor="#C55A11", facecolor="none"))
ax.text(dw_j, dw_i - 0.42, "DeepWalk", ha="center", fontsize=7,
        color="#C55A11", fontweight="bold")

# Highlight best Node2Vec cell
ax.add_patch(Rectangle((best_j - 0.5, best_i - 0.5), 1, 1, linewidth=2.5,
                       edgecolor="#1E8449", facecolor="none"))
ax.text(best_j, best_i - 0.42, "best", ha="center", fontsize=7,
        color="#1E8449", fontweight="bold")

fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03).set_label("Accuracy")
fig.tight_layout()
plt.savefig("pq_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pq_heatmap.png")


# =============================================================================
# PART 3 – Movie recommender system (MovieLens 100K + Node2Vec)
# =============================================================================

print("\n" + "=" * 60)
print("PART 3 – Movie recommender system")
print("=" * 60)

# ── Download MovieLens 100K ───────────────────────────────────────────────────

import os
if not os.path.isdir("ml-100k"):
    print("Downloading MovieLens 100K …")
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    with urlopen(url) as zurl:
        with ZipFile(BytesIO(zurl.read())) as zfile:
            zfile.extractall(".")
else:
    print("MovieLens 100K already extracted, skipping download.")

ratings = pd.read_csv("ml-100k/u.data", sep="\t",
                      names=["user_id", "movie_id", "rating", "timestamp"])
movies  = pd.read_csv("ml-100k/u.item", sep="|",
                      usecols=range(2),
                      names=["movie_id", "title"],
                      encoding="latin-1")

print(f"Loaded {len(ratings)} ratings, {len(movies)} movies")

# Keep only positive ratings (4 and 5)
ratings = ratings[ratings["rating"] >= 4]
print(f"Positive ratings (≥4): {len(ratings)}")

# ── Build co-like graph ───────────────────────────────────────────────────────

pairs = defaultdict(int)
for _, group in ratings.groupby("user_id"):
    user_movies = list(group["movie_id"])
    for i in range(len(user_movies)):
        for j in range(i + 1, len(user_movies)):
            pairs[(user_movies[i], user_movies[j])] += 1

G_movies = nx.Graph()
for (movie1, movie2), score in pairs.items():
    if score >= 20:
        G_movies.add_edge(movie1, movie2, weight=score)

print(f"Movie graph: {G_movies.number_of_nodes()} nodes, "
      f"{G_movies.number_of_edges()} edges")

# ── Convert to PyG and train Node2Vec ────────────────────────────────────────

# Build node index mappings before converting
node_list   = list(G_movies.nodes())
idx_to_movie = {i: m for i, m in enumerate(node_list)}
movie_to_idx = {m: i for i, m in idx_to_movie.items()}

# ── Hybrid training: nx generates weight-aware walks, PyG trains ────────────
#
# PyG's Node2Vec ignores `edge_weight` in its internal random walks, which
# flattens highly-weighted "saga" neighbours (e.g. sequels) into the same
# bucket as weak co-likes. We work around this with a hybrid approach:
#   - networkx generates the biased random walks via `random_walk()` (Part 1),
#     which multiplies each transition probability by the edge weight.
#   - PyG's `Node2Vec` is used only as an embedding + loss container;
#     we feed our own walks to `rec_model.loss(pos_rw, neg_rw)`.
# Everything else (embedding layer, skip-gram loss, optimizer) stays PyG.

# Re-seed before Part 3 so the recommender is reproducible regardless of
# how much RNG state Part 2's grid search consumed.
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

WALKS_PER_NODE = 200
WALK_LENGTH    = 20
CONTEXT_SIZE   = 10
NUM_NEG        = 5
EMB_DIM        = 64
P_BIAS, Q_BIAS = 2.0, 1.0
NUM_NODES      = len(node_list)

print("Generating weight-aware biased walks (networkx) …")
all_walks = []
for mid in node_list:
    for _ in range(WALKS_PER_NODE):
        w = random_walk(G_movies, mid, WALK_LENGTH, P_BIAS, Q_BIAS)
        # Pad dead-end walks by repeating the last node
        while len(w) < WALK_LENGTH + 1:
            w.append(w[-1])
        all_walks.append([movie_to_idx[int(n)] for n in w])

walks = torch.tensor(all_walks, dtype=torch.long)
del all_walks
print(f"  walks tensor: {tuple(walks.shape)}")

# Mikolov's unigram^0.75 distribution for negative sampling. Uniform negatives
# let the model collapse all popular movies onto the same "blockbuster"
# direction; sampling negatives proportional to corpus frequency forces it
# to keep them apart, which is what surfaces sequels (Terminator 2, etc.)
# instead of generic co-liked hits.
node_freq   = walks.view(-1).bincount(minlength=NUM_NODES).float()
neg_weights = node_freq.pow(0.75)
neg_weights = neg_weights / neg_weights.sum()

# Slide a context_size window over each walk (PyG's internal sub-walk format).
num_sub_per_walk = walks.size(1) - CONTEXT_SIZE + 1
pos_subs = torch.cat(
    [walks[:, j:j + CONTEXT_SIZE] for j in range(num_sub_per_walk)], dim=0)
del walks
print(f"  pos sub-walks: {tuple(pos_subs.shape)}")

# PyG Node2Vec — used only for `embedding` + `loss`; a dummy edge_index
# satisfies the constructor (we never call its internal random walks).
dummy_ei  = torch.stack([torch.arange(NUM_NODES), torch.arange(NUM_NODES)])
rec_model = Node2Vec(
    dummy_ei,
    embedding_dim        = EMB_DIM,
    walk_length          = WALK_LENGTH,
    context_size         = CONTEXT_SIZE,
    walks_per_node       = WALKS_PER_NODE,
    num_negative_samples = NUM_NEG,
    p                    = P_BIAS,
    q                    = Q_BIAS,
    sparse               = True,
).to(device)

# Small uniform init — without it the embedding saturates at epoch 1 and
# collapses onto a "popularity" manifold, losing the saga/theme axis.
# Same trick as Chapter 3.
with torch.no_grad():
    rec_model.embedding.weight.data.uniform_(-0.5 / EMB_DIM, 0.5 / EMB_DIM)

LR_REC = 0.005
EPOCHS = 10
BATCH  = 1024

rec_optimizer = torch.optim.SparseAdam(rec_model.parameters(), lr=LR_REC)
rec_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    rec_optimizer, T_max=EPOCHS, eta_min=LR_REC * 0.05)

print(f"Training hybrid Node2Vec on movie graph "
      f"({pos_subs.size(0)} sub-walks/epoch, {EPOCHS} epochs) …")

num_sub = pos_subs.size(0)
for epoch in range(1, EPOCHS + 1):
    rec_model.train()
    perm          = torch.randperm(num_sub)
    total, steps  = 0.0, 0
    for i in range(0, num_sub, BATCH):
        idx    = perm[i:i + BATCH]
        pos_rw = pos_subs[idx].to(device)

        # Negative sub-walks: same target, context sampled from unigram^0.75.
        B = pos_rw.size(0)
        neg_rw = torch.multinomial(
            neg_weights, B * NUM_NEG * CONTEXT_SIZE, replacement=True,
        ).view(B * NUM_NEG, CONTEXT_SIZE).to(device)
        neg_rw[:, 0] = pos_rw[:, 0].repeat_interleave(NUM_NEG)

        rec_optimizer.zero_grad()
        loss = rec_model.loss(pos_rw, neg_rw)
        loss.backward()
        rec_optimizer.step()
        total += loss.item()
        steps += 1
    rec_scheduler.step()
    print(f"  Epoch {epoch:3d} | Loss: {total/steps:.4f}")

del pos_subs

# ── Recommendation function ───────────────────────────────────────────────────

rec_model.eval()
with torch.no_grad():
    all_emb = rec_model().detach().cpu()

emb_norm = F.normalize(all_emb, dim=1)


def recommend(title: str, top_k: int = 5):
    """Print the top_k most similar movies to the given title."""
    row = movies[movies["title"] == title]
    if row.empty:
        print(f"Movie not found: {title}")
        return
    movie_id = row["movie_id"].values[0]
    if movie_id not in movie_to_idx:
        print(f'"{title}" not in graph (too few co-likes)')
        return

    idx  = movie_to_idx[movie_id]
    sims = (emb_norm @ emb_norm[idx]).numpy()
    top  = np.argsort(-sims)[1:top_k + 1]

    print(f'\nMovies most similar to "{title}":')
    for rank, i in enumerate(top, 1):
        mid    = idx_to_movie[i]
        mtitle = movies[movies["movie_id"] == mid]["title"].values[0]
        print(f"  {rank}. {mtitle:<45s} (similarity: {sims[i]:.2f})")


recommend("Star Wars (1977)")
recommend("Toy Story (1995)")

print("\nDone. Outputs: karate_club_ch4.png, pq_heatmap.png")