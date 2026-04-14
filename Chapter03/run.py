"""
Chapter 3 – Creating Node Representations with DeepWalk
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch-geometric scikit-learn umap-learn networkx matplotlib
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_networkx
import umap

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# PART 1 – Word2Vec from scratch in PyTorch
# =============================================================================

print("=" * 60)
print("PART 1 – Word2Vec skip-gram (pure PyTorch)")
print("=" * 60)

# ── Skip-gram data generation ─────────────────────────────────────────────────

CONTEXT_SIZE = 2

text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc eu sem
scelerisque, dictum eros aliquam, accumsan quam. Pellentesque tempus, lorem
ut semper fermentum, ante turpis accumsan ex, sit amet ultricies tortor erat
quis nulla. Nunc consectetur ligula sit amet purus porttitor, vel tempus tortor
scelerisque. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices
posuere cubilia curae; Quisque suscipit ligula nec faucibus accumsan. Duis
vulputate massa sit amet viverra hendrerit. Integer maximus quis sapien id
convallis. Donec elementum placerat ex laoreet gravida. Praesent quis enim
facilisis, bibendum est nec, pharetra ex. Etiam pharetra congue justo, eget
imperdiet diam varius non. Mauris dolor lectus, interdum in laoreet quis,
faucibus vitae velit. Donec lacinia dui eget maximus cursus. Class aptent
taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos.
Vivamus tincidunt velit eget nisi ornare convallis. Pellentesque habitant morbi
tristique senectus et netus et malesuada fames ac turpis egestas. Donec
tristique ultrices tortor at accumsan.""".split()

skipgrams = []
for i in range(CONTEXT_SIZE, len(text) - CONTEXT_SIZE):
    context = [text[j] for j in
               np.arange(i - CONTEXT_SIZE, i + CONTEXT_SIZE + 1) if j != i]
    skipgrams.append((text[i], context))

print(f"Generated {len(skipgrams)} skip-gram pairs")
print(f"First two: {skipgrams[:2]}")

# ── Vocabulary ────────────────────────────────────────────────────────────────

vocab      = sorted(set(text))
VOCAB_SIZE = len(vocab)
EMBED_DIM  = 10

word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

print(f"\nVocabulary size: {VOCAB_SIZE}")

# ── Skip-gram model ───────────────────────────────────────────────────────────

class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear     = nn.Linear(embed_dim, vocab_size)

    def forward(self, target_idx: torch.Tensor) -> torch.Tensor:
        embed  = self.embeddings(target_idx)   # (batch, embed_dim)
        logits = self.linear(embed)            # (batch, vocab_size)
        return logits


model_w2v = SkipGram(VOCAB_SIZE, EMBED_DIM)
optimizer = optim.Adam(model_w2v.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("\nTraining skip-gram model …")
for epoch in range(100):
    total_loss = 0.0
    for target_word, context_words in skipgrams:
        target_idx = torch.tensor([word2idx[target_word]])
        for ctx in context_words:
            ctx_idx = torch.tensor([word2idx[ctx]])
            optimizer.zero_grad()
            logits = model_w2v(target_idx)
            loss   = criterion(logits, ctx_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d} | Loss: {total_loss:.4f}")

print("\nWord embedding (first word):")
print(model_w2v.embeddings.weight[0].detach().numpy())


# =============================================================================
# PART 2 – DeepWalk on Zachary's Karate Club (PyTorch Geometric)
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 – DeepWalk via PyTorch Geometric Node2Vec (p=q=1)")
print("=" * 60)

# ── Dataset ───────────────────────────────────────────────────────────────────

dataset = KarateClub()
data    = dataset[0]
labels  = data.y.numpy()

print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
print(f"Label distribution: {np.bincount(labels)}")

# ── Plot Karate Club ──────────────────────────────────────────────────────────

G_karate = to_networkx(data, to_undirected=True)
pos_k    = nx.spring_layout(G_karate, seed=SEED)

# Color by historical Mr. Hi (0) / Officer (1) factions, like fig3_6.
# data.y has 4 modularity-based community labels — different concept.
_G_zachary = nx.karate_club_graph()
factions_k = [1 if _G_zachary.nodes[n]['club'] == 'Officer' else 0
              for n in range(data.num_nodes)]
colors_k   = ["#2E75B6" if f == 0 else "#C0392B" for f in factions_k]

plt.figure(figsize=(8, 7), dpi=150)
plt.axis("off")
nx.draw_networkx(G_karate, pos=pos_k, node_color=colors_k,
                 node_size=600, font_size=9, font_color="white",
                 font_weight="bold", edge_color="#EDEDED")
plt.title("Zachary's Karate Club (Mr. Hi vs Officer)", fontsize=13)
plt.tight_layout()
plt.savefig("karate_club.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved karate_club.png")

# ── DeepWalk model ────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

EMB_DIM, EPOCHS, LR = 32, 100, 0.025

model_dw = Node2Vec(
    data.edge_index,
    embedding_dim        = EMB_DIM,
    walk_length          = 10,
    context_size         = 3,     # narrow window: limits cross-community pollution
    walks_per_node       = 20,
    num_negative_samples = 5,
    p                    = 1.0,   # DeepWalk: p=q=1
    q                    = 1.0,
    sparse               = True,
).to(device)

# gensim-style small uniform init: U[-0.5/d, 0.5/d] avoids sigmoid saturation
# at start. Without this, the model gets stuck and embeddings don't separate.
with torch.no_grad():
    model_dw.embedding.weight.data.uniform_(-0.5/EMB_DIM, 0.5/EMB_DIM)

# batch_size=1 because PyG's loader iterates over nodes (range(num_nodes)),
# not over walk pairs. With 34 nodes and bs=1 we get 34 gradient steps/epoch.
loader    = model_dw.loader(batch_size=1, shuffle=True, num_workers=0)
optimizer = torch.optim.SparseAdam(model_dw.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=LR * 0.02)

print("Training DeepWalk …")
for epoch in range(1, EPOCHS + 1):
    model_dw.train()
    total_loss = 0.0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model_dw.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d} | Loss: {total_loss/len(loader):.4f}")

# ── Extract embeddings ────────────────────────────────────────────────────────

model_dw.eval()
with torch.no_grad():
    embeddings = model_dw().detach()

emb_np   = embeddings.cpu().numpy()
emb_norm = F.normalize(embeddings, dim=1)

# ── Most similar nodes ────────────────────────────────────────────────────────

node0_sim = (emb_norm @ emb_norm[0]).cpu().numpy()
top10     = np.argsort(-node0_sim)[1:11]

print("\nNodes most similar to node 0 (instructor):")
for n in top10:
    print(f"  Node {n:2d} | cosine similarity: {node0_sim[n]:.4f} | faction: {labels[n]}")

sim_0_4 = F.cosine_similarity(embeddings[0].unsqueeze(0),
                               embeddings[4].unsqueeze(0))
print(f"\nSimilarity between node 0 and 4: {sim_0_4.item():.4f}")

# ── t-SNE + UMAP visualisation ────────────────────────────────────────────────

print("\nComputing t-SNE projection …")
tsne_proj = TSNE(n_components=2, init="pca",
                 learning_rate="auto", random_state=SEED,
                 perplexity=10).fit_transform(emb_np)

print("Computing UMAP projection …")
umap_proj = umap.UMAP(n_components=2, random_state=SEED,
                       n_neighbors=8, min_dist=0.3).fit_transform(emb_np)

# PyG's KarateClub.data.y has 4 community labels (modularity-based,
# from Kipf & Welling's GCN paper) — not the historical 2-class Mr.Hi/Officer.
PALETTE = ["#2E75B6", "#C0392B", "#27AE60", "#F39C12"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
for ax, proj, title in zip(axes,
                             [tsne_proj, umap_proj],
                             ["t-SNE", "UMAP"]):
    for cls in range(4):
        mask = labels == cls
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   s=140, color=PALETTE[cls], edgecolors="white",
                   linewidths=0.8, label=f"Class {cls} (n={mask.sum()})",
                   zorder=3)
    for i, (x, y) in enumerate(proj):
        ax.text(x, y, str(i), ha="center", va="center",
                fontsize=6, color="white", fontweight="bold", zorder=4)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.axis("off")

plt.suptitle("DeepWalk node embeddings – Zachary's Karate Club (4 communities)",
             fontsize=11, fontstyle="italic", y=0.02)
plt.tight_layout()
plt.savefig("tsne_umap_embeddings.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved tsne_umap_embeddings.png")

# ── 2-class historical split (Mr. Hi vs Officer) ──────────────────────────────
G_zachary = nx.karate_club_graph()
labels_2c = np.array([1 if G_zachary.nodes[n]['club'] == 'Officer' else 0
                       for n in range(data.num_nodes)])

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
for ax, proj, title in zip(axes, [tsne_proj, umap_proj], ["t-SNE", "UMAP"]):
    ax.scatter(proj[labels_2c == 0, 0], proj[labels_2c == 0, 1],
               s=140, color="#2E75B6", edgecolors="white",
               linewidths=0.8, label="Mr. Hi", zorder=3)
    ax.scatter(proj[labels_2c == 1, 0], proj[labels_2c == 1, 1],
               s=140, color="#C0392B", edgecolors="white",
               linewidths=0.8, label="Officer", zorder=3)
    for i, (x, y) in enumerate(proj):
        ax.text(x, y, str(i), ha="center", va="center",
                fontsize=6, color="white", fontweight="bold", zorder=4)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axis("off")

plt.suptitle("DeepWalk node embeddings – Zachary's Karate Club (2 factions)",
             fontsize=11, fontstyle="italic", y=0.02)
plt.tight_layout()
plt.savefig("tsne_umap_embeddings_2class.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved tsne_umap_embeddings_2class.png")

# ── Node classification ───────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    emb_np, labels, test_size=0.5, stratify=labels, random_state=SEED
)

clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nNode classification results:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

print("\nDone. Outputs: karate_club.png, tsne_umap_embeddings.png, "
      "tsne_umap_embeddings_2class.png")
