"""
Chapter 3 - Creating Node Representations with DeepWalk
Hands-On Graph Neural Networks Using Python, Second Edition

Runs the code of the chapter in order and prints the outputs shown in the book.
Saves Figures 3.5, 3.6 and 3.7 to figures/.

Usage:
    python run.py
"""

import os
import random
import sys

import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_networkx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'tools'))
from bookstyle import (apply, figure, FONT, BG, DPI, INK, INK_SOFT,
                       G3, G4, G5, G6)

plt = apply()

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

G0, G2 = INK, INK_SOFT
# The two factions have to be told apart in a grayscale print, so they differ
# by two steps of the ramp and both carry dark labels. A reversed white label
# on a dark node closes up at this point size once the ink spreads.
CLASS_A, CLASS_B = G4, G6


def save(fig, name):
    """Like bookstyle.save, but shows the figure inline before closing it."""
    fig.savefig(os.path.join(FIG_DIR, name), dpi=DPI, facecolor=BG,
                edgecolor="none")
    print(f"[saved figures/{name}]")
    if "ipykernel" in sys.modules:
        plt.show()
    else:
        plt.close(fig)


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
# PART 1 - Creating skip-grams
# =============================================================================
banner("PART 1 - Creating skip-grams")

CONTEXT_SIZE = 2
text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc eu sem
scelerisque, dictum eros aliquam, accumsan quam. Pellentesque
tempus, lorem ut semper fermentum, ante turpis accumsan ex,
sit amet ultricies tortor erat quis nulla.
""".split()

skipgrams = []
for i in range(CONTEXT_SIZE, len(text) - CONTEXT_SIZE):
    array = [
        text[j] for j in np.arange(
            i - CONTEXT_SIZE, i + CONTEXT_SIZE + 1)
        if j != i
    ]
    skipgrams.append((text[i], array))

print(skipgrams[0:2])

vocab = sorted(set(text))
VOCAB_SIZE = len(vocab)
print(f'Vocabulary size: {VOCAB_SIZE}')


# =============================================================================
# PART 2 - The skip-gram model
# =============================================================================
banner("PART 2 - The skip-gram model")

word2idx = {w: i for i, w in enumerate(vocab)}
EMBED_DIM = 10


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, target_idx):
        embed = self.embeddings(target_idx)
        logits = self.linear(embed)
        return logits


torch.manual_seed(0)
model = SkipGram(VOCAB_SIZE, EMBED_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
n_pairs = sum(len(ctx) for _, ctx in skipgrams)

for epoch in range(100):
    total_loss = 0
    for target_word, context_words in skipgrams:
        target_idx = torch.tensor([word2idx[target_word]])
        for ctx in context_words:
            ctx_idx = torch.tensor([word2idx[ctx]])
            optimizer.zero_grad()
            logits = model(target_idx)
            loss = criterion(logits, ctx_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1:3d} | Loss: {total_loss / n_pairs:.4f}')

print('Word embedding =')
print(model.embeddings.weight[0].detach().numpy())


# =============================================================================
# PART 3 - Random walks
# =============================================================================
banner("PART 3 - Random walks")

random.seed(0)
np.random.seed(0)

G = nx.erdos_renyi_graph(10, 0.3, seed=1, directed=False)
pos = nx.spring_layout(G, seed=0)

fig, ax = figure(aspect=0.78)
ax.axis("off")
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=G4, width=0.9)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=320, node_color=G5,
                       edgecolors=INK, linewidths=0.8)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=8.5, font_color=INK,
                        font_weight="bold", font_family=FONT)
ax.margins(0.09)
save(fig, "fig3_5_random_graph.png")


def random_walk(G, start, length):
    walk = [str(start)]
    for _ in range(length):
        neighbors = list(G.neighbors(start))
        if not neighbors:
            break
        next_node = np.random.choice(neighbors)
        walk.append(str(next_node))
        start = next_node
    return walk


print(random_walk(G, start=0, length=10))


# =============================================================================
# PART 4 - Zachary's Karate Club
# =============================================================================
banner("PART 4 - Zachary's Karate Club")

dataset = KarateClub()
data = dataset[0]

G_nx = nx.karate_club_graph()
labels = np.array([1 if G_nx.nodes[n]['club'] == 'Officer' else 0
                   for n in G_nx.nodes])

print(f'Nodes: {data.num_nodes}, Edges: {data.num_edges}')

G = to_networkx(data, to_undirected=True)
pos = nx.spring_layout(G, seed=0)
node_colors = [CLASS_A if l == 0 else CLASS_B for l in labels]

fig, ax = figure(aspect=0.84)
ax.axis("off")
# 32 and 33 are 0.25in apart once the layout is stretched to the column, so a
# node wider than that runs its neighbour's label into its own.
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=G6, width=0.7)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=200, node_color=node_colors,
                       edgecolors=INK, linewidths=0.7)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color=INK,
                        labels={n: str(n) for n in G.nodes},
                        font_weight="bold", font_family=FONT)
# networkx fits the data exactly, which clips the outermost nodes in half.
ax.margins(0.09)
ax.legend(handles=[mpatches.Patch(facecolor=CLASS_A, edgecolor=G3,
                                  label="Mr. Hi's faction"),
                   mpatches.Patch(facecolor=CLASS_B, edgecolor=G3,
                                  label="Officer's faction")],
          loc="upper left", fontsize=8, framealpha=1.0, edgecolor=G6)
save(fig, "fig3_6_karate_club.png")


# =============================================================================
# PART 5 - Training the DeepWalk model
# =============================================================================
banner("PART 5 - Training the DeepWalk model")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)
model = Node2Vec(
    data.edge_index,
    embedding_dim=128,
    walk_length=10,
    context_size=10,
    walks_per_node=80,
    num_negative_samples=5,
    p=1.0,
    q=1.0,
    sparse=True,
).to(device)

with torch.no_grad():
    model.embedding.weight.uniform_(-0.5 / 128, 0.5 / 128)

loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)


def train_epoch():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


for epoch in range(1, 101):
    loss = train_epoch()
    if epoch == 1 or epoch % 20 == 0:
        print(f'Epoch {epoch:3d} | Loss: {loss:.4f}')


# =============================================================================
# PART 6 - Similar nodes
# =============================================================================
banner("PART 6 - Similar nodes")

model.eval()
with torch.no_grad():
    embeddings = model().detach()

emb_norm = F.normalize(embeddings, dim=1)
node0_sim = (emb_norm @ emb_norm[0]).cpu().numpy()

top10 = np.argsort(-node0_sim)[1:11]
print('Nodes most similar to node 0:')
for n in top10:
    print(f'   ({n}, {node0_sim[n]:.4f})')

sim_0_4 = F.cosine_similarity(embeddings[0].unsqueeze(0),
                              embeddings[4].unsqueeze(0))
print(f'Similarity between node 0 and 4: {sim_0_4.item():.4f}')


# =============================================================================
# PART 7 - Visualizing the embeddings
# =============================================================================
banner("PART 7 - Visualizing the embeddings")

emb_np = embeddings.cpu().numpy()
tsne_proj = TSNE(n_components=2, init='pca', perplexity=10,
                 learning_rate='auto', random_state=0).fit_transform(emb_np)
umap_proj = umap.UMAP(n_components=2,
                      random_state=0).fit_transform(emb_np)

# The two panels stay side by side, as the caption says. They are 2.75in each,
# where a number inside every marker cannot be set above 6.5 pt and collides
# with its neighbour's; the chapter's own code does not draw those numbers
# either. Only node 8 is named, because the text singles it out, and it is
# annotated beside its marker rather than inside it.
fig, axes = figure(aspect=0.46, ncols=2)
for ax, proj, title in zip(axes, [tsne_proj, umap_proj], ["t-SNE", "UMAP"]):
    ax.axis("off")
    ax.set_title(title, fontweight="bold", color=INK, pad=4, fontfamily=FONT)
    ax.scatter(proj[labels == 0, 0], proj[labels == 0, 1], s=100,
               color=CLASS_A, edgecolors=INK, linewidths=0.6,
               label="Mr. Hi", zorder=3)
    ax.scatter(proj[labels == 1, 0], proj[labels == 1, 1], s=100,
               color=CLASS_B, edgecolors=INK, linewidths=0.6,
               label="Officer", zorder=3)
    ax.annotate("node 8", xy=proj[8], xytext=(12, 10),
                textcoords="offset points", fontsize=8, color=INK,
                fontfamily=FONT, zorder=5,
                arrowprops=dict(arrowstyle="-", color=INK_SOFT, lw=0.7))
    ax.margins(0.16)
    ax.legend(fontsize=8, framealpha=1.0, edgecolor=G6, loc="best")
save(fig, "fig3_7_tsne_umap.png")


# =============================================================================
# PART 8 - Node classification
# =============================================================================
banner("PART 8 - Node classification")

X_train, X_test, y_train, y_test = train_test_split(
    emb_np, labels,
    test_size=0.5, stratify=labels, random_state=0
)

clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))