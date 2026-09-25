import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import networkx as nx
import random, os

random.seed(0)
np.random.seed(0)

OUT   = os.path.dirname(os.path.abspath(__file__))
FONT  = "DejaVu Sans"
BG    = "white"
BLUE  = "#2E75B6"
ORANGE= "#C55A11"
GRAY  = "#595959"
LGRAY = "#EDEDED"
BLACK = "#1A1A1A"

from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

G = nx.karate_club_graph()
labs = np.array([1 if G.nodes[n]['club']=='Officer' else 0 for n in G.nodes])

def next_node_biased(G, previous, current, p, q):
    neighbors = list(G.neighbors(current))
    alphas = []
    for nbr in neighbors:
        if nbr == previous:
            alphas.append(1/p)
        elif previous is not None and G.has_edge(nbr, previous):
            alphas.append(1.0)
        else:
            alphas.append(1/q)
    probs = [a/sum(alphas) for a in alphas]
    return np.random.choice(neighbors, p=probs)

def random_walk_biased(G, start, length, p, q):
    walk = [start]
    for _ in range(length):
        current  = walk[-1]
        previous = walk[-2] if len(walk) > 1 else None
        nbrs = list(G.neighbors(current))
        if not nbrs: break
        walk.append(next_node_biased(G, previous, current, p, q))
    return [str(x) for x in walk]

p_values = [1, 2, 4, 7]
q_values = [1, 2, 4, 7]
N_RUNS   = 10   # reduced for speed

print("Running p/q grid …")
results = np.zeros((len(p_values), len(q_values)))

for pi, p in enumerate(p_values):
    for qi, q in enumerate(q_values):
        accs = []
        for run in range(N_RUNS):
            np.random.seed(run); random.seed(run)
            walks = []
            for node in G.nodes:
                for _ in range(80):
                    walks.append(random_walk_biased(G, node, 10, p, q))

            model = Word2Vec(
                sentences=walks, vector_size=64, window=5,
                min_count=0, sg=1, hs=0, negative=5,
                workers=1, seed=run, epochs=20
            )
            emb = np.array([model.wv[str(i)] for i in range(len(G.nodes))])

            X_tr, X_te, y_tr, y_te = train_test_split(
                emb, labs, test_size=0.4, stratify=labs, random_state=run)
            clf = RandomForestClassifier(n_estimators=50, random_state=run)
            clf.fit(X_tr, y_tr)
            accs.append(accuracy_score(y_te, clf.predict(X_te)))

        results[pi, qi] = np.mean(accs)
        print(f"  p={p}, q={q} → {results[pi,qi]:.3f}")

# Plot
fig, ax = plt.subplots(figsize=(7, 5.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

im = ax.imshow(results, cmap='Blues', vmin=0.70, vmax=0.96,
               aspect='auto', origin='upper')

for i in range(len(p_values)):
    for j in range(len(q_values)):
        val = results[i, j]
        tc  = 'white' if val > 0.87 else BLACK
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=12, color=tc, fontweight='bold', fontfamily=FONT)

ax.set_xticks(range(len(q_values)))
ax.set_yticks(range(len(p_values)))
ax.set_xticklabels([f'q={q}' for q in q_values], fontsize=11, fontfamily=FONT)
ax.set_yticklabels([f'p={p}' for p in p_values], fontsize=11, fontfamily=FONT)
ax.set_xlabel('In-out parameter q  (lower → more exploration / DFS)\n'
              'Orange box: DeepWalk (p = q = 1)',
              fontsize=11, color=BLACK, fontfamily=FONT, labelpad=8)
ax.set_ylabel('Return parameter p  (higher → less backtracking)',
              fontsize=11, color=BLACK, fontfamily=FONT, labelpad=8)
ax.set_title("Mean accuracy for different p and q values\n"
             "Zachary's Karate Club — 10 runs, 60/40 train/test split",
             fontsize=11, fontweight='bold', color=BLACK,
             fontfamily=FONT, pad=10)

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label('Accuracy', fontsize=10, fontfamily=FONT)

# Highlight DeepWalk cell (labelled in the x-axis caption to avoid overlap)
rect = Rectangle((-0.5,-0.5), 1, 1, linewidth=2.5,
                 edgecolor=ORANGE, facecolor='none')
ax.add_patch(rect)

fig.tight_layout()
path = f"{OUT}/fig4_5_pq_heatmap.png"
fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close(fig)
print(f"  saved fig4_5_pq_heatmap.png")
