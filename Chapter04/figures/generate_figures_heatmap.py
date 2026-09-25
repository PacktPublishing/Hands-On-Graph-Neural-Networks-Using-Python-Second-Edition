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

# Shared grayscale palette (same values as chapters 06+).
G0 = "#111111"; G1 = "#333333"; G2 = "#555555"
G3 = "#777777"; G4 = "#999999"; G5 = "#BBBBBB"; G6 = "#DDDDDD"

INK       = G0        # titles, axis labels, dark cell annotations
CMAP      = "Greys"   # perceptually monotonic: higher accuracy → darker cell
HL_DARK   = G0        # highlight box drawn over a light cell
HL_LIGHT  = "white"   # highlight box drawn over a dark cell
# Cell luminance below which white ink reads better than dark ink; 0.18 is where
# the two contrast ratios cross. Replaces the old fixed 0.87 accuracy cut-off,
# which was tuned for the Blues ramp.
LUM_FLIP  = 0.18

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

# Hue is gone, so the tones carry the ranking on their own. The old 0.70-0.96
# window pushed all sixteen cells into the black end of Greys and flattened
# them out; fitting the ramp to the measured band instead would magnify
# differences of ~0.01 that are within run-to-run noise. 0.90-1.00 is a
# compromise: an absolute scale the reader can trust, on which the accuracies
# still separate.
VMIN, VMAX = 0.90, 1.00

im = ax.imshow(results, cmap=CMAP, vmin=VMIN, vmax=VMAX,
               aspect='auto', origin='upper')

def ink_on(val):
    """White or dark numerals, whichever contrasts better with that cell."""
    r, g, b = im.cmap(im.norm(val))[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 'white' if lum < LUM_FLIP else INK

for i in range(len(p_values)):
    for j in range(len(q_values)):
        val = results[i, j]
        tc  = ink_on(val)
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=12, color=tc, fontweight='bold', fontfamily=FONT)

ax.set_xticks(range(len(q_values)))
ax.set_yticks(range(len(p_values)))
ax.set_xticklabels([f'q={q}' for q in q_values], fontsize=11, fontfamily=FONT)
ax.set_yticklabels([f'p={p}' for p in p_values], fontsize=11, fontfamily=FONT)
ax.set_xlabel('In-out parameter q  (lower → more exploration / DFS)\n'
              'Outlined box: DeepWalk (p = q = 1)',
              fontsize=11, color=INK, fontfamily=FONT, labelpad=8)
ax.set_ylabel('Return parameter p  (higher → less backtracking)',
              fontsize=11, color=INK, fontfamily=FONT, labelpad=8)
ax.set_title("Mean accuracy for different p and q values\n"
             "Zachary's Karate Club — 10 runs, 60/40 train/test split",
             fontsize=11, fontweight='bold', color=INK,
             fontfamily=FONT, pad=10)

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label('Accuracy', fontsize=10, fontfamily=FONT)

# Highlight DeepWalk cell (labelled in the x-axis caption to avoid overlap).
# Without a hue to fall back on, the outline flips to white on dark cells so it
# stays visible whatever accuracy the p=q=1 run reaches.
hl = HL_LIGHT if ink_on(results[0, 0]) == 'white' else HL_DARK
rect = Rectangle((-0.5,-0.5), 1, 1, linewidth=2.5,
                 edgecolor=hl, facecolor='none')
ax.add_patch(rect)

fig.tight_layout()
path = f"{OUT}/fig4_5_pq_heatmap.png"
fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close(fig)
print(f"  saved fig4_5_pq_heatmap.png")
