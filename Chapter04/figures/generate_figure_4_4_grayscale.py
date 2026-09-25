"""Grayscale version of Figure 4.4 (Zachary's Karate Club)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

OUT  = os.path.dirname(os.path.abspath(__file__))
FONT = "DejaVu Sans"
BG   = "white"

# Two-tone grayscale: dark for Mr. Hi, light for Officer.
DARK  = "#404040"
LIGHT = "#A0A0A0"
LGRAY = "#D8D8D8"

G   = nx.karate_club_graph()
pos = nx.spring_layout(G, seed=0)

labels = [1 if G.nodes[n]["club"] == "Officer" else 0 for n in G.nodes]
colors = [DARK if l == 0 else LIGHT for l in labels]

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis("off")

nx.draw_networkx_edges(G, pos, ax=ax, edge_color=LGRAY, width=1.2, alpha=0.9)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600, node_color=colors,
                       edgecolors="white", linewidths=1.5)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="white",
                        font_weight="bold", font_family=FONT)

ax.legend(handles=[mpatches.Patch(color=DARK,  label="Mr. Hi's faction"),
                   mpatches.Patch(color=LIGHT, label="Officer's faction")],
          loc="lower right", fontsize=10, framealpha=0.9, edgecolor=LGRAY)

fig.tight_layout()
path = f"{OUT}/fig4_4_karate_club_gray.png"
fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none")
plt.close(fig)
print(f"Saved {path}")
