"""
Chapter 9 – All figures in grayscale.
Figs 9.1–9.2: pure matplotlib diagrams (conceptual, no data).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

OUT  = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

FONT = "DejaVu Sans"
BG   = "white"
G0   = "#111111"; G1 = "#333333"; G2 = "#555555"
G3   = "#777777"; G4 = "#999999"; G5 = "#BBBBBB"; G6 = "#DDDDDD"

def save(fig, name, dpi=200):
    fig.savefig(f"{OUT}/{name}", dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"  saved {name}")

def rbox(ax, x, y, w, h, label, fill=G2, fc='white', fs=9, r=0.04):
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle=f"round,pad=0.02,rounding_size={r}",
                          facecolor=fill, edgecolor=G4,
                          linewidth=1.1, zorder=3)
    ax.add_patch(rect)
    # Draw the label as one block so matplotlib spaces the lines according to
    # the font size; a fixed offset per line overlaps at larger font sizes.
    ax.text(x, y, label, ha='center', va='center', linespacing=1.5,
            fontsize=fs, color=fc, fontweight='bold',
            fontfamily=FONT, zorder=4)

def arr(ax, x1, y1, x2, y2, color=G2, lw=1.4):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12), zorder=2)


# ── Fig 9.1: Ingestion pipeline from OGB to Neo4j to GDS to PyG ──────────────
print("Figure 9.1 …")
fig, ax = plt.subplots(figsize=(12, 3.2))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 12); ax.set_ylim(0, 3.2)

stages = [
    (1.4,  "OGB",   "PygNodePropPredDataset", G1),
    (3.6,  "CSV",   "papers.csv\ncitations.csv", G2),
    (5.8,  "Neo4j", "LOAD CSV\n:Paper, :CITES", G0),
    (8.0,  "GDS",   "in-memory\nprojected graph", G2),
    (10.2, "PyG",   "Data(x, y,\nedge_index)", G1),
]

for i, (x, title, subtitle, fill) in enumerate(stages):
    rbox(ax, x, 2.2, 1.7, 0.55, title, fill=fill, fc='white', fs=11)
    rbox(ax, x, 1.35, 1.7, 0.85, subtitle, fill=G6, fc=G0, fs=8)
    if i > 0:
        x_prev = stages[i-1][0]
        arr(ax, x_prev+0.85, 1.65, x-0.85, 1.65, color=G2, lw=1.6)

ax.text(6.0, 0.5, "The graph moves from the OGB download through disk "
        "into Neo4j, then into GDS, then into PyG.",
        ha='center', fontsize=9, color=G3, fontfamily=FONT, fontstyle='italic')
ax.set_title("Ingestion pipeline — OGBN-arxiv reaches PyG through Neo4j and GDS",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig09_1_ingestion_pipeline.png")


# ── Fig 9.2: PyG Remote Backend flow ─────────────────────────────────────────
print("Figure 9.2 …")
fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 11); ax.set_ylim(0, 6.5)

# Database (bottom)
rbox(ax, 5.5, 0.9, 3.4, 1.0, "Graph database\n(e.g., K\u00f9zu)",
     fill=G0, fc='white', fs=10)

# Two stores (middle row)
rbox(ax, 2.5, 3.0, 3.0, 1.2,
     "FeatureStore\nget_tensor(node_ids)", fill=G2, fc='white', fs=9)
rbox(ax, 8.5, 3.0, 3.0, 1.2,
     "GraphStore\nget_edge_index(nodes)", fill=G2, fc='white', fs=9)

# NeighborLoader (upper middle)
rbox(ax, 5.5, 4.9, 3.4, 1.1,
     "NeighborLoader\nsamples seeds, assembles batches",
     fill=G1, fc='white', fs=10)

# Output batch (top)
rbox(ax, 5.5, 6.05, 2.6, 0.5, "PyG mini-batch",
     fill=G3, fc='white', fs=10)

# DB up to the two stores
arr(ax, 4.4, 1.4, 3.3, 2.4, color=G3, lw=1.5)
arr(ax, 6.6, 1.4, 7.7, 2.4, color=G3, lw=1.5)
ax.text(3.4, 1.95, "features",  ha='right', fontsize=8.5,
        color=G3, fontfamily=FONT, fontstyle='italic')
ax.text(7.6, 1.95, "topology",  ha='left',  fontsize=8.5,
        color=G3, fontfamily=FONT, fontstyle='italic')

# Stores up to NeighborLoader
arr(ax, 3.3, 3.6, 4.5, 4.35, color=G2, lw=1.5)
arr(ax, 7.7, 3.6, 6.5, 4.35, color=G2, lw=1.5)

# NeighborLoader to output batch
arr(ax, 5.5, 5.45, 5.5, 5.8, color=G1, lw=1.6)

ax.text(5.5, 0.15, "The full graph never leaves the database — "
        "each mini-batch queries only the neighborhoods it needs.",
        ha='center', fontsize=9, color=G3, fontfamily=FONT, fontstyle='italic')
ax.set_title("PyG Remote Backend — training reads from the graph database on demand",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig09_2_remote_backend.png", dpi=300)

print(f"\nAll figures saved to {OUT}")
