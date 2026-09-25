import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FONT="DejaVu Sans"; BG="white"
G0="#222222"; G2="#666666"; G4="#AAAAAA"   # GCN darkest, GNN mid, MLP light
BLACK="#1A1A1A"

datasets = ["Cora", "Facebook"]
models   = ["MLP", "Vanilla GNN", "GCN"]
means = {
    "MLP":         [53.77, 77.45],
    "Vanilla GNN": [75.09, 86.58],
    "GCN":         [80.41, 90.00],
}
stds = {
    "MLP":         [1.41, 0.25],
    "Vanilla GNN": [1.33, 1.50],
    "GCN":         [0.53, 0.13],
}
colors = {"MLP": G4, "Vanilla GNN": G2, "GCN": G0}

x = np.arange(len(datasets))
w = 0.26

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

for i, m in enumerate(models):
    offset = (i - 1) * w
    bars = ax.bar(x + offset, means[m], w, yerr=stds[m], capsize=4,
                  color=colors[m], edgecolor="white", linewidth=0.8,
                  label=m, error_kw=dict(ecolor=BLACK, lw=1.2))
    for b, mu in zip(bars, means[m]):
        ax.text(b.get_x() + b.get_width()/2, mu + 2.0, f"{mu:.1f}",
                ha="center", va="bottom", fontsize=9, color=BLACK,
                fontfamily=FONT, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=11, fontfamily=FONT)
ax.set_ylabel("Mean test accuracy (%)", fontsize=11, fontfamily=FONT)
ax.set_ylim(40, 100)
ax.set_title("Mean test accuracy over 20 runs — MLP vs Vanilla GNN vs GCN",
             fontsize=12, fontweight="bold", color=BLACK, fontfamily=FONT, pad=12)
ax.legend(fontsize=10, framealpha=0.95, edgecolor="#CCCCCC")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/mlp_gnn_gcn_comparison.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
print("saved")
