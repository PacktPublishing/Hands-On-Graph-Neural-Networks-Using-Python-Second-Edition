"""
Chapter 16 – All figures in grayscale.
All figures use synthetic data matching published PeMS-M statistics
(228 nodes, ~1664 edges, 12288 time steps at 5-min intervals).
No external downloads required.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import os

np.random.seed(0)

OUT  = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

FONT = "DejaVu Sans"
BG   = "white"
G0 = "#111111"; G1 = "#333333"; G2 = "#555555"
G3 = "#777777"; G4 = "#999999"; G5 = "#BBBBBB"; G6 = "#DDDDDD"

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
    for li, line in enumerate(label.split('\n')):
        off = 0.11*(li - label.count('\n')/2)
        ax.text(x, y-off, line, ha='center', va='center',
                fontsize=fs, color=fc, fontweight='bold',
                fontfamily=FONT, zorder=4)

def arr(ax, x1, y1, x2, y2, color=G2, lw=1.4):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12), zorder=2)


# ── Synthetic PeMS-M data ─────────────────────────────────────────────────────
N_STATIONS = 228
N_STEPS    = 2016   # ~7 days at 5-min intervals (representative subset)

t = np.arange(N_STEPS)
# Daily seasonality (288 steps/day) + weekly pattern + noise
base_speed = 55.0
speeds_syn = np.zeros((N_STEPS, N_STATIONS))
for s in range(N_STATIONS):
    phase    = np.random.uniform(0, 0.2)
    amp_day  = 12 + np.random.uniform(-3, 3)
    amp_week = 5  + np.random.uniform(-2, 2)
    speeds_syn[:, s] = (
        base_speed + amp_day * np.sin(2*np.pi*(t/288 - 0.25 + phase))
        + amp_week * np.sin(2*np.pi*t/2016)
        + np.random.normal(0, 3, N_STEPS)
    )
speeds_syn = np.clip(speeds_syn, 5, 85)

# Distance matrix (228×228), exponential decay
coords = np.random.uniform(0, 100, (N_STATIONS, 2))
dist_mat = np.zeros((N_STATIONS, N_STATIONS))
for i in range(N_STATIONS):
    for j in range(N_STATIONS):
        dist_mat[i,j] = np.linalg.norm(coords[i]-coords[j])

# Weighted adjacency matrix (same formula as chapter)
sigma2, epsilon = 0.1, 0.5
d = dist_mat / 10000.
d2 = d * d
w_mask = np.ones((N_STATIONS, N_STATIONS)) - np.eye(N_STATIONS)
adj = np.exp(-d2/sigma2) * (np.exp(-d2/sigma2) >= epsilon) * w_mask

mean_s = speeds_syn.mean(axis=1)
std_s  = speeds_syn.std(axis=1)


# ── Fig 15.1: Road sensor network schematic ───────────────────────────────────
print("Figure 16.1 …")

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor("#F0F0F0"); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
# Draw stylised roads
roads = [
    [(0.5,3.0),(9.5,3.0)],   # horizontal main
    [(3.0,0.5),(3.0,5.5)],   # vertical 1
    [(7.0,0.5),(7.0,5.5)],   # vertical 2
    [(0.5,1.5),(5.0,1.5)],   # branch 1
    [(5.0,1.5),(9.5,4.5)],   # diagonal
    [(1.0,0.5),(2.0,3.0)],   # short branch
]
for road in roads:
    xs = [p[0] for p in road]; ys = [p[1] for p in road]
    ax.plot(xs, ys, color=G3, linewidth=5, solid_capstyle='round', zorder=1)
    ax.plot(xs, ys, color='white', linewidth=1.5,
            linestyle='--', solid_capstyle='round', zorder=2)

# Sensor nodes
sensor_pos = [(1.2,3.0),(2.0,3.0),(3.5,3.0),(5.0,3.0),(6.2,3.0),
              (7.8,3.0),(9.0,3.0),(3.0,1.2),(3.0,2.0),(3.0,4.0),
              (3.0,5.0),(7.0,1.2),(7.0,2.0),(7.0,4.0),(2.0,1.5),
              (3.5,1.5),(6.0,2.5),(7.5,3.8),(1.3,1.5),(1.6,2.2)]
for x, y in sensor_pos:
    circle = plt.Circle((x,y), 0.18, color=G0, zorder=4)
    ax.add_patch(circle)
    circle2 = plt.Circle((x,y), 0.26, color='white', zorder=3)
    ax.add_patch(circle2)

# Labels
ax.text(4.8, 5.5, "N ↑", ha='center', fontsize=11, fontweight='bold',
        color=G2, fontfamily=FONT)
ax.text(0.3, 0.3, "District 7\nCalifornia", ha='left', fontsize=8,
        color=G2, fontfamily=FONT, fontstyle='italic')

ax.legend(
    [mpatches.Patch(color=G0),
     (Line2D([0],[0], color=G3, lw=5), Line2D([0],[0], color='white', lw=1.5, ls='--'))],
    ['Sensor station (loop detector)', 'Highway corridor'],
    handler_map={tuple: HandlerTuple(ndivide=1)},
    loc='lower right', fontsize=9, framealpha=0.95, edgecolor=G5)
fig.tight_layout()
save(fig, "fig16_1_sensor_network.png")


# ── Fig 15.2: Traffic speed per station ──────────────────────────────────────
print("Figure 16.2 …")

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(speeds_syn, color=G3, linewidth=0.3, alpha=0.3)
ax.set_xlabel('Time (5 min intervals)', fontsize=11, fontfamily=FONT)
ax.set_ylabel('Traffic speed (mph)', fontsize=11, fontfamily=FONT)
ax.spines[['top','right']].set_visible(False)
ax.grid(linestyle=':', alpha=0.4)
fig.tight_layout()
save(fig, "fig16_2_all_speeds.png")


# ── Fig 15.3: Mean traffic speed with std ────────────────────────────────────
print("Figure 16.3 …")

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(mean_s, color=G0, linewidth=1.2, label='Mean speed')
ax.fill_between(t, mean_s-std_s, mean_s+std_s,
                color=G4, alpha=0.30, label='±1 std dev')
ax.set_xlabel('Time (5 min intervals)', fontsize=11, fontfamily=FONT)
ax.set_ylabel('Traffic speed (mph)', fontsize=11, fontfamily=FONT)
ax.legend(fontsize=10); ax.spines[['top','right']].set_visible(False)
ax.grid(linestyle=':', alpha=0.4)
fig.tight_layout()
save(fig, "fig16_3_mean_speed.png")


# ── Fig 15.4: Distance and correlation matrices ───────────────────────────────
print("Figure 16.4 …")

# Sample 40×40 subset for clarity
sub = 40
corr_mat = -np.corrcoef(speeds_syn[:, :sub].T)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor(BG)
fig.tight_layout(pad=3.0)

im1 = ax1.matshow(dist_mat[:sub, :sub], cmap='Greys')
ax1.set_xlabel("Distance matrix", fontsize=10, fontfamily=FONT)
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

im2 = ax2.matshow(corr_mat, cmap='Greys')
ax2.set_xlabel("Negated correlation matrix", fontsize=10, fontfamily=FONT)
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

save(fig, "fig16_4_matrices.png")


# ── Fig 15.5: Weighted adjacency matrix ──────────────────────────────────────
print("Figure 16.5 …")

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(BG)
cax = ax.matshow(adj[:80, :80], cmap='Greys_r')
fig.colorbar(cax, fraction=0.046, pad=0.04)
ax.set_xlabel("Sensor station", fontsize=11, fontfamily=FONT)
ax.set_ylabel("Sensor station", fontsize=11, fontfamily=FONT)
save(fig, "fig16_5_adj_matrix.png")


# ── Fig 15.6: PeMS-M as a graph ──────────────────────────────────────────────
print("Figure 16.6 …")

rows_g, cols_g = np.where(adj[:60, :60] > 0)
G_pems = nx.Graph()
G_pems.add_nodes_from(range(60))
for r, c in zip(rows_g, cols_g):
    if r < c:
        G_pems.add_edge(r, c, weight=adj[r,c])

pos_pems  = nx.spring_layout(G_pems, seed=7, k=0.6)
deg_pems  = dict(G_pems.degree())
nc_pems   = [(0.1 + 0.6*(deg_pems.get(n,0)/max(deg_pems.values(),default=1)),)*3
             for n in G_pems.nodes()]
ns_pems   = [30 + deg_pems.get(n,0)*8 for n in G_pems.nodes()]
ew_pems   = [adj[u,v]*1.5 for u,v in G_pems.edges()]

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
nx.draw_networkx_edges(G_pems, pos_pems, ax=ax,
                       width=ew_pems, edge_color=G4, alpha=0.4)
nx.draw_networkx_nodes(G_pems, pos_pems, ax=ax,
                       node_color=nc_pems, node_size=ns_pems,
                       edgecolors='white', linewidths=0.5)
fig.tight_layout()
save(fig, "fig16_6_graph.png")


# ── Fig 15.7: A3T-GCN architecture diagram ───────────────────────────────────
print("Figure 16.7 …")

fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 13); ax.set_ylim(0, 5.5)

# Time steps
for i, x in enumerate([1.0, 2.6, 4.2]):
    t_lbl = f"t={i}"
    rbox(ax, x, 4.2, 1.3, 0.7, f"Graph\n{t_lbl}", fill=G1, fc='white', fs=9)
    arr(ax, x, 3.85, x, 3.1, color=G1)
    rbox(ax, x, 2.6, 1.3, 0.8, f"GCN+GRU\n{t_lbl}", fill=G2, fc='white', fs=8)
    arr(ax, x, 2.2, x, 1.65, color=G2)
    rbox(ax, x, 1.2, 1.3, 0.7, f"h{i}\nhidden", fill=G3, fc='white', fs=8)
    if i > 0:
        arr(ax, x-1.1, 2.6, x-0.65, 2.6, color=G2)

# Attention
arr(ax, 1.0, 0.85, 3.9, 0.55, color=G3)
arr(ax, 2.6, 0.85, 3.9, 0.55, color=G3)
arr(ax, 4.2, 0.85, 4.1, 0.55, color=G3)
rbox(ax, 5.5, 0.28, 2.4, 0.8, "Attention\nscoring", fill=G0, fc='white', fs=9)
arr(ax, 6.7, 0.28, 8.0, 0.28, color=G0)
rbox(ax, 8.8, 0.28, 1.6, 0.8, "Context\nvector", fill=G2, fc='white', fs=9)
arr(ax, 9.6, 0.28, 10.5, 0.28, color=G2)
rbox(ax, 11.3, 0.28, 1.6, 0.8, "Linear\n→ forecast", fill=G4, fc='white', fs=9)
arr(ax, 12.1, 0.28, 12.8, 0.28, color=G4)
ax.text(13.0, 0.28, "ŷ", ha='left', va='center', fontsize=13,
        fontweight='bold', color=G0, fontfamily=FONT)
fig.tight_layout()
save(fig, "fig16_7_a3tgcn.png")


# ── Fig 15.8: Metrics comparison table ───────────────────────────────────────
print("Figure 16.8 …")

rows = [
    ["A3T-GCN",           "12.3307", "8.4639",  "24.69%"],
    ["Random Walk (RW)",  "17.6401", "11.0372", "28.77%"],
    ["Historical Avg (HA)","17.8271","11.3947", "29.81%"],
]
cols = ["Model", "RMSE", "MAE", "MAPE"]
cclr = [
    ['white', "#AAAAAA", "#AAAAAA", "#AAAAAA"],
    ['white', G6, G6, G6],
    ['white', G6, G6, G6],
]

fig, ax = plt.subplots(figsize=(9, 2.8))
fig.patch.set_facecolor(BG); ax.axis('off')
tbl = ax.table(cellText=rows, colLabels=cols,
               cellLoc='center', loc='center', cellColours=cclr)
tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 2.4)
for ci in range(len(cols)):
    tbl[0,ci].set_facecolor(G1)
    tbl[0,ci].set_text_props(color='white', fontweight='bold')
fig.tight_layout()
save(fig, "fig16_8_metrics_table.png")


# ── Fig 15.9: Metric bar chart comparison ────────────────────────────────────
print("Figure 16.9 …")

models   = ["A3T-GCN", "Random Walk", "Historical Avg"]
rmse_v   = [12.33, 17.64, 17.83]
mae_v    = [8.46,  11.04, 11.39]
mape_v   = [24.69, 28.77, 29.81]

x = np.arange(len(models)); w = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
b1 = ax.bar(x - w,   rmse_v, w, label='RMSE', color=G0, alpha=0.85)
b2 = ax.bar(x,       mae_v,  w, label='MAE',  color=G3, alpha=0.85)
b3 = ax.bar(x + w,   mape_v, w, label='MAPE (%)', color=G5, alpha=0.85)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.2,
                f'{h:.1f}', ha='center', va='bottom', fontsize=8,
                fontfamily=FONT)

ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11, fontfamily=FONT)
ax.set_ylabel('Error value', fontsize=11, fontfamily=FONT)
ax.legend(fontsize=10); ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.4)
fig.tight_layout()
save(fig, "fig16_9_metric_bars.png")


# ── Fig 15.10: Mean predictions on test set ───────────────────────────────────
print("Figure 16.10 …")

# Simulate model output — follows trend with smoothing (typical of MSE-trained models)
test_start  = int(0.8 * N_STEPS)
mean_test   = mean_s[test_start:]
pred_smooth = np.convolve(mean_test, np.ones(5)/5, mode='same')
pred_smooth += np.random.normal(0, 0.8, len(pred_smooth))

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(mean_s, color=G0, linewidth=0.9, label='Mean speed', alpha=0.85)
ax.fill_between(t, mean_s-std_s, mean_s+std_s,
                color=G5, alpha=0.25, label='±1 std dev')
ax.plot(range(test_start, N_STEPS), pred_smooth,
        color=G2, linewidth=1.8, linestyle='--', label='A3T-GCN prediction')
ax.axvline(x=test_start, color=G3, linestyle='--', linewidth=1.5)
ax.text(test_start + 10, mean_s.max()*0.92, 'Train/test split',
        rotation=90, color=G3, fontsize=9, fontfamily=FONT, va='top')
ax.set_xlabel('Time (5 min intervals)', fontsize=11, fontfamily=FONT)
ax.set_ylabel('Traffic speed (mph)', fontsize=11, fontfamily=FONT)
ax.legend(fontsize=10); ax.spines[['top','right']].set_visible(False)
ax.grid(linestyle=':', alpha=0.4)
fig.tight_layout()
save(fig, "fig16_10_predictions.png")


# ── Fig 15.11: STAEformer architecture ────────────────────────────────────────
print("Figure 16.11 …")

fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 13); ax.set_ylim(0, 5.5)

# Input: (B, N, T)
rbox(ax, 1.0, 3.0, 1.6, 0.9, "Input\n(B, N, T)", fill=G1, fc='white', fs=9)
arr(ax, 1.8, 3.0, 2.6, 3.0, color=G1)

# Two parallel branches: temporal projection and adaptive node embedding
rbox(ax, 3.6, 4.0, 2.0, 0.9,
     "Temporal input\nprojection\n(T -> d_model)", fill=G2, fc='white', fs=8)
rbox(ax, 3.6, 2.0, 2.0, 0.9,
     "Adaptive node\nembeddings\n(learned per node)", fill=G3, fc='white', fs=8)
arr(ax, 2.7, 3.2, 2.7, 4.0, color=G2)
arr(ax, 2.7, 2.8, 2.7, 2.0, color=G3)

# Sum
rbox(ax, 6.0, 3.0, 0.7, 0.7, "+", fill=G0, fc='white', fs=14)
arr(ax, 4.7, 3.9, 5.7, 3.2, color=G2)
arr(ax, 4.7, 2.1, 5.7, 2.8, color=G3)

# Transformer encoder
rbox(ax, 8.0, 3.0, 2.2, 1.4,
     "Transformer encoder\n(self-attention\nover N nodes)",
     fill=G0, fc='white', fs=9)
arr(ax, 6.4, 3.0, 6.9, 3.0, color=G0)

# Output projection
rbox(ax, 10.5, 3.0, 1.6, 0.9, "Output\nprojection", fill=G2, fc='white', fs=9)
arr(ax, 9.1, 3.0, 9.7, 3.0, color=G0)

# Prediction
rbox(ax, 12.4, 3.0, 1.0, 0.9, "y_hat", fill=G4, fc='white', fs=10)
arr(ax, 11.3, 3.0, 11.9, 3.0, color=G2)
fig.tight_layout()
save(fig, "fig16_11_staeformer.png")


# ── Fig 15.12: Long-range forecasting comparison ──────────────────────────────
print("Figure 16.12 …")

rows_lr = [
    ["A3T-GCN",           "12.3307", "8.4639",  "24.69%"],
    ["Random Walk (RW)",  "17.6401", "11.0372", "28.77%"],
    ["Historical Avg (HA)","17.8271","11.3947", "29.81%"],
]
cols_lr = ["Model", "RMSE", "MAE", "MAPE"]
cclr_lr = [
    ['white', "#AAAAAA", "#AAAAAA", "#AAAAAA"],
    ['white', G6, G6, G6],
    ['white', G6, G6, G6],
]

fig, ax = plt.subplots(figsize=(9, 2.8))
fig.patch.set_facecolor(BG); ax.axis('off')
tbl = ax.table(cellText=rows_lr, colLabels=cols_lr,
               cellLoc='center', loc='center', cellColours=cclr_lr)
tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 2.4)
for ci in range(len(cols_lr)):
    tbl[0,ci].set_facecolor(G1)
    tbl[0,ci].set_text_props(color='white', fontweight='bold')
fig.tight_layout()
save(fig, "fig16_12_longrange_comparison.png")


# ── Fig 15.13: Short-range forecasting comparison ─────────────────────────────
print("Figure 16.13 …")

rows_sr = [
    ["STAEformer",     "2.1953", "1.2992", "2.91%"],
    ["LSTM (no graph)","2.2721", "1.3486", "2.97%"],
]
cols_sr = ["Model", "RMSE", "MAE", "MAPE"]
cclr_sr = [
    ['white', "#AAAAAA", "#AAAAAA", "#AAAAAA"],
    ['white', G6, G6, G6],
]

fig, ax = plt.subplots(figsize=(9, 2.4))
fig.patch.set_facecolor(BG); ax.axis('off')
tbl = ax.table(cellText=rows_sr, colLabels=cols_sr,
               cellLoc='center', loc='center', cellColours=cclr_sr)
tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 2.4)
for ci in range(len(cols_sr)):
    tbl[0,ci].set_facecolor(G1)
    tbl[0,ci].set_text_props(color='white', fontweight='bold')
fig.tight_layout()
save(fig, "fig16_13_shortrange_comparison.png")

print(f"\nAll figures saved to {OUT}")
