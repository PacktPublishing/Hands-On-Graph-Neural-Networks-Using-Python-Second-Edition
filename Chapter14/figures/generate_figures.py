"""
Chapter 14 – All figures in grayscale.
Figs 14.1–14.3, 14.9, 14.13: pure matplotlib diagrams.
Figs 14.4–14.8, 14.10–14.12, 14.14–14.15: synthetic data matching
published dataset statistics (no external downloads required).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import os

np.random.seed(0)

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
    for li, line in enumerate(label.split('\n')):
        offset = 0.11*(li - label.count('\n')/2)
        ax.text(x, y-offset, line, ha='center', va='center',
                fontsize=fs, color=fc, fontweight='bold',
                fontfamily=FONT, zorder=4)

def arr(ax, x1, y1, x2, y2, color=G2, lw=1.4):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12), zorder=2)


# ── Fig 14.1: EvolveGCN overview ──────────────────────────────────────────────
print("Figure 14.1 …")
fig, ax = plt.subplots(figsize=(12, 4.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 12); ax.set_ylim(0, 4.5)

for i, (x, lbl, fill) in enumerate([
    (1.5,  "Graph\nt=0", G1),
    (3.5,  "Graph\nt=1", G2),
    (5.5,  "Graph\nt=2", G2),
    (7.5,  "Graph\n...",  G3),
    (9.5,  "Graph\nt=T", G4),
]):
    rbox(ax, x, 3.0, 1.6, 1.0, lbl, fill=fill, fc='white', fs=10)
    if i > 0:
        arr(ax, x-1.0+0.8, 3.0, x-0.8, 3.0, color=G2)

# GCN evolves
for x in [1.5, 3.5, 5.5, 7.5, 9.5]:
    arr(ax, x, 2.5, x, 1.9, color=G1)
    rbox(ax, x, 1.4, 1.6, 0.85, "GCN(t)\nembeddings", fill=G0, fc='white', fs=8)

# RNN evolution arrows between GCNs
for x in [1.5, 3.5, 5.5, 7.5]:
    ax.annotate("", xy=(x+1.8, 1.4), xytext=(x+0.8, 1.4),
                arrowprops=dict(arrowstyle="-|>", color=G3,
                                lw=1.2, mutation_scale=10,
                                connectionstyle="arc3,rad=-0.3"))

ax.text(5.5, 0.45, "RNN updates GCN weight matrices over time",
        ha='center', fontsize=9, color=G3, fontfamily=FONT, fontstyle='italic')
ax.set_title("EvolveGCN — the GCN weight matrices evolve over time via an RNN",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig14_1_evolvegcn_overview.png")


# ── Fig 14.2: EvolveGCN-H ─────────────────────────────────────────────────────
print("Figure 14.2 …")
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 11); ax.set_ylim(0, 5)

for i, x in enumerate([1.5, 5.5, 9.5]):
    t = f"t={i}"
    rbox(ax, x, 4.0, 2.0, 0.75, f"H^l(t)  W^l(t)\n{t}", fill=G1, fc='white', fs=9)
    rbox(ax, x, 2.4, 2.0, 0.75, f"GRU\n{t}", fill=G3, fc='white', fs=9)
    rbox(ax, x, 0.9, 2.0, 0.75, f"GCNConv\n{t}", fill=G0, fc='white', fs=9)
    arr(ax, x, 3.6, x, 2.78, color=G1)
    arr(ax, x, 2.0, x, 1.28, color=G3)

for x1, x2 in [(1.5, 5.5), (5.5, 9.5)]:
    arr(ax, x1+1.0, 2.4, x2-1.0, 2.4, color=G2)
    arr(ax, x1+1.0, 0.9, x2-1.0, 0.9, color=G2)

ax.text(5.5, 0.12, "Node embeddings propagate forward in time",
        ha='center', fontsize=8.5, color=G3, fontfamily=FONT, fontstyle='italic')
ax.set_title("EvolveGCN-H — GRU receives both node embeddings and GCN weights",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig14_2_evolvegcn_h.png")


# ── Fig 14.3: EvolveGCN-O ─────────────────────────────────────────────────────
print("Figure 14.3 …")
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 11); ax.set_ylim(0, 5)

for i, x in enumerate([1.5, 5.5, 9.5]):
    t = f"t={i}"
    rbox(ax, x, 3.8, 2.0, 0.75, f"W^l(t-1)\n{t}", fill=G2, fc='white', fs=9)
    rbox(ax, x, 2.4, 2.0, 0.75, f"LSTM\n{t}", fill=G3, fc='white', fs=9)
    rbox(ax, x, 1.0, 2.0, 0.75, f"GCNConv\n{t}", fill=G0, fc='white', fs=9)
    arr(ax, x, 3.4, x, 2.78, color=G2)
    arr(ax, x, 2.0, x, 1.38, color=G3)

for x1, x2 in [(1.5, 5.5), (5.5, 9.5)]:
    arr(ax, x1+1.0, 2.4, x2-1.0, 2.4, color=G2)

ax.text(5.5, 0.22, "LSTM only receives GCN weights (no node embeddings)",
        ha='center', fontsize=8.5, color=G3, fontfamily=FONT, fontstyle='italic')
ax.set_title("EvolveGCN-O — LSTM receives only GCN weight matrices",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig14_3_evolvegcn_o.png")


# ── Fig 14.4: WikiMaths graph ─────────────────────────────────────────────────
print("Figure 14.4 …")
rng = np.random.default_rng(7)
G_wm = nx.barabasi_albert_graph(120, m=4, seed=7)
deg_wm = dict(G_wm.degree())
pos_wm = nx.spring_layout(G_wm, seed=3, k=0.5)
nc_wm  = [(0.1 + 0.7*(deg_wm[n]/max(deg_wm.values())),)*3 for n in G_wm.nodes()]
ns_wm  = [10 + deg_wm[n]*4 for n in G_wm.nodes()]

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
nx.draw_networkx_edges(G_wm, pos_wm, ax=ax, alpha=0.12, edge_color=G4, width=0.6)
nx.draw_networkx_nodes(G_wm, pos_wm, ax=ax, node_color=nc_wm,
                       node_size=ns_wm, edgecolors='white', linewidths=0.4)
sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys,
                            norm=plt.Normalize(0, max(deg_wm.values())))
sm.set_array([])
fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02,
             label='Node degree (number of Wikipedia links)')
ax.set_title("WikiMaths dataset — 1,068 Wikipedia articles (illustrative subgraph)\n"
             "Node shade proportional to number of links (t = 0)",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
ax.text(0.02, 0.02, "Illustrative — run chapter14.py locally for real graph",
        transform=ax.transAxes, fontsize=7.5, color=G3, fontstyle='italic')
fig.tight_layout()
save(fig, "fig14_4_wikimaths_graph.png")


# ── Helper: synthetic WikiMaths time series ───────────────────────────────────
t = np.arange(731)
trend    = 0.0003 * t
seasonal = 0.18 * np.sin(2*np.pi*t/365) + 0.08*np.sin(2*np.pi*t/7)
noise    = np.random.normal(0, 0.12, 731)
mean_ts  = trend + seasonal + noise + 0.5
std_ts   = 0.25 + 0.05*np.sin(2*np.pi*t/365)
roll7    = np.convolve(mean_ts, np.ones(7)/7, mode='same')
split    = 360

# ── Fig 14.5: WikiMaths time series ───────────────────────────────────────────
print("Figure 14.5 …")
fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(t, mean_ts, color=G1, linewidth=0.8, label='Mean', alpha=0.8)
ax.plot(t, roll7,   color=G0, linewidth=2.0, label='7-day moving average')
ax.fill_between(t, mean_ts-std_ts, mean_ts+std_ts,
                color=G4, alpha=0.25, label='±1 std dev')
ax.axvline(x=split, color=G2, linestyle='--', linewidth=1.5)
ax.text(split+5, mean_ts.max()*0.9, 'Train/test split',
        rotation=90, color=G2, fontsize=9, fontfamily=FONT, va='top')
ax.set_xlabel('Time (days)', fontsize=11, fontfamily=FONT)
ax.set_ylabel('Normalised number of visits', fontsize=11, fontfamily=FONT)
ax.set_title('WikiMaths — mean normalised number of visits',
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
ax.legend(fontsize=10); ax.spines[['top','right']].set_visible(False)
ax.grid(linestyle=':', alpha=0.5)
ax.text(0.98, 0.97, "Illustrative — run chapter14.py locally for real values",
        transform=ax.transAxes, ha='right', va='top',
        fontsize=7.5, color=G3, fontstyle='italic')
fig.tight_layout()
save(fig, "fig14_5_wikimaths_ts.png")


# ── Fig 14.6: WikiMaths with predictions ──────────────────────────────────────
print("Figure 14.6 …")
preds = roll7[split:] + np.random.normal(0, 0.08, len(roll7[split:]))

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(t, mean_ts, color=G1, linewidth=0.8, label='Mean', alpha=0.8)
ax.plot(t, roll7,   color=G0, linewidth=2.0, label='Moving average')
ax.fill_between(t, mean_ts-std_ts, mean_ts+std_ts,
                color=G4, alpha=0.25, label='±1 std dev')
ax.plot(t[split:], preds, color=G2, linewidth=1.8,
        linestyle='--', label='EvolveGCN prediction')
ax.axvline(x=split, color=G3, linestyle='--', linewidth=1.5)
ax.text(split+5, mean_ts.max()*0.9, 'Train/test split',
        rotation=90, color=G3, fontsize=9, fontfamily=FONT, va='top')
ax.set_xlabel('Time (days)', fontsize=11, fontfamily=FONT)
ax.set_ylabel('Normalised number of visits', fontsize=11, fontfamily=FONT)
ax.set_title('WikiMaths — EvolveGCN predicted mean normalised visits',
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
ax.legend(fontsize=10); ax.spines[['top','right']].set_visible(False)
ax.grid(linestyle=':', alpha=0.5)
ax.text(0.98, 0.97, "Illustrative — run chapter14.py locally for real values",
        transform=ax.transAxes, ha='right', va='top',
        fontsize=7.5, color=G3, fontstyle='italic')
fig.tight_layout()
save(fig, "fig14_6_wikimaths_pred.png")


# ── Fig 14.7: WikiMaths scatter (no seaborn) ──────────────────────────────────
print("Figure 14.7 …")
y_true_wm = mean_ts[split:split+50]
y_pred_wm = y_true_wm + np.random.normal(0, 0.15, 50)
m_fit = np.polyfit(y_true_wm, y_pred_wm, 1)
x_line = np.linspace(y_true_wm.min(), y_true_wm.max(), 100)

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.scatter(y_true_wm, y_pred_wm, color=G1, alpha=0.6, s=40, edgecolors='none')
ax.plot(x_line, np.polyval(m_fit, x_line), color=G0, linewidth=2.0,
        label='Regression line')
ax.plot([x_line[0],x_line[-1]], [x_line[0],x_line[-1]],
        color=G4, linewidth=1.2, linestyle='--', label='Perfect prediction')
ax.set_xlabel('Ground truth (normalised visits)', fontsize=11, fontfamily=FONT)
ax.set_ylabel('Predicted value', fontsize=11, fontfamily=FONT)
ax.set_title('WikiMaths — predicted vs ground truth (single snapshot)',
             fontsize=11, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
ax.legend(fontsize=10); ax.spines[['top','right']].set_visible(False)
ax.text(0.02, 0.97, "Illustrative — run chapter14.py locally",
        transform=ax.transAxes, ha='left', va='top',
        fontsize=7.5, color=G3, fontstyle='italic')
fig.tight_layout()
save(fig, "fig14_7_wikimaths_scatter.png")


# ── Fig 14.8: England regions graph ───────────────────────────────────────────
print("Figure 14.8 …")
rng2 = np.random.default_rng(42)
G_eng = nx.watts_strogatz_graph(129, k=6, p=0.3, seed=42)
pos_eng = nx.kamada_kawai_layout(G_eng)
deg_eng = dict(G_eng.degree())
nc_eng  = [(0.1 + 0.7*(deg_eng[n]/max(deg_eng.values())),)*3 for n in G_eng.nodes()]
ns_eng  = [30 + deg_eng[n]*8 for n in G_eng.nodes()]
edge_w  = [0.2 + rng2.random()*0.8 for _ in G_eng.edges()]

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
nx.draw_networkx_edges(G_eng, pos_eng, ax=ax, alpha=0.15,
                       edge_color=G4, width=0.7)
nx.draw_networkx_nodes(G_eng, pos_eng, ax=ax, node_color=nc_eng,
                       node_size=ns_eng, edgecolors='white', linewidths=0.5)
ax.set_title("England Covid dataset — 129 NUTS 3 regions (nodes)\n"
             "Edges represent population movement between regions (dynamic weights)",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
ax.text(0.02, 0.02, "Illustrative — spatial arrangement is schematic",
        transform=ax.transAxes, fontsize=7.5, color=G3, fontstyle='italic')
fig.tight_layout()
save(fig, "fig14_8_england_graph.png")


# ── Fig 14.9: MPNN-LSTM architecture ─────────────────────────────────────────
print("Figure 14.9 …")
fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 13); ax.set_ylim(0, 5)

boxes = [
    (1.2, 2.5, "Graph input\nx_t, edge_index\nedge_weight", G1),
    (3.5, 2.5, "GCN layer 1\n+ BatchNorm\n+ Dropout",      G2),
    (5.9, 2.5, "GCN layer 2\n+ BatchNorm\n+ Dropout",      G2),
    (8.3, 2.5, "2-layer\nLSTM",                             G0),
    (10.7,2.5, "Linear\n+ tanh",                            G3),
]
for x, y, lbl, fill in boxes:
    rbox(ax, x, y, 1.9, 1.5, lbl, fill=fill, fc='white', fs=9)

for i in range(len(boxes)-1):
    arr(ax, boxes[i][0]+0.95, 2.5, boxes[i+1][0]-0.95, 2.5, color=G2, lw=1.5)

# Time loop arrow back
ax.annotate("", xy=(1.2, 1.2), xytext=(10.7, 1.2),
            arrowprops=dict(arrowstyle="-|>", color=G3, lw=1.4,
                            mutation_scale=12,
                            connectionstyle="arc3,rad=0.0"))
ax.text(5.9, 0.6, "Repeat for each time step t=1…T",
        ha='center', fontsize=8.5, color=G3, fontfamily=FONT, fontstyle='italic')

ax.set_title("MPNN-LSTM architecture — GCN captures spatial, LSTM captures temporal",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig14_9_mpnn_lstm.png")


# ── Helper: synthetic England Covid time series ───────────────────────────────
n_snap = 71   # March 3 – May 12, 2020
t_cov  = np.arange(n_snap)
# Exponential growth then plateau — realistic COVID first wave shape
mean_cov = 0.05 * np.exp(0.08*t_cov) / (1 + 0.012*np.exp(0.08*t_cov))
mean_cov += np.random.normal(0, 0.02, n_snap)
std_cov   = 0.08 + 0.04*np.sin(2*np.pi*t_cov/14)
split_cov = int(0.8*n_snap)   # 57 train, 14 test

# ── Fig 14.10: England Covid time series ──────────────────────────────────────
print("Figure 14.10 …")
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(t_cov, mean_cov, color=G0, linewidth=1.6, label='Mean cases')
ax.fill_between(t_cov, mean_cov-std_cov, mean_cov+std_cov,
                color=G4, alpha=0.25, label='±1 std dev')
ax.axvline(x=split_cov, color=G2, linestyle='--', linewidth=1.5)
ax.text(split_cov+0.5, mean_cov.max()*0.9, 'Train/test split',
        rotation=90, color=G2, fontsize=9, fontfamily=FONT, va='top')
ax.set_xlabel('Days since 3 March 2020', fontsize=11, fontfamily=FONT)
ax.set_ylabel('Normalised number of cases', fontsize=11, fontfamily=FONT)
ax.set_title('England Covid dataset — mean normalised number of reported cases',
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
ax.legend(fontsize=10); ax.spines[['top','right']].set_visible(False)
ax.grid(linestyle=':', alpha=0.5)
ax.text(0.98, 0.97, "Illustrative — run chapter14.py locally for real values",
        transform=ax.transAxes, ha='right', va='top',
        fontsize=7.5, color=G3, fontstyle='italic')
fig.tight_layout()
save(fig, "fig14_10_covid_ts.png")


# ── Fig 14.11: COVID predictions ──────────────────────────────────────────────
print("Figure 14.11 …")
# Model learns mean but underestimates variance
cov_pred = np.full(n_snap - split_cov,
                   mean_cov[split_cov:].mean() + 0.03) + \
           np.random.normal(0, 0.04, n_snap - split_cov)

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(t_cov, mean_cov, color=G0, linewidth=1.6, label='True cases')
ax.fill_between(t_cov, mean_cov-std_cov, mean_cov+std_cov,
                color=G5, alpha=0.3, label='±1 std dev')
ax.plot(t_cov[split_cov:], cov_pred, color=G2, linewidth=2.0,
        linestyle='--', label='MPNN-LSTM prediction')
ax.axvline(x=split_cov, color=G3, linestyle='--', linewidth=1.5)
ax.text(split_cov+0.5, mean_cov.max()*0.9, 'Train/test split',
        rotation=90, color=G3, fontsize=9, fontfamily=FONT, va='top')
ax.set_xlabel('Days since 3 March 2020', fontsize=11, fontfamily=FONT)
ax.set_ylabel('Normalised number of cases', fontsize=11, fontfamily=FONT)
ax.set_title('England Covid — MPNN-LSTM predicted vs true (mean across 129 regions)',
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
ax.legend(fontsize=10); ax.spines[['top','right']].set_visible(False)
ax.grid(linestyle=':', alpha=0.5)
ax.text(0.98, 0.97, "Illustrative — run chapter14.py locally for real values",
        transform=ax.transAxes, ha='right', va='top',
        fontsize=7.5, color=G3, fontstyle='italic')
fig.tight_layout()
save(fig, "fig14_11_covid_pred.png")


# ── Fig 14.12: COVID scatter ──────────────────────────────────────────────────
print("Figure 14.12 …")
true_s  = mean_cov[split_cov:]
pred_s  = cov_pred
m_fit2  = np.polyfit(true_s, pred_s, 1)
xl2     = np.linspace(true_s.min(), true_s.max(), 100)

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.scatter(true_s, pred_s, color=G1, alpha=0.7, s=50, edgecolors='none')
ax.plot(xl2, np.polyval(m_fit2, xl2), color=G0, linewidth=2.0,
        label='Regression line')
ax.plot([xl2[0],xl2[-1]], [xl2[0],xl2[-1]],
        color=G4, linewidth=1.2, linestyle='--', label='Perfect prediction')
ax.set_xlabel('Ground truth (normalised cases)', fontsize=11, fontfamily=FONT)
ax.set_ylabel('Predicted value', fontsize=11, fontfamily=FONT)
ax.set_title('England Covid — predicted vs ground truth\n(first test snapshot)',
             fontsize=11, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
ax.legend(fontsize=10); ax.spines[['top','right']].set_visible(False)
ax.text(0.02, 0.97, "Illustrative — run chapter14.py locally",
        transform=ax.transAxes, ha='left', va='top',
        fontsize=7.5, color=G3, fontstyle='italic')
fig.tight_layout()
save(fig, "fig14_12_covid_scatter.png")


# ── Fig 14.13: TGN architecture ──────────────────────────────────────────────
print("Figure 14.13 …")
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
ax.set_xlim(0, 13); ax.set_ylim(0, 6)

# Event stream (left)
ax.text(0.8, 5.4, "Event stream", ha='center', fontsize=9,
        color=G3, fontfamily=FONT, fontstyle='italic')
for i, (y, lbl) in enumerate([(4.6,"(u,v,t₁)"), (3.8,"(v,w,t₂)"),
                                (3.0,"(u,w,t₃)"), (2.2,"...")]):
    rbox(ax, 0.9, y, 1.4, 0.55, lbl, fill=G2, fc='white', fs=9)

arr(ax, 1.6, 3.8, 2.2, 3.8, color=G2, lw=1.6)

# Memory module
rbox(ax, 3.2, 3.8, 1.8, 3.0,
     "Memory\nmodule\ns_u, s_v\n(per node)", fill=G0, fc='white', fs=9)
arr(ax, 4.1, 3.8, 4.8, 3.8, color=G1, lw=1.6)

# Message function
rbox(ax, 5.7, 3.8, 1.8, 1.2,
     "Message\nfunction\nm(s_u,s_v,Δt,e_uv)", fill=G1, fc='white', fs=8)
arr(ax, 6.6, 3.8, 7.3, 3.8, color=G2, lw=1.6)

# Memory updater (GRU)
rbox(ax, 8.2, 3.8, 1.8, 1.2,
     "Memory\nupdater\n(GRU)", fill=G2, fc='white', fs=9)
# Feedback from updater to memory
ax.annotate("", xy=(3.2, 2.2), xytext=(8.2, 2.4),
            arrowprops=dict(arrowstyle="-|>", color=G3, lw=1.3,
                            mutation_scale=11,
                            connectionstyle="arc3,rad=0.25"))

arr(ax, 9.1, 3.8, 9.9, 3.8, color=G2, lw=1.6)

# Embedding module
rbox(ax, 10.8, 3.8, 1.8, 1.2,
     "Embedding\nmodule\n(Graph Attn)", fill=G3, fc='white', fs=8)
arr(ax, 11.7, 3.8, 12.3, 3.8, color=G3, lw=1.6)
ax.text(12.6, 3.8, "z(t)", ha='left', va='center', fontsize=11,
        fontweight='bold', color=G0, fontfamily=FONT)

# Labels
ax.text(3.2, 5.5, "Memory", ha='center', fontsize=9,
        color=G3, fontfamily=FONT, fontstyle='italic')
ax.text(5.7, 5.1, "Message", ha='center', fontsize=9,
        color=G3, fontfamily=FONT, fontstyle='italic')
ax.text(8.2, 5.1, "Updater", ha='center', fontsize=9,
        color=G3, fontfamily=FONT, fontstyle='italic')
ax.text(10.8, 5.1, "Embedding", ha='center', fontsize=9,
        color=G3, fontfamily=FONT, fontstyle='italic')

ax.set_title("Temporal Graph Network (TGN) — four modules process a continuous event stream",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
fig.tight_layout()
save(fig, "fig14_13_tgn.png")


# ── Fig 14.14: TGN training loss curve ───────────────────────────────────────
print("Figure 14.14 …")
epochs    = np.arange(1, 51)
tgn_loss  = 0.70 * np.exp(-0.06*epochs) + 0.08 + \
            0.03*np.random.randn(50)
val_loss  = tgn_loss + 0.04 + 0.02*np.random.randn(50)

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(epochs, tgn_loss, color=G0, linewidth=2.0, label='Train loss')
ax.plot(epochs, val_loss,  color=G3, linewidth=2.0,
        linestyle='--', label='Val loss')
ax.set_xlabel('Epoch', fontsize=11, fontfamily=FONT)
ax.set_ylabel('Binary cross-entropy loss', fontsize=11, fontfamily=FONT)
ax.set_title('TGN training on Wikipedia interaction dataset',
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, pad=10)
ax.legend(fontsize=10); ax.spines[['top','right']].set_visible(False)
ax.grid(linestyle=':', alpha=0.4)
ax.text(0.98, 0.97, "Illustrative — run chapter14.py locally for real values",
        transform=ax.transAxes, ha='right', va='top',
        fontsize=7.5, color=G3, fontstyle='italic')
fig.tight_layout()
save(fig, "fig14_14_tgn_loss.png")


# ── Fig 14.15: Snapshot vs event-based comparison ────────────────────────────
print("Figure 14.15 …")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG)

t_axis = np.linspace(0, 10, 500)

# Left: snapshot-based (discrete steps)
ax = axes[0]; ax.set_facecolor(BG)
for snap_t in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    ax.axvline(x=snap_t, color=G3, linewidth=0.8, linestyle='-', alpha=0.6)
    ax.scatter([snap_t], [0.5], color=G0, s=120, zorder=3)
ax.plot(t_axis, 0.5 + 0*t_axis, color=G5, linewidth=0.5)
ax.set_xlim(0, 10); ax.set_ylim(0, 1)
ax.set_xlabel('Time', fontsize=11, fontfamily=FONT)
ax.set_title('Snapshot-based\n(EvolveGCN, MPNN-LSTM)', fontsize=11,
             fontweight='bold', color=G0, fontfamily=FONT, pad=8)
ax.text(0.5, 0.15, "Graph observed at fixed intervals only",
        ha='center', fontsize=8.5, color=G3, fontfamily=FONT,
        fontstyle='italic', transform=ax.transAxes)
ax.set_yticks([]); ax.spines[['top','right','left']].set_visible(False)

# Right: event-based (continuous stream)
ax = axes[1]; ax.set_facecolor(BG)
event_times = sorted(np.random.uniform(0, 10, 22))
for et in event_times:
    ax.axvline(x=et, color=G4, linewidth=0.6, linestyle=':', alpha=0.4)
    ax.scatter([et], [0.5], color=G1, s=80, zorder=3, marker='|', linewidths=2)
ax.plot(t_axis, 0.5 + 0*t_axis, color=G5, linewidth=0.5)
ax.set_xlim(0, 10); ax.set_ylim(0, 1)
ax.set_xlabel('Time', fontsize=11, fontfamily=FONT)
ax.set_title('Event-based\n(TGN)', fontsize=11,
             fontweight='bold', color=G0, fontfamily=FONT, pad=8)
ax.text(0.5, 0.15, "Model updates at every interaction event",
        ha='center', fontsize=8.5, color=G3, fontfamily=FONT,
        fontstyle='italic', transform=ax.transAxes)
ax.set_yticks([]); ax.spines[['top','right','left']].set_visible(False)

fig.suptitle("Snapshot-based vs event-based temporal modelling",
             fontsize=12, fontweight='bold', color=G0, fontfamily=FONT, y=0.97)
fig.tight_layout()
save(fig, "fig14_15_snapshot_vs_event.png")

print(f"\nAll figures saved to {OUT}")
