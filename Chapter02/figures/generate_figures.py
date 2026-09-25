import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

NODE_BLUE = '#3a82d6'
EDGE_BLUE = '#5a9bd5'
R = 0.30

fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=300)
ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis('off'); ax.set_aspect('equal')

# ---- LEFT graph (disconnected) ----
L = {1:(2.3,4.0), 2:(0.9,1.2), 3:(3.5,1.2), 4:(4.7,2.6), 5:(4.7,1.2)}
L_edges = [(1,2),(1,3),(2,3),(4,5)]   # 4-5 now SOLID

# ---- RIGHT graph (connected) ----
Rg = {1:(8.0,4.0), 2:(6.6,1.2), 3:(9.2,1.2), 4:(10.4,2.6)}
R_edges = [(1,2),(1,3),(2,3),(1,4)]

def draw(pos, edges):
    for a, b in edges:
        (x1,y1), (x2,y2) = pos[a], pos[b]
        ax.plot([x1,x2], [y1,y2], color=EDGE_BLUE, lw=2.2, zorder=1,
                solid_capstyle='round')
    for n, (x,y) in pos.items():
        ax.add_patch(Circle((x,y), R, facecolor=NODE_BLUE, edgecolor='none', zorder=2))
        ax.text(x, y, str(n), color='white', ha='center', va='center',
                fontsize=12, fontweight='bold', zorder=3)

draw(L, L_edges)
draw(Rg, R_edges)

# subtle vertical divider like the original
ax.plot([5.8,5.8], [0.3,4.7], color='#cfd8e3', lw=1.0, zorder=0)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('figure_connected_vs_disconnected.png',
            dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')