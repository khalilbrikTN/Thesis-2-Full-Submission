import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, Rectangle
from pathlib import Path

FIGURES = Path('figures')
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family'       : 'serif',
    'font.size'         : 8,
    'figure.dpi'        : 300,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
    'savefig.pad_inches': 0.05,
})

fig, ax = plt.subplots(figsize=(3.5, 2.5))

# ── Room ──────────────────────────────────────────────────────────────────────
RW, RH = 8.0, 5.0
room = Rectangle((0, 0), RW, RH,
                 linewidth=1.5, edgecolor='#333333', facecolor='#f6f6f6')
ax.add_patch(room)

# ── Colors ────────────────────────────────────────────────────────────────────
C_AP      = '#1a9641'   # green  — APs
C_MP      = '#d7191c'   # red    — MPs
C_FRESNEL = '#756bb1'   # purple — Fresnel zones
C_ACTIVE  = '#252525'   # near-black — active links
C_FADED   = '#bdbdbd'   # light grey — inactive links

# ── Node positions ────────────────────────────────────────────────────────────
aps = np.array([[0.5, 1.0],
                [0.5, 2.5],
                [0.5, 4.0]])

mps = np.array([[7.5, 1.0],
                [7.5, 2.5],
                [7.5, 4.0]])

# ── Helpers ───────────────────────────────────────────────────────────────────
def draw_link(p1, p2, color, lw, alpha=1.0):
    ax.annotate('', xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, alpha=alpha,
                                shrinkA=6, shrinkB=6),
                zorder=2)

def draw_fresnel(p1, p2, minor=0.85):
    cx, cy = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    dist   = np.hypot(dx, dy)
    angle  = np.degrees(np.arctan2(dy, dx))
    # Use 82 % of the link length so the ellipse ends clear the node markers,
    # then clip to the room rectangle so nothing bleeds through the walls.
    el = Ellipse((cx, cy), width=dist * 0.82, height=minor, angle=angle,
                 facecolor=C_FRESNEL, alpha=0.20,
                 edgecolor=C_FRESNEL, linewidth=0.7,
                 linestyle='--', zorder=1)
    ax.add_patch(el)
    el.set_clip_path(room)   # hard-clip to room boundary

# ── Inactive links ────────────────────────────────────────────────────────────
inactive = [(0, 0), (2, 2), (0, 1), (2, 1)]
for (i, j) in inactive:
    draw_link(aps[i], mps[j], C_FADED, lw=0.7, alpha=0.8)

# ── Active links + Fresnel zones (cross-diagonal — diverse angles) ────────────
active = [(0, 2), (1, 1), (2, 0)]
for (i, j) in active:
    draw_fresnel(aps[i], mps[j])
    draw_link(aps[i], mps[j], C_ACTIVE, lw=1.1)

# ── AP nodes ──────────────────────────────────────────────────────────────────
for k, (x, y) in enumerate(aps):
    ax.plot(x, y, 's', color=C_AP, markersize=6.5,
            markeredgecolor='black', markeredgewidth=0.5, zorder=5)
    ax.text(x - 0.18, y, f'AP{k+1}',
            ha='right', va='center', fontsize=6, color=C_AP)

# ── MP nodes ──────────────────────────────────────────────────────────────────
for k, (x, y) in enumerate(mps):
    ax.plot(x, y, 'o', color=C_MP, markersize=6.5,
            markeredgecolor='black', markeredgewidth=0.5, zorder=5)
    ax.text(x + 0.18, y, f'MP{k+1}',
            ha='left', va='center', fontsize=6, color=C_MP)

# ── Person (stick figure) inside the Fresnel overlap region ──────────────────
px, py = 5.1, 2.45
PC = '#111111'
# head
ax.add_patch(plt.Circle((px, py + 0.58), 0.21, color=PC, zorder=6))
# body
ax.plot([px, px], [py + 0.37, py - 0.32], color=PC, lw=1.3, zorder=6)
# arms
ax.plot([px - 0.30, px + 0.30], [py + 0.08, py + 0.08],
        color=PC, lw=1.3, zorder=6)
# legs
ax.plot([px, px - 0.24], [py - 0.32, py - 0.80], color=PC, lw=1.3, zorder=6)
ax.plot([px, px + 0.24], [py - 0.32, py - 0.80], color=PC, lw=1.3, zorder=6)

# "person" label
ax.text(px + 0.35, py + 0.62, 'person',
        fontsize=5.5, color='#444444', style='italic', va='center')

# ── Fresnel zone annotation ───────────────────────────────────────────────────
ax.annotate('Fresnel\nzone', fontsize=5.5, color=C_FRESNEL, style='italic',
            xy=(2.6, 3.55), xytext=(1.2, 4.55),
            va='center',
            arrowprops=dict(arrowstyle='->', color=C_FRESNEL,
                            lw=0.6, connectionstyle='arc3,rad=0.15'))

# ── Legend ────────────────────────────────────────────────────────────────────
ap_patch  = mpatches.Patch(facecolor=C_AP,      edgecolor='black',
                            linewidth=0.5, label='AP (transmitter)')
mp_patch  = mpatches.Patch(facecolor=C_MP,      edgecolor='black',
                            linewidth=0.5, label='MP (receiver)')
fz_patch  = mpatches.Patch(facecolor=C_FRESNEL, alpha=0.4,
                            edgecolor=C_FRESNEL, label='Fresnel zone')
ax.legend(handles=[ap_patch, mp_patch, fz_patch],
          loc='lower right', fontsize=5.5,
          framealpha=0.9, edgecolor='0.75',
          handlelength=0.9, handletextpad=0.4,
          borderpad=0.5, labelspacing=0.3)

# ── Frame ─────────────────────────────────────────────────────────────────────
ax.set_xlim(-1.1, RW + 1.1)
ax.set_ylim(-0.6, RH + 0.9)
ax.set_aspect('equal')
ax.axis('off')

fig.tight_layout()
fig.savefig(FIGURES / 'fig0_dfl_deployment.png')
plt.close(fig)
print('Saved fig0_dfl_deployment.png')
