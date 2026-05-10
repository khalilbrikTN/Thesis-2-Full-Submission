"""
Presentation figure: Coverage comparison for a fixed budget.

Three configurations of 5 active links are shown side-by-side:
  A) Clustered   → poor  spatial coverage
  B) Parallel    → medium spatial coverage
  C) Diverse     → good  spatial coverage

Each panel shows:
  • All 10 sensor nodes (available candidates)
  • 5 active links + their Fresnel zones (ellipses)
  • A 22×22 grid of candidate coverage points, coloured green (covered)
    or grey (uncovered)
  • Coverage percentage = fraction of grid points in at least one Fresnel zone

Output: figures/fig_coverage_comparison.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, FancyBboxPatch
import matplotlib.patches as mpatches
from pathlib import Path

FIGURES = Path('figures')
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family'       : 'serif',
    'font.size'         : 9,
    'figure.dpi'        : 300,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
    'savefig.pad_inches': 0.05,
})

# ── Actual sensor node positions (10 nodes, metres) ────────────────────────────
nodes = np.array([
    [0.0000, 3.4544],   # 0  left-middle
    [0.8128, 0.0000],   # 1  bottom-left
    [4.0386, 0.3302],   # 2  bottom-centre
    [4.9276, 2.0320],   # 3  centre-low
    [8.1280, 0.7239],   # 4  bottom-right
    [8.1280, 3.9624],   # 5  right-middle
    [1.1684, 7.6835],   # 6  top-left
    [4.1656, 6.4262],   # 7  top-centre-left
    [5.2070, 6.3246],   # 8  top-centre-right
    [6.7564, 7.8232],   # 9  top-right
])

# ── Room extent (metres) ───────────────────────────────────────────────────────
RX1, RX2 = 0.0, 8.5
RY1, RY2 = 0.0, 8.2

# ── Grid: the discretisation of the monitored area ────────────────────────────
GRID_N  = 22                               # points per axis (22×22 = 484 total)
gx = np.linspace(0.20, 8.30, GRID_N)
gy = np.linspace(0.20, 8.00, GRID_N)
GX, GY   = np.meshgrid(gx, gy)
grid_pts = np.column_stack([GX.ravel(), GY.ravel()])    # (484, 2)

# ── Fresnel-zone coverage model ────────────────────────────────────────────────
# A grid point q is "covered" by link (A,B) if it falls inside the first
# Fresnel zone:  d(q,A) + d(q,B)  ≤  d(A,B) + λ_eff
# λ_eff = 0.40 m  (slightly exaggerated from 2.4 GHz for legibility)
LAMBDA_EFF = 0.40

def covered_by(pts, p1, p2, lam=LAMBDA_EFF):
    d  = np.linalg.norm(p2 - p1)
    d1 = np.linalg.norm(pts - p1, axis=1)
    d2 = np.linalg.norm(pts - p2, axis=1)
    return (d1 + d2) <= (d + lam)

def coverage_mask(links):
    mask = np.zeros(len(grid_pts), dtype=bool)
    for i, j in links:
        mask |= covered_by(grid_pts, nodes[i], nodes[j])
    return mask

def fresnel_ellipse(ax, p1, p2, room_clip, lam=LAMBDA_EFF):
    """Draw the Fresnel-zone ellipse for link p1→p2, clipped to room."""
    d   = np.linalg.norm(p2 - p1)
    a   = (d + lam) / 2
    c   = d / 2
    b   = np.sqrt(max(a**2 - c**2, 1e-9))
    ang = np.degrees(np.arctan2(*(p2 - p1)[::-1]))
    el  = Ellipse((p1 + p2) / 2, 2*a, 2*b, angle=ang,
                  fc='#756bb1', alpha=0.18, ec='#756bb1', lw=0.9,
                  ls='--', zorder=1)
    ax.add_patch(el)
    el.set_clip_path(room_clip)

# ── Three configurations (same budget: 5 active links) ────────────────────────
configs = [
    dict(
        label  = 'POOR COVERAGE',
        lcolor = '#cb181d',
        title  = 'Configuration A  —  Clustered links',
        sub    = 'All links confined to the bottom strip',
        links  = [(1, 4), (2, 4), (3, 4), (1, 3), (2, 3)],
    ),
    dict(
        label  = 'MEDIUM COVERAGE',
        lcolor = '#d95f0e',
        title  = 'Configuration B  —  Parallel links',
        sub    = 'Links run in similar directions, limited angular diversity',
        links  = [(6, 9), (0, 5), (1, 4), (7, 3), (8, 4)],
    ),
    dict(
        label  = 'GOOD COVERAGE',
        lcolor = '#238b45',
        title  = 'Configuration C  —  Spatially diverse links',
        sub    = 'Cross-cutting links span the entire monitored area',
        links  = [(6, 4), (1, 9), (0, 5), (6, 2), (9, 2)],
    ),
]

# ── Colours ────────────────────────────────────────────────────────────────────
C_NODE_ON  = '#2166ac'
C_NODE_OFF = '#aaaaaa'
C_ACTIVE   = '#111111'
C_COV      = '#41ab5d'
C_UCOV     = '#d4d4d4'
C_ROOM_BG  = '#f8f8f8'

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.8))

for ax, cfg in zip(axes, configs):
    links = cfg['links']
    mask  = coverage_mask(links)
    score = mask.mean()
    n_cov = int(mask.sum())
    n_tot = len(grid_pts)

    # ── Room background ────────────────────────────────────────────────────────
    room = Rectangle((RX1, RY1), RX2 - RX1, RY2 - RY1,
                     lw=1.8, edgecolor='#555', facecolor=C_ROOM_BG, zorder=0)
    ax.add_patch(room)

    # ── Grid points (discretisation) ──────────────────────────────────────────
    ax.scatter(grid_pts[~mask, 0], grid_pts[~mask, 1],
               s=14, c=C_UCOV, marker='s', lw=0, zorder=2)
    ax.scatter(grid_pts[ mask, 0], grid_pts[ mask, 1],
               s=14, c=C_COV,  marker='s', lw=0, zorder=3)

    # ── Fresnel zones ─────────────────────────────────────────────────────────
    for i, j in links:
        fresnel_ellipse(ax, nodes[i], nodes[j], room)

    # ── Active links (arrows) ─────────────────────────────────────────────────
    for i, j in links:
        ax.annotate('', xy=nodes[j], xytext=nodes[i],
                    arrowprops=dict(arrowstyle='->', color=C_ACTIVE,
                                   lw=1.3, shrinkA=5, shrinkB=5), zorder=4)

    # ── Sensor nodes ──────────────────────────────────────────────────────────
    active_set = set(n for pair in links for n in pair)
    for k, (x, y) in enumerate(nodes):
        on = k in active_set
        ax.plot(x, y, 'o',
                c=C_NODE_ON if on else C_NODE_OFF,
                ms=6.5, mec='black', mew=0.55, zorder=5)
        ax.text(x + 0.22, y + 0.18, str(k),
                fontsize=5.5,
                color=C_NODE_ON if on else C_NODE_OFF,
                fontweight='bold' if on else 'normal',
                ha='left', va='bottom', zorder=6)

    # ── Quality badge (coloured box inside axes, top-centre) ──────────────────
    ax.text(0.50, 0.975, cfg['label'],
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10, fontweight='bold', color=cfg['lcolor'],
            bbox=dict(boxstyle='round,pad=0.25',
                      fc='white', ec=cfg['lcolor'], lw=1.2, alpha=0.92),
            zorder=10)

    # ── Configuration title (inside axes, just below badge) ───────────────────
    ax.text(0.50, 0.895, cfg['title'],
            transform=ax.transAxes, ha='center', va='top',
            fontsize=8.5, fontweight='bold', color='#222', zorder=10)

    ax.text(0.50, 0.840, cfg['sub'],
            transform=ax.transAxes, ha='center', va='top',
            fontsize=7.5, color='#555', style='italic', zorder=10)

    # ── Coverage score (below room) ────────────────────────────────────────────
    ax.text(0.50, -0.030,
            f'Coverage  =  {score:.0%}   ({n_cov} / {n_tot} grid points)',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9, color='#222',
            bbox=dict(boxstyle='round,pad=0.3',
                      fc='white', ec='#ccc', lw=0.8, alpha=0.9))

    ax.set_xlim(RX1 - 0.5, RX2 + 0.5)
    ax.set_ylim(RY1 - 0.5, RY2 + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')

# ── Shared legend ──────────────────────────────────────────────────────────────
h_on   = plt.Line2D([0],[0], marker='o', color='w', mfc=C_NODE_ON,
                    ms=6.5, mec='black', mew=0.55, label='Active sensor node')
h_off  = plt.Line2D([0],[0], marker='o', color='w', mfc=C_NODE_OFF,
                    ms=6.5, mec='black', mew=0.55, label='Inactive candidate node')
h_link = plt.Line2D([0],[0], color=C_ACTIVE, lw=1.3,
                    label='Active link  (budget = 5)')
h_fz   = mpatches.Patch(fc='#756bb1', alpha=0.5, ec='#756bb1',
                         ls='--', lw=0.9, label='Fresnel zone')
h_cov  = mpatches.Patch(fc=C_COV,  label='Covered grid point')
h_ucov = mpatches.Patch(fc=C_UCOV, label='Uncovered grid point')

fig.legend(
    handles=[h_on, h_off, h_link, h_fz, h_cov, h_ucov],
    loc='lower center', ncol=6, fontsize=8.5,
    framealpha=0.97, edgecolor='#bbb',
    handlelength=1.4, handletextpad=0.5,
    borderpad=0.5, columnspacing=1.2,
    bbox_to_anchor=(0.5, 0.0),
)

fig.suptitle(
    'Deployment Coverage: Same Budget (5 Active Links) — Three Different Configurations',
    fontsize=12, fontweight='bold', y=0.99,
)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.12, top=0.94, wspace=0.06)

out = FIGURES / 'fig_coverage_comparison.png'
fig.savefig(out)
plt.close(fig)
print(f'Saved {out}')
