"""
System-level architecture diagram for the QUBO-based DFL node placement solution.
Slide-quality output (300 DPI, 16:9 aspect ratio).
Output: figures/fig_system_diagram.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Canvas ─────────────────────────────────────────────────────────────────────
FW, FH = 14.0, 8.6
fig, ax = plt.subplots(figsize=(FW, FH), dpi=150)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')
BG = '#F0F4F8'
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── Palette ────────────────────────────────────────────────────────────────────
C_INPUT  = '#BFDBFE'   # blue-200
C_METRIC = '#FDE68A'   # amber-200
C_QUBO   = '#DDD6FE'   # violet-200
C_SOLVER = '#FBCFE8'   # pink-200
C_BUDGET = '#BBF7D0'   # green-200
C_OUTPUT = '#FED7AA'   # orange-200
C_KNN    = '#99F6E4'   # teal-200
C_LOOP   = '#EFF6FF'   # blue-50 (loop band)
INK      = '#0F172A'   # text
ARROW    = '#475569'   # arrow shaft
LOOP_C   = '#1D4ED8'   # loop border / label
BACK_C   = '#DC2626'   # feedback arrow

# ── Geometry ───────────────────────────────────────────────────────────────────
BH   = 0.90    # standard box height
BW   = 12.0    # main box width
LM   = 1.0     # left margin (left edge of boxes)
RM   = LM + BW # right edge of boxes  = 13.0
CX   = LM + BW / 2  # centre x = 7.0

# Y = bottom edge of each box
Y_DATA   = 7.20
Y_METRIC = 5.90
Y_QUBO   = 4.60
Y_SOLVER = 3.30
Y_OUT    = 2.00
Y_KNN    = 0.70

# Metric sub-boxes
MW  = (BW - 0.55) / 2
MX0 = LM
MX1 = LM + MW + 0.55

# Solver / Budget sub-boxes (inside loop row)
SLW  = 6.55    # solver box width
BUDW = 5.20    # budget box width
GAP  = BW - SLW - BUDW   # = 0.25
SLX  = LM
BUDX = LM + SLW + GAP    # = 1 + 6.55 + 0.25 = 7.80

# ── Helpers ────────────────────────────────────────────────────────────────────

def box(x, y, w, h=BH, color='white', ec=INK, lw=2.0, rad=0.13, z=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad={rad}',
        facecolor=color, edgecolor=ec,
        linewidth=lw, zorder=z))

def t(x, y, s, sz=11, w='normal', c=INK, ha='center', va='center', **kw):
    ax.text(x, y, s, fontsize=sz, fontweight=w,
            color=c, ha=ha, va=va, zorder=6, **kw)

def arr(x1, y1, x2, y2, c=ARROW, lw=2.1, ms=16):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                mutation_scale=ms), zorder=5)

# ── Binary Search Loop Background ─────────────────────────────────────────────
loop_yb = Y_SOLVER - 0.14
loop_yt = Y_QUBO   + BH + 0.14
ax.add_patch(FancyBboxPatch(
    (LM - 0.35, loop_yb), BW + 0.35 + 0.75, loop_yt - loop_yb,
    boxstyle='round,pad=0.05',
    facecolor=C_LOOP, edgecolor=LOOP_C,
    linewidth=2.2, linestyle='--', zorder=1, alpha=0.95))

t(LM - 0.10, loop_yt - 0.11,
  'Binary Search Loop  (7 iterations · ε = 0.01)',
  sz=8.5, c=LOOP_C, w='bold', ha='left')

# ── Row 1: RSS Fingerprint Database ───────────────────────────────────────────
box(LM, Y_DATA, BW, color=C_INPUT, lw=2.6)
t(CX, Y_DATA + BH*0.66, 'RSS Fingerprint Database', sz=14, w='bold')
t(CX, Y_DATA + BH*0.28,
  r'$N=10$ sensor nodes  ·  $K=21$ reference locations  ·  '
  r'$P = N(N-1) = 90$ directed AP-MP pairs',
  sz=10.5, c='#1E3A5F')

# ── Row 2: Metrics ─────────────────────────────────────────────────────────────
box(MX0, Y_METRIC, MW, color=C_METRIC, lw=2.0)
t(MX0 + MW/2, Y_METRIC + BH*0.67, r'Link Importance  $I_p$', sz=12, w='bold')
t(MX0 + MW/2, Y_METRIC + BH*0.28,
  r'$I_p = \mathrm{Var}_k\!\left[\,\mathrm{RSS}(k,i,j)\,\right]$',
  sz=10.5, c='#78350F')

box(MX1, Y_METRIC, MW, color=C_METRIC, lw=2.0)
t(MX1 + MW/2, Y_METRIC + BH*0.67, r'Link Redundancy  $R_{pq}$', sz=12, w='bold')
t(MX1 + MW/2, Y_METRIC + BH*0.28,
  r'$R_{pq} = |\,\mathrm{corr}(\mathrm{RSS}_p,\,\mathrm{RSS}_q)\,|$',
  sz=10.5, c='#78350F')

# ── Row 3: QUBO (inside loop) ─────────────────────────────────────────────────
box(LM, Y_QUBO, BW, color=C_QUBO, lw=2.4)
t(CX, Y_QUBO + BH*0.67, r'QUBO Objective  $Q(\mathbf{x},\,\alpha)$', sz=14, w='bold')
t(CX, Y_QUBO + BH*0.28,
  r'$Q(\mathbf{x},\alpha)\ =\ -\alpha\sum_p I_p\,x_p'
  r'\ +\ (1-\alpha)\sum_{p \neq q}R_{pq}\,x_p x_q$'
  r'  ·  $\alpha\in[0,1]$ balances importance vs. redundancy',
  sz=10.5, c='#3B0764')

# ── Row 4: Solver + Budget (inside loop) ──────────────────────────────────────
box(SLX, Y_SOLVER, SLW, color=C_SOLVER, lw=2.0)
t(SLX + SLW/2, Y_SOLVER + BH*0.70, 'QUBO Solver',           sz=12.5, w='bold')
t(SLX + SLW/2, Y_SOLVER + BH*0.40,
  'SA:  OpenJij SASampler  (1 000 sweeps · 100 reads)',       sz=9.5,  c='#831843')
t(SLX + SLW/2, Y_SOLVER + BH*0.16,
  r'QA:  D-Wave (projected)  $\tau = 20\,\mu\mathrm{s}$ per solve',
  sz=9.5, c='#831843')

box(BUDX, Y_SOLVER, BUDW, color=C_BUDGET, lw=2.0)
t(BUDX + BUDW/2, Y_SOLVER + BH*0.70, 'Budget Check', sz=12.5, w='bold')
t(BUDX + BUDW/2, Y_SOLVER + BH*0.40,
  r'AP count $\leq n=5$  and  MP count $\leq m=5$',
  sz=9.5, c='#14532D')
t(BUDX + BUDW/2, Y_SOLVER + BH*0.14,
  r'Feasible: $a \leftarrow \alpha$   ·   Infeasible: $b \leftarrow \alpha$',
  sz=9,  c='#166534')

# ── Row 5: Optimal Deployment ─────────────────────────────────────────────────
box(LM, Y_OUT, BW, color=C_OUTPUT, lw=2.4)
t(CX, Y_OUT + BH*0.67, r'Optimal Deployment  $\mathbf{x}^*$', sz=14, w='bold')
t(CX, Y_OUT + BH*0.28,
  r'Best feasible solution tracked across all binary search iterations'
  r'  ($\alpha^* = a_{\rm final}$,  $\approx 6$ active pairs)',
  sz=10.5, c='#431407')

# ── Row 6: k-NN Localization ──────────────────────────────────────────────────
box(LM, Y_KNN, BW, color=C_KNN, lw=2.4)
t(CX, Y_KNN + BH*0.67,
  r'$k$-NN Localization  ($k=3$, Euclidean on RSS vectors)',
  sz=13, w='bold')
t(CX, Y_KNN + BH*0.28,
  r'QA: $2.66\pm0.09\,\mathrm{m}$  ·  SA: $2.69\pm0.08\,\mathrm{m}$  ·  '
  r'QA per-solve speedup: $\mathbf{129\!\times}$  ·  Total search speedup: $\mathbf{5\,839\!\times}$',
  sz=10.5, c='#064E3B')

# ── Arrows ─────────────────────────────────────────────────────────────────────

# Data → two metrics (diverging diagonals)
arr(CX,          Y_DATA,          MX0 + MW/2, Y_METRIC + BH)
arr(CX,          Y_DATA,          MX1 + MW/2, Y_METRIC + BH)

# Two metrics → QUBO (converging diagonals)
arr(MX0 + MW/2,  Y_METRIC,        CX,         Y_QUBO   + BH)
arr(MX1 + MW/2,  Y_METRIC,        CX,         Y_QUBO   + BH)

# QUBO → Solver (straight down, left-aligned with solver centre)
arr(SLX + SLW/2, Y_QUBO,          SLX + SLW/2, Y_SOLVER + BH)

# Solver → Budget (short horizontal)
arr(SLX + SLW,   Y_SOLVER + BH/2, BUDX,         Y_SOLVER + BH/2)

# Solver+Budget → Optimal Deployment (centred)
arr(CX,          Y_SOLVER,        CX,           Y_OUT    + BH)

# Output → KNN
arr(CX,          Y_OUT,           CX,           Y_KNN    + BH)

# ── Feedback Loop Arrow (right side) ──────────────────────────────────────────
# From right edge of Budget box, go right, up, then back left into QUBO right edge
LX   = RM + 0.52          # x of the vertical segment
Y_LO = Y_SOLVER + BH*0.62 # start y (upper part of budget/solver row)
Y_HI = Y_QUBO   + BH*0.50 # end y (mid-height of QUBO box)

# Three line segments: right → up → left (back to RM)
ax.plot([RM, LX],    [Y_LO, Y_LO], color=BACK_C, lw=2.4, solid_capstyle='round', zorder=5)
ax.plot([LX, LX],    [Y_LO, Y_HI], color=BACK_C, lw=2.4, solid_capstyle='round', zorder=5)
ax.plot([LX, RM+0.05],[Y_HI, Y_HI], color=BACK_C, lw=2.4, solid_capstyle='round', zorder=5)
# Arrowhead pointing left at the end
arr(RM + 0.18, Y_HI, RM, Y_HI, c=BACK_C, lw=2.4, ms=14)

# Label on vertical segment
t(LX + 0.13, (Y_LO + Y_HI)/2,
  r'$\alpha$  update',
  sz=9.5, c=BACK_C, w='bold',
  ha='left', va='center', rotation=90)

# ── Save ───────────────────────────────────────────────────────────────────────
out = Path('figures/fig_system_diagram.png')
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved {out}')
plt.close()
