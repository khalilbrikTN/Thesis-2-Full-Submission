import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

DATA    = Path('data')
FIGURES = Path('figures')
FIGURES.mkdir(exist_ok=True)

# ── IEEE rcParams ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'        : 'serif',
    'font.size'          : 9,
    'axes.labelsize'     : 10,
    'xtick.labelsize'    : 9,
    'ytick.labelsize'    : 9,
    'legend.fontsize'    : 9,
    'legend.framealpha'  : 0.9,
    'legend.edgecolor'   : '0.8',
    'figure.dpi'         : 300,
    'savefig.dpi'        : 300,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.05,
    'lines.linewidth'    : 1.5,
    'axes.linewidth'     : 0.8,
    'xtick.direction'    : 'in',
    'ytick.direction'    : 'in',
    'xtick.major.width'  : 0.8,
    'ytick.major.width'  : 0.8,
    'xtick.minor.width'  : 0.5,
    'ytick.minor.width'  : 0.5,
    'xtick.major.size'   : 4,
    'ytick.major.size'   : 4,
    'xtick.minor.size'   : 2,
    'ytick.minor.size'   : 2,
})

# ── Load data ──────────────────────────────────────────────────────────────────
df_summary  = pd.read_csv(DATA / 'summary.csv')
df_log      = pd.read_csv(DATA / 'binary_search_log.csv')
df_multirun = pd.read_csv(DATA / 'multirun_results.csv')

sa_row = df_summary[df_summary['solver'] == 'SA'].iloc[0]
qa_row = df_summary[df_summary['solver'] == 'QA'].iloc[0]
df_qa  = df_log[df_log['solver'] == 'QA'].sort_values('alpha').reset_index(drop=True)

sa_errors = df_multirun[df_multirun['solver'] == 'SA']['best_acc_m'].values
qa_errors = df_multirun[df_multirun['solver'] == 'QA']['best_acc_m'].values

C_SA = '#2166ac'   # blue  — SA
C_QA = '#d6604d'   # red   — QA
C_AP = '#1a9641'   # green — AP
C_MP = '#d7191c'   # red   — MP

FIG_W = 3.5   # single-column IEEE width (inches)
FIG_H = 2.4

GRID_KW = dict(linestyle='--', linewidth=0.5, alpha=0.5, color='0.7')

# ── Fig 1: Localization Accuracy (mean ± std over 20 runs) ────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

means  = [sa_errors.mean(), qa_errors.mean()]
stds   = [sa_errors.std(),  qa_errors.std()]
labels = ['SA', 'QA']
colors = [C_SA, C_QA]

bars = ax.bar(labels, means, yerr=stds, capsize=4,
              color=colors, width=0.45,
              edgecolor='black', linewidth=0.8,
              error_kw=dict(elinewidth=0.8, ecolor='black'))

for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2,
            m + s + 0.02,
            f'{m:.2f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Solver')
ax.set_ylabel('Mean Localization Error (m)')
ax.set_ylim(0, max(means) * 1.35)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.tick_params(axis='both', which='both', direction='in')
ax.grid(axis='y', **GRID_KW)

fig.tight_layout()
fig.savefig(FIGURES / 'fig1_accuracy.png')
plt.close(fig)
print('Saved fig1_accuracy.png')

# ── Fig 2: Time-to-Solution ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

tts  = [sa_row['tts_99_ms'], qa_row['tts_99_ms']]
bars = ax.bar(['SA', 'QA'], tts,
              color=[C_SA, C_QA], width=0.45,
              edgecolor='black', linewidth=0.8)

for bar, val in zip(bars, tts):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            f'{val:.3f} ms', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Solver')
ax.set_ylabel(r'$\mathrm{TTS}_{99}$ (ms)')
ax.set_yscale('log')
ax.set_ylim(0.005, 10)
ax.tick_params(axis='both', which='both', direction='in')
ax.grid(axis='y', which='both', **GRID_KW)

fig.tight_layout()
fig.savefig(FIGURES / 'fig2_tts.png')
plt.close(fig)
print('Saved fig2_tts.png')

# ── Fig 3: Total Pairs Selected vs α (QA) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

ax.plot(df_qa['alpha'], df_qa['pairs_selected'],
        color=C_QA, marker='o', markersize=5, linestyle='-', linewidth=1.5)

ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('Pairs Selected')
ax.set_xlim(df_qa['alpha'].min() - 0.02, df_qa['alpha'].max() + 0.03)
ax.set_ylim(0, df_qa['pairs_selected'].max() + 2)
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.tick_params(axis='both', which='both', direction='in')
ax.tick_params(axis='x', which='minor', bottom=False)
ax.grid(**GRID_KW)

fig.tight_layout()
fig.savefig(FIGURES / 'fig3_pairs_vs_alpha.png')
plt.close(fig)
print('Saved fig3_pairs_vs_alpha.png')

# ── Fig 4: AP count and MP count vs α (QA) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

ax.plot(df_qa['alpha'], df_qa['AP_count'],
        color=C_AP, marker='s', markersize=5, linestyle='-',
        linewidth=1.5, label='AP nodes')
ax.plot(df_qa['alpha'], df_qa['MP_count'],
        color=C_MP, marker='^', markersize=5, linestyle='--',
        linewidth=1.5, label='MP nodes')

ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('Node Count')
ax.set_xlim(df_qa['alpha'].min() - 0.02, df_qa['alpha'].max() + 0.03)
ax.set_ylim(0, max(df_qa['AP_count'].max(), df_qa['MP_count'].max()) + 2)
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.tick_params(axis='both', which='both', direction='in')
ax.tick_params(axis='x', which='minor', bottom=False)
ax.grid(**GRID_KW)
ax.legend(loc='upper left')

fig.tight_layout()
fig.savefig(FIGURES / 'fig4_ap_mp_vs_alpha.png')
plt.close(fig)
print('Saved fig4_ap_mp_vs_alpha.png')

# ── Fig 5: AP/MP node counts per solver (grouped bar) ─────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

x       = np.arange(2)          # two groups: AP Nodes, MP Nodes
width   = 0.32
sa_vals = [int(sa_row['AP_count']), int(sa_row['MP_count'])]
qa_vals = [int(qa_row['AP_count']), int(qa_row['MP_count'])]

bars_sa = ax.bar(x - width / 2, sa_vals, width, color=C_SA, hatch='//',
                 edgecolor='black', linewidth=0.8, label='SA')
bars_qa = ax.bar(x + width / 2, qa_vals, width, color=C_QA,
                 edgecolor='black', linewidth=0.8, label='QA')

for bar, val in zip(list(bars_sa) + list(bars_qa), sa_vals + qa_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.06,
            str(val), ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(['AP Nodes', 'MP Nodes'])
ax.set_xlabel('Node Type')
ax.set_ylabel('Node Count')
ax.set_ylim(0, max(sa_vals + qa_vals) + 1.8)
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.tick_params(axis='both', which='both', direction='in')
ax.grid(axis='y', **GRID_KW)
ax.legend(loc='upper left')

fig.tight_layout()
fig.savefig(FIGURES / 'fig5_ap_mp_per_solver.png')
plt.close(fig)
print('Saved fig5_ap_mp_per_solver.png')

# ── Fig 6: Total time = search (all iterations) + TTS at α* ──────────────────
QA_ANNEAL_MS = 0.020

sa_search_ms = sa_row['time_s'] * 1000
sa_tts_ms    = sa_row['tts_99_ms']
sa_combined  = sa_search_ms + sa_tts_ms

qa_search_ms = int(qa_row['iterations']) * QA_ANNEAL_MS
qa_tts_ms    = qa_row['tts_99_ms']
qa_combined  = qa_search_ms + qa_tts_ms

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

totals = [sa_combined, qa_combined]
bars   = ax.bar(['SA', 'QA'], totals,
                color=[C_SA, C_QA], width=0.45,
                edgecolor='black', linewidth=0.8)

for bar, val in zip(bars, totals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            f'{val:.3f} ms', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Solver')
ax.set_ylabel('Total Time (ms)')
ax.set_yscale('log')
ax.set_ylim(0.005, 5000)
ax.tick_params(axis='both', which='both', direction='in')
ax.grid(axis='y', which='both', **GRID_KW)

fig.tight_layout()
fig.savefig(FIGURES / 'fig6_total_time.png')
plt.close(fig)
print('Saved fig6_total_time.png')

# ── Fig 7: CDF of localization error over 20 seeds ────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

for errors, label, color, ls in [
    (sa_errors, 'SA', C_SA, '-'),
    (qa_errors, 'QA', C_QA, '--'),
]:
    sorted_e = np.sort(errors)
    cdf      = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
    ax.step(sorted_e, cdf, where='post',
            color=color, linestyle=ls, linewidth=1.5, label=label)

ax.set_xlabel('Localization Error (m)')
ax.set_ylabel('CDF')
ax.set_xlim(2.35, 2.85)
ax.set_ylim(0, 1.05)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.tick_params(axis='both', which='both', direction='in')
ax.grid(**GRID_KW)
ax.legend(loc='upper left')

fig.tight_layout()
fig.savefig(FIGURES / 'fig7_cdf.png')
plt.close(fig)
print('Saved fig7_cdf.png')

print('\nAll 7 figures saved to figures/')
