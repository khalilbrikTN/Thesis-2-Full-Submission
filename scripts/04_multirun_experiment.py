"""
Multi-run experiment: repeat the full binary-search pipeline N_RUNS times
with different random seeds to produce mean +/- std statistics.
Greedy is deterministic (run once, replicated across rows for consistency).
Outputs:
  data/multirun_results.csv   -- one row per (run, solver)
  data/multirun_summary.csv   -- mean / std / min / max per solver
"""
import numpy as np
import pandas as pd
import openjij as oj
import time
from pathlib import Path

DATA   = Path('data')
N_RUNS = 100

# ── Load data ──────────────────────────────────────────────────────────────────
df_pairs      = pd.read_csv(DATA / 'pairs.csv')
df_importance = pd.read_csv(DATA / 'importance.csv')
df_redundancy = pd.read_csv(DATA / 'redundancy.csv', index_col=0)

rss_train    = pd.read_csv(DATA / 'fp_rss_train.csv').values
rss_test     = pd.read_csv(DATA / 'fp_rss_test.csv').values
coords_train = pd.read_csv(DATA / 'fp_coords_train.csv').values
coords_test  = pd.read_csv(DATA / 'fp_coords_test.csv').values

I     = df_importance['importance'].values
R     = df_redundancy.values
pairs = list(zip(df_pairs['ap'], df_pairs['mp']))
L     = len(pairs)

# ── Parameters (same as run_02.py) ────────────────────────────────────────────
AP_BUDGET        = 5
MP_BUDGET        = 5
EPSILON          = 0.01
KNN_K            = 3
NUM_READS_SA     = 100
NUM_READS_QA     = 10
NUM_SWEEPS       = 1000
P_TARGET         = 0.99
QA_ANNEAL_TIME_S = 20e-6   # 20 µs projected D-Wave anneal time

print(f'Dataset: L={L} pairs, train={rss_train.shape}, test={rss_test.shape}')
print(f'Budget: AP<={AP_BUDGET}, MP<={MP_BUDGET}, eps={EPSILON}, N_RUNS={N_RUNS}')

# ── Helpers ────────────────────────────────────────────────────────────────────
def build_qubo(alpha):
    Q = {}
    for p in range(L):
        Q[(p, p)] = -alpha * I[p]
    coeff = 2.0 * (1.0 - alpha)
    if coeff != 0.0:
        for p in range(L):
            for q in range(p + 1, L):
                v = coeff * R[p, q]
                if v != 0.0:
                    Q[(p, q)] = v
    return Q

def count_active(sample):
    aps, mps = set(), set()
    for p, v in sample.items():
        if v == 1:
            aps.add(pairs[p][0])
            mps.add(pairs[p][1])
    return len(aps), len(mps)

def sample_to_vec(sample):
    x = np.zeros(L, dtype=int)
    for p, v in sample.items():
        x[p] = int(v)
    return x

def knn_error(x_vec):
    sel = np.where(x_vec == 1)[0]
    if len(sel) == 0:
        return np.inf
    errors = []
    for t in range(len(rss_test)):
        dists = np.linalg.norm(rss_train[:, sel] - rss_test[t, sel], axis=1)
        nn    = np.argsort(dists)[:KNN_K]
        errors.append(np.linalg.norm(coords_train[nn].mean(0) - coords_test[t]))
    return float(np.mean(errors))

def tts(p_s, tau):
    if p_s <= 0:
        return np.inf
    if p_s >= 1.0:
        return tau
    return tau * np.log(1 - P_TARGET) / np.log(1 - p_s)

# ── Binary search (single run) ─────────────────────────────────────────────────
def run_binary_search(sampler, num_reads, anneal_time, seed):
    a, b = 0.0, 1.0
    x_star = None
    best_acc = np.inf
    alpha_star = None
    best_resp = best_t_qubo = None
    t_start = time.time()

    while b - a >= EPSILON:
        alpha = (a + b) / 2.0
        Q     = build_qubo(alpha)
        t0    = time.time()
        resp  = sampler.sample_qubo(Q, num_reads=num_reads,
                                    num_sweeps=NUM_SWEEPS, seed=seed)
        t_q   = time.time() - t0

        samp  = resp.first.sample
        x_vec = sample_to_vec(samp)
        ap_c, mp_c = count_active(samp)

        if ap_c <= AP_BUDGET and mp_c <= MP_BUDGET:
            a   = alpha
            acc = knn_error(x_vec)
            if acc < best_acc:
                best_acc   = acc
                x_star     = x_vec.copy()
                alpha_star = alpha
                best_resp  = resp
                best_t_qubo = t_q
        else:
            b = alpha

    t_total = time.time() - t_start

    # TTS at alpha*
    p_s_val = tts_ms = np.nan
    ap_out = mp_out = pairs_out = None

    if x_star is not None:
        best_e = best_resp.first.energy
        n_ok   = sum(1 for e in best_resp.record.energy
                     if np.isclose(e, best_e, atol=1e-6))
        p_s_val = n_ok / num_reads
        tau     = anneal_time if anneal_time is not None else (best_t_qubo / num_reads)
        tts_ms  = tts(p_s_val, tau) * 1000
        sd      = {p: int(x_star[p]) for p in range(L)}
        ap_out, mp_out = count_active(sd)
        pairs_out = int(x_star.sum())

    return {
        'alpha_star'     : alpha_star,
        'AP_count'       : ap_out,
        'MP_count'       : mp_out,
        'pairs_selected' : pairs_out,
        'best_acc_m'     : best_acc,
        'time_s'         : t_total,
        'tts_99_ms'      : tts_ms,
        'p_s'            : p_s_val,
    }

# ── Greedy baseline (deterministic, seed-independent) ─────────────────────────
df_sorted = df_importance.sort_values('importance', ascending=False)
sel_g, ap_g, mp_g = [], set(), set()
for _, row in df_sorted.iterrows():
    ap, mp = int(row['ap']), int(row['mp'])
    if len(ap_g | {ap}) <= AP_BUDGET and len(mp_g | {mp}) <= MP_BUDGET:
        sel_g.append(int(row['var_index']))
        ap_g.add(ap)
        mp_g.add(mp)

x_g      = np.zeros(L, dtype=int)
x_g[sel_g] = 1
greedy_acc = knn_error(x_g)
print(f'\nGreedy (deterministic): acc={greedy_acc:.4f} m  '
      f'pairs={len(sel_g)}  APs={len(ap_g)}  MPs={len(mp_g)}')

# ── Multi-run loop ─────────────────────────────────────────────────────────────
records = []
sa_sampler = oj.SASampler()
qa_sampler = oj.SQASampler()

for run in range(N_RUNS):
    seed = run
    print(f'\n--- Run {run + 1:2d}/{N_RUNS}  (seed={seed}) ---')

    r_sa = run_binary_search(sa_sampler, NUM_READS_SA, None, seed)
    print(f'  SA: acc={r_sa["best_acc_m"]:.4f} m  '
          f'pairs={r_sa["pairs_selected"]}  '
          f'alpha*={r_sa["alpha_star"]}  '
          f'p_s={r_sa["p_s"]:.3f}')
    records.append({'run': run, 'solver': 'SA', 'seed': seed, **r_sa})

    r_qa = run_binary_search(qa_sampler, NUM_READS_QA, QA_ANNEAL_TIME_S, seed)
    print(f'  QA: acc={r_qa["best_acc_m"]:.4f} m  '
          f'pairs={r_qa["pairs_selected"]}  '
          f'alpha*={r_qa["alpha_star"]}  '
          f'p_s={r_qa["p_s"]:.3f}')
    records.append({'run': run, 'solver': 'QA', 'seed': seed, **r_qa})

# Greedy: same result every time; still add a row per run for consistent indexing
for run in range(N_RUNS):
    records.append({
        'run': run, 'solver': 'Greedy', 'seed': run,
        'alpha_star': None,
        'AP_count': len(ap_g), 'MP_count': len(mp_g),
        'pairs_selected': len(sel_g),
        'best_acc_m': greedy_acc,
        'time_s': 0.0,
        'tts_99_ms': None,
        'p_s': None,
    })

df_runs = pd.DataFrame(records)
df_runs.to_csv(DATA / 'multirun_results.csv', index=False)
print(f'\nSaved multirun_results.csv  ({len(df_runs)} rows)')

# ── Summary statistics ─────────────────────────────────────────────────────────
rows_sum = []
for solver in ['Greedy', 'SA', 'QA']:
    sub = df_runs[df_runs['solver'] == solver]
    acc  = sub['best_acc_m']
    tts_ = sub['tts_99_ms']
    pairs= sub['pairs_selected']
    rows_sum.append({
        'solver'        : solver,
        'acc_mean_m'    : acc.mean(),
        'acc_std_m'     : acc.std(ddof=1),
        'acc_min_m'     : acc.min(),
        'acc_max_m'     : acc.max(),
        'pairs_mean'    : pairs.mean(),
        'pairs_std'     : pairs.std(ddof=1),
        'AP_mean'       : sub['AP_count'].mean(),
        'MP_mean'       : sub['MP_count'].mean(),
        'tts_mean_ms'   : tts_.mean() if solver != 'Greedy' else None,
        'tts_std_ms'    : tts_.std(ddof=1) if solver != 'Greedy' else None,
        'time_mean_s'   : sub['time_s'].mean(),
    })

df_sum = pd.DataFrame(rows_sum)
df_sum.to_csv(DATA / 'multirun_summary.csv', index=False)

print('\n' + '='*60)
print('MULTI-RUN SUMMARY')
print('='*60)
for _, row in df_sum.iterrows():
    print(f"\n{row['solver']}:")
    print(f"  Accuracy  : {row['acc_mean_m']:.4f} +/- {row['acc_std_m']:.4f} m"
          f"  (min={row['acc_min_m']:.4f}, max={row['acc_max_m']:.4f})")
    print(f"  Pairs     : {row['pairs_mean']:.1f} +/- {row['pairs_std']:.1f}")
    print(f"  AP / MP   : {row['AP_mean']:.1f} / {row['MP_mean']:.1f}")
    if row['solver'] != 'Greedy':
        print(f"  TTS_99    : {row['tts_mean_ms']:.4f} +/- {row['tts_std_ms']:.4f} ms")
    print(f"  Time/run  : {row['time_mean_s']:.3f} s")

print('\nSaved multirun_summary.csv')
print('Done.')
