import numpy as np
import pandas as pd
import openjij as oj
import time
from pathlib import Path

DATA = Path('data')
OUT  = Path('data')
print('openjij loaded OK')

# ── 1. Load Precomputed Data ──────────────────────────────────────────────────
df_pairs       = pd.read_csv(DATA / 'pairs.csv')
df_importance  = pd.read_csv(DATA / 'importance.csv')
df_redundancy  = pd.read_csv(DATA / 'redundancy.csv', index_col=0)

fp_rss_train    = pd.read_csv(DATA / 'fp_rss_train.csv').values    # (K, L)
fp_rss_test     = pd.read_csv(DATA / 'fp_rss_test.csv').values     # (K, L)
fp_coords_train = pd.read_csv(DATA / 'fp_coords_train.csv').values # (K, 2)
fp_coords_test  = pd.read_csv(DATA / 'fp_coords_test.csv').values  # (K, 2)

I     = df_importance['importance'].values   # (L,)
R     = df_redundancy.values                 # (L, L)
pairs = list(zip(df_pairs['ap'], df_pairs['mp']))
L     = len(pairs)
N     = 10

print(f'Pairs L={L},  train shape={fp_rss_train.shape},  test shape={fp_rss_test.shape}')

# ── 2. Parameters ─────────────────────────────────────────────────────────────
AP_BUDGET          = 5
MP_BUDGET          = 5
EPSILON            = 0.01
KNN_K              = 3
NUM_READS_SA       = 100    # SA reads per QUBO call
NUM_READS_QA       = 10     # QA reads per QUBO call
NUM_SWEEPS_SA      = 1000
NUM_SWEEPS_QA      = 1000
SEED               = 42
P_TARGET           = 0.99   # desired success probability for TTS
QA_ANNEAL_TIME_S   = 20e-6  # D-Wave hardware anneal time: 20 µs (projected)

print(f'AP budget={AP_BUDGET},  MP budget={MP_BUDGET},  epsilon={EPSILON}')
print(f'SA reads={NUM_READS_SA} sweeps={NUM_SWEEPS_SA},  QA reads={NUM_READS_QA} sweeps={NUM_SWEEPS_QA}')
print(f'QA projected anneal time={QA_ANNEAL_TIME_S*1e6:.1f} µs')

# ── 3. TTS Helper ────────────────────────────────────────────────────────────
def calculate_tts(p_s, avg_time, p_target=0.99):
    """
    TTS(τ, p_R) = τ × ln(1 - p_R) / ln(1 - p_s)
    - p_s == 0  : returns np.inf  (solver never succeeds)
    - p_s == 1  : returns avg_time (one run is always enough → TTS = τ)
    - 0 < p_s < 1: standard formula
    avg_time: anneal time per single run (τ)
    """
    if p_s == 0:
        return np.inf
    if p_s >= 1.0:
        return avg_time   # one run is enough → TTS = τ (theoretical minimum)
    try:
        return avg_time * np.log(1 - p_target) / np.log(1 - p_s)
    except (ValueError, ZeroDivisionError):
        return np.inf


# ── 4. QUBO Builder ───────────────────────────────────────────────────────────
def build_qubo(alpha, I, R):
    Q = {}
    L = len(I)
    for p in range(L):
        Q[(p, p)] = -alpha * I[p]
    coeff_scale = 2.0 * (1.0 - alpha)
    if coeff_scale != 0.0:
        for p in range(L):
            for q in range(p + 1, L):
                v = coeff_scale * R[p, q]
                if v != 0.0:
                    Q[(p, q)] = v
    return Q

# ── 4. Helpers ────────────────────────────────────────────────────────────────
def count_active(sample, pairs):
    active_ap, active_mp = set(), set()
    for p, val in sample.items():
        if val == 1:
            i, j = pairs[p]
            active_ap.add(i)
            active_mp.add(j)
    return len(active_ap), len(active_mp)

def sample_to_vector(sample, L):
    x = np.zeros(L, dtype=int)
    for p, val in sample.items():
        x[p] = int(val)
    return x

# ── 5. KNN Localizer ─────────────────────────────────────────────────────────
def localizer(x_vec, fp_rss_train, fp_coords_train,
              fp_rss_test, fp_coords_test, K=3):
    selected = np.where(x_vec == 1)[0]
    if len(selected) == 0:
        return np.inf
    train_feat = fp_rss_train[:, selected]
    test_feat  = fp_rss_test[:, selected]
    errors = []
    for k in range(len(test_feat)):
        dists   = np.linalg.norm(train_feat - test_feat[k], axis=1)
        knn_idx = np.argsort(dists)[:K]
        pred    = fp_coords_train[knn_idx].mean(axis=0)
        true    = fp_coords_test[k]
        errors.append(np.linalg.norm(pred - true))
    return float(np.mean(errors))

x_all = np.ones(L, dtype=int)
baseline_acc = localizer(x_all, fp_rss_train, fp_coords_train,
                          fp_rss_test, fp_coords_test, K=KNN_K)
print(f'Baseline KNN error (all {L} pairs): {baseline_acc:.4f} m')

# ── 6. Binary Search ─────────────────────────────────────────────────────────
def binary_search(sampler, sampler_name, I, R, pairs, L,
                  n_budget, m_budget, epsilon,
                  fp_rss_train, fp_coords_train,
                  fp_rss_test, fp_coords_test,
                  knn_k=3, num_reads=100, num_sweeps=1000, seed=42,
                  anneal_time=None):

    a, b       = 0.0, 1.0
    x_star     = None
    best_acc   = np.inf
    alpha_star = None
    log        = []
    iteration  = 0
    t_start    = time.time()

    while b - a >= epsilon:
        iteration += 1
        alpha = (a + b) / 2.0

        Q          = build_qubo(alpha, I, R)
        t0         = time.time()
        response   = sampler.sample_qubo(Q, num_reads=num_reads,
                                         num_sweeps=num_sweeps, seed=seed)
        t_qubo     = time.time() - t0
        best_sample = response.first.sample
        x_vec       = sample_to_vector(best_sample, L)

        ap_count, mp_count = count_active(best_sample, pairs)
        feasible = (ap_count <= n_budget) and (mp_count <= m_budget)

        acc = np.nan
        if feasible:
            a   = alpha
            acc = localizer(x_vec, fp_rss_train, fp_coords_train,
                            fp_rss_test, fp_coords_test, K=knn_k)
            if acc < best_acc:
                best_acc      = acc
                x_star        = x_vec.copy()
                alpha_star    = alpha
                best_response = response
                best_t_qubo   = t_qubo
            status = f'FEASIBLE   acc={acc:.4f} m'
        else:
            b      = alpha
            status = 'OVER-BUDGET'

        print(f'  [{sampler_name}] iter={iteration:2d}  alpha={alpha:.4f}  '
              f'AP={ap_count} MP={mp_count}  pairs={int(x_vec.sum())}  {status}')

        log.append({
            'solver'         : sampler_name,
            'iteration'      : iteration,
            'alpha'          : alpha,
            'a'              : a,
            'b'              : b,
            'AP_count'       : ap_count,
            'MP_count'       : mp_count,
            'feasible'       : feasible,
            'acc_m'          : acc,
            'pairs_selected' : int(x_vec.sum()),
        })

    t_total = time.time() - t_start

    # ── TTS at alpha* ─────────────────────────────────────────────────────────
    tts_99 = np.nan
    p_s    = np.nan
    if x_star is not None:
        best_energy = best_response.first.energy
        n_success   = sum(1 for e in best_response.record.energy
                          if np.isclose(e, best_energy, atol=1e-6))
        p_s         = n_success / num_reads
        # τ: for QA use hardware anneal time; for SA use measured wall-clock per read
        tau = anneal_time if anneal_time is not None else (best_t_qubo / num_reads)
        tts_99 = calculate_tts(p_s, tau, P_TARGET)
        print(f'  [{sampler_name}] TTS99: p_s={p_s:.3f}  '
              f'tau={tau*1e6:.2f} µs  TTS={tts_99*1e6:.3f} µs')

    return {
        'alpha_star' : alpha_star,
        'x_star'     : x_star,
        'best_acc'   : best_acc,
        'time_s'     : t_total,
        'iterations' : iteration,
        'p_s'        : p_s,
        'tts_99_ms'  : tts_99 * 1000 if not np.isnan(tts_99) else np.nan,
        'log'        : log,
    }

# ── 7. Run SA ─────────────────────────────────────────────────────────────────
print('\n=== Simulated Annealing ===')
sa_result = binary_search(
    oj.SASampler(), 'SA', I, R, pairs, L,
    AP_BUDGET, MP_BUDGET, EPSILON,
    fp_rss_train, fp_coords_train, fp_rss_test, fp_coords_test,
    knn_k=KNN_K, num_reads=NUM_READS_SA, num_sweeps=NUM_SWEEPS_SA, seed=SEED,
    anneal_time=None,              # SA: use measured wall-clock time per read
)
print(f'\nSA  alpha*={sa_result["alpha_star"]}  '
      f'best_acc={sa_result["best_acc"]:.4f} m  '
      f'time={sa_result["time_s"]:.2f}s  iters={sa_result["iterations"]}')

# ── 8. Run QA ─────────────────────────────────────────────────────────────────
print('\n=== Simulated Quantum Annealing ===')
qa_result = binary_search(
    oj.SQASampler(), 'QA', I, R, pairs, L,
    AP_BUDGET, MP_BUDGET, EPSILON,
    fp_rss_train, fp_coords_train, fp_rss_test, fp_coords_test,
    knn_k=KNN_K, num_reads=NUM_READS_QA, num_sweeps=NUM_SWEEPS_QA, seed=SEED,
    anneal_time=QA_ANNEAL_TIME_S,  # QA: D-Wave hardware anneal time (20 µs)
)
print(f'\nQA  alpha*={qa_result["alpha_star"]}  '
      f'best_acc={qa_result["best_acc"]:.4f} m  '
      f'time={qa_result["time_s"]:.2f}s  iters={qa_result["iterations"]}')

# ── 9. Save Results ───────────────────────────────────────────────────────────
all_logs = sa_result['log'] + qa_result['log']
df_log = pd.DataFrame(all_logs)
df_log.to_csv(OUT / 'binary_search_log.csv', index=False)
print('\nSaved binary_search_log.csv')

for name, result in [('sa', sa_result), ('qa', qa_result)]:
    if result['x_star'] is not None:
        sel_idx = np.where(result['x_star'] == 1)[0]
        df_sel  = df_pairs.iloc[sel_idx].copy()
        df_sel['importance'] = I[sel_idx]
        df_sel.to_csv(OUT / f'selected_pairs_{name}.csv', index=False)
        print(f'Saved selected_pairs_{name}.csv  ({len(sel_idx)} pairs)')

rows = []
for name, result in [('SA', sa_result), ('QA', qa_result)]:
    x = result['x_star']
    if x is not None:
        sd = {p: int(x[p]) for p in range(L)}
        ap_c, mp_c = count_active(sd, pairs)
    else:
        ap_c, mp_c = None, None
    rows.append({
        'solver'         : name,
        'alpha_star'     : result['alpha_star'],
        'AP_count'       : ap_c,
        'MP_count'       : mp_c,
        'pairs_selected' : int(x.sum()) if x is not None else None,
        'best_acc_m'     : result['best_acc'],
        'time_s'         : result['time_s'],
        'iterations'     : result['iterations'],
        'p_s'            : result['p_s'],
        'tts_99_ms'      : result['tts_99_ms'],
    })

df_summary = pd.DataFrame(rows)
df_summary.to_csv(OUT / 'summary.csv', index=False)
print('\nSummary:')
print(df_summary.to_string(index=False))
print('\nDone.')
