"""
Greedy baseline for AP-MP node placement.
Selects pairs by descending importance (RSS variance) while respecting
the same AP and MP budgets used in the QUBO experiments.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA     = Path('data')
AP_BUDGET = 5
MP_BUDGET = 5
KNN_K     = 3

# ── Load precomputed data ──────────────────────────────────────────────────────
df_imp   = pd.read_csv(DATA / 'importance.csv')          # var_index, ap, mp, importance
rss_train = pd.read_csv(DATA / 'fp_rss_train.csv').values  # K_train × P
rss_test  = pd.read_csv(DATA / 'fp_rss_test.csv').values  # K_test  × P
coords_train = pd.read_csv(DATA / 'fp_coords_train.csv').values  # K_train × 2
coords_test  = pd.read_csv(DATA / 'fp_coords_test.csv').values  # K_test  × 2

# ── KNN localizer (identical to run_02.py) ─────────────────────────────────────
def knn_accuracy(selected_indices, k=KNN_K):
    if len(selected_indices) == 0:
        return np.inf
    train = rss_train[:, selected_indices]   # K_train × |S|
    test  = rss_test[:,  selected_indices]   # K_test  × |S|
    errors = []
    for t in range(len(test)):
        dists = np.linalg.norm(train - test[t], axis=1)
        nn    = np.argsort(dists)[:k]
        pred  = coords_train[nn].mean(axis=0)
        errors.append(np.linalg.norm(pred - coords_test[t]))
    return float(np.mean(errors))

# ── Greedy selection ───────────────────────────────────────────────────────────
df_sorted = df_imp.sort_values('importance', ascending=False).reset_index(drop=True)

selected = []
ap_nodes = set()
mp_nodes = set()

for _, row in df_sorted.iterrows():
    ap, mp = int(row['ap']), int(row['mp'])
    new_aps = ap_nodes | {ap}
    new_mps = mp_nodes | {mp}
    if len(new_aps) <= AP_BUDGET and len(new_mps) <= MP_BUDGET:
        selected.append(int(row['var_index']))
        ap_nodes = new_aps
        mp_nodes = new_mps

# ── Evaluate ───────────────────────────────────────────────────────────────────
acc = knn_accuracy(selected)

print(f"Greedy baseline")
print(f"  Pairs selected : {len(selected)}")
print(f"  AP nodes       : {len(ap_nodes)}  {sorted(ap_nodes)}")
print(f"  MP nodes       : {len(mp_nodes)}  {sorted(mp_nodes)}")
print(f"  Mean error (m) : {acc:.4f}")
print()

# Compare against QUBO results
df_summary = pd.read_csv(DATA / 'summary.csv')
sa = df_summary[df_summary['solver'] == 'SA'].iloc[0]
qa = df_summary[df_summary['solver'] == 'QA'].iloc[0]

print("Comparison (same n=5, m=5 budget):")
print(f"  Greedy : {acc:.4f} m   pairs={len(selected)}")
print(f"  SA     : {sa['best_acc_m']:.4f} m   pairs={int(sa['pairs_selected'])}")
print(f"  QA     : {qa['best_acc_m']:.4f} m   pairs={int(qa['pairs_selected'])}")

# ── Save result ────────────────────────────────────────────────────────────────
pd.DataFrame([{
    'solver': 'Greedy',
    'pairs_selected': len(selected),
    'AP_count': len(ap_nodes),
    'MP_count': len(mp_nodes),
    'best_acc_m': acc,
}]).to_csv(DATA / 'greedy_result.csv', index=False)
print("\nSaved to data/greedy_result.csv")
