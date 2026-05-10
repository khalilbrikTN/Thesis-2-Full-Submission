# QUBO-Based DFL Node Placement — Final Submission

## Project Overview
This project formulates the Distributed Fingerprint Localization (DFL) sensor
node placement problem as a Quadratic Unconstrained Binary Optimization (QUBO)
problem and solves it using Simulated Annealing (SA) and Quantum Annealing (QA)
via the OpenJij library.

---

## Folder Structure

```
Final_Submission/
├── README.md                          <- This file
├── notebooks/                         <- Interactive Jupyter notebooks
│   ├── 01_precompute_fingerprint_data.ipynb
│   └── 02_qubo_solver_exploration.ipynb
├── scripts/                           <- Standalone Python scripts (run in order)
│   ├── 01_qubo_binary_search_solver.py
│   ├── 02_plot_result_figures.py
│   ├── 03_greedy_baseline.py
│   ├── 04_multirun_experiment.py
│   ├── 05_plot_coverage_comparison.py
│   ├── 06_plot_dfl_deployment.py
│   └── 07_plot_system_diagram.py
├── latex/                             <- LaTeX snippets used in the thesis
│   ├── algorithm.tex
│   └── qubo_formulation.tex
├── data/
│   ├── input/                         <- Raw / precomputed input data
│   │   ├── fingerprint_rss_train.csv
│   │   ├── fingerprint_rss_test.csv
│   │   ├── fingerprint_coords_train.csv
│   │   ├── fingerprint_coords_test.csv
│   │   ├── ap_mp_pairs.csv
│   │   ├── importance_scores.csv
│   │   └── redundancy_matrix.csv
│   └── output/                        <- Results produced by the scripts
│       ├── binary_search_log.csv
│       ├── greedy_result.csv
│       ├── multirun_results.csv
│       ├── multirun_summary.csv
│       ├── selected_pairs_qa.csv
│       ├── selected_pairs_sa.csv
│       └── summary.csv
└── figures/                           <- All output figures
    ├── fig00_dfl_deployment.png
    ├── fig01_accuracy.png / .pdf
    ├── fig02_time_to_solution.png / .pdf
    ├── fig03_pairs_vs_alpha.png / .pdf
    ├── fig04_ap_mp_vs_alpha.png / .pdf
    ├── fig05_ap_mp_per_solver.png
    ├── fig06_total_computation_time.png
    ├── fig07_empirical_cdf.png
    ├── fig08_coverage_comparison.png
    ├── fig09_system_architecture_diagram.png
    └── coverage_animation.html
```

---

## How to Run

### Prerequisites
```
pip install numpy pandas matplotlib openjij
```

### Execution Order

| Step | Script | Description |
|------|--------|-------------|
| 1 | `notebooks/01_precompute_fingerprint_data.ipynb` | Precompute pairs, importance, redundancy from raw fingerprint data |
| 2 | `scripts/01_qubo_binary_search_solver.py` | Run SA/QA binary search to find optimal alpha |
| 3 | `scripts/02_plot_result_figures.py` | Generate main result figures (fig01–fig07) |
| 4 | `scripts/03_greedy_baseline.py` | Run greedy baseline for comparison |
| 5 | `scripts/04_multirun_experiment.py` | Run 100-trial statistical experiment |
| 6 | `scripts/05_plot_coverage_comparison.py` | Generate coverage comparison figure |
| 7 | `scripts/06_plot_dfl_deployment.py` | Generate DFL deployment diagram |
| 8 | `scripts/07_plot_system_diagram.py` | Generate system architecture diagram |

All scripts should be run from the `Final_Submission/` root directory.
Output files are written to `data/output/` and `figures/`.
