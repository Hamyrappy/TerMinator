import numpy as np

from data import generate_synthetic_matrix
from metrics import frobenius_error, compute_quantization_metrics
from solvers import ALSQuantizer, NaiveQuantizer, SparseQuantizedDecomposition


def run_experiment():
    # 1. Setup Data
    print("Generating data...")
    W = generate_synthetic_matrix("gaussian", (1024, 1024))

    # 2. Setup Solvers
    solvers = [
        NaiveQuantizer(),
        ALSQuantizer(max_iter=5, row_wise=False),
        ALSQuantizer(max_iter=5, row_wise=True),
    ]

    # 3. Run Loop
    results = []

    for solver in solvers:
        Q, alpha, stats = solver.quantize(W)

        error = frobenius_error(W, Q, alpha)
        spec = compute_quantization_metrics(W, Q, alpha, k=50)

        results.append((solver, stats, error, spec))

    sparse_q = SparseQuantizedDecomposition(1)
    Q, alpha, stats = sparse_q.quantize(W)
    Q_hat = sparse_q.reconstruct(Q, stats.misc["W_sparse"], alpha)
    error = frobenius_error(W, Q_hat, 1)
    spec = compute_quantization_metrics(W, Q_hat, 1,k=50)
    results.append((sparse_q, stats, error, spec))

    used_sparse = min(Q.shape) > 1024
    header_cols = ['Solver', 'Time(s)', 'FrobErr', 'DirErr', 'RelSpecGap']
    if not used_sparse:
        header_cols.extend(['CondErr','EffRankΔ'])
    header = f"{header_cols[0]:<25} | "
    header += " | ".join(f"{col:<8}" for col in header_cols[1:])
    print(header)
    print("-" * len(header))
    for solver, stats, error, spec in results:
        row_values = (
            f"{solver.name:<25} | "
            f"{stats.time:<8.3f} | "
            f"{error:<8.4f} | "
            f"{spec['weighted_dir_err']:<8.2f} | "
            f"{spec['rel_gap_error']:<10.4f} | "
        )
        if not used_sparse:
            row_values += (
                f"{spec['condition_number_error']:<8.4f} | "
                f"{spec.get('effective_rank_change', 0):<8.4f}"
                )
        print(row_values)

if __name__ == "__main__":
    run_experiment()
