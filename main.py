import numpy as np

from data import generate_synthetic_matrix
from metrics import frobenius_error, spectral_analysis
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
        spec = spectral_analysis(W, Q, alpha)

        results.append((solver, stats, error, spec))

    sparse_q = SparseQuantizedDecomposition(1)
    Q, alpha, stats = sparse_q.quantize(W)
    Q_hat = sparse_q.reconstruct(Q, stats.misc["W_sparse"], alpha)
    error = frobenius_error(W, Q_hat, 1)
    spec = spectral_analysis(W, Q_hat, 1)
    results.append((sparse_q, stats, error, spec))

    header = (
        f"{'Solver':<25} | {'Error':<9} | {'σ1 err':<8} | "
        f"{'SpecGapΔ':<8} | {'SpecRMSE':<9} | {'EnergyΔ':<9}| {'Time(s)':<8}"
    )
    print(header)
    print("-" * len(header))
    for solver, stats, error, spec in results:
        print(
            f"{solver.name:<25} | "
            f"{error:<9.4f} | "
            f"{spec['top_singular_value_err']:<8.4f} | "
            f"{spec['spectral_gap_diff']:<8.4f} | "
            f"{spec['singular_values_rmse']:<9.4f} | "
            f"{spec['spectral_energy_err']:<9.4f} | "
            f"{stats.time:<8.3f}"
        )


if __name__ == "__main__":
    run_experiment()
