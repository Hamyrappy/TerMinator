import numpy as np

from data import generate_synthetic_matrix
from metrics import frobenius_error, spectral_analysis
from solvers import ALSQuantizer, NaiveQuantizer, SparseQuantizedDecomposition


def run_experiment():
    # 1. Setup Data
    print("Generating data...")
    W = generate_synthetic_matrix("uniform", (1024, 1024))

    # 2. Setup Solvers
    solvers = [
        NaiveQuantizer(),
        ALSQuantizer(max_iter=5, row_wise=False),
        ALSQuantizer(max_iter=5, row_wise=True),
        SparseQuantizedDecomposition(0.05),
    ]

    # 3. Run Loop
    results = {}
    header = (
        f"{'Solver':<20} | {'Error':<9} | {'σ1 err':<8} | "
        f"{'SpecGapΔ':<8} | {'SpecRMSE':<9} | {'EnergyΔ':<9}| {'Time(s)':<8}"
    )
    print(header)
    print("-" * len(header))

    for solver in solvers:
        # The Interface Contract in action:
        Q, alpha, stats = solver.quantize(W)

        # The Metrics Contract in action:
        error = frobenius_error(W, Q, alpha)
        spec = spectral_analysis(W, Q, alpha)

        print(
            f"{solver.name:<20} | "
            f"{error:<9.4f} | "
            f"{spec['top_singular_value_err']:<8.4f} | "
            f"{spec['spectral_gap_diff']:<8.4f} | "
            f"{spec['singular_values_rmse']:<9.4f} | "
            f"{spec['spectral_energy_err']:<9.4f} | "
            f"{stats.time:<8.3f}"
        )
        results[solver.name] = {
            "frobenius_error": float(error),
            **spec,
            "time": float(stats.time),
        }


if __name__ == "__main__":
    run_experiment()
