import numpy as np
import matplotlib.pyplot as plt

from data import generate_synthetic_matrix
from metrics import frobenius_error, compute_quantization_metrics
from solvers import ALSQuantizer, NaiveQuantizer, SparseQuantizedDecomposition


def run_experiment(save_path="singular_values.png"):
    # 1. Setup Data
    print("Generating data...")
    W = generate_synthetic_matrix("gaussian", (1024, 1024))

    # 2. Setup Solvers
    solvers = [
        NaiveQuantizer(),
        #NaiveQuantizer(init_with_power=True),
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
    sv_dict = {}
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
        sv_dict[solver.name] = spec['s_Q']
        sv_dict['original'] = spec['s_W']
        print(row_values)
        
    plt.figure(figsize=(10,6))
    for name, s_Q in sv_dict.items():
        plt.hist(np.log1p(s_Q), bins=50, alpha=0.5, density=True, label=name)
    
    plt.xlabel('log(1 + singular value)')
    plt.ylabel('Density')
    plt.title('Singular Value Distributions of Quantized Matrices')
    plt.legend()
    plt.grid(True, ls='--', lw=0.5)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")


if __name__ == "__main__":
    run_experiment()
