from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data import generate_synthetic_matrix, get_real_weights
from metrics import compute_quantization_metrics, frobenius_error
from solvers import (
    ALSQuantizer,
    BaseQuantizer,
    NaiveQuantizer,
    SparseQuantizedDecomposition,
)


@dataclass
class SolverResult:
    name: str
    frob_err: float
    weighted_dir_err: float
    rel_gap_err: float
    cond_err: float | None
    eff_rank_delta: float | None
    time: float
    s_W: np.ndarray | None = None
    s_Q: np.ndarray | None = None


np.random.seed(69)


def make_solvers() -> List[BaseQuantizer]:
    """List of all solvers to compare (except sparse+quantized, handled separately)."""
    return [
        NaiveQuantizer(),
        ALSQuantizer(row_wise=False),
        ALSQuantizer(row_wise=True),
    ]


def run_on_matrix(
    W: np.ndarray,
    with_singulars: bool = False,
    k: int = 50,
) -> List[SolverResult]:
    """
    Run all quantizers on a single weight matrix W and return per-solver metrics.
    Includes:
      - Naive
      - ALS (global / row-wise)
      - Sparse + Quantized (bitnet)
      - Sparse + Quantized (ALS)
    """
    results: List[SolverResult] = []

    # Normal solvers
    for solver in make_solvers():
        Q, alpha, stats = solver.quantize(W)
        frob = float(frobenius_error(W, Q, alpha))
        spec = compute_quantization_metrics(W, Q, alpha, k=k)

        results.append(
            SolverResult(
                name=solver.name,
                frob_err=frob,
                weighted_dir_err=float(spec["weighted_dir_err"]),
                rel_gap_err=float(spec["rel_gap_error"]),
                cond_err=(
                    float(spec["condition_number_error"])
                    if spec["condition_number_error"] is not None
                    else None
                ),
                eff_rank_delta=(
                    float(spec["effective_rank_change"])
                    if spec["effective_rank_change"] is not None
                    else None
                ),
                time=float(stats.time),
                s_W=spec["s_W"] if with_singulars else None,
                s_Q=spec["s_Q"] if with_singulars else None,
            )
        )

    # Sparse + quantized (bitnet inner)
    sparse_bitnet = SparseQuantizedDecomposition(adaptive=True)
    Q_b, alpha_b, stats_b = sparse_bitnet.quantize(
        W,
        quantizer="bitnet",
        tau_quantile=0.95,
    )
    W_hat_b = sparse_bitnet.reconstruct(Q_b, stats_b.misc["W_sparse"], alpha_b)
    frob_b = float(frobenius_error(W, W_hat_b, 1.0))
    spec_b = compute_quantization_metrics(W, W_hat_b, 1.0, k=k)

    results.append(
        SolverResult(
            name=sparse_bitnet.name + " [bitnet]",
            frob_err=frob_b,
            weighted_dir_err=float(spec_b["weighted_dir_err"]),
            rel_gap_err=float(spec_b["rel_gap_error"]),
            cond_err=(
                float(spec_b["condition_number_error"])
                if spec_b["condition_number_error"] is not None
                else None
            ),
            eff_rank_delta=(
                float(spec_b["effective_rank_change"])
                if spec_b["effective_rank_change"] is not None
                else None
            ),
            time=float(stats_b.time),
            s_W=spec_b["s_W"] if with_singulars else None,
            s_Q=spec_b["s_Q"] if with_singulars else None,
        )
    )

    # Sparse + quantized (ALS inner)
    sparse_als = SparseQuantizedDecomposition(adaptive=True)
    Q_a, alpha_a, stats_a = sparse_als.quantize(
        W,
        quantizer="als",
        tau_quantile=0.95,
    )
    W_hat_a = sparse_als.reconstruct(Q_a, stats_a.misc["W_sparse"], alpha_a)
    frob_a = float(frobenius_error(W, W_hat_a, 1.0))
    spec_a = compute_quantization_metrics(W, W_hat_a, 1.0, k=k)

    results.append(
        SolverResult(
            name=sparse_als.name + " [ALS]",
            frob_err=frob_a,
            weighted_dir_err=float(spec_a["weighted_dir_err"]),
            rel_gap_err=float(spec_a["rel_gap_error"]),
            cond_err=(
                float(spec_a["condition_number_error"])
                if spec_a["condition_number_error"] is not None
                else None
            ),
            eff_rank_delta=(
                float(spec_a["effective_rank_change"])
                if spec_a["effective_rank_change"] is not None
                else None
            ),
            time=float(stats_a.time),
            s_W=spec_a["s_W"] if with_singulars else None,
            s_Q=spec_a["s_Q"] if with_singulars else None,
        )
    )

    return results


def aggregate_results(all_results: Iterable[List[SolverResult]]) -> List[SolverResult]:
    """
    all_results: list over matrices, each element a list of SolverResult.
    Returns per-solver averages (singular-value arrays are not averaged).
    """
    by_name: Dict[str, List[SolverResult]] = {}

    for results in all_results:
        for r in results:
            by_name.setdefault(r.name, []).append(r)

    averaged: List[SolverResult] = []
    for name, res_list in by_name.items():
        arr = np.array(
            [
                [
                    r.frob_err,
                    r.weighted_dir_err,
                    r.rel_gap_err,
                    r.cond_err if r.cond_err is not None else np.nan,
                    r.eff_rank_delta if r.eff_rank_delta is not None else np.nan,
                    r.time,
                ]
                for r in res_list
            ],
            dtype=float,
        )
        # compute nanmean for cond_err / eff_rank_delta
        mean = np.nanmean(arr, axis=0)
        averaged.append(
            SolverResult(
                name=name,
                frob_err=float(mean[0]),
                weighted_dir_err=float(mean[1]),
                rel_gap_err=float(mean[2]),
                cond_err=float(mean[3]),
                eff_rank_delta=float(mean[4]),
                time=float(mean[5]),
                s_W=None,
                s_Q=None,
            )
        )

    averaged.sort(key=lambda r: r.name)
    return averaged


def print_results_table(results: List[SolverResult], title: str | None = None) -> None:
    if title:
        print(f"\n=== {title} ===")

    header_cols = [
        "Solver",
        "Time(s)",
        "FrobErr",
        "DirErr",
        "RelSpecGap",
        "CondErr",
        "EffRankΔ",
    ]
    header = f"{header_cols[0]:<30} | " + " | ".join(
        f"{col:<10}" for col in header_cols[1:]
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<30} | "
            f"{r.time:<10.3f} | "
            f"{r.frob_err:<10.4f} | "
            f"{r.weighted_dir_err:<10.4f} | "
            f"{r.rel_gap_err:<10.4f} | "
            f"{(r.cond_err if r.cond_err is not None else float('nan')):<10.4f} | "
            f"{(r.eff_rank_delta if r.eff_rank_delta is not None else float('nan')):<10.4f}"
        )


def save_matrix_as_gray_image(W: np.ndarray, path: str) -> None:
    """
    Save a 2D matrix W as a grayscale image (for visualization only).
    Rescales values to [0,255].
    """
    W_min, W_max = float(W.min()), float(W.max())
    if W_max - W_min < 1e-8:
        arr = np.zeros_like(W, dtype=np.uint8)
    else:
        arr_01 = (W - W_min) / (W_max - W_min)
        arr = (arr_01 * 255.0).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    img.save(path)


def plot_singular_value_histograms(
    W: np.ndarray,
    results: List[SolverResult],
    save_path: str = "singular_values.png",
) -> None:
    """
    Plot histograms of log(1 + singular values) for original W and each quantized solver.
    Uses SolverResult.s_W / s_Q if they are provided; otherwise computes s_W once.
    """
    plt.figure(figsize=(10, 6))

    # Original singular values
    s_W_any = None
    for r in results:
        if r.s_W is not None:
            s_W_any = r.s_W
            break
    if s_W_any is None:
        _, s_W_any, _ = np.linalg.svd(W, full_matrices=False)
    plt.hist(np.log1p(s_W_any), bins=50, alpha=0.5, density=True, label="original")

    # Quantized
    for r in results:
        if r.s_Q is None:
            continue
        plt.hist(
            np.log1p(r.s_Q),
            bins=50,
            alpha=0.5,
            density=True,
            label=r.name,
        )

    plt.xlabel("log(1 + singular value)")
    plt.ylabel("Density")
    plt.title("Singular Value Distributions of Quantized Matrices")
    plt.legend()
    plt.grid(True, ls="--", lw=0.5)

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Figure saved to {save_path}")


def run_on_real_weights() -> None:
    real_weights = get_real_weights()
    all_results: List[List[SolverResult]] = []

    for i, W in enumerate(real_weights[:10]):
        print(i)
        all_results.append(run_on_matrix(W))

    avg_results = aggregate_results(all_results)
    print_results_table(avg_results, title="AVERAGED OVER REAL WEIGHTS")


def run_on_synthetic_with_plots() -> None:
    """
    Run on one synthetic matrix, print detailed metrics, save singular-value histogram
    and original matrix as an image.
    """
    print("Generating synthetic data...")
    W = generate_synthetic_matrix("gaussian", (1024, 1024))

    save_matrix_as_gray_image(W, "original_gray.png")
    print("Original matrix saved as grayscale image to original_gray.png")

    results = run_on_matrix(W, with_singulars=True, k=50)
    print_results_table(results, title="SYNTHETIC (gaussian, 1024x1024)")

    plot_singular_value_histograms(W, results, save_path="singular_values.png")


def run_on_synthetic_average() -> None:
    """
    Run on multiple synthetic matrices and print per-solver averages.
    """
    all_results: List[List[SolverResult]] = []

    for dist in ["gaussian", "laplace", "uniform"]:
        for size in [10, 100, 1000]:
            W = generate_synthetic_matrix(dist, (size, size))
            results = run_on_matrix(W)
            print_results_table(results, title=f"{dist}_{size}")
            all_results.append(results)

    avg_results = aggregate_results(all_results)
    print_results_table(avg_results, title="AVERAGED OVER ALL SYNTHETICS")


if __name__ == "__main__":
    # 1) Single synthetic matrix with images + singular-value plots:
    # run_on_synthetic_with_plots()

    # 2) Or averaged benchmarks:
    run_on_real_weights()
    # run_on_synthetic_average()
