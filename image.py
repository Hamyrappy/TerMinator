from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image

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


def load_image_as_matrix(path: str) -> tuple[np.ndarray, tuple[int, int], float]:
    """
    Load image as grayscale float32, subtract mean so values are centered around 0.
    Returns centered matrix W (H, W), original size, and mean (in [0,255]).
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)  # [0,255]
    mean = float(arr.mean())
    W = arr - mean  # roughly symmetric around 0
    return W, img.size, mean  # size = (W, H)


def save_matrix_as_image(
    W_hat: np.ndarray, size: tuple[int, int], out_path: str
) -> None:
    """
    Save 2D matrix W_hat (in pixel space, roughly [0,255]) as grayscale image.
    """
    W_hat_clipped = np.clip(W_hat, 0.0, 255.0).astype(np.uint8)
    img_q = Image.fromarray(W_hat_clipped)
    img_q = img_q.resize(size)
    img_q.save(out_path)


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


def quantize_image(path: str, out_dir: str = "quantized_out", k: int = 50) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Save original for visual comparison
    orig_img = Image.open(path).convert("L")
    orig_img.save(os.path.join(out_dir, "img_original_gray.png"))

    # Centered weights W ∈ ℝ^{H×W}, mean in [0,255]
    W, size, mean = load_image_as_matrix(path)

    results: List[SolverResult] = []

    # 1) Naive (BitNet-ish ternary)
    naive = NaiveQuantizer()
    Q_n, alpha_n, stats_n = naive.quantize(W)  # Q_n ∈ {-1,0,1}
    W_hat_naive = alpha_n * Q_n + mean  # back to pixel space
    save_matrix_as_image(W_hat_naive, size, os.path.join(out_dir, "img_naive.png"))

    frob = float(frobenius_error(W, Q_n, alpha_n))
    spec = compute_quantization_metrics(W, Q_n, alpha_n, k=k)
    results.append(
        SolverResult(
            name=naive.name,
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
            time=float(stats_n.time),
        )
    )

    # 2a) ALS (global alpha)
    als_global = ALSQuantizer(max_iter=50, row_wise=False)
    Q_ag, alpha_ag, stats_ag = als_global.quantize(W)
    W_hat_als_global = alpha_ag * Q_ag + mean
    save_matrix_as_image(
        W_hat_als_global, size, os.path.join(out_dir, "img_als_global.png")
    )

    frob = float(frobenius_error(W, Q_ag, alpha_ag))
    spec = compute_quantization_metrics(W, Q_ag, alpha_ag, k=k)
    results.append(
        SolverResult(
            name=als_global.name,
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
            time=float(stats_ag.time),
        )
    )

    # 2b) ALS (row-wise alpha)
    als_row = ALSQuantizer(max_iter=50, row_wise=True)
    Q_ar, alpha_ar, stats_ar = als_row.quantize(W)
    W_hat_als_row = alpha_ar * Q_ar + mean
    save_matrix_as_image(
        W_hat_als_row, size, os.path.join(out_dir, "img_als_row_wise.png")
    )

    frob = float(frobenius_error(W, Q_ar, alpha_ar))
    spec = compute_quantization_metrics(W, Q_ar, alpha_ar, k=k)
    results.append(
        SolverResult(
            name=als_row.name,
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
            time=float(stats_ar.time),
        )
    )

    # 3a) Sparse + Quantized decomposition (adaptive tau, bitnet inner)
    sparse_bitnet = SparseQuantizedDecomposition(adaptive=True)
    Q_db, alpha_db, stats_db = sparse_bitnet.quantize(
        W,
        quantizer="bitnet",
        tau_quantile=0.95,
    )
    W_sparse_b = stats_db.misc["W_sparse"]
    W_hat_sparse_b = sparse_bitnet.reconstruct(Q_db, W_sparse_b, alpha_db) + mean
    save_matrix_as_image(
        W_hat_sparse_b, size, os.path.join(out_dir, "img_sparse_bitnet.png")
    )

    frob = float(frobenius_error(W, W_hat_sparse_b - mean, 1.0))
    spec = compute_quantization_metrics(W, W_hat_sparse_b - mean, 1.0, k=k)
    results.append(
        SolverResult(
            name=sparse_bitnet.name + " [bitnet]",
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
            time=float(stats_db.time),
        )
    )

    # 3b) Sparse + Quantized decomposition (adaptive tau, ALS inner)
    sparse_als = SparseQuantizedDecomposition(adaptive=True)
    Q_da, alpha_da, stats_da = sparse_als.quantize(
        W,
        quantizer="als",
        tau_quantile=0.95,
    )
    W_sparse_a = stats_da.misc["W_sparse"]
    W_hat_sparse_a = sparse_als.reconstruct(Q_da, W_sparse_a, alpha_da) + mean
    save_matrix_as_image(
        W_hat_sparse_a, size, os.path.join(out_dir, "img_sparse_als.png")
    )

    frob = float(frobenius_error(W, W_hat_sparse_a - mean, 1.0))
    spec = compute_quantization_metrics(W, W_hat_sparse_a - mean, 1.0, k=k)
    results.append(
        SolverResult(
            name=sparse_als.name + " [ALS]",
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
            time=float(stats_da.time),
        )
    )

    print_results_table(results, title="IMAGE QUANTIZATION METRICS")
    print(f"Saved quantized images to: {out_dir}")


if __name__ == "__main__":
    quantize_image("./shreck.jpg")
