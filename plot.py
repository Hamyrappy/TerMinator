import matplotlib.pyplot as plt
import numpy as np

from data import generate_synthetic_matrix, get_real_weights
from solvers import SparseQuantizedDecomposition

np.random.seed(69)


def run_tau_sweep():
    shape = (100, 100)
    distributions: list[Literal["gaussian", "laplace", "uniform"]] = [
        "gaussian",
        "laplace",
        "uniform",
    ]

    # add “real-bert” as a fourth curve
    dist_labels = {
        "gaussian": "Gaussian",
        "laplace": "Laplace",
        "uniform": "Uniform",
        "real-bert": "BERT weights",
    }

    tau_quantiles = np.linspace(0.01, 0.99, 99)

    results: dict[str, dict[str, list[float]]] = {
        dist: {"q": [], "rel_err": [], "sparse_frac": []}
        for dist in list(distributions) + ["real-bert"]
    }

    # synthetic distributions
    for dist in distributions:
        W = generate_synthetic_matrix(dist, shape)
        sq = SparseQuantizedDecomposition(adaptive=True)

        for q in tau_quantiles:
            Q_dense, alpha, stats = sq.quantize(
                W,
                quantizer="als",  # or "bitnet"
                tau_quantile=float(q),
                quantizer_init_kwargs={"max_iter": 20},
                quantizer_kwargs={},
            )
            W_sparse = stats.misc["W_sparse"]
            W_hat = sq.reconstruct(Q_dense, W_sparse, beta=alpha)
            rel_err = np.linalg.norm(W - W_hat, "fro") / np.linalg.norm(W, "fro")

            results[dist]["q"].append(q)
            results[dist]["rel_err"].append(rel_err)
            results[dist]["sparse_frac"].append(stats.misc["sparse_fraction"])

    # real BERT weights: take one Linear layer and crop/resize to 100x100
    bert_weights = get_real_weights("bert-base-uncased")
    if len(bert_weights) == 0:
        raise RuntimeError("No Linear weights found in BERT model")

    W_full = bert_weights[0]  # first Linear layer
    # center crop or pad to 100x100
    n, m = W_full.shape
    W = np.zeros(shape, dtype=W_full.dtype)
    nn = min(shape[0], n)
    mm = min(shape[1], m)
    W[:nn, :mm] = W_full[:nn, :mm]

    sq_real = SparseQuantizedDecomposition(adaptive=True)
    dist = "real-bert"

    for q in tau_quantiles:
        Q_dense, alpha, stats = sq_real.quantize(
            W,
            quantizer="als",
            tau_quantile=float(q),
            quantizer_init_kwargs={"max_iter": 20},
            quantizer_kwargs={},
        )
        W_sparse = stats.misc["W_sparse"]
        W_hat = sq_real.reconstruct(Q_dense, W_sparse, beta=alpha)
        rel_err = np.linalg.norm(W - W_hat, "fro") / np.linalg.norm(W, "fro")

        results[dist]["q"].append(q)
        results[dist]["rel_err"].append(rel_err)
        results[dist]["sparse_frac"].append(stats.misc["sparse_fraction"])

    # Plot: relative error vs tau-quantile
    plt.figure(figsize=(6, 4))
    for key in ["gaussian", "laplace", "uniform", "real-bert"]:
        plt.plot(results[key]["q"], results[key]["rel_err"], label=dist_labels[key])
    plt.xlabel("τ quantile (|W|)")
    plt.ylabel("Relative Frobenius error")
    plt.title("Reconstruction error for different τ (100×100, ALS on dense part)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("error_for_different.pdf")


run_tau_sweep()
