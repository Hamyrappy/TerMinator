import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svdvals, svd

def frobenius_error(W: np.ndarray, Q: np.ndarray, alpha: float) -> float:
    """Relative Frobenius Norm Error: ||W - aQ||_F / ||W||_F"""
    reconstruction = alpha * Q
    # Note: alpha broadcasting works automatically for both scalar and (N,1) vector
    diff = W - reconstruction
    return np.linalg.norm(diff, 'fro') / np.linalg.norm(W, 'fro')

def compute_quantization_metrics(W: np.ndarray, Q: np.ndarray, alpha: float, k: int = 50, eps: float = 1e-8, X: np.ndarray = None) -> dict:
    """
    Advanced NLA metrics.
    Compares Singular Values of W and Reconstruction.
    """
    # TODO: Use scipy.sparse.linalg.svds for speed if matrix is huge
    # 1. Compute svd(W) -> s_W
    # 2. Compute svd(alpha * Q) -> s_rec
    # 3. Return difference stats

    W_hat = alpha * Q
    min_dim = min(W.shape)
    use_sparse = min_dim > 1024
    k = min(k, min_dim - 1)

    if use_sparse:
        U_W, s_W, _ = svds(W, k=k)
        U_Q, s_Q, _ = svds(W_hat, k=k)
        order_W = np.argsort(s_W)[::-1]
        order_Q = np.argsort(s_Q)[::-1]
        U_W, s_W = U_W[:, order_W], s_W[order_W]
        U_Q, s_Q = U_Q[:, order_Q], s_Q[order_Q]
        effective_rank_change = None
        cond_err = None
    else:
        U_W, s_W, _ = svd(W, full_matrices=False)
        U_Q, s_Q, _ = svd(W_hat, full_matrices=False)
        eff_rank_W = np.sum(s_W**2) / (s_W[0]**2 + eps)
        eff_rank_Q = np.sum(s_Q**2) / (s_Q[0]**2 + eps)
        effective_rank_change = abs(eff_rank_W - eff_rank_Q)
        kappa_W = s_W[0] / max(s_W[-1], eps)
        kappa_Q = s_Q[0] / max(s_Q[-1], eps)
        cond_err = abs(kappa_W - kappa_Q) / kappa_W

    k_sub = min(k, U_W.shape[1], U_Q.shape[1])

    alignments = np.abs(np.sum(U_W[:, :k_sub] * U_Q[:, :k_sub], axis=0))
    weights = s_W[:k_sub] / np.sum(s_W[:k_sub])
    weighted_dir_err = np.sum(weights * (1.0 - alignments))

    gap_W = s_W[0] - s_W[1] if len(s_W) > 1 else 0
    gap_Q = s_Q[0] - s_Q[1] if len(s_Q) > 1 else 0
    gap_err = abs(gap_W - gap_Q)
    rel_gap_err =  gap_err/ (gap_W + eps * (len(s_W) == 1))

    rho_out = None
    if X is not None:
        Y = W @ X
        Y_hat = W_hat @ X
        numerator = np.trace(Y.T @ Y_hat)
        denominator = np.linalg.norm(Y, 'fro') * np.linalg.norm(Y_hat, 'fro')
        rho_out = numerator / denominator


    return {
        "weighted_dir_err": float(weighted_dir_err),
        "condition_number_error": float(cond_err) if cond_err is not None else None,
        "rel_gap_error": float(rel_gap_err),
        "effective_rank_change": float(effective_rank_change) if effective_rank_change is not None else None,
        "out_correlation": float(rho_out) if rho_out is not None else None,
        "k_used": k_sub,
        "is_sparse_mode": use_sparse
    }

