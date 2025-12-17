import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svdvals

def frobenius_error(W: np.ndarray, Q: np.ndarray, alpha: float) -> float:
    """Relative Frobenius Norm Error: ||W - aQ||_F / ||W||_F"""
    reconstruction = alpha * Q
    # Note: alpha broadcasting works automatically for both scalar and (N,1) vector
    diff = W - reconstruction
    return np.linalg.norm(diff, 'fro') / np.linalg.norm(W, 'fro')

def spectral_analysis(W: np.ndarray, Q: np.ndarray, alpha: float, k: int = None) -> dict:
    """
    Advanced NLA metrics.
    Compares Singular Values of W and Reconstruction.
    """
    # TODO: Use scipy.sparse.linalg.svds for speed if matrix is huge
    # 1. Compute svd(W) -> s_W
    # 2. Compute svd(alpha * Q) -> s_rec
    # 3. Return difference stats
    min_dim = min(W.shape)
    use_sparse = min_dim > 1024
    if k is None:
        k = min(50, min_dim - 1)
    if use_sparse:
        s_w = svds(W, k=k, return_singular_vectors=False)
        s_rec = svds(alpha * Q, k=k, return_singular_vectors=False)
        s_w = np.sort(s_w)[::-1]
        s_rec = np.sort(s_rec)[::-1]
    else:
        s_w = svdvals(W)[:k]
        s_rec = svdvals(alpha * Q)[:k]
    
    gap_W = s_w[0] - s_w[1] if len(s_w) > 1 else 0
    gap_rec = s_rec[0] - s_rec[1] if len(s_rec) > 1 else 0
    
    top1_error = np.abs(s_w[0] - s_rec[0]) / s_w[0]

    min_len = min(len(s_w), len(s_rec))
    sv_rmse = np.sqrt(np.mean((s_w[:min_len] - s_rec[:min_len])**2))

    energy_w = np.cumsum(s_w[:min_len] ** 2)
    energy_rec = np.cumsum(s_rec[:min_len] ** 2)
    energy_err = np.mean(np.abs(energy_w - energy_rec) / energy_w)

    return {
        "top_singular_value_err": float(top1_error),
        "spectral_gap_diff": float(np.abs(gap_W - gap_rec)),
        "singular_values_rmse": float(sv_rmse),
        "spectral_energy_err": float(energy_err),
        "k_used": k,
        "is_sparse_mode": use_sparse
    }
