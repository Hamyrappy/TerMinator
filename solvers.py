from __future__ import annotations

import time
from typing import Any, Literal, Tuple, cast, override

import numpy as np

from base import BaseQuantizer, ScaleType, Stats


# see https://arxiv.org/pdf/2402.17764
class NaiveQuantizer(BaseQuantizer):
    def __init__(self, eps: float = 1e-8):
        super().__init__("Naive (BitNet-ish)")
        self.eps = eps

    @override
    def quantize(self, W: np.ndarray) -> Tuple[np.ndarray, ScaleType, Stats]:
        start_time = time.time()

        alpha = float(np.mean(np.abs(W)))
        if alpha <= 0.0:
            Q = np.zeros_like(W)
            elapsed = time.time() - start_time
            return Q, 0.0, Stats(time=elapsed, misc={"loss": 0.0})

        Q = np.clip(np.rint(W / (alpha + self.eps)), -1.0, 1.0).astype(
            W.dtype, copy=False
        )

        elapsed = time.time() - start_time
        return Q, alpha, Stats(time=elapsed)


# https://math.mit.edu/~stevenj/18.335/norm-equivalence.pdf
class ALSQuantizer(BaseQuantizer):
    def __init__(
        self,
        max_iter: int = 50,
        row_wise: bool = False,
        eps: float = 1e-8,
        enforce_nonneg_alpha: bool = True,
        keep_alpha_on_zero_q: bool = False,
    ):
        name = f"ALS{' (Row-wise)' if row_wise else ''}"
        super().__init__(name)
        self.max_iter = int(max_iter)
        self.row_wise = bool(row_wise)
        self.eps = float(eps)
        self.enforce_nonneg_alpha = bool(enforce_nonneg_alpha)
        self.keep_alpha_on_zero_q = bool(keep_alpha_on_zero_q)

    def _init_alpha(self, W: np.ndarray) -> np.ndarray | float:
        if self.row_wise:
            alpha = np.mean(np.abs(W), axis=1, keepdims=True)
            return alpha + self.eps
        return float(np.mean(np.abs(W)) + self.eps)

    def _update_q(self, W: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
        X = W / (alpha + self.eps)
        Q = np.clip(np.rint(X), -1.0, 1.0).astype(W.dtype, copy=False)
        return Q

    def _update_alpha(
        self, W: np.ndarray, Q: np.ndarray, alpha_prev: np.ndarray | float
    ) -> np.ndarray | float:
        if self.row_wise:
            num = np.sum(W * Q, axis=1, keepdims=True)
            den = np.sum(Q * Q, axis=1, keepdims=True)

            if self.keep_alpha_on_zero_q:
                alpha = np.where(den > 0, num / (den + self.eps), alpha_prev)
            else:
                alpha = np.where(den > 0, num / den, 0.0)

            if self.enforce_nonneg_alpha:
                alpha = np.maximum(alpha, 0.0)
            return alpha

        num = float(np.sum(W * Q))
        den = float(np.sum(Q * Q))
        if den > 0.0:
            alpha = num / den
        else:
            alpha = float(alpha_prev) if self.keep_alpha_on_zero_q else 0.0

        if self.enforce_nonneg_alpha:
            alpha = max(alpha, 0.0)
        return alpha

    @override
    def quantize(self, W: np.ndarray) -> Tuple[np.ndarray, ScaleType, Stats]:
        """
        Minimize ||W - alpha * Q||_F^2 over ternary Q ∈ {-1,0,1}^{N×M} and scale(s) alpha.
        If row_wise is False: alpha is scalar.
        If row_wise is True: alpha is (N, 1) per-row scale.
        """
        start = time.time()

        alpha = self._init_alpha(W)
        loss_history: list[float] = []

        for _ in range(self.max_iter):
            Q = self._update_q(W, alpha)
            alpha_new = self._update_alpha(W, Q, alpha)

            W_hat = alpha_new * Q
            diff = W - W_hat
            loss = float(np.square(diff).sum())
            loss_history.append(loss)

            alpha = alpha_new

        elapsed = time.time() - start
        return Q, alpha, Stats(time=elapsed, misc={"loss_history": loss_history})


class SparseQuantizedDecomposition(BaseQuantizer):
    tau: float
    adaptive: bool

    def __init__(self, tau: float | None = None, adaptive: bool = False):
        """
        If adaptive is False: use fixed tau (must be not None).
        If adaptive is True: ignore tau and choose it from W statistics.
        """
        name = f"Sparse + Quantized" if adaptive else f"Sparse + Quantized (tau={tau})"
        super().__init__(name)
        self.tau = float(tau) if tau is not None else 0.0
        self.adaptive = adaptive

    def _compute_tau(self, W: np.ndarray, q: float) -> float:
        """
        Choose tau based on |W| statistics.
        Examples:
          - std-based: mu + k * sigma
          - quantile-based: e.g. 99.9% quantile (keep top 0.1% as sparse),
            similar to “≈0.1% of feature dimensions” in LLM.int8(). [web:64]
        """
        absW = np.abs(W).ravel()

        tau_q = float(np.quantile(absW, q))

        return tau_q

    @override
    def quantize(
        self,
        W: np.ndarray,
        quantizer: Literal["bitnet", "als"] = "bitnet",
        tau_quantile: float = 0.99,
        quantizer_init_kwargs: dict[Any, Any] | None = None,
        quantizer_kwargs: dict[Any, Any] | None = None,
    ) -> Tuple[np.ndarray, ScaleType, Stats]:
        start_time = time.time()
        if quantizer_init_kwargs is None:
            quantizer_init_kwargs = {}
        if quantizer_kwargs is None:
            quantizer_kwargs = {}

        if self.adaptive:
            tau = self._compute_tau(W, q=tau_quantile)
        else:
            tau = self.tau

        mask = np.abs(W) > tau
        W_sparse = W * mask
        W_dense = W - W_sparse

        if quantizer == "bitnet":
            solver = NaiveQuantizer(**quantizer_init_kwargs)
        elif quantizer == "als":
            solver = ALSQuantizer(**quantizer_init_kwargs)
        else:
            raise ValueError(f"Unknown quantizer '{quantizer}'")

        Q_dense, alpha, inner_stats = solver.quantize(W_dense, **quantizer_kwargs)

        elapsed = time.time() - start_time
        return (
            Q_dense,
            alpha,
            Stats(
                time=elapsed,
                misc={
                    "mask": mask,
                    "W_sparse": W_sparse,
                    "tau": tau,
                    "inner_stats": inner_stats,
                    "sparse_fraction": float(mask.sum()) / mask.size,
                },
            ),
        )

    def reconstruct(
        self, Q_dense: np.ndarray, W_sparse: np.ndarray, beta: float
    ) -> np.ndarray:
        return W_sparse + beta * Q_dense
