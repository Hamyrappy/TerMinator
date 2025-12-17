from __future__ import annotations

import time
from typing import Tuple, cast, override

import numpy as np

from base import BaseQuantizer, ScaleType, Stats
from metrics import frobenius_error


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
        max_iter: int = 5,
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
        # Simple, robust initialization: mean absolute value.
        if self.row_wise:
            # (N, 1)
            alpha = np.mean(np.abs(W), axis=1, keepdims=True)
            return alpha + self.eps
        return float(np.mean(np.abs(W)) + self.eps)

    def _update_q(self, W: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
        # Q-step: ternary projection minimizing ||W - alpha*Q||_F^2 given alpha.
        # Uses thresholding via round(W/alpha) then clipping to {-1,0,1}.
        X = W / (alpha + self.eps)
        # Keep Q float dtype to avoid repeated casts; or switch to int8 if you prefer.
        Q = np.clip(np.rint(X), -1.0, 1.0).astype(W.dtype, copy=False)
        return Q

    def _update_alpha(
        self, W: np.ndarray, Q: np.ndarray, alpha_prev: np.ndarray | float
    ) -> np.ndarray | float:
        # alpha-step: least squares fit.
        if self.row_wise:
            # (N,1)
            num = np.sum(W * Q, axis=1, keepdims=True)
            den = np.sum(Q * Q, axis=1, keepdims=True)

            if self.keep_alpha_on_zero_q:
                # Keep previous alpha when den == 0
                alpha = np.where(den > 0, num / (den + self.eps), alpha_prev)
            else:
                # Set alpha to 0 when den == 0 (well-defined; avoids blow-ups)
                alpha = np.where(den > 0, num / den, 0.0)

            if self.enforce_nonneg_alpha:
                alpha = np.maximum(alpha, 0.0)
            return alpha

        # scalar
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

            # Compute loss efficiently: sum of squares
            W_hat = alpha_new * Q
            diff = W - W_hat
            loss = float(np.square(diff).sum())
            loss_history.append(loss)

            alpha = alpha_new

        elapsed = time.time() - start
        return Q, alpha, Stats(time=elapsed, misc={"loss_history": loss_history})


# https://arxiv.org/pdf/2208.07339
class SparseQuantizedDecomposition(BaseQuantizer):
    tau: float

    def __init__(self, tau: float):
        super().__init__("Sparse + Quantized")
        self.tau = float(tau)

    @override
    def quantize(self, W: np.ndarray) -> Tuple[np.ndarray, float, Stats]:
        start_time = time.time()

        mask = np.abs(W) > self.tau
        W_sparse = W * mask
        W_dense = W - W_sparse

        Q_dense, beta, stats_dense = NaiveQuantizer().quantize(W_dense)

        elapsed = time.time() - start_time
        return (
            Q_dense,
            beta,
            Stats(
                time=elapsed,
                misc={
                    "mask": mask,
                    "W_sparse": W_sparse,
                    "dense_stats": getattr(stats_dense, "misc", None),
                },
            ),
        )
