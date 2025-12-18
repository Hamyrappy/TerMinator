from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np

# Scale can be a vector if compression is Row-wise
type ScaleType = float | np.ndarray


@dataclass
class Stats(object):
    time: float
    misc: dict[Any, Any] = field(default_factory=dict)


class BaseQuantizer(ABC):
    """
    Abstract Base Class that all solvers must inherit from.
    Enforces a strict interface for collaboration.
    """

    name: str

    def __init__(self, name: str, init_with_power: bool = False):
        self.name = name + " svd" if init_with_power else name
        self.init_with_power = init_with_power

    @abstractmethod
    def quantize(self, W: np.ndarray) -> Tuple[np.ndarray, ScaleType, Stats]:
        """
        Main method to compress the matrix.

        Args:
            W (np.ndarray): High-precision input matrix (N, M).

        Returns:
            Q (np.ndarray): Ternary matrix {-1, 0, 1} of shape (N, M).
            alpha (ScaleType): Scaling factor(s).
                               Scalar if global, (N, 1) if row-wise.
            stats (Dict): Logs for analysis (e.g., {'loss_history': [...], 'time': 0.5})
        """
        pass

    def __repr__(self):
        return f"Quantizer[{self.name}]"
    

    def initialize_with_svd(
        self,
        W: np.ndarray, 
        energy_threshold: float = 0.99,
        max_rank: int = None
    ) -> np.ndarray:
        """
        Low-rank approximation of W using truncated SVD.
        
        Args:
            W: matrix to approximate
            energy_threshold: fraction of total energy to preserve
            num_iter: iterations (not used here, left for Power Method)
            eps: small number to avoid division by zero
            max_rank: optional upper bound on rank
        
        Returns:
            W_approx: low-rank approximation preserving energy_threshold
        """
        u, s, vh = np.linalg.svd(np.array(W), full_matrices=False)

        energy = np.cumsum(s**2) / np.sum(s**2)

        k = np.searchsorted(energy, energy_threshold) + 1
        if max_rank is not None:
            k = min(k, max_rank)

        W_approx = u[:, :k] @ np.diag(s[:k]) @ vh[:k, :]
        return W_approx



    def initialize_with_power_method(
        self,
        W: np.ndarray, 
        energy_threshold: float = 0.99,
        num_iter: int = 20,
        eps: float = 1e-8,
        max_rank: int = None
    ) -> np.ndarray:
        """
        Low-rank approximation of W using truncated SVD.
        
        Args:
            W: matrix to approximate
            energy_threshold: fraction of total energy to preserve
            num_iter: iterations (not used here, left for Power Method)
            eps: small number to avoid division by zero
            max_rank: optional upper bound on rank
        
        Returns:
            W_approx: low-rank approximation preserving energy_threshold
        """
        u, s, vh = np.linalg.svd(np.array(W), full_matrices=False)

        energy = np.cumsum(s**2) / np.sum(s**2)

        k = np.searchsorted(energy, energy_threshold) + 1
        if max_rank is not None:
            k = min(k, max_rank)

        W_approx = u[:, :k] @ np.diag(s[:k]) @ vh[:k, :]
        return W_approx


    def initialize_with_power_method(self, W: np.ndarray, num_iter: int = 20, eps: float = 1e-8) -> np.ndarray:
        """
        Spectral initialization for W using Power Method.

        Returns a rank-1 approximation W_hat = sigma * u v^T
        which can be used as initialization for quantization.

        Args:
            W: High-precision input matrix (N, M).
            num_iter: Number of power iterations
            eps: Small epsilon to avoid division by zero

        Returns:
            W_hat: Rank-1 approximation of W
        """
        m, n = W.shape
        v = np.random.randn(n)
        v /= np.linalg.norm(v) + eps

        for _ in range(num_iter):
            u = W @ v
            u_norm = np.linalg.norm(u) + eps
            u /= u_norm

            v_next = W.T @ u
            v_next /= np.linalg.norm(v_next) + eps

            if np.linalg.norm(v - v_next) < eps:
                break
            v = v_next

        sigma = u.T @ W @ v
        W_hat = sigma * np.outer(u, v)
        return W_hat

