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

    def __init__(self, name: str):
        self.name = name

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
