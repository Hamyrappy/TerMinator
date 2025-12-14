import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Union, Optional

# Тип для Alpha: может быть числом (Global) или вектором (Row-wise)
ScaleType = Union[float, np.ndarray]

class BaseQuantizer(ABC):
    """
    Abstract Base Class that all solvers must inherit from.
    Enforces a strict interface for collaboration.
    """
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def quantize(self, W: np.ndarray) -> Tuple[np.ndarray, ScaleType, Dict]:
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