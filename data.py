from typing import Literal

import numpy as np
import torch
from transformers import AutoModel


# ! Пока не напишем подтягивание нормальных данных - используем эту заглушку.
def get_dummy_data(shape: tuple[int, int]) -> np.ndarray:
    """Fast random data for debugging pipelines."""
    N, M = shape
    return np.random.randn(N, M)


def generate_synthetic_matrix(
    distribution: Literal["gaussian", "laplace", "uniform"],
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Generate synthetic weights.
    Args:
        distribution: 'gaussian', 'laplace', 'uniform'
        shape: (N, M)
    """
    if distribution == "gaussian":
        return np.random.randn(*shape)
    elif distribution == "laplace":
        return np.random.laplace(loc=0.0, scale=1.0, size=shape)
    elif distribution == "uniform":
        return np.random.uniform(size=shape)

    raise ValueError(f"Unknown dist: {distribution}")


def get_real_weights(
    model_name: Literal["bert-base-uncased"] = "bert-base-uncased",
) -> list[np.ndarray]:
    """
    Load real weights from HuggingFace (requires `transformers` lib).
    Returns a list of matrices (layers) as NumPy arrays.
    """
    model = AutoModel.from_pretrained(model_name)

    weight_matrices: list[np.ndarray] = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            W = module.weight.detach().cpu().numpy().astype(np.float32)
            weight_matrices.append(W)

    return weight_matrices
