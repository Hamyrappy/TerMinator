import numpy as np

# ! Пока не напишем подтягивание нормальных данных - используем эту заглушку. 
def get_dummy_data(N=512, M=512) -> np.ndarray:
    """Fast random data for debugging pipelines."""
    return np.random.randn(N, M)

def get_synthetic_matrix(distribution: str, shape: tuple, **kwargs) -> np.ndarray:
    """
    Generate synthetic weights.
    Args:
        distribution: 'gaussian', 'laplace', 'uniform'
        shape: (N, M)
    """
    # TODO: Implement proper distributions with heavy tails if needed
    if distribution == 'gaussian':
        return np.random.randn(*shape)
    elif distribution == 'laplace':
        return np.random.laplace(loc=0.0, scale=1.0, size=shape)
    else:
        raise ValueError(f"Unknown dist: {distribution}")

def get_real_weights(model_name="bert-base-uncased") -> list[np.ndarray]:
    """
    Load real weights from HuggingFace (requires `transformers` lib).
    Returns a list of matrices (layers).
    """
    # TODO: Load model, extract Linear layer weights, convert to numpy
    # return [W1, W2, ...]
    pass