import numpy as np


def relu(x: np.array) -> np.array:
    return np.maximum(0, x)
