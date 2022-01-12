import numpy as np


def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))
