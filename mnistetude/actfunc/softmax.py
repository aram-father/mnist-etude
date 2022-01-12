import numpy as np


def softmax(x: np.array) -> np.array:
    constant = np.max(x)
    numerator = np.exp(x - constant)
    denominator = np.sum(numerator)
    return numerator / denominator
