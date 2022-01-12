import numpy as np


def step(x: np.array) -> np.array:
    return (x > 0).astype(np.int)
