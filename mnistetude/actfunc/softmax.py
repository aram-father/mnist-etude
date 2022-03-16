import numpy as np
from . import iactfunc


class Softmax(iactfunc.IActFunc):
    def __init__(self):
        pass

    def activate(self, x: np.array) -> np.array:
        constant = np.max(x, axis=1).reshape(-1, 1)
        numerator = np.exp(x - constant)
        denominator = np.sum(numerator, axis=1).reshape(-1, 1)
        return numerator / denominator
