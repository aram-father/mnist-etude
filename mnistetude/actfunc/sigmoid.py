import numpy as np
from . import iactfunc


class Sigmoid(iactfunc.IActFunc):
    def __init__(self):
        pass

    def activate(self, x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))
