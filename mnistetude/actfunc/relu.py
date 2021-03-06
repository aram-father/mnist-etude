import numpy as np
from . import iactfunc


class Relu(iactfunc.IActFunc):
    def __init__(self):
        pass

    def activate(self, x: np.array) -> np.array:
        return np.maximum(0, x)
