import numpy as np
from . import iactfunc


class Identity(iactfunc.IActFunc):
    def __init__(self):
        pass

    def activate(self, x: np.array) -> np.array:
        return x
