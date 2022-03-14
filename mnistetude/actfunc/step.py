import numpy as np
from . import iactfunc


class Step(iactfunc.IActFunc):
    def __init__(self):
        pass

    def activate(self, x: np.array) -> np.array:
        return (x > 0).astype(np.int)
