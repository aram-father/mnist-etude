import numpy as np


class ILossFunc:
    def __init__(self):
        pass

    def get_loss(self, prediction: np.array, label: np.array) -> np.array:
        raise
