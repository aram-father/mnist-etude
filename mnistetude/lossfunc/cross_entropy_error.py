import numpy as np
from . import ilossfunc


class CrossEntropyError(ilossfunc.ILossFunc):
    def __init__(self):
        pass

    def get_loss(self, prediction: np.array, label: np.array) -> np.array:
        if prediction.ndim == 1:
            prediction = prediction.reshape(1, prediction.size)
            label = label.reshape(1, label.size)

        batch_size = prediction.shape[0]

        eps = 1e-7
        return -np.sum(label * np.log(prediction + eps)) / batch_size
