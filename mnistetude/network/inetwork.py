import numpy as np
from .. import layer
from .. import lossfunc
from typing import List


class INetwork:
    def __init__(self, layers: List[layer.ILayer]):
        self.layers = layers

    def predict(self, image: np.array) -> np.array:
        raise

    def get_loss(self, image: np.array, label: np.array) -> np.array:
        prediction = self.predict(image)
        return lossfunc.CrossEntropyError().get_loss(prediction, label)
