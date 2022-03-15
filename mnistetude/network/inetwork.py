import numpy as np
from .. import layer
from typing import List


class INetwork:
    def __init__(self, layers: List[layer.ILayer]):
        self.layers = layers

    def predict(self, image: np.array) -> np.array:
        raise
