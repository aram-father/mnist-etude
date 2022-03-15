from . import inetwork
from .. import layer
from typing import List
import numpy as np


class MultiLayerPerceptron(inetwork.INetwork):
    def __init__(self, layers: List[layer.ILayer]):
        super().__init__(layers)

    def predict(self, image: np.array) -> np.array:
        forwarded = image
        for layer in self.layers:
            forwarded = layer.forward(forwarded)

        return forwarded
