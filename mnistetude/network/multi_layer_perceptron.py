from .. import layer
from typing import List
import numpy as np


class MultiLayerPerceptron:
    def __init__(self, layers: List[layer.Perceptron]):
        self.layers = layers

    def predict(self, image: np.array) -> np.array:
        forwarded = image
        for layer in self.layers:
            forwarded = layer.forward(forwarded)

        return forwarded
