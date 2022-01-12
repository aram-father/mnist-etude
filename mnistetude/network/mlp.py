from ..actfunc import *
from ..layer import *
from typing import Dict
import numpy as np


class Mlp:
    def __init__(self, params: Dict[str, np.array]):
        self.input_layer = Perceptron(
            params['input_layer_weights'], params['input_layer_biases'], sigmoid)
        self.hidden_layer = Perceptron(
            params['hidden_layer_weights'], params['hidden_layer_biases'], sigmoid)
        self.output_layer = Perceptron(
            params['output_layer_weights'], params['output_layer_biases'], softmax)

    def predict(self, image: np.array) -> np.array:
        neurons_0_1 = self.input_layer.forward(image)
        neurons_1_2 = self.hidden_layer.forward(neurons_0_1)
        return self.output_layer.forward(neurons_1_2)
