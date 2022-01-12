import numpy as np
from typing import Callable


class Perceptron:
    def __init__(self, weight: np.array, bias: np.array, actfunc: Callable):
        self.num_of_inputs, self.num_of_outputs = weight.shape
        self.weight = weight
        self.bias = bias
        self.actfunc = actfunc

    def forward(self, inputs: np.array) -> np.array:
        return self.actfunc(inputs.dot(self.weight) + self.bias)
