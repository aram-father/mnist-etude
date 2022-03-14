import numpy as np
from .. import actfunc


class Perceptron:
    def __init__(self, weight: np.array, bias: np.array, actfunc: actfunc.IActFunc):
        self.num_of_inputs, self.num_of_outputs = weight.shape
        self.weight = weight
        self.bias = bias
        self.actfunc = actfunc

    def forward(self, inputs: np.array) -> np.array:
        return self.actfunc.activate(inputs.dot(self.weight) + self.bias)
