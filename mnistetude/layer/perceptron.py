import numpy as np
from . import ilayer
from .. import actfunc


class Perceptron(ilayer.ILayer):
    def __init__(self, weight: np.array, bias: np.array, actfunc: actfunc.IActFunc):
        super().__init__(weight, bias, actfunc)

    def forward(self, inputs: np.array) -> np.array:
        return self.actfunc.activate(inputs.dot(self.weight) + self.bias)
