import numpy as np
from . import itrainer
from .. import network
from typing import Callable, List, Tuple


class StochasticGradientDescent(itrainer.ITrainer):
    @staticmethod
    def numerical_gradient(f: Callable, x: np.array, delta: float = 1e-4) -> np.array:
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            center_x = float(x[idx])
            x[idx] = center_x + delta
            f_front = f(x)
            x[idx] = center_x - delta
            f_rear = f(x)
            x[idx] = center_x
            grad[idx] = (f_front - f_rear) / (2 * delta)
            it.iternext()

        return grad

    def __init__(self, learning_rate: float = 1e-1, batch_size: int = 100):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def train(self, network: network.INetwork, train_image: np.array, train_label: np.array):
        for iteration in range(max(1, int(train_image.shape[0] / self.batch_size))):
            print(
                f'iteration: {iteration} / {int(train_image.shape[0] / self.batch_size)}')
            mask = np.random.choice(train_image.shape[0], self.batch_size)
            batch_image = train_image[mask]
            batch_label = train_label[mask]

            grad_weights, grad_biases = self.get_gradients(
                network, batch_image, batch_label)

            for layer, grad_weight, grad_bias in zip(network.layers, grad_weights, grad_biases):
                layer.weight -= self.learning_rate * grad_weight
                layer.bias -= self.learning_rate * grad_bias

            print(network.get_loss(batch_image, batch_label))

    def get_gradients(self, network: network.INetwork, image: np.array, label: np.array) -> Tuple[List[np.array], List[np.array]]:
        def get_loss(x): return network.get_loss(image, label)

        grad_weights = []
        grad_biases = []

        for layer in network.layers:
            grad_weights.append(
                StochasticGradientDescent.numerical_gradient(get_loss, layer.weight))
            grad_biases.append(
                StochasticGradientDescent.numerical_gradient(get_loss, layer.bias))

        return grad_weights, grad_biases
