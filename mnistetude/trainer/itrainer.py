import numpy as np
from .. import network


class ITrainer:
    def __init__(self):
        pass

    def train(self, network: network.INetwork, train_image: np.array, train_label: np.array):
        raise
