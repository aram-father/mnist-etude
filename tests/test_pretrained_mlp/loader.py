import pickle
import numpy as np
from typing import Dict
from os import path

data_path = path.abspath(path.join(path.dirname(__file__), '../data'))
parameter_path = path.abspath(path.join(path.dirname(__file__), './parameters'))
number_of_training_images = 60000
number_of_test_images = 10000
image_dimension = (1, 28, 28)
image_size = np.prod(image_dimension)


def _load_image(fname: str) -> np.array:
    with open(fname, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    return data.reshape(-1, image_size)


def _load_label(fname: str) -> np.array:
    with open(fname, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)

    return data


def _normalize(image: np.array) -> np.array:
    return image.astype(np.float32) / 255.0


def _one_hot_label(label: np.array) -> np.array:
    one_hot = np.zeros((label.size, 10))
    for idx, row in enumerate(one_hot):
        row[label[idx]] = 1

    return one_hot


def _deflatten(image: np.array) -> np.array:
    return np.reshape(-1, image_dimension)


def load_dataset(**kwarg) -> Dict[str, np.array]:
    dataset = {}
    dataset['train_images'] = _load_image(
        '/'.join([data_path, 'train-images.idx3-ubyte']))
    dataset['train_labels'] = _load_label(
        '/'.join([data_path, 'train-labels.idx1-ubyte']))
    dataset['test_images'] = _load_image(
        '/'.join([data_path, 't10k-images.idx3-ubyte']))
    dataset['test_labels'] = _load_label(
        '/'.join([data_path, 't10k-labels.idx1-ubyte']))

    if kwarg.get('normalize', False):
        dataset['train_images'] = _normalize(dataset['train_images'])
        dataset['test_images'] = _normalize(dataset['test_images'])

    if kwarg.get('one_hot_label', False):
        dataset['train_labels'] = _one_hot_label(dataset['train_labels'])
        dataset['test_labels'] = _one_hot_label(dataset['test_labels'])

    if not kwarg.get('flatten', True):
        dataset['train_images'] = _deflatten(dataset['train_images'])
        dataset['test_images'] = _deflatten(dataset['test_images'])

    return dataset


def load_parameters() -> Dict[str, np.array]:
    with open('/'.join([parameter_path, 'mlp_parameters.pkl']), 'rb') as f:
        params = pickle.load(f)

    return params
