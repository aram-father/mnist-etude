from types import new_class
import numpy as np
from PIL import Image
from context import loader, mnistetude

if __name__ == '__main__':
    dataset = loader.load_dataset(normalize=True)
    parameters = loader.load_parameters()

    network = mnistetude.Mlp(parameters)

    acc = 0
    for image, label in zip(dataset['test_images'], dataset['test_labels']):
        prediction = np.argmax(network.predict(image))
        if prediction == label:
            acc += 1

    print(f'Accuracy: {float(100 * acc) / loader.number_of_test_images} %')
