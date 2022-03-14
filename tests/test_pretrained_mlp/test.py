import numpy as np
from PIL import Image
from context import loader, mnistetude

if __name__ == '__main__':
    dataset = loader.load_dataset(normalize=True)
    parameters = loader.load_parameters()

    network = mnistetude.Mlp(parameters)

    batch_predictions = np.argmax(network.predict(dataset['test_images']), axis=1)
    acc = np.sum(batch_predictions == dataset['test_labels'])

    print(f'Accuracy: {float(100 * acc) / loader.number_of_test_images} %')
