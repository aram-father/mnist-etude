import numpy as np
from PIL import Image
import context
import loader
import mnistetude


if __name__ == '__main__':
    dataset = loader.load_dataset(normalize=True)
    params = loader.load_parameters()

    network = mnistetude.network.MultiLayerPerceptron([
        mnistetude.layer.Perceptron(params['input_layer_weights'],
                                    params['input_layer_biases'],
                                    mnistetude.actfunc.Sigmoid()),
        mnistetude.layer.Perceptron(params['hidden_layer_weights'],
                                    params['hidden_layer_biases'],
                                    mnistetude.actfunc.Sigmoid()),
        mnistetude.layer.Perceptron(params['output_layer_weights'],
                                    params['output_layer_biases'],
                                    mnistetude.actfunc.Softmax())
    ])

    batch_predictions = np.argmax(
        network.predict(dataset['test_images']), axis=1)
    acc = np.sum(batch_predictions == dataset['test_labels'])

    print(f'Accuracy: {float(100 * acc) / loader.number_of_test_images} %')
