import numpy as np
from PIL import Image
import context
import loader
import mnistetude
from matplotlib import pyplot as plt
import pickle

if __name__ == '__main__':
    np.random.seed(10)

    dataset = loader.load_dataset(
        normalize=True, one_hot_label=True, flatten=True)

    network = mnistetude.network.MultiLayerPerceptron([
        mnistetude.layer.Perceptron(
            0.01 * np.random.randn(784, 50), np.zeros(50), mnistetude.actfunc.Sigmoid()),
        mnistetude.layer.Perceptron(
            0.01 * np.random.randn(50, 10), np.zeros(10), mnistetude.actfunc.Softmax())
    ])

    trainer = mnistetude.trainer.StochasticGradientDescent(0.1, 100)

    progress_test = []
    progress_train = []

    for epoch in range(3):
        print(f'epoch: {epoch}')
        trainer.train(
            network, dataset['train_images'], dataset['train_labels'])

        batch_predictions = np.argmax(
            network.predict(dataset['test_images']), axis=1)
        batch_answers = np.argmax(dataset['test_labels'], axis=1)
        acc = np.sum(batch_predictions == batch_answers)
        progress_test.append(acc * 100.0 / loader.number_of_test_images)

        batch_predictions = np.argmax(
            network.predict(dataset['train_images']), axis=1)
        batch_answers = np.argmax(dataset['train_labels'], axis=1)
        acc = np.sum(batch_predictions == batch_answers)
        progress_train.append(acc * 100.0 / loader.number_of_training_images)

    network_file = open('network.pkl', 'wb')
    pickle.dump(network, network_file)

    plt.plot(range(1, len(progress_test) + 1), progress_test, label='test')
    plt.plot(range(1, len(progress_train) + 1), progress_train, label='train')
    plt.xlabel('epoch')
    plt.ylabel('accuary')
    plt.legend()
    plt.show()
