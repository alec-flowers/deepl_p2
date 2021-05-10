from utils import MODELS_DIR, count_errors, save_pickle,\
    load_pickle, generate_disc_set, check_pred_target
from module import *
from solvers import *
import matplotlib.pyplot as plt
import numpy as np


def train_network(module_list, criterion, batch_size=5,
                  lr=1e-5, epochs=100, num_train=1000,
                  num_test=1000, normalize=True,
                  one_hot=True, network_name=None):
    """
    Builds and trains a neural network on generated data.

    :param module_list:     list of ordered modules to feed network
    :param criterion:       which optimizer to use
    :param batch_size:      size of batches
    :param lr:              learning rate
    :param epochs:          number epochs
    :param num_train:       size of training data
    :param num_test:        size of test data
    :param normalize:       normalize train and test data
    :param one_hot:         one-hot encode train and test data

    :return neuralnet:      return trained neural net
    """
    print(f"== Training Network: {network_name} ==")
    train_data, train_labels = generate_disc_set(num_train,
                                                 normalize=normalize,
                                                 one_hot=one_hot)

    neuralnet = Sequential(module_list)
    gradient_descent = BatchStochaticGradientDescent(neuralnet, criterion,
                                                     lr, batch_size)
    for epoch in range(epochs):
        losses = []
        nb_train_errors = []
        for input, target in zip(train_data.split(batch_size),
                                 train_labels.split(batch_size)):
            output = neuralnet.forward(input)
            loss = criterion.forward(output, target)
            gradient_descent.gd_reset()
            neuralnet.backward(criterion.backward())
            gradient_descent.step()

            losses.append(loss)
            nb_train_errors.append(check_pred_target(output, target).item())
        if epoch % 10 == 0:
            print(f" Epoch {epoch} || Train Loss:\
                  {(sum(losses)/train_labels.size(0)).item():.03f}\ || Train\
                  Accuracy: {1 - sum(nb_train_errors)/train_labels.size(0):.03f} %")
    print(f"== End Training: {network_name} ==")
    if network_name is not None:
        save_pickle(neuralnet, MODELS_DIR, network_name)

    return neuralnet


def test_network(neuralnet=None, model_name=None,
                 num_test=1000, normalize=True, one_hot=True):
    """
    Test a trained neural network on new data.
    Can either pass in network or load from a file.

    :param neuralnet:       Sequential object to test
    :param model_name:      filename to load from models directory
    :param num_test:        number test data points
    :param normalize:       normalize test data points
    :param one_hot:         one hot encode labels of test data

    :return: None
    """
    test_data, test_labels = generate_disc_set(num_test, normalize=normalize,
                                               one_hot=one_hot)
    if neuralnet is not None:
        test_net = neuralnet
    elif model_name is not None:
        test_net = load_pickle(MODELS_DIR, model_name)
    else:
        raise AssertionError

    test_error = count_errors(test_net, test_data, test_labels, batch_size=10)
    print(f"Test Accuracy: {1 - test_error/test_labels.size(0):.03f} %")


if __name__ == "__main__":
    n_input = 2
    n_output = 2

    criterion = NLLLoss()
    mod_list = [Linear(n_input, 10, 0.0, 1.0), LeakyRelu(),
                Linear(10, 50, 0.0, 1.0), LeakyRelu(),
                Linear(50, 50, 0.0, 1.0), LeakyRelu(),
                Linear(50, 10, 0.0, 1.0), LeakyRelu(),
                Linear(10, n_output, 0.0, 1.0), Softmax()]

    mod_1 = Sequential(mod_list)

    criterion2 = MSEloss()
    mod_list2 = [Linear(n_input, 10, 0.0, 1.0), Tanh(),
                 Linear(10, 50, 0.0, 1.0), Tanh(),
                 Linear(50, 50, 0.0, 1.0), Tanh(),
                 Linear(50, 10, 0.0, 1.0), Relu(),
                 Linear(10, n_output, 0.0, 1.0), Sigmoid()]

    net = train_network(mod_list, criterion, lr=1e-5,
                        network_name='cross_entropy', batch_size=5)
    test_network(model_name='cross_entropy')

    net2 = train_network(mod_list2, criterion2, lr=1e-3,
                         network_name='mse', batch_size=5)
    test_network(model_name='mse')
