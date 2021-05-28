from utils import MODELS_DIR, count_errors, save_pickle, load_pickle, generate_disc_set, check_pred_target
from solvers import *
from module import Sequential


def train_network(module_list, criterion, train_data, train_labels, batch_size=5,
                  lr=1e-5, epochs=100, network_name=None):
    """
    Builds and trains a neural network on generated data.

    :param module_list:     list of ordered modules to feed network
    :param criterion:       which optimizer to use
    :param train_data       training data
    :param train_labels     training data labels
    :param batch_size:      size of batches
    :param lr:              learning rate
    :param epochs:          number epochs
    :param network_name:    name to save network under

    :return neuralnet:      return trained neural net
    """
    print(f"== Training Network: {network_name} ==")

    neuralnet = Sequential(module_list)
    gradient_descent = BatchStochaticGradientDescent(neuralnet, criterion,
                                                     lr,
                                                     batch_size)
    for epoch in range(epochs):
        losses = []
        nb_train_errors = []
        for input, target in zip(train_data.split(batch_size), train_labels.split(batch_size)):
            # forward pass
            output = neuralnet.forward(input)

            # calculate loss and gradients
            loss = criterion.forward(output, target)
            gradient_descent.gd_reset()
            neuralnet.backward(criterion.backward())

            # update parameters
            gradient_descent.step()

            # save data
            losses.append(loss)
            nb_train_errors.append(check_pred_target(output, target).item())

        if epoch % 10 == 0:
            print(f" Epoch {epoch} || Train Loss: {(sum(losses)/train_labels.size(0)).item():.03f}\
             || Train Accuracy: {1 - sum(nb_train_errors)/train_labels.size(0):.03f} %")
    print(f"== End Training: {network_name} ==")
    print(f"Last Train Accuracy: {1 - sum(nb_train_errors)/train_labels.size(0):.03f} %")
    if network_name is not None:
        save_pickle(neuralnet, MODELS_DIR, network_name)

    return neuralnet


def test_network(neuralnet=None, model_name=None, num_test=1000, normalize=True, one_hot=True):
    """
    Test a trained neural network on new data. Can either pass in network or load from a file.

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
    print(f"Test Accuracy: {1 - test_error/test_labels.size(0):.03f} % \n")