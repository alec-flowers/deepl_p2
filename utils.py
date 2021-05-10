import torch
import math
import matplotlib.pyplot as plt
import pathlib
import pickle
import os

torch.set_grad_enabled(False)


REPO_ROOT = pathlib.Path(__file__).absolute().parents[0].absolute().resolve()
assert (REPO_ROOT.exists())
MODELS_DIR = (REPO_ROOT / "models").absolute().resolve()
assert (MODELS_DIR.exists())


def generate_disc_set(nb, x=.5, y=.5, plot=False,
                      normalize = False, one_hot=False):
    '''
    Create data points randomly on [0,1].
    Classify those inside circle as 1 and outside as 0.

    :param nb: number of datapoints to create
    :param x: x-coordinate of circle
    :param y: y-coordinate of circle
    :param plot: boolean to output graph of data
    :param normalize: boolean to normalized input data
    :param one_hot: boolean to one hot encode output
    :return: data, labels
    '''
    radius = 1 / math.sqrt(2 * math.pi)

    data = torch.empty(nb, 2, dtype=torch.float32).uniform_(0, 1)
    x_scale = data[:, 0] - x
    y_scale = data[:, 1] - y

    labels = torch.where(x_scale.square().add(y_scale.square()).sqrt() > radius,
                         1, 0)
    if one_hot:
        one_hot = torch.zeros(nb, 2)
        one_hot[(range(one_hot.shape[0])), labels] = 1
        labels = one_hot

    if plot:
        circle1 = plt.Circle((x, y), radius, color='black',
                             fill=False, linewidth=3)
        fig, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='RdYlGn', marker='.')
        ax.add_patch(circle1)
        plt.show()

    if normalize:
        mu, std = data.mean(), data.std()
        data.sub_(mu).div_(std)

    return data, labels


def check_pred_target(output, target):
    """
    Checks the one-hot predictions and targets

    :param output:      predictions
    :param target:      labels
    :return:            total number of prediction errors
    """
    errors = []
    for i in range(output.size(0)):
        errors.append(torch.argmax(output[i]) != torch.argmax(target[i]))
    return sum(errors)


def count_errors(model, data, labels, batch_size):
    """
    Batches and checks the test error.

    :param model:       neural network
    :param data:        test data
    :param labels:      test labels
    :param batch_size:  size of batch
    :return:            total number of prediciton errors
    """
    error = 0
    for d, label in zip(data.split(batch_size), labels.split(batch_size)):
        output = model.forward(d)
        predict = torch.argmax(output, dim=1)
        label = torch.argmax(label, dim=1)
        for i in range(label.size(0)):
            if predict[i] != label[i]:
                error += 1
    return error


def save_pickle(item, path, name):
    """
    Save a file as .pickle

    :param item:    item to save
    :param path:    location to save
    :param name:    name of file
    :return:        None
    """
    filename = os.path.join(path, f'{name}.pickle')
    with open(filename, "wb") as f:
        pickle.dump(item, f)


def load_pickle(path, name):
    """
    Load pickle file

    :param path:    where file is located
    :param name:    name of file
    :return:        saved item
    """
    file_path = os.path.join(path, name + '.pickle')
    with open(file_path, "rb") as f:
        item = pickle.load(f)
    return item
