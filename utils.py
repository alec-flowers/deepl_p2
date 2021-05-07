import torch
import math
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)


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

    labels = torch.where(x_scale.square().add(y_scale.square()).sqrt() > radius, 1, 0)
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

# def train_model(model, train_data, train_labels, optimizer, lr, epochs, batch_size):
#     train_data_count = train_data.size(0)
#     for e in range(epochs):
#         for j in range(1, train_data_count, batch_size):
#             optimizer.gd_reset()
#             optimizer.gd_step(train_data, train_labels)
#         optimizer.gd_epoch_reset()
#
#         print(f" in epoch {e}: train error count:\
#         {optimizer.nb_train_errors[e]}")
#         # gradient_descent.nb_train_errors = 0


def check_pred_target(output, target):
    """
    @brief      checks the on-hot predictions and targets
    """
    errors = []
    for i in range(output.size(0)):
        errors.append(torch.argmax(output[i]) != torch.argmax(target[i]))
    return sum(errors)


def count_errors(model, data, labels, batch_size):
    error = 0
    for d, label in zip(data.split(batch_size), labels.split(batch_size)):
        output = model.forward(d)
        predict = torch.argmax(output, dim=1)
        label = torch.argmax(label, dim=1)
        for i in range(label.size(0)):
            if predict[i] != label[i]:
                error += 1
    return error