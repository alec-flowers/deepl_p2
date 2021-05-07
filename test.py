import utils
from module import *
from solvers import *
import matplotlib.pyplot as plt
import numpy as np


def train_network(n_input=2, n_output=2, batch_size=5,
                  lr=1e-5, epochs=100, num_train=1000,
                  num_test=1000, normalize=True, one_hot=True,):

    train_data, train_labels = utils.generate_disc_set(num_train,
                                                       normalize=normalize, one_hot=one_hot)
    test_data, test_labels = utils.generate_disc_set(num_test, normalize=normalize,
                                                     one_hot=one_hot)


    mod_list = [Linear(n_input, 10, 0.0, 1.0), Relu(),
                Linear(10, 10, 0.0, 1.0), Relu(),
                Linear(10, 2, 0.0, n_output), Softmax()]

    seq = Sequential(mod_list)
    criterion = NLLLoss()
    gradient_descent = BatchStochaticGradientDescent(seq, criterion,
                                                     lr,
                                                     batch_size)
    for i in range(epochs):
        losses = []
        nb_train_errors = []
        for input, target in zip(train_data.split(batch_size), train_labels.split(batch_size)):
            output = seq.forward(input)
            loss = criterion.forward(output, target)
            gradient_descent.gd_reset()
            seq.backward(criterion.backward())
            gradient_descent.step()

            losses.append(loss)
            nb_train_errors.append(utils.check_pred_target(output, target).item())

        print(f" Epoch {i} - train loss: {(sum(losses)/train_labels.size(0)).item():.03f} \
        train accuracy: {1 - sum(nb_train_errors)/train_labels.size(0):.03f} %")

    test_error = utils.count_errors(seq, test_data, test_labels, batch_size)
    print("========================")
    print(f"Test Accuracy: {1 - test_error/test_labels.size(0):.03f} %")