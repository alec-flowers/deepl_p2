import utils
from module import *
from solvers import *
import matplotlib.pyplot as plt
import numpy as np

epochs = 200
train_data_count = 1000
learning_rate = 0.002
batch_size = 1

train_data, train_labels = utils.generate_disc_set(train_data_count,
                                                   normalize=True)
test_data, test_labels = utils.generate_disc_set(1000)


train_targets = convert_to_one_hot_labels(train_data, train_labels)
test_targets = convert_to_one_hot_labels(test_data, test_labels)
hidden_size = 5
mod_list = [Linear(2, hidden_size, 0.0, 1.0), Relu(),
            Linear(hidden_size, hidden_size, 0.0, 1.0), Relu(),
            # Linear(hidden_size, hidden_size, 0.0, 1.0), Relu(),
            Linear(hidden_size, 2, 0.0, 1.0), Tanh()]

seq = Sequential(mod_list)
loss = MSEloss()
gradient_descent = BatchStochaticGradientDescent(seq, loss,
                                                 learning_rate,
                                                 batch_size)
for i in range(epochs):
    for j in range(1, train_data_count, batch_size):
        gradient_descent.gd_reset()
        gradient_descent.gd_step(train_data, train_targets)
    gradient_descent.gd_epoch_reset()

    print(f" in epoch {i}: train error count:\
    {gradient_descent.nb_train_errors[i]}")
    # gradient_descent.nb_train_errors = 0
