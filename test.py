import utils
from module import *
from solvers import *

epochs = 100
train_data_count = 1000
learning_rate = 0.02
batch_size = 1

train_data, train_labels = utils.generate_disc_set(1000)
test_data, test_labels = utils.generate_disc_set(1000)


mod_list = [Linear(2, 25), Relu(),
            Linear(25, 50), Relu(),
            Linear(50, 25), Relu(),
            Linear(25, 25), Relu(),
            Linear(25, 2), Sigmoid()]

seq = Sequential(mod_list)
loss = MSEloss()
gradient_descent = BatchStochaticGradientDescent(seq, loss,
                                                 learning_rate,
                                                 batch_size)
for i in range(epochs):
    gradient_descent.gd_reset()
    for j in range(1, train_data_count, batch_size):
        gradient_descent.gd_step(train_data, train_labels)


print(gradient_descent.nb_train_errors)
