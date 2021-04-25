import utils
from module import *
from solvers import *

epochs = 100

train_data, train_labels = utils.generate_disc_set(1000)
test_data, test_labels = utils.generate_disc_set(1000)


mod_list = [Linear(2, 25), Tanh(),
            Linear(25, 50), Tanh(),
            Linear(50, 25), Tanh(),
            Linear(25, 25), Tanh(),
            Linear(25, 2), Relu()]
seq = Sequential(mod_list)
loss = MSEloss()
gradient_descent = BatchStochaticGradientDescent(seq, loss, 0.02)
for i in range(epochs):
    gradient_descent.gd_reset()
    gradient_descent.gd_step(train_data, train_labels)

print(gradient_descent.nb_train_errors)
