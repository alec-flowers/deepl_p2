"""
This file contains different solvers that will be used for optimizing our
home baked NN.
"""


def gradient_descent_worker(weight, grad, learning_rate, batch_size):
    """
    Worker for gradient descent step

    :param weight:          weights to be updated
    :param grad:            grads of the model weights
    :param learning_rate:   the learning rate (coeff. of the update)
    :param batch_size:      batch size
    """
    if (not(weight is None) and not(grad is None)):
        weight.add_(-(learning_rate/batch_size) * grad)


class Solver(object):
    """
    Abstract Documentation for Solver class

    :param module:          neural network
    :param criterion:       loss to use
    :param learning_rate:   learning rate
    """

    def __init__(self, module, criterion, learning_rate):
        self.module = module
        self.criterion = criterion
        self.lr = learning_rate
        self.epoch_counter = 0
        self.iter_counter = 0

    def update_lr(self, new_lr):
        self.lr = new_lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        """
        Reset the grads of all of the parameters of the module of the solver
        """
        params = self.module.get_param()
        for param in params:
            _, grad = param
            if not (grad is None):
                grad.zero_()


class BatchStochaticGradientDescent(Solver):
    """
    Batch Gradient Descent Solver

    :param module:          neural network
    :param criterion:       loss to use
    :param learning_rate:   learning rate
    :param batch_size:      batch size (1: plain SGD, train size: GD, between: batch gd)
    """

    def __init__(self, module, criterion, learning_rate, batch_size=1):
        super(BatchStochaticGradientDescent,
              self).__init__(module, criterion, learning_rate)
        self.batch_size = batch_size

    def step(self):
        """
        Applies the gradient step on the parameters of the module
        wᵢ = wᵢ₋₁ - α ∂l/∂w|ᵢ₋₁
        """
        params = self.module.get_param()
        for param in params:
            weight, grad = param
            gradient_descent_worker(weight, grad, self.lr, self.batch_size)

    def gd_reset(self):
        self.zero_grad()


