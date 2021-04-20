
"""
This file contains different solvers that will be used for optimizing our
home baked NN.
"""
import torch
from module import *


class Solver(object):
    """
    Documentation for Solver

    """

    def __init__(self, module, learning_rate):
        """
        Inputs:
        module: a sequential module object
        learning_rate: the constant by which the
                       gradient of the parameters are applied
        """
        self.module = module
        self.lr = learning_rate
        self.epoch_counter = 0
        self.iter_counter = 0

    def update_lr(self, new_lr):
        """
        updates the learning rate according to given new_lr
        """
        self.lr = new_lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        """
        reset the grads of all of the parameters of the module of the solver
        """
        params = self.module.get_param()
        for param in params:
            _, grad = param
            if grad is None:
                continue
            else:
                grad.zero_()


class GradientDescent(Solver):
    """
    Vanila Gradient Descent Solver

    """

    def __init__(self, module, learning_rate):
        """
        Inputs:
        module: a sequential module object
        learning_rate: the constant by which the
                       gradient of the parameters are applied
        """
        super(GradientDescent, self).__init__(module, learning_rate)

    def step(self):
        """
        applies the gradient step on the parameters of the module
        wᵢ = wᵢ₋₁ - α ∂l/∂w|ᵢ₋₁
        """
        params = self.module.get_param()
        for param in params:
            weight, grad = param
            if ((weight is None) or (grad is None)):
                continue
            else:
                weight.add_(-self.lr * grad)



mod_list = [Linear(10, 5), Tanh(), Linear(5, 2), Relu()]
seq = Sequential(mod_list)
seq.forward(torch.empty((10, 1)))
seq.backward(torch.empty((2, 1)))

gradient_descent = GradientDescent(seq, 0.02)
gradient_descent.step()
gradient_descent.zero_grad()
