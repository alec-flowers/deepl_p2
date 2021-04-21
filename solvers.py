
"""
This file contains different solvers that will be used for optimizing our
home baked NN.
"""
import torch
from module import *


def gradient_descent_worker(weight, grad, learning_rate, batch_size):
    """
    @brief      worker for gradient descent step

    @param      weights to be updated
    @param      grads of the model weights
    @param      the learning rate (coeff. of the update)

    """
    if (not(weight is None) and not(grad is None)):
        weight.add_(-(learning_rate/batch_size) * grad)


def make_grad_zero_worker(grad):
    """
    @brief      worker for making gradient zero

    @param      grads of the model weights
    """
    if not (grad is None):
        grad.zero_()


def check_output_target(train_target_one_hot, output_one_hot):
    """
    @brief      checks the on-hot predictions and targets

    @param      grads of the model weights
    """
    output_list = [output_one_hot[0], output_one_hot[1]]
    prediction = output_list.index(max(output_list))

    train_targets_list = [train_target_one_hot[0], train_target_one_hot[1]]
    correct = train_targets_list.index(max(train_targets_list))

    return int(correct) != int(prediction)


def MSEloss(pred, target):
    """
    Calculate MSEloss

    Outputs:
    loss :  float
    """
    return (pred - target.float()).pow(2).sum()


def dMSEloss(pred, target):
    """
    Calculate derivative of MSEloss

    Outputs:
    derivative :  FloatTensor with same dimension as input
    """
    return 2*(pred - target.float())


class Solver(object):
    """
    Abstract Documentation for Solver class

    """

    def __init__(self, module, criterion, learning_rate):
        """
        Inputs:
        module: a sequential module object
        learning_rate: the constant by which the
                       gradient of the parameters are applied
        """
        self.module = module
        self.criterion = criterion
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
            make_grad_zero_worker(grad)


class BatchStochaticGradientDescent(Solver):
    """
    @description    Batch Gradient Descent Solver

    @param          module
    @param          learning rate
    @param          batch size (1: plain SGD,
                                train size: GD,
                                bet.1 & train_size batch-SGD)
    """

    def __init__(self, module, learning_rate, batch_size=1):
        """
        Inputs:
        module: a sequential module object
        learning_rate: the constant by which the
                       gradient of the parameters are applied
        """
        super(BatchStochaticGradientDescent,
              self).__init__(module, learning_rate)
        self.nb_train_errors = 0
        self.tot_loss = 0
        self.call_count = 0
        self.batch_size = batch_size

    def step(self, batch_size=1):
        """
        applies the gradient step on the parameters of the module
        wᵢ = wᵢ₋₁ - α ∂l/∂w|ᵢ₋₁
        """
        params = self.module.get_param()
        for param in params:
            weight, grad = param
            gradient_descent_worker(weight, grad, self.lr, batch_size)

    def gd_step(self, train_inps, train_targets):
        self.nb_train_errors = 0
        self.tot_d_loss = torch.empty((train_targets.size(0)))
        start = self.call_count * self.batch_size
        stop = max(self.call_count * self.batch_size, train_inps.size(0))
        for i in range(start, stop, 1):
            output = self.module.forward(train_inps[i])
            if check_output_target(train_targets[i], output):
                self.nb_train_errors += 1
            self.tot_loss += criterion.forward(output,
                                                train_targets[i].float())
            d_loss_d_output = criterion.backward(output,
                                                 train_targets[i].float())
            self.module.backward(d_loss_d_output)
        self.step(self.batch_size)
        self.call_count += 1

    def gd_reset(self):
        self.zero_grad()
        self.call_count = 0


mod_list = [Linear(10, 5), Tanh(), Linear(5, 2), Relu()]
seq = Sequential(mod_list)
inp = torch.empty((10, 1)).normal_()
out = torch.empty((2, 1)).normal_()
seq.forward(inp)
seq.backward(out)


gradient_descent = BatchStochaticGradientDescent(seq, 0.02)
gradient_descent.gd_reset()
gradient_descent.step()
gradient_descent.zero_grad()
gradient_descent.gd_reset()
inps = torch.empty((5, 10, 1)).normal_()
outs = torch.empty((5, 2, 1)).normal_()
gradient_descent.gd_step(inps, outs)
