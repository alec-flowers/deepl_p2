
"""
This file contains different solvers that will be used for optimizing our
home baked NN.
"""
import torch
from module import *


def convert_to_one_hot_labels(input, target):
    """
    Convertes targets to one-hot labels of -1 and 1

    Output:
    one_hot_labels : nx2 dimension FloatTensor
    """
    one_hot_ret = input.new(target.size(0), target.max() + 1).fill_(-1)
    one_hot_ret.scatter_(1, target.view(-1, 1), 1.0)
    return one_hot_ret


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


def check_pred_target(train_target_one_hot, output_one_hot):
    """
    @brief      checks the on-hot predictions and targets

    @param      grads of the model weights
    """
    # output_list = [output_one_hot[0], output_one_hot[1]]
    # prediction = output_list.index(max(output_list))

    # # correct = train_target_not_one_hot
    # train_targets_list = list(train_target_one_hot)
    # correct = train_targets_list.index(max(train_targets_list))
    #
    # return int(correct) != int(prediction)
    return torch.argmax(train_target_one_hot) != torch.argmax(output_one_hot)


def mse_loss_(pred, target):
    """
    Calculate MSEloss

    Outputs:
    loss :  float
    """
    return (pred - target.float().view_as(pred)).pow(2).sum()


def d_mse_loss_(pred, target):
    """
    Calculate derivative of MSEloss

    Outputs:
    derivative :  FloatTensor with same dimension as input
    """
    return 2*(pred - target.float().view_as(pred))


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

    def __init__(self, module, criterion, learning_rate, batch_size=1):
        """
        Inputs:
        module: a sequential module object
        learning_rate: the constant by which the
                       gradient of the parameters are applied
        """
        super(BatchStochaticGradientDescent,
              self).__init__(module, criterion, learning_rate)
        self.nb_train_errors = []
        self.tot_loss = []
        self.nb_train_errors_epoch = 0
        self.tot_loss_epoch = 0.
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
        start = self.call_count * self.batch_size
        stop = min((self.call_count+1) * self.batch_size, train_inps.size(0))
        for i in range(start, stop, 1):
            output = self.module.forward(train_inps[i])
            if check_pred_target(train_targets[i], output):
                self.nb_train_errors_epoch += 1
            loss = mse_loss_(output, train_targets[i])
            self.tot_loss_epoch += loss.float()
            d_loss_d_output = d_mse_loss_(output, train_targets[i])
            d_loss_d_input = self.module.backward(d_loss_d_output)
        self.step(stop - start)
        self.call_count += 1


    def gd_reset(self):
        self.zero_grad()

    def gd_epoch_reset(self):
        self.call_count = 0
        self.nb_train_errors.append(self.nb_train_errors_epoch)
        self.nb_train_errors_epoch = 0
        self.tot_loss.append(self.tot_loss_epoch)
        self.tot_loss_epoch = 0.


if __name__ == "__main__":
    mod_list = [Linear(10, 5), Tanh(), Linear(5, 2), Relu()]
    seq = Sequential(mod_list)
    inp = torch.empty((10, 1)).normal_()
    out = torch.empty((2, 1)).normal_()
    seq.forward(inp)
    seq.backward(out)
    loss = MSEloss()

    gradient_descent = BatchStochaticGradientDescent(seq, loss, 0.02)
    gradient_descent.gd_reset()
    gradient_descent.step()
    gradient_descent.zero_grad()
    gradient_descent.gd_reset()
    inps = torch.empty((5, 10, 1)).normal_()
    outs = torch.empty((5, 2, 1)).normal_()
    gradient_descent.gd_step(inps, outs)
