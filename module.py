from torch import empty

class Module(object):
    def forward(self, * input):
        raise NotImplementedError

    def backward(self, * gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

## TODO Implement these modules

class Linear(Module):

class ReLU(Module):

class Tanh(Module):

class Sequential(Module):

class LossMSE(Module):
