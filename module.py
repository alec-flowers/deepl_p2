import torch


class Module(object):
    def __init__(self):
        pass
    def forward(self, * input):
        raise NotImplementedError

    def backward(self, * gradwrtoutput):
        raise NotImplementedError

    def get_param(self):
        return []


class Linear(Module):
    def __init__(self, input_dim, output_dim, mean=0, std=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = torch.empty((output_dim, input_dim)).normal_(mean, std)
        self.b = torch.empty((output_dim, 1)).normal_(mean, std)
        self.grad_w = torch.empty(self.w.size())
        self.grad_b = torch.empty(self.b.size())
        self.x = torch.empty((input_dim, 1))

    def forward(self, x):
        self.x = x
        return torch.mm(self.w, x) + self.b

    def backward(self, gradwrtoutput):
        self.grad_w = torch.mm(self.x, gradwrtoutput.T)
        self.grad_b = gradwrtoutput
        return torch.mm(self.w.T, gradwrtoutput)

    def get_param(self):
        return [[self.w, self.grad_w], [self.b, self.grad_b]]

# lin = Linear(5,5)
# input = torch.Tensor([[1],[2],[3],[4],[5]])
# print(input.size())
# a = lin.forward(input)


class Relu(Module):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        return torch.clamp(x, min=0)

    def backward(self, gradwrtoutput):
        return torch.torch.clamp(self.x, min=0, max=1).ceil() * gradwrtoutput

    def get_param(self):
        return [[None, None]]


class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        exp = torch.exp(2*self.x)
        return (exp-1)/(exp+1)

    def backward(self, gradwrtoutput):
        return 2/(torch.exp(self.x)+torch.exp(-self.x)) * gradwrtoutput

    def get_param(self):
        return [[None, None]]

class MSEloss(Module):
    def __init__(self):
        super().__init__()
        self.x = None
        self.target = None

    def forward(self, x, target):
        self.x = x
        self.target = target
        return sum((self.x - self.target)**2) / self.x.size(0).item()

    def backward(self):
        return 2*(self.x - self.target) / self.x.size(0).item()

    def get_param(self):
        return [[None, None]]


class Sequential(Module):

    def __init__(self):
        super().__init__()
        self.modules = []

    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def add_module(self, new_module):
        self.modules.append(new_module)

    def forward(self, input):
        x = input
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, gradwrtoutput):
        y = gradwrtoutput
        print(self.modules[::-1])
        for module in self.modules[::-1]:
            y = module.backward(y)
        return y

    def get_param(self):
        param_list = []
        for module in self.modules:
            for item in module.get_param():
                param_list.append(item)
        return param_list

mod_list = [Linear(10,5), Tanh(), Linear(5,2), Relu()]
seq = Sequential(mod_list)
seq.forward(torch.empty((10, 1)))
seq.backward(torch.empty((2, 1)))