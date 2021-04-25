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
    """
    Fully connected linear layer
    Constructor parameters: dimensions of input and output

    Returns:
    forward  :  FloatTensor of size m (m: input_dim)
    backward :  FloatTensor of size n (n: output_dim)
    """

    def __init__(self, input_dim, output_dim, mean=0, std=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = torch.empty((output_dim, input_dim)).normal_(mean, std)
        self.b = torch.empty((output_dim, 1)).normal_(mean, std)
        self.grad_w = torch.empty(self.w.size())
        self.grad_b = torch.empty(self.b.size())
        self.x = torch.empty((input_dim, 1))

    def forward(self, inp):
        # xᴸ⁻¹
        self.x = inp
        # sᴸ = wᴸ xᴸ⁻¹ + bᴸ
        if len(list(self.x.size())) == 1:
            self.x = self.x.view(-1, 1)
        return torch.mm(self.w, self.x) + self.b

    def backward(self, gradwrtoutput):
        # Populating the values ∂l/∂w and ∂l/∂b of current layer

        # ∂l/∂wᵢⱼᴸ⁻¹ = ∑ ∂l/∂sᵢᴸ . xⱼᴸ⁻¹
        self.grad_w.add_(torch.mm(gradwrtoutput, self.x.T))

        # ∂l/∂bᵢᴸ⁻¹ = ∑ ∂l/∂sᵢᴸ
        self.grad_b.add_(gradwrtoutput)

        # return ∂L/∂xⱼᴸ⁻¹ = wᵢⱼ . ∂L/∂sⱼᴸ
        return torch.mm(self.w.T, gradwrtoutput)

    def get_param(self):
        return [(self.w, self.grad_w), (self.b, self.grad_b)]

# lin = Linear(5,5)
# input = torch.Tensor([[1],[2],[3],[4],[5]])
# print(input.size())
# a = lin.forward(input)


class Relu(Module):
    """
    ReLU activation module

    Returns:
    forward  :  FloatTensor of size m (m: number of units)
    backward :  FloatTensor of size m (m: number of units)
    """

    def __init__(self):
        super().__init__()
        self.s = None

    def forward(self, inp):
        self.s = inp
        # xᴸ = σ(sᴸ)
        return torch.clamp(self.s, min=0)

    def backward(self, gradwrtoutput):
        # return ∂l/∂sᵢᴸ = ∂l/∂xᵢᴸ * σ'(sᵢᴸ)
        return torch.torch.clamp(self.s, min=0, max=1).ceil() * gradwrtoutput

    def get_param(self):
        return [(None, None)]


class Tanh(Module):
    """
    Tanh activation module

    Returns:
    forward  :  FloatTensor of size m (m: number of units)
    backward :  FloatTensor of size m (m: number of units)
    """

    def __init__(self):
        super().__init__()
        self.s = None

    def forward(self, s):
        self.s = s
        # xᴸ = σ(sᴸ)
        exp = torch.exp(2*self.s)
        # tanh(x) = ((-1. + e⁻²ˣ)/(1. + e⁻²ˣ)).
        return (exp-1)/(exp+1)

    def backward(self, gradwrtoutput):
        # return ∂l/∂sᵢᴸ = ∂l/∂xᵢᴸ * σ'(sᵢᴸ)
        return 2/(torch.exp(self.s)+torch.exp(-self.s)) * gradwrtoutput

    def get_param(self):
        return [(None, None)]


class MSEloss(Module):
    """
    Mean Squared loss module

    Returns:
    forward  :  MSELoss: l = (x - _x_)²/n (Tensor of size of 1)
    backward :  ∂l/∂xₙᴸ = 2. (x - _x_)/n (Tensor of size of n)
    """

    def __init__(self):
        super().__init__()
        self.x = None
        self.target = None

    def forward(self, x, target):
        self.x = x
        self.target = target
        return sum((self.x - self.target)**2) / self.x.size(0)

    def backward(self):
        return 2*(self.x - self.target) / self.x.size(0)

    def get_param(self):
        return [(None, None)]


class Sequential(Module):
    """
    A module for combining the several modules in a sequential structure

    Returns:
    forward  :  Apply the sequence of forward of the underlying modules
    In the forward process in addition to constructing and returning
    the values of the last layer the node activation values are also stored

    backward :  apply the back-propagation of the layers going through the
    network in a reverse order and fills the gradient member of the underlying
    module objects
    """

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
        for module in self.modules[::-1]:
            y = module.backward(y)
        return y

    def get_param(self):
        param_list = []
        for module in self.modules:
            for item in module.get_param():
                param_list.append(item)
        return param_list
