import math
import torch

#debugging stuff
old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info

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
        self.w = torch.empty((self.output_dim, self.input_dim)).normal_(mean, std)
        self.b = torch.empty(self.output_dim).normal_(mean, std)
        self.grad_w = torch.empty(self.w.size())
        self.grad_b = torch.empty(self.b.size())
        self.x = None

    def forward(self, inp):
        # xᴸ⁻¹
        self.x = inp
        # sᴸ = wᴸ xᴸ⁻¹ + bᴸ
        # if len(list(self.x.size())) == 1:
        #     self.x = self.x.view(-1, 1)
        return torch.addmm(self.b, self.x, self.w.T)

    def backward(self, gradwrtoutput):
        # Populating the values ∂l/∂w and ∂l/∂b of current layer

        # ∂l/∂wᵢⱼᴸ⁻¹ = ∑ ∂l/∂sᵢᴸ . xⱼᴸ⁻¹
        self.grad_w.add_(torch.mm(gradwrtoutput.T, self.x))

        # ∂l/∂bᵢᴸ⁻¹ = ∑ ∂l/∂sᵢᴸ
        #TODO unsure about this
        self.grad_b.add_(torch.sum(gradwrtoutput, dim=0))

        # return ∂L/∂xⱼᴸ⁻¹ = wᵢⱼ . ∂L/∂sⱼᴸ
        a = torch.mm(gradwrtoutput, self.w)
        return torch.mm(gradwrtoutput, self.w)

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
        return self.s.clamp(min=0)

    def backward(self, gradwrtoutput):
        # return ∂l/∂sᵢᴸ = ∂l/∂xᵢᴸ * σ'(sᵢᴸ)
        return self.s.sign().clamp(min=0) * gradwrtoutput

    def get_param(self):
        return [(None, None)]


class LeakyRelu(Module):
    """
    LeakyReLU activation module

    Returns:
    forward  :  FloatTensor of size m (m: number of units)
    backward :  FloatTensor of size m (m: number of units)
    """

    def __init__(self, leaky_factor=0.01):
        super().__init__()
        self.s = None
        self.leaky_factor = leaky_factor

    def forward(self, inp):
        self.s = inp
        return torch.where(self.s > 0.0, self.s, self.s * self.leaky_factor)

    def backward(self, gradwrtoutput):
        return torch.where(self.s > 0.0, 1.0, self.leaky_factor) * gradwrtoutput

    def get_param(self):
        return [(None, None)]


class Sigmoid(Module):
    """
    Sigmoid activation module

    Returns:
    forward  :  FloatTensor of size m (m: number of units)
    backward :  FloatTensor of size m (m: number of units)
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = None

    def forward(self, inp):
        # if inp.item >= 0:
        self.sigmoid = 1. / (1. + torch.exp(-inp))
        # else:
        #     self.sigmoid = torch.exp(inp) / (1. + torch.exp(inp))
        return self.sigmoid

    def backward(self, gradwrtoutput):
        return self.sigmoid * (1-self.sigmoid) * gradwrtoutput

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
        # tanh(x) = ((-1. - e⁻²ˣ)/(1. + e⁻²ˣ)).
        return (1. - (2. / (1. + exp)))

    def backward(self, gradwrtoutput):
        # return ∂l/∂sᵢᴸ = ∂l/∂xᵢᴸ * σ'(sᵢᴸ)
        # return 2/(torch.exp(self.s)+torch.exp(-self.s)) * gradwrtoutput
        return (4 * ((self.s.exp() + self.s.mul(-1).exp()).pow(-2)) *
                gradwrtoutput)

    def get_param(self):
        return [(None, None)]


class Softmax(Module):
    """

    """

    def __init__(self):
        super().__init__()
        self.s = None

    def forward(self, inp):
        # Numerical stability, subtract max before applying softmax
        shift = inp - torch.max(inp, 1)[0].view(-1, 1)
        self.s = torch.exp(shift)
        return self.s / torch.sum(self.s, 1, keepdim=True)

    def backward(self, gradwrtoutput):
        # p = (self.s/torch.sum(self.s)).view(1, -1)
        # gradwrtoutput = gradwrtoutput.view(1, -1)
        # jacobian = p * torch.eye(p.size(0)) - torch.mm(p.T, p)
        # grad = gradwrtoutput @ jacobian
        #TODO only works with NLLLoss
        return self.s - gradwrtoutput

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
        self.pred = None
        self.target = None

    def forward(self, pred, target):
        self.pred = pred
        self.target = target.float().view_as(pred)
        a = torch.sum((self.pred - self.target)**2) / self.pred.size(0)
        return torch.sum((self.pred - self.target)**2) / self.pred.size(0)

    def backward(self):
        a = 2*(self.pred - self.target) / self.pred.size(0)
        return 2*(self.pred - self.target) / self.pred.size(0)

    def get_param(self):
        return [(None, None)]


class NLLLoss(Module):
    """
    Negative Log Liklihood Loss

    """

    def __init__(self):
        super().__init__()
        self.pred = None
        self.target = None

    def forward(self, pred, target):
        epsilon = 1e-5
        self.pred = pred
        self.target = target.float().view_as(pred)
        #add epsilon so log(0) !-> -inf
        return torch.sum(-torch.log(self.pred * self.target + epsilon))

    def backward(self):
        epsilon = 1e-6
        #-(1/self.target.size(0)) * (self.target / (self.pred + epsilon))
        #TODO only works with softmax ouput layer
        return self.target

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
