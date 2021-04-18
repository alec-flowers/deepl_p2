import torch
import math

torch.set_grad_enabled(False)


def generate_disc_set(nb):
    # Edit this for disc centered at (0.5, 0.5)
    tensor1 = torch.empty(nb, 2, dtype=torch.float32).uniform_(0, 1)
    radius = 1/math.sqrt(2*math.pi)
    tensor2 = torch.where(tensor1[:, 0].square().add(tensor1[:, 1].square()).sqrt() > radius, 1, 0)

    return tensor1, tensor2
