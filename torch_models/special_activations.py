import torch
import numpy as np
from torch import nn


### gaussian error linear unit (gelu) activation applied to x
def gelu(x):
    cdf = 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    return x * cdf


def d_gelu_dx(x):
    raise NotImplementedError


# Swish activation (similar to GELU). When beta=1, equivalent to Sigmoid Linear Unit (SiLU).
def swish(x):
    return x / (1 + np.exp(-x))


def d_swish_dx(x):
    sigmoid = 1 / (1 + np.exp(-x))
    swish = x * sigmoid
    return swish + (sigmoid * (1 - swish))


class SwiGLU(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear_1 = nn.Linear(dimension, dimension)
        self.linear_2 = nn.Linear(dimension, dimension)

    def forward(self, x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        swiglu = swish * self.linear_2(x)

        return swiglu


class GLU(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear_1 = nn.Linear(dimension, dimension)
        self.linear_2 = nn.Linear(dimension, dimension)

    def forward(self, x):
        output = self.linear_1(x)
        sigmoid = torch.sigmoid(output)
        glu = sigmoid * self.linear_2(x)

        return glu
