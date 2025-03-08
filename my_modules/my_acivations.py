import numpy as np
from numba import njit
from scipy.special import erf


@njit
def relu(wx_b):
    return np.maximum(wx_b, 0)


@njit
def d_relu_dz(z):
    return z > 0


@njit
def prelu(z, a):
    return np.maximum(z, a * z)


@njit
def d_prelu_dz(z, a):
    if z < 0:
        return a * z
    return 1.0


### sigmoid applied to z
@njit
def sigmoid(z):
    return (1 + np.exp(-z)) ** -1


def d_sigmoid_dz(z):
    raise NotImplementedError


### stable softmax applied to z
@njit
def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)


def d_softmax_dz(z):
    raise NotImplementedError


### approx. gaussian error linear unit (gelu) activation applied to z
@njit
def gelu_approx(z):
    term = 1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3))
    return 0.5 * z * term


def d_gelu_approx_dz(z):
    raise NotImplementedError


### gaussian error linear unit (gelu) activation applied to z
@njit
def gelu(z):
    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    return z * cdf


def d_gelu_dz(z):
    raise NotImplementedError


### swish activation applied to z, equals silu when beta = 1
@njit
def swish(z, beta: float = 1.0):
    return z * ((1 + np.exp(-beta * z)) ** -1)


@njit
def d_swish_dz(z):
    denom = 4 * (np.cosh(z / 2) ** 2)
    numer = z + np.sinh(z)
    return 0.5 + numer / denom
