import math

import numpy as np
from numba import njit


### relu applied to z
@njit
def relu(z):
    return np.maximum(z, 0)


@njit
def d_relu_dz(z):
    return z > 0


### prelu applied to z
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
    return 1 / (1 + np.exp(-z))


@njit
def d_sigmoid_dz(z):
    sig = 1 / (1 + np.exp(-z))
    return sig * (1 - sig)


### stable softmax applied to z
def softmax(z):
    z_maxz = z - np.max(z)
    e_z = np.exp(z_maxz)
    return e_z / np.sum(e_z)


@njit
def d_softmax_dz(z):
    raise NotImplementedError


@njit
def softmax_jit(z):
    num_s = z.shape[1]
    sftmx = np.ones(z.shape)
    for n in range(num_s):
        sftmx[:, n] = np.exp(z[:, n]) / np.sum(np.exp(z[:, n]))
    return sftmx


### approx. gaussian error linear unit (gelu) activation applied to z
@njit
def gelu_approx(z):
    term = 1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3))
    return 0.5 * z * term


@njit
def d_erf_dz(x):
    return (2 / np.sqrt(np.pi)) * np.exp(-(x**2))


@njit
def d_gelu_approx_dz(z):
    s = z / np.sqrt(2)
    approx = np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3))
    return 0.5 + 0.5 * approx + ((0.5 * z * d_erf_dz(s)) / np.sqrt(2))


### gaussian error linear unit (gelu) activation applied to z
@njit
def gelu(z):
    s = z / np.sqrt(2)
    return z * 0.5 * (1.0 + math.erf(s))


@njit
def d_gelu_dz(z):
    s = z / np.sqrt(2)
    return 0.5 + 0.5 * math.erf(s) + ((0.5 * z * d_erf_dz(s)) / np.sqrt(2))


### swish activation applied to z, equals silu when beta = 1
@njit
def swish(z, beta: float = 1.0):
    return z * ((1 + np.exp(-beta * z)) ** -1)


@njit
def d_swish_dz(z):
    denom = 4 * (np.cosh(z / 2) ** 2)
    numer = z + np.sinh(z)
    return 0.5 + numer / denom


def main() -> None:
    print("my_activations.py is a module")


if __name__ == "__main__":
    main()
