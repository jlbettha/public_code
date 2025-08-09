#!/usr/bin/env python3
"""Betthauser, J. 2025 - This module contains various activation functions and their derivatives."""
import time

import numpy as np
from numba import njit


@njit
def erf(x):
    # save the sign of x
    sign = np.where(x >= 0, 1.0, -1.0)  # np.array([1.0 if x >= 0 else -1.0 for x in x])
    x = np.abs(x)

    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y  # erf(-x) = -erf(x)


### relu applied to z
@njit
def relu(z):
    return np.maximum(z, 0)


@njit
def d_relu_dz(z):
    return 1.0 * (z > 0)


### prelu applied to z
@njit
def prelu(z, a):
    return np.maximum(z, a * z)


@njit
def d_prelu_dz(z, a):
    return np.where(z < 0, a * z, 1.0)


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


# @njit
# def d_softmax_dz(z):
#     # the derivative of softmax (likely never used)
#     return NotImplementedError("Derivative of softmax is not implemented.")


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
def erf_prime(x):
    return (2 / np.sqrt(np.pi)) * np.exp(-(x**2))


@njit
def d_gelu_approx_dz(z):
    s = z / np.sqrt(2)
    approx = np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3))
    return 0.5 + 0.5 * approx + ((0.5 * z * erf_prime(s)) / np.sqrt(2))


### gaussian error linear unit (gelu) activation applied to z
@njit
def gelu(z):
    s = z / np.sqrt(2)
    return z * 0.5 * (1.0 + erf(s))


@njit
def d_gelu_dz(z):
    s = z / np.sqrt(2)
    return 0.5 + 0.5 * erf(s) + ((0.5 * z * erf_prime(s)) / np.sqrt(2))


### swish activation applied to z, equals silu when beta = 1
@njit
def swish(z, beta: float = 1.0):
    return z * ((1 + np.exp(-beta * z)) ** -1)


@njit
def d_swish_dz(z):
    denom = 4 * (np.cosh(z / 2) ** 2)
    number = z + np.sinh(z)
    return 0.5 + number / denom


def main() -> None:
    print_vals = False
    rng = np.random.default_rng(42)
    z = rng.uniform(size=(32, 10)).astype(np.float64)
    if print_vals:
        print("z:", z)
        print("erf(z):", erf(z))
        print("erf_prime(z):", erf_prime(z))

        print("relu(z):", relu(z))
        print("d_relu_dz(z):", d_relu_dz(z))

        print("prelu(z, 0.1):", prelu(z, 0.1))
        print("d_prelu_dz(z, 0.1):", d_prelu_dz(z, 0.1))

        print("sigmoid(z):", sigmoid(z))
        print("d_sigmoid_dz(z):", d_sigmoid_dz(z))

        # print("softmax(z):", softmax(z))
        print("softmax_jit(z):", softmax_jit(z))

        print("gelu_approx(z):", gelu_approx(z))
        print("d_gelu_approx_dz(z):", d_gelu_approx_dz(z))
        print("gelu(z):", gelu(z))
        print("d_gelu_dz(z):", d_gelu_dz(z))

        print("swish(z):", swish(z))
        print("d_swish_dz(z):", d_swish_dz(z))
    else:
        _ = erf(z)
        _ = erf_prime(z)

        _ = relu(z)
        _ = d_relu_dz(z)

        _ = prelu(z, 0.1)
        _ = d_prelu_dz(z, 0.1)

        _ = sigmoid(z)
        _ = d_sigmoid_dz(z)

        # _ = softmax(z)
        _ = softmax_jit(z)

        _ = gelu_approx(z)
        _ = d_gelu_approx_dz(z)
        _ = gelu(z)
        _ = d_gelu_dz(z)

        _ = swish(z)
        _ = d_swish_dz(z)


if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"Time taken for first run: {t1 - t0:.5f} seconds")
    print(f"Time taken for second run: {t2 - t1:.5f} seconds")
    print(f"JIT speedup: {(t1 - t0) / (t2 - t1):.2f}x")
