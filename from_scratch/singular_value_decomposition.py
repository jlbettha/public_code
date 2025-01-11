# pylint: disable=C0103
""" Betthauser, 2020: 
SVD: sigular value decomposition, A = U @ SIGMA @ V.T
"""

import time
import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def singular_value_decomposition(
    A: NDArray[np.float64], tolerance: float = 1e-8
) -> tuple[NDArray[np.float64]]:
    """SVD of matrix A

    Args:
        A (NDArray[np.float64]): m x n matrix to decompose
        tolerance (float): optional, default = 1e-9

    Returns:
        tuple[NDArray[np.float64]]: U, sigmas, V
    """
    m, n = A.shape

    ## manually compute m eigenvalues and eigenvectors
    B = A.copy()
    sigmas = np.zeros(m)
    U = np.zeros((m, m))
    V = np.zeros((n, n))
    k = np.min([m, n])
    for i in range(m):  # m):
        randn_vec = np.random.normal(size=n)
        x_0 = randn_vec / np.sqrt(np.sum(randn_vec**2))

        if i == 0:
            B = A.copy()
        else:
            B = B - sigmas[i - 1] * np.expand_dims(
                U[:, i - 1], axis=1
            ) @ np.expand_dims(V[:, i - 1], axis=0)

        BTB = B.T @ B

        M = BTB.copy()

        last_val = 0
        x = x_0.copy()
        while True:
            x = np.dot(M, x)
            x = x / np.sqrt(np.sum(x**2))
            check = np.sum(np.abs(x - last_val))
            if check < tolerance:
                break

            last_val = x

        v = x / np.sqrt(np.sum(x**2))
        V[:, i] = v
        Av = A @ v
        sigmas[i] = np.sqrt(np.sum(Av**2))
        U[:, i] = Av / sigmas[i]

    return U, sigmas, V


def main() -> None:
    """SVD: sigular value decomposition, A = U @ SIGMA @ V.T"""
    # TODO: account for both m > n AND n > m

    ## init vars
    m = 3
    n = 4
    tolerance = 1e-9

    A = np.random.uniform(0, 10, size=(m, n))

    ## manual implementation of SVD solver
    U, S, V = singular_value_decomposition(A)
    print("my U:", U)
    print("my sig:", S)
    print("my V:", V, flush=True)
    # print("mine:", A - U @ S @ V.T)

    ## sanity check with built-in SVD solver
    rU, rsig, rV = np.linalg.svd(A)
    print("\nbuilt-in U:", rU)
    print("built-in sig:", rsig)
    print("built-in V:", rV)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time()-t0:.3f} seconds.")
