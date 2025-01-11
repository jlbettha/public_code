# pylint: disable=C0103
""" Betthauser, 2020: 
SVD: sigular value decomposition, A = U @ SIGMA @ V.T
"""

import time
import numpy as np
from numpy.typing import NDArray


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
    k = np.min([m, n])

    ## manually compute m eigenvalues and eigenvectors of
    ## matrix A (copied to alias 'B' to keep original A intact)
    B = A.copy()
    sigmas = np.zeros(k)
    U = np.zeros((m, k))
    VT = np.zeros((k, n))

    for i in range(k):  # m):
        randn_vec = np.random.normal(size=n)
        x_0 = randn_vec / np.sqrt(np.sum(randn_vec**2))

        if i == 0:
            B = A.copy()
        else:
            # remove influence of previous eigenval/vec from B
            B = B - sigmas[i - 1] * np.expand_dims(
                U[:, i - 1], axis=1
            ) @ np.expand_dims(VT[i - 1, :], axis=0)

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
        VT[i, :] = v
        Av = A @ v
        sigmas[i] = np.sqrt(np.sum(Av**2))
        U[:, i] = Av / sigmas[i]

    return U, sigmas, VT


def main() -> None:
    """SVD: sigular value decomposition, A = U @ SIGMA @ V.T"""

    ## init vars
    m = 20
    n = 15

    A = np.random.uniform(0, 10, size=(m, n))

    ## manual implementation of SVD solver
    U, sigmas, VT = singular_value_decomposition(A)

    ## sanity check with built-in SVD solver
    rU, rsig, rVT = np.linalg.svd(A, full_matrices=False)

    # print("my U:", U)
    # print("my sig:", sigmas)
    # print("my V^T:", VT, flush=True)

    # print("\nbuilt-in U:", rU)
    # print("built-in sig:", rsig)
    # print("built-in V^T:", rVT)

    print(f"Check: SVD_mine == A ? {np.allclose(A, U @ np.diag(sigmas) @ VT)}")
    print(f"Check: SVD_numpy == A ? {np.allclose(A, rU @ np.diag(rsig) @ rVT)}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time()-t0:.3f} seconds.")
