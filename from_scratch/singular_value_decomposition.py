# pylint: disable=C0103
"""
Betthauser, 2020:
SVD: sigular value decomposition, A = U @ SIGMA @ V.T
"""

import time

import numpy as np


def singular_value_decomposition(a: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    """
    SVD of matrix A

    Args:
        a (np.ndarray[float]): m x n matrix to decompose
        tolerance (float): optional, default = 1e-9

    Returns:
        tuple[np.ndarray[float]]: u, sigmas, v

    """
    m, n = a.shape
    k = np.min([m, n])

    ## manually compute m eigenvalues and eigenvectors of
    ## matrix A (copied to alias 'b' to keep original A intact)
    b = a.copy()
    sigmas = np.zeros(k)
    u = np.zeros((m, k))
    vt = np.zeros((k, n))

    rng = np.random.default_rng()
    for i in range(k):  # m):
        randn_vec = rng.normal(size=n)
        x_0 = randn_vec / np.sqrt(np.sum(randn_vec**2))

        if i == 0:
            b = a.copy()
        else:
            # remove influence of previous eigenval/vec from b
            b = b - sigmas[i - 1] * np.expand_dims(u[:, i - 1], axis=1) @ np.expand_dims(vt[i - 1, :], axis=0)

        btb = b.T @ b

        m = btb.copy()

        last_val = 0
        x = x_0.copy()
        while True:
            x = np.dot(m, x)
            x = x / np.sqrt(np.sum(x**2))
            check = np.sum(np.abs(x - last_val))
            if check < tolerance:
                break

            last_val = x

        v = x / np.sqrt(np.sum(x**2))
        vt[i, :] = v
        av = a @ v
        sigmas[i] = np.sqrt(np.sum(av**2))
        u[:, i] = av / sigmas[i]

    return u, sigmas, vt


def main() -> None:
    """SVD: sigular value decomposition, A = U @ SIGMA @ V.T"""
    ## init vars
    m = 20
    n = 15

    rng = np.random.default_rng()
    a = rng.uniform(0, 10, size=(m, n))

    ## manual implementation of SVD solver
    u, sigmas, vt = singular_value_decomposition(a)

    ## sanity check with built-in SVD solver
    ru, rsig, rvt = np.linalg.svd(a, full_matrices=False)

    # print("my U:", u)
    # print("my sig:", sigmas)
    # print("my V^T:", vt, flush=True)

    # print("\nbuilt-in U:", ru)
    # print("built-in sig:", rsig)
    # print("built-in V^T:", rvt)

    print(f"Check: SVD_mine == A ? {np.allclose(a, u @ np.diag(sigmas) @ vt)}")
    print(f"Check: SVD_numpy == A ? {np.allclose(a, ru @ np.diag(rsig) @ rvt)}")


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter() - tmain:.3f} seconds.")
