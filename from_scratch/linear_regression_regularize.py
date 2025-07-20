import time

import matplotlib.pyplot as plt
import numpy as np
from pylops import MatrixMult
from pylops.optimization.sparsity import fista, omp
from sklearn.datasets import make_regression


def ordinary_least_squares(y: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Regular least squares
    ### solve y = Ax --> x = A^-1 y = (A^T A)^-1 A^T y
    Args:
        y (np.ndarray): _description_
        a (np.ndarray): _description_

    Returns:
        np.ndarray: n x 1 weights vector x

    """
    pseudo_inverse = np.linalg.inv(a.T @ a) @ a.T
    return pseudo_inverse @ y


def lsq_ell_2_ridge(
    y: np.ndarray,
    a: np.ndarray,
    lambda2: float = 0.2,
) -> np.ndarray:
    """
    Ridge Regression: least squares w/ L2 penalty
    ### let y = Ax --> solve argmin_x: 1/(2n)*||Ax-y||^2_2 + lambda*||x||^2_2
    ### Tikhonov closed-form: x = (lambda*I + A^T A)^-1 A^T y
    Args:
        y (np.ndarray): _description_
        a (np.ndarray): _description_
        lambda2 (float, optional): regularization param. Defaults to 0.2.

    Returns:
        np.ndarray: n x 1 weights vector x

    """
    ###

    tikhonov = np.linalg.inv(lambda2 * np.eye(a.shape[1]) + a.T @ a) @ a.T
    return tikhonov @ y


def sparse_ell_1_lasso(y: np.ndarray, a: np.ndarray, lambda1: float = 0.25) -> np.ndarray:
    """
    Lasso Regression: least squares w/ L1 penalty
    ### let y = Ax --> solve argmin_x: 1/(2n)*||Ax-y||^2_2 + lambda*||x||_1

    Args:
        y (np.ndarray): _description_
        a (np.ndarray): _description_
        lambda1 (float, optional): regularization param. Defaults to 0.2.

    Returns:
        np.ndarray: n x 1 weights vector x

    """
    aop = MatrixMult(a)
    x_l1, _, _ = fista(aop, y, niter=1500, eps=lambda1, tol=1e-7)
    return x_l1


def sparse_ell_0_omp(y: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Sparse Regression: least squares w/ L0 penalty
    ### let y = Ax --> solve argmin_x: 1/(2n)*||Ax-y||^2_2 + lambda*||x||_0
    Args:
        y (np.ndarray): m x 1
        a (np.ndarray): m x n

    Returns:
        np.ndarray: n x 1 weights vector x

    """
    aop = MatrixMult(a)
    x_l0, _, _ = omp(aop, y)
    return x_l0


def main() -> None:
    """_summary_"""
    n = 50
    noise_level = 10
    num_features = 100  # value > 1
    a_mat, ys_noisy = make_regression(n_samples=n, n_features=num_features, noise=noise_level, random_state=42)

    # A_mat = A_mat / np.linalg.norm(A_mat, axis=0)

    ### Regular least squares
    x_reg = ordinary_least_squares(ys_noisy, a_mat)
    print(
        f"\nLSq -- sparsity {100 * (num_features - np.count_nonzero(x_reg)) / num_features:.1f}%,  Sum|weights| = {np.sum(np.abs(x_reg))}\n"
    )
    ys_reg = a_mat @ x_reg  # noqa: F841

    ### Ridge Regression: least squares w/ L2 penalty
    x_l2 = lsq_ell_2_ridge(ys_noisy, a_mat, lambda2=0.5)
    print(
        f"L2 --- sparsity {100 * (num_features - np.count_nonzero(x_l2)) / num_features:.1f}%,  Sum|weights| = {np.sum(np.abs(x_l2))}\n"
    )
    ys_l2 = a_mat @ x_l2  # noqa: F841

    ### Lasso Regression: least squares w/ L1 penalty
    x_l1 = sparse_ell_1_lasso(ys_noisy, a_mat, lambda1=0.8)
    print(
        f"L1 --- sparsity {100 * (num_features - np.count_nonzero(x_l1)) / num_features:.1f}%, Sum|weights| = {np.sum(np.abs(x_l1))}\n"
    )
    ys_l1 = a_mat @ x_l1  # noqa: F841

    ### Sparse Regression: least squares w/ L0 penalty
    x_l0 = sparse_ell_0_omp(ys_noisy, a_mat)
    print(
        f"L0 --- sparsity {100 * (num_features - np.count_nonzero(x_l0)) / num_features:.1f}%, Sum|weights| = {np.sum(np.abs(x_l0))}\n"
    )
    ys_l0 = a_mat @ x_l0  # noqa: F841

    # plots
    plt.figure(figsize=(5, 6))
    plt.subplot(4, 1, 1)
    plt.stem(x_reg, linefmt="--k", basefmt="--k", markerfmt="ko", label="LSq")
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.stem(x_l2, linefmt="--c", basefmt="--c", markerfmt="co", label="L2")
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.stem(x_l1, linefmt="--g", basefmt="--g", markerfmt="go", label="L1")
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.stem(x_l0, linefmt="--r", basefmt="--r", markerfmt="ro", label="L0")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter() - tmain:.3f} seconds.")
