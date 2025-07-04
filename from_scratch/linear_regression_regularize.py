import time
import numpy as np
import matplotlib.pyplot as plt
from pylops.optimization.sparsity import omp, fista
from pylops import MatrixMult
from sklearn.datasets import make_regression


def ordinary_least_squares(
    y: np.ndarray[float], A: np.ndarray[float]
) -> np.ndarray[float]:
    """Regular least squares
    ### solve y = Ax --> x = A^-1 y = (A^T A)^-1 A^T y
    Args:
        y (np.ndarray[float]]): _description_
        A (np.ndarray[float]): _description_

    Returns:
        np.ndarray[float]: n x 1 weights vector x
    """

    pseudo_inverse = np.linalg.inv(A.T @ A) @ A.T
    x_reg = pseudo_inverse @ y
    return x_reg


def lsq_ell_2_ridge(
    y: np.ndarray[float],
    A: np.ndarray[float],
    lambda2: float = 0.2,
) -> np.ndarray[float]:
    """Ridge Regression: least squares w/ L2 penalty
    ### let y = Ax --> solve argmin_x: 1/(2n)*||Ax-y||^2_2 + lambda*||x||^2_2
    ### Tikhonov closed-form: x = (lambda*I + A^T A)^-1 A^T y
    Args:
        y (np.ndarray[float]): _description_
        A (np.ndarray[float]): _description_
        lambda2 (float, optional): regularization param. Defaults to 0.2.

    Returns:
        np.ndarray[float]: n x 1 weights vector x
    """
    ###

    tikhonov = np.linalg.inv(lambda2 * np.eye(A.shape[1]) + A.T @ A) @ A.T
    x_l2 = tikhonov @ y
    return x_l2


def sparse_ell_1_lasso(
    y: np.ndarray[float], A: np.ndarray[float], lambda1: float = 0.25
) -> np.ndarray[float]:
    """Lasso Regression: least squares w/ L1 penalty
    ### let y = Ax --> solve argmin_x: 1/(2n)*||Ax-y||^2_2 + lambda*||x||_1

    Args:
        y (np.ndarray[float]): _description_
        A (np.ndarray[float]): _description_
        lambda1 (float, optional): regularization param. Defaults to 0.2.

    Returns:
        np.ndarray[float]: n x 1 weights vector x
    """

    Aop = MatrixMult(A)
    x_l1, _, _ = fista(Aop, y, niter=1500, eps=lambda1, tol=1e-7)
    return x_l1


def sparse_ell_0_omp(y: np.ndarray[float], A: np.ndarray[float]) -> np.ndarray[float]:
    """Sparse Regression: least squares w/ L0 penalty
    ### let y = Ax --> solve argmin_x: 1/(2n)*||Ax-y||^2_2 + lambda*||x||_0
    Args:
        y (np.ndarray[float]): m x 1
        A (np.ndarray[float]): m x n

    Returns:
        np.ndarray[float]: n x 1 weights vector x
    """

    Aop = MatrixMult(A)
    x_l0, _, _ = omp(Aop, y)
    return x_l0


def main() -> None:
    """_summary_"""
    N = 50
    noise_level = 10
    num_features = 100  # value > 1
    A_mat, ys_noisy = make_regression(
        n_samples=N, n_features=num_features, noise=noise_level, random_state=42
    )

    # A_mat = A_mat / np.linalg.norm(A_mat, axis=0)

    ### Regular least squares
    x_reg = ordinary_least_squares(ys_noisy, A_mat)
    print(
        f"\nLSq -- sparsity {100*(num_features-np.count_nonzero(x_reg))/num_features:.1f}%,  Sum|weights| = {np.sum(np.abs(x_reg))}\n"
    )
    ys_reg = A_mat @ x_reg

    ### Ridge Regression: least squares w/ L2 penalty
    x_l2 = lsq_ell_2_ridge(ys_noisy, A_mat, lambda2=0.5)
    print(
        f"L2 --- sparsity {100*(num_features-np.count_nonzero(x_l2))/num_features:.1f}%,  Sum|weights| = {np.sum(np.abs(x_l2))}\n"
    )
    ys_l2 = A_mat @ x_l2

    ### Lasso Regression: least squares w/ L1 penalty
    x_l1 = sparse_ell_1_lasso(ys_noisy, A_mat, lambda1=0.8)
    print(
        f"L1 --- sparsity {100*(num_features-np.count_nonzero(x_l1))/num_features:.1f}%, Sum|weights| = {np.sum(np.abs(x_l1))}\n"
    )
    ys_l1 = A_mat @ x_l1

    ### Sparse Regression: least squares w/ L0 penalty
    x_l0 = sparse_ell_0_omp(ys_noisy, A_mat)
    print(
        f"L0 --- sparsity {100*(num_features-np.count_nonzero(x_l0))/num_features:.1f}%, Sum|weights| = {np.sum(np.abs(x_l0))}\n"
    )
    ys_l0 = A_mat @ x_l0

    # plots
    fig = plt.figure(figsize=(5, 6))
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
    print(f"Program took {time.perf_counter()-tmain:.3f} seconds.")
