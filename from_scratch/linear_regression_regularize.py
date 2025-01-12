import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from pylops.optimization.sparsity import omp, fista


def ssd(arr1: NDArray[np.float64], arr2: NDArray[np.float64]) -> float:
    """sum of squared difference between two arrays

    Args:
        arr1 (NDArray[np.float64]): an array
        arr2 (NDArray[np.float64]): another array

    Returns:
        float: sum of squared difference
    """
    return np.sum((arr1 - arr2) ** 2)


def main() -> None:
    """_summary_"""

    lambda0 = 0.1
    lambda1 = 0.1
    lambda2 = 0.1

    b1 = 0.23
    b0 = -2.21
    N = 200
    noise_level = 2
    ones = np.ones(N)
    xs = np.linspace(0, 10, N)
    ys = b0 * xs + b1
    ys_noisy = ys + noise_level * np.random.randn(N)

    ### Regular least squares
    ### solve y = Xb --> b = X^-1 y = (X^T X)^-1 X^T y
    x_mat = np.stack((xs, ones)).T
    pseudo_inverse = (
        np.linalg.inv(x_mat.T @ x_mat) @ x_mat.T
    )  # full pinv calculation (X^T X)^-1 X^T
    b_reg = np.squeeze(pseudo_inverse @ ys)
    print(b_reg)
    ys_reg = b_reg[0] * xs + b_reg[1]

    ### Ridge Regression: least squares w/ L2 penalty
    ### let y = Xb --> solve argmin_b: 1/(2n)*||Xb-y||^2_2 + lambda*||b||^2_2
    ### Tikhonov closed-form: b = (lambda*I + X^T X)^-1 X^T y

    ### Lasso Regression: least squares w/ L1 penalty
    ### let y = Xb --> solve argmin_b: 1/(2n)*||Xb-y||^2_2 + lambda*||b||_1

    ### S-sparse Regression: least squares w/ L0 penalty
    ### let y = Xb --> solve argmin_b: 1/(2n)*||Xb-y||^2_2 + lambda*||b||_0

    plt.figure()
    plt.scatter(xs, ys_noisy)
    plt.plot(
        xs,
        ys_reg,
        c="k",
        label=f"Least squares: y={b_reg[0]:.2f}x+{b_reg[1]:.2f}",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time() - t0:.3f} seconds.")
