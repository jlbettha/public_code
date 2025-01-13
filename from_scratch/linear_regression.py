import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from sklearn.datasets import make_regression


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
    N = 80
    noise_level = 15
    num_features = 1
    xs, ys_noisy = make_regression(
        n_samples=N, n_features=num_features, noise=noise_level, random_state=42
    )
    xs = np.squeeze(xs)

    ### let y = Xb --> b = X^-1 y = (X^T X)^-1 X^T y
    x_mat = np.stack((xs, np.ones(N))).T
    pseudo_inverse = (
        np.linalg.inv(x_mat.T @ x_mat) @ x_mat.T
    )  # full pinv calculation (X^T X)^-1 X^T
    b_vec = np.squeeze(pseudo_inverse @ ys_noisy)
    print(b_vec)
    ys_linalg = b_vec[0] * xs + b_vec[1]

    plt.figure()
    plt.scatter(xs, ys_noisy)
    plt.plot(
        xs,
        ys_linalg,
        c="k",
        label=f"Least squares: y={b_vec[0]:.2f}x+{b_vec[1]:.2f}",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time() - t0:.3f} seconds.")
