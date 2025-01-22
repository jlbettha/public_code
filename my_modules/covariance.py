import time
from numpy.typing import NDArray
import numpy as np


def covariance(x_mat: NDArray[np.float64]) -> NDArray[np.float64]:
    """_summary_

    Args:
        x_mat (NDArray[float]): (n x m) matrix where n is number of samples, and m is number of features

    Returns:
        x_cov (NDArray[float]): (m x m) covariance matrix
    """
    mu = np.mean(x_mat, axis=0)
    N = x_mat.shape[0]
    x_centered = np.subtract(x_mat, mu)
    x_cov = (1 / (N - 1)) * x_centered.T @ x_centered
    return x_cov


if __name__ == "__main__":
    x_mat1 = np.random.rand(1000, 5)
    x_matT = x_mat1.T

    t0 = time.time()
    mycov = covariance(x_mat1)
    tfcov = time.time() - t0

    t1 = time.time()
    npcov = np.cov(x_matT)
    tcov = time.time() - t1

    print("my_cov: ", mycov)
    print("np_cov: ", npcov)

    print(f"my_cov time: {tfcov}")
    print(f"np_cov time: {tcov}")

    if np.sum(np.abs(mycov - npcov)) < 1e-9:
        print("sanity achieved!")
