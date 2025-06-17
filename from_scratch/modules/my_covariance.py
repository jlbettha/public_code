import time
from numpy.typing import NDArray
import numpy as np


def covariance(x_mat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute covariance of a matrix
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


def main() -> None:
    x_mat1 = np.random.rand(1000, 200)
    x_matT = x_mat1.T

    t0 = time.perf_counter()
    mycov = covariance(x_mat1)
    tfcov = time.perf_counter() - t0

    t1 = time.perf_counter()
    npcov = np.cov(x_matT)
    tcov = time.perf_counter() - t1

    # print("my_cov: ", mycov)
    # print("np_cov: ", npcov)

    print(f"my_cov([{x_mat1.shape[0]} x {x_mat1.shape[1]}]) time: {tfcov}")
    print(f"np.cov([{x_mat1.shape[0]} x {x_mat1.shape[1]}]) time: {tcov}")

    if np.sum(np.abs(mycov - npcov)) < 1e-9:
        print("my_cov() == np.cov(): sanity achieved!")


if __name__ == "__main__":
    main()
