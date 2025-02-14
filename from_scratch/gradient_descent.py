import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from sklearn.datasets import make_regression
from numba import njit


@njit
def ssd(arr1: NDArray[np.float64], arr2: NDArray[np.float64]) -> float:
    """SSD: sum of squared differences between two arrays

    Args:
        arr1 (NDArray[np.float64]): an array
        arr2 (NDArray[np.float64]): another array

    Returns:
        float: sum of squared differences
    """
    return np.sum((arr1 - arr2) * (arr1 - arr2))


def main() -> None:
    """_summary_"""
    N = 200
    noise_level = 15
    num_features = 1
    xs, ys_noisy = make_regression(
        n_samples=N, n_features=num_features, noise=noise_level
    )
    xs = np.squeeze(xs)

    ### ordinary least squares est.
    x_mat = np.stack((xs, np.ones(N))).T
    b_vec = np.squeeze(np.linalg.pinv(x_mat) @ ys_noisy)  # easy-mode pinv calculation
    ys_linalg = b_vec[0] * xs + b_vec[1]

    ### Derivative calcs
    # obj = SUM_n (y - (xb0 + b1))**2 --> SUM_n (y - xb0 - b1)**2
    # der_obj_db0 -->  -2 SUM x(y-xb0-b1)
    # der_obj_db1 -->  -2 SUM (y-xb0-b1)

    ### gradient descent
    learning_rate = 0.03
    b_est = [1, 1]
    sse = ssd(xs * b_est[0] + b_est[1], ys_noisy) / N
    last_sse = sse
    k = 0
    while True:
        k += 1
        b_est[0] = b_est[0] - learning_rate * (
            -2 * np.sum(xs * (ys_noisy - xs * b_est[0] - b_est[1]) / N)
        )
        b_est[1] = b_est[1] - learning_rate * (
            -2 * np.sum(ys_noisy - xs * b_est[0] - b_est[1]) / N
        )
        sse = ssd(xs * b_est[0] + b_est[1], ys_noisy) / N
        # print(sse)

        sse_diff = np.abs(last_sse - sse)
        last_sse = sse
        if sse_diff < 1e-6 or np.isnan(sse) or np.isinf(sse):
            break

    print(b_vec)
    print(b_est)
    print(k)

    y_est = b_est[0] * xs + b_est[1]

    plt.figure()
    plt.scatter(xs, ys_noisy)
    plt.plot(
        xs,
        ys_linalg,
        c="k",
        label=f"Ordinary least squares: y={b_vec[0]:.2f}x+{b_vec[1]:.2f}",
    )
    plt.plot(
        xs,
        y_est,
        c="r",
        ls=":",
        label=f"Gradient descent: y={b_est[0]:.2f}x+{b_est[1]:.2f}, {k} iters.",
    )
    plt.legend()

    plt.show()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time() - t0:.3f} seconds.")
