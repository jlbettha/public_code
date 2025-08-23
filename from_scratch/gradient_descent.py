import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from sklearn.datasets import make_regression


@njit
def ssd(arr1: np.ndarray[float], arr2: np.ndarray[float]) -> float:
    """
    SSD: sum of squared differences between two arrays

    Args:
        arr1 (np.ndarray[float]): an array
        arr2 (np.ndarray[float]): another array

    Returns:
        float: sum of squared differences

    """
    diff = arr1 - arr2
    return np.sum(diff * diff)


@njit
def gradient_d_ssd_dw(x, y, w):
    return -2 * x.T @ (y - x @ w)


def main() -> None:
    """_summary_"""
    n = 200
    noise_level = 15
    num_features = 1
    xs, ys_noisy = make_regression(
        n_samples=n, n_features=num_features, noise=noise_level
    )
    x_mat = np.c_[xs, np.ones(xs.shape[0])]

    ### ordinary least squares est.
    w_vec = np.squeeze(np.linalg.pinv(x_mat) @ ys_noisy)  # easy-mode pinv calculation
    ys_linalg = x_mat @ w_vec

    ### Derivative calcs
    # obj = SUM_n (y - (xb0 + b1))**2 --> SUM_n (y - xb0 - b1)**2
    # der_obj_db0 -->  -2 SUM x(y-xb0-b1)
    # der_obj_db1 -->  -2 SUM (y-xb0-b1)

    ### gradient descent
    learning_rate = 0.02
    rng = np.random.default_rng()
    w_est = rng.normal(size=2)
    sse = ssd(x_mat @ w_est, ys_noisy) / n
    last_sse = sse
    k = 0
    tol = 1e-5
    while True:
        k += 1
        # dw = gradient_d_ssd_dw(x_mat, ys_noisy, w_est)
        dw = -2 * x_mat.T @ (ys_noisy - x_mat @ w_est)
        w_est = w_est - learning_rate * dw / n  # w
        sse = ssd(x_mat @ w_est, ys_noisy) / n
        # print(sse)

        sse_diff = np.abs(last_sse - sse)

        if sse_diff < tol or np.isnan(sse) or np.isinf(sse):
            break

        last_sse = sse

        if k % 5 == 0:
            y_est = x_mat @ w_est
            plt.cla()
            plt.scatter(xs, ys_noisy)
            plt.plot(
                xs,
                ys_linalg,
                c="k",
                label=f"Ordinary least squares: y={w_vec[0]:.2f}x+{w_vec[1]:.2f}",
            )
            plt.plot(
                xs,
                y_est,
                c="r",
                ls=":",
                label=f"Gradient descent: y={w_est[0]:.2f}x+{w_est[1]:.2f}, {k} iters.",
            )
            plt.pause(0.001)

    plt.close()
    y_est = x_mat @ w_est

    plt.figure()
    plt.scatter(xs, ys_noisy)
    plt.plot(
        xs,
        ys_linalg,
        c="k",
        label=f"Ordinary least squares: y={w_vec[0]:.2f}x+{w_vec[1]:.2f}",
    )
    plt.plot(
        xs,
        y_est,
        c="r",
        ls=":",
        label=f"Gradient descent: y={w_est[0]:.2f}x+{w_est[1]:.2f}, {k} iters.",
    )
    plt.legend()

    plt.show()


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter()-tmain:.3f} seconds.")
