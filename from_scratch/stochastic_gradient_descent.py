import time
import numpy as np
import matplotlib.pyplot as plt

# import sympy as sym


def ssd(arr1, arr2) -> float:
    """_summary_

    Args:
        arr1 (_type_): _description_
        arr2 (_type_): _description_

    Returns:
        float: _description_
    """
    return np.sum((arr1 - arr2) ** 2)


def main(b0: float = 0.73, b1: float = 4.21) -> None:
    """_summary_"""
    N = 10 * 2000
    mini_batch_size = 20
    noise_level = 1
    ones = np.ones(N)
    xs = np.linspace(0, 10, N)
    ys = b0 * xs + b1
    ys_noisy = ys + noise_level * np.random.randn(len(xs))

    ### pure linalg least squares est.
    x_mat = np.stack((xs, ones)).T
    b_vec = np.squeeze(np.linalg.pinv(x_mat) @ ys)  # easy-mode pinv calculation
    ys_linalg = b_vec[0] * xs + b_vec[1]

    ### Derivative calcs
    # obj = SUM_n (y - (xb0 + b1))**2 --> SUM_n (y - xb0 - b1)**2
    # der_obj_db0 -->  -2 SUM x(y-xb0-b1)
    # der_obj_db1 -->  -2 SUM (y-xb0-b1)

    ### gradient descent
    learning_rate = 0.01
    mini_idx = np.random.randint(0, N, size=mini_batch_size)
    b_est = [1, 1]
    sse = ssd(xs[mini_idx] * b_est[0] + b_est[1], ys_noisy[mini_idx]) / mini_batch_size
    last_sse = sse
    k = 0
    while True:
        k += 1
        b_est[0] = b_est[0] - learning_rate * (
            -2 * np.sum( xs[mini_idx] * (ys_noisy[mini_idx] - xs[mini_idx] * b_est[0] - b_est[1]) ) / mini_batch_size
        )
        b_est[1] = b_est[1] - learning_rate * (
            -2 * np.sum(ys_noisy[mini_idx] - xs[mini_idx] * b_est[0] - b_est[1]) / mini_batch_size
        )
        sse = ssd(xs[mini_idx] * b_est[0] + b_est[1], ys_noisy[mini_idx]) / mini_batch_size
        
        mini_idx = np.random.randint(0, N, size=mini_batch_size)
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
        xs, ys_linalg, c="k", label=f"Least squares: y={b_vec[0]:.2f}x+{b_vec[1]:.2f}"
    )
    plt.plot(
        xs, y_est, c="r", label=f"Stochastic gradient descent: y={b_est[0]:.2f}x+{b_est[1]:.2f}, {k} iters.",
    )
    plt.legend()

    plt.show()


if __name__ == "__main__":
    t0 = time.time()
    main(b0=0.23, b1=-2.21)
    print(f"Program took {time.time() - t0:.3f} seconds.")