import time
import numpy as np
import matplotlib.pyplot as plt

# import sympy as sym


def ssd(arr1, arr2):
    return np.sum((arr1 - arr2) ** 2)


def main(b0: float, b1: float) -> None:
    """_summary_"""
    N = 10 * 20
    noise_level = 2
    ones = np.ones(N)
    xs = np.linspace(0, 10, N)
    ys = b0 * xs + b1
    ys_noisy = ys + noise_level * np.random.randn(len(xs))

    ### let y = Xb --> b = X^-1 y = (X^T X)^-1 X^T y
    x_mat = np.stack((xs, ones)).T
    pseudo_inverse = (
        np.linalg.inv(x_mat.T @ x_mat) @ x_mat.T
    )  # full pinv calculation (X^T X)^-1 X^T
    b_vec = np.squeeze(pseudo_inverse @ ys)
    # b_vec = np.squeeze(np.linalg.pinv(x_mat) @ ys) # easy-mode pinv calculation
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
    main(b1=0.23, b0=-2.21)
    print(f"Program took {time.time() - t0:.3f} seconds.")
