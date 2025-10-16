"""Betthauser, 2020: support vector machine, RBF kernel (for now)"""

import time
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from sklearn.datasets import make_circles, make_moons, make_multilabel_classification  # noqa: F401
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # noqa: F401

EPS = 1e-12


@njit
def radial_basis_func(
    a: float | np.ndarray[float], b: float | np.ndarray[float], gamma: float = 1.0
) -> float | np.ndarray[float]:
    """
    RBF: radial basis function

    Args:
        a (float | np.ndarray[float]): input vector/scalar
        b (float | np.ndarray[float]): input vector/scalar to compare with 'a'
        gamma (float, optional): gamma hyperparameter, usually tuned via validation. Defaults to 1.0.

    Returns:
        float |np.ndarray[float]: weight(s)

    """
    vec = a - b
    return np.exp(-gamma * np.sum(vec * vec))


@njit
def d_rbf_dx(
    a: float | np.ndarray[float], b: float | np.ndarray[float], gamma: float = 1.0
) -> float | np.ndarray[float]:
    """
    _summary_

    Args:
        a (float | np.ndarray[float]): _description_
        b (float | np.ndarray[float]): _description_
        gamma (float, optional): _description_. Defaults to 1.0.

    Returns:
        float | np.ndarray[float]: _description_

    """
    vec = a - b
    ssd = np.sum(vec * vec)
    dist = np.sqrt(ssd)
    return -2 * gamma * dist * np.exp(-gamma * ssd)


@njit
def compute_kernel_matrix(x: np.ndarray[float], kernel_function: Callable, gamma: float = 1.0) -> np.ndarray[float]:
    """
    Compute the kernel matrix for a dataset x using a given kernel function.

    Args:
        x (np.ndarray[float]): Input dataset of shape (n_samples, n_features).
        kernel_function (Callable): Kernel function to use for computing the kernel matrix.
        gamma (float, optional): Kernel parameter, typically used for RBF kernels. Defaults to 1.0.

    Returns:
        np.ndarray[float]: Kernel matrix of shape (n_samples, n_samples).

    """
    n_samples = x.shape[0]
    kernel_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            kernel_matrix[i, j] = kernel_function(x[i], x[j], gamma)

    return kernel_matrix


@njit
def encode_one_hot(y: np.ndarray[int]) -> np.ndarray[int]:
    """
    _summary_

    Args:
        y (np.ndarray[int]): _description_

    Returns:
        np.ndarray[int]: _description_

    """
    num_classes = len(np.unique(y))
    y_one_hot = np.zeros((y.shape[0], num_classes))
    for n in range(y.shape[0]):
        y_one_hot[n, y[n]] = 1
    return y_one_hot


@njit
def hinge_loss(
    w: np.ndarray[float],
    x: np.ndarray[float],
    y_one_hot: np.ndarray[int],
    lambda1: float = 0.05,
) -> float:
    """
    _summary_

    Args:
        w (np.ndarray[float]): _description_
        x (np.ndarray[float]): _description_
        y_one_hot (np.ndarray[int]): _description_
        lambda1 (float, optional): _description_. Defaults to 0.05.

    Returns:
        float: _description_

    """
    regularize_term = 0.0
    if lambda1 > EPS:
        regularize_term = 0.5 * np.sum(w.T @ w)
    z = w @ x.T
    t = y_one_hot * z.T
    hinge_term = np.ones(t.shape) - t
    hinge_term = np.maximum(hinge_term, 0)

    return lambda1 * regularize_term + np.sum(hinge_term)


def support_vector_machine(
    x: np.ndarray[float],
    y: np.ndarray[int],
    learning_rate: float = 1e-3,
    iters: float = 1000,
    c: float = 1.0,
    tol: float = 1e-3,
) -> None:
    """
    _summary_

    Args:
        x (np.ndarray[float]): _description_
        y (np.ndarray[int]): _description_
        learning_rate (float, optional): _description_. Defaults to 1e-3.
        iters (float, optional): _description_. Defaults to 1000.
        c (float, optional): _description_. Defaults to 1.0.
        tol (float, optional): _description_. Defaults to 1e-3.

    """
    num_classes = len(np.unique(y))
    _, num_features = x.shape

    rng = np.random.default_rng()
    weights = rng.uniform(size=(num_classes, num_features))

    y_one_hot = encode_one_hot(y)
    y_one_hot[y_one_hot == 0] = -1

    loss = hinge_loss(weights, x, y_one_hot, 0.00)
    last_loss = loss
    min_loss = loss
    min_it = 0
    # losses = []
    for it in range(iters):
        z = weights @ x.T
        t = (y_one_hot * z.T).T
        ids = np.where(t > 1)

        gradient_weights = weights - c * y_one_hot.T @ x
        gradient_weights[ids] = weights[ids]

        weights = weights - learning_rate * gradient_weights * (0.9999**it)

        loss = hinge_loss(weights, x, y_one_hot, 0.00)
        if loss < min_loss:
            min_loss = loss
            min_it = it
            final_weights = weights

        # if it % 10_000 == 0:
        #     losses.append(loss)
        #     # print(f"loss: {loss:.7f}")
        #     plt.cla()
        #     plt.semilogy(losses)
        #     plt.pause(0.001)

        if np.abs(loss - last_loss) < tol or (loss - last_loss) > 0:
            # plt.close()
            # print(f"SVM-RBF took {min_it+1} iterations.")
            return final_weights, min_it

        last_loss = loss

    print(f"SVM-RBF took {min_it + 1} iterations.")
    return final_weights, min_it


def main() -> None:  # noqa: PLR0915
    """_summary_"""
    ## init vars
    learning_rate = 1e-3
    gamma = 15
    iters = 500_000
    c = 0.1
    tol = 1e-8

    x, y = make_moons(n_samples=600, noise=0.1, random_state=22)
    # x, y = make_circles(n_samples=600, shuffle=True, noise=0.05, random_state=42)
    y = np.squeeze(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

    # scaler = MinMaxScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)

    t0 = time.perf_counter()
    x_train_rbf = compute_kernel_matrix(x_train, radial_basis_func, gamma=gamma)
    print(f"build rbf kernel: {time.perf_counter() - t0:.3f} seconds")

    x_train_rbf = np.c_[x_train_rbf, np.ones(x_train_rbf.shape[0])]

    t0 = time.perf_counter()
    weights, its = support_vector_machine(x_train_rbf, y_train, learning_rate=learning_rate, iters=iters, c=c, tol=tol)
    print(f"train time: {time.perf_counter() - t0:.3f} seconds in {its + 1} iterations")

    y_pred = []
    # x_test, y_test = x_train, y_train  # test on training data, should be close to 100%

    t0 = time.perf_counter()
    for i in range(len(y_test)):
        test_pt = x_test[i, :]
        test_pt_rbf = np.array([radial_basis_func(test_pt, x_train[j, :], gamma=gamma) for j in range(len(y_train))])
        test_pt_rbf = np.append(test_pt_rbf, 1.0)
        one_hot_test = weights @ test_pt_rbf
        y_pred.append(np.argmax(one_hot_test))
        # print(np.argmax(one_hot_test), y_test[i])

    print(f"avg predict time: {(time.perf_counter() - t0) / len(y_test):.3f} seconds")
    print(f"predict time for {len(y_test)} examples: {(time.perf_counter() - t0):.3f} seconds")
    print(f"test accuracy: {accuracy_score(np.array(y_pred), y_test):.4f}")

    # Create a grid to plot the decision boundary
    delta = 0.025
    x_min, x_max = x_test[:, 0].min() - 0.1, x_test[:, 0].max() + 0.1
    y_min, y_max = x_test[:, 1].min() - 0.1, x_test[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, delta), np.arange(y_min, y_max, delta))
    xx_long, yy_long = xx.ravel(), yy.ravel()
    z = []
    t0 = time.perf_counter()
    for i in range(len(xx_long)):
        test_pt = np.c_[xx_long[i], yy_long[i]]
        test_pt_rbf = np.array([radial_basis_func(test_pt, x_train[j, :], gamma=gamma) for j in range(len(y_train))])
        test_pt_rbf = np.append(test_pt_rbf, 1.0)
        one_hot_test = weights @ test_pt_rbf
        # print(one_hot_test)
        z.append(one_hot_test[1] - one_hot_test[0])  # np.argmax(one_hot_test))

    print(f"avg predict time: {(time.perf_counter() - t0) / len(xx_long):.3f} seconds")
    print(f"predict time for {len(xx_long)} examples: {(time.perf_counter() - t0):.3f} seconds")

    z = np.array(z).reshape(xx.shape)
    z = (z - z.min()) / (z.max() - z.min())

    # plt.contourf(xx, yy, z > 0.5, alpha=0.3)
    plt.contourf(xx, yy, z, levels=np.linspace(0, 1, 20), alpha=0.3)
    plt.contour(xx, yy, z, colors="k", levels=1, linewidths=1)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=20, edgecolor="k", linewidth=0.5)
    # plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter() - tmain:.3f} seconds.")
