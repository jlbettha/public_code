""" Betthauser, 2020: support vector machine, RBF kernel (for now)
"""

import time
import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import Callable


def radial_basis_func(
    a: float | NDArray[np.float64], b: float | NDArray[np.float64], gamma: float = 1.0
) -> float | NDArray[np.float64]:
    """RBF: radial basis function

    Args:
        a (float | NDArray[np.float64]): input vector/scalar
        b (float | NDArray[np.float64]): input vector/scalar to compare with 'a'
        gamma (float, optional): gamma hyperparameter, usually tuned via validation. Defaults to 1.0.

    Returns:
        float | NDArray[np.float64]: weight(s)
    """
    return np.exp(-gamma * (np.linalg.norm(a - b) ** 2) / 2.0)


def d_rbf_dx(
    a: float | NDArray[np.float64], b: float | NDArray[np.float64], gamma: float = 1.0
) -> float | NDArray[np.float64]:
    """_summary_

    Args:
        a (float | NDArray[np.float64]): _description_
        b (float | NDArray[np.float64]): _description_
        gamma (float, optional): _description_. Defaults to 1.0.

    Returns:
        float | NDArray[np.float64]: _description_
    """
    dist = np.linalg.norm(a - b)
    return -gamma * dist * np.exp(-gamma * (dist**2) / 2.0)


def compute_kernel_matrix(X, kernel_function: Callable, gamma=1.0):
    """
    Compute the kernel matrix for a dataset X using a given kernel function.

    Parameters:
    - X: Input dataset (n_samples, n_features).
    - kernel_function: Kernel function to use.
    - gamma: Kernel parameter.

    Returns:
    - Kernel matrix (n_samples, n_samples).
    """
    n_samples = X.shape[0]
    kernel_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            kernel_matrix[i, j] = kernel_function(X[i], X[j], gamma)

    return kernel_matrix


def encode_one_hot(y, num_classes):
    y_one_hot = np.zeros((y.shape[0], num_classes))
    for n in range(y.shape[0]):
        y_one_hot[n, y[n]] = 1
    return np.vstack(y_one_hot)


def hinge_loss(W, X, y, lambda1: float = 0.01):
    norm_2 = 0.5 * np.mean(W.T @ W)

    fx = W @ X.T
    t = y * fx.T
    dists = np.ones(t.shape) - t
    dists[dists < 0] = 0

    return lambda1 * norm_2 + np.mean(dists)


# def plot_decision_boundary(model, X, y, title):
#     # Create a grid to plot the decision boundary
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)

#     plt.contourf(xx, yy, Z, alpha=0.3)
#     plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor="k")
#     plt.title(title)
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.xlim(x_min, x_max)
#     plt.ylim(y_min, y_max)
#     plt.gca().set_aspect("equal", adjustable="box")


def support_vector_machine(
    X: NDArray[np.float64],
    y: NDArray[np.uint8],
    learning_rate: float = 0.001,
    lambda1: float = 0.1,
    iters: float = 1000,
) -> None:

    num_classes = len(np.unique(y))
    num_samples, num_features = X.shape

    weights = np.random.randn(num_classes, num_features)
    # biases = np.zeros((num_classes, 1))

    y_one_hot = encode_one_hot(y, num_classes)
    y_one_hot[y_one_hot == 0] = -1

    # print(y_one_hot)

    # = ∑ai – ½∑aiaj yiyjK(xi•xj)
    loss = hinge_loss(weights, X, y_one_hot, lambda1)
    last_loss = loss
    losses = []
    for it in range(iters):
        # TODO: fix iteration updates (runs, but not correctly)
        indicator = y_one_hot * (weights @ X.T).T
        indicator[indicator < 1] = 1
        indicator[indicator > 1] = 0

        gradient_weights = -(y_one_hot * indicator).T @ X + 2 * lambda1 * weights

        weights = weights - learning_rate * gradient_weights

        loss = hinge_loss(weights, X, y_one_hot, lambda1)
        if it % 500 == 0:
            losses.append(loss)
            # print(f"loss: {loss:.7f}")
            plt.cla()
            plt.semilogy(losses)
            plt.pause(0.001)

        if np.abs(loss - last_loss) < 1e-9:
            plt.close()
            print(f"SVM-RBF took {it+1} iterations.")
            return weights

        last_loss = loss

    print(f"SVM-RBF took {it+1} iterations.")
    return weights


def main() -> None:
    """_summary_"""
    ## init vars
    learning_rate = 0.0001
    lambda1 = 0.01
    gamma = 1 / 4
    iters = 55000

    X, y = load_iris(return_X_y=True)
    # X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    num_classes = len(np.unique(y))

    def rbf_kernel(x1, x2, gamma=1.0):
        squared_distance = np.sum((x1 - x2) ** 2)
        return np.exp(-gamma * squared_distance)

    X_train_rbf = compute_kernel_matrix(X_train, rbf_kernel, gamma=gamma)
    print("Kernel Matrix (RBF Kernel):")
    print(X_train_rbf.shape)

    # exit()

    # X = np.c_[X, X**2]
    X_train_rbf = np.c_[X_train_rbf, np.ones(X_train_rbf.shape[0])]

    weights = support_vector_machine(
        X_train_rbf, y_train, learning_rate, lambda1, iters
    )
    print(weights.shape)
    for i in range(1):  # len(y_test)):
        test_pt = X_test[i, :]
        test_pt_rbf = np.array(
            [
                radial_basis_func(test_pt, X_train[j, :], gamma=gamma)
                for j in range(len(y_train))
            ]
        )
        test_pt_rbf = np.append(test_pt_rbf, 1.0)
        print(test_pt_rbf.shape)
        one_hot_test = weights @ test_pt_rbf
        print(one_hot_test)
        print(y_test[i])
    # x = np.linspace(0, 10, 1000)
    # dists = radial_basis_func(x, gamma=1 / num_features)
    # print(dists)
    # plt.plot(x, dists)
    # plt.show()

    return NotImplementedError


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time()-t0:.3f} seconds.")
