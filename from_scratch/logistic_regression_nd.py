"""Betthauser, 2020: logistic regresssion"""

import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from sklearn.datasets import make_circles, make_moons, make_multilabel_classification  # noqa: F401

EPS = 1e-12


@njit
def _sigmoid(x: float | np.ndarray[float]) -> float:
    """
    sigmoid(x) = 1/(1+e^-x)

    Args:
        x (float | NDArray[floats]): input to sigmoid, input is wx+b for logistic regression

    Returns:
        float: sigmoid(x)

    """
    return 1 / (1 + np.exp(-x))


def logistic_regression(
    xs: np.ndarray[float],
    ys: np.ndarray[float],
    learning_rate: float = 0.01,
    tolerance: float = 1e-7,
    plot: bool = False,
) -> tuple:
    """
    logistic_regression

    Args:
        xs (np.ndarray[float]): data points + appended bias column of 1s
        ys (np.ndarray[float]): labels
        learning_rate (float): gradient descent learning rate
        tolerance (float): convergence threshold, default is 1e-6
        plot (bool, optional): _description_. Defaults to False.

    Returns:
        Any: _description_

    """
    num_pts = xs.shape[0]
    dim = xs.shape[1]

    # init conditions
    rng = np.random.default_rng()
    wts = rng.normal(size=dim)
    wx_b = np.dot(xs, wts)
    sigmoid_all_wx_b = _sigmoid(wx_b)
    print(f"{wx_b.shape=}, {sigmoid_all_wx_b.shape=}, {wts.shape=}, {xs.shape=}, {ys.shape=}")

    pointwise_costs1 = ys * np.log(sigmoid_all_wx_b)
    pointwise_costs2 = (1 - ys) * np.log(1 - sigmoid_all_wx_b + EPS)
    j_cost = -np.sum(pointwise_costs1 + pointwise_costs2) / num_pts

    likelihoods = []
    iters = 0
    while True:
        iters = iters + 1

        # update weights with gradient descent,  rule: d_loss/dwt_j = (1/N)*SUM[sigmoid(x_i)-y_i)*x_j_i]
        # z = np.dot(xs.T, sigmoid_all_wx_b - ys)
        # wts = wts - learning_rate * _d_sigmoid_dz(sigmoid_all_wx_b - ys) / num_pts
        wts = wts - learning_rate * np.dot(xs.T, sigmoid_all_wx_b - ys) / num_pts

        wx_b = np.dot(xs, wts)
        sigmoid_all_wx_b = _sigmoid(wx_b)

        pointwise_costs1 = ys * np.log(sigmoid_all_wx_b)
        pointwise_costs2 = (1 - ys) * np.log(1 - sigmoid_all_wx_b + EPS)

        ll_last = j_cost
        j_cost = -np.sum(pointwise_costs1 + pointwise_costs2) / num_pts
        likelihoods.append(j_cost)

        # return, if required.
        if np.abs(j_cost - ll_last) < tolerance:
            print(f"logistic regression took {iters} iterations.")
            return wts, likelihoods

        if iters % 50000 == 0:
            print(
                f"Iter: {iters} -- J_cost: {j_cost:.4f}, weights: {wts}, diff: {np.abs(j_cost - ll_last)}",
                flush=True,
            )

        # outputs, if desired.
        if plot:
            plt.subplot(1, 2, 1)
            plt.cla()
            plt.plot(np.arange(0, len(likelihoods), 1), likelihoods)
            plt.ylabel("log-likelihood")

            plt.subplot(1, 2, 2)
            plt.cla()
            plt.scatter(xs[:, 0], ys, c=ys, cmap="Spectral", s=8, label="labels")
            plt.plot(xs, sigmoid_all_wx_b, label="estimate")
            plt.legend(loc=4, bbox_to_anchor=(1, 0.1))
            plt.pause(0.001)


def main() -> None:
    """_summary_"""
    # generate data
    num_points = 400
    ndim = 2

    # xs, labels = make_circles(n_samples=num_points, shuffle=True, noise=0.1, random_state=42)
    # xs, labels = make_moons(n_samples=num_points, shuffle=True, noise=0.1, random_state=42)
    xs, labels = make_multilabel_classification(n_samples=num_points, n_features=ndim, n_classes=1, random_state=1123)

    xs = np.concatenate((xs, np.ones((num_points, 1))), axis=1)

    labels = np.squeeze(labels)
    weights, _ = logistic_regression(xs, labels, learning_rate=0.003, tolerance=5e-8, plot=False)

    num_steps = 100
    x0 = np.linspace(np.min(xs[:, 0]) - 0.2, np.max(xs[:, 0]) + 0.2, num_steps)
    x1 = np.linspace(np.min(xs[:, 1]) - 0.2, np.max(xs[:, 1]) + 0.2, num_steps)
    xx0, xx1 = np.meshgrid(x0, x1)
    wwxx_bb = weights[0] * xx0 + weights[1] * xx1 + weights[2]
    est_prob_class = _sigmoid(wwxx_bb)

    # plot results
    plt.figure(figsize=(5, 5))
    plt.scatter(
        xs[:, 0],
        xs[:, 1],
        c=labels,
        cmap="rainbow",
        s=8,
    )

    # est_prob_class[est_prob_class>0.5] = 1
    # est_prob_class[est_prob_class<=0.5] = 0
    plt.contourf(xx0, xx1, est_prob_class, levels=np.linspace(0, 1, 20), cmap="plasma", alpha=0.3)
    plt.contour(xx0, xx1, est_prob_class, colors="k", levels=1, linewidths=1)
    plt.show()


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter() - tmain:.3f} seconds.")
