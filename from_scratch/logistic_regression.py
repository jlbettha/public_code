""" Betthauser, 2020: logistic regresssion
"""

import time
import math
from typing import Any
from itertools import repeat
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def _sigmoid(x: float | NDArray[np.float64]) -> float:
    """sigmoid(x) = 1/(1+e^-x)

    Args:
        x (float | NDArray[floats]): input to sigmoid, input is wx+b for logistic regression

    Returns:
        float: sigmoid(x)
    """
    # sigx = (1 + np.exp(x)) ** -1
    sigx = 1.0 / (1 + np.exp(-x))
    return sigx


def logistic_regression(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    learning_rate: float = 0.1,
    tolerance: float = 1e-6,
    plot: bool = False,
) -> Any:
    """_summary_

    Args:
        xs (NDArray[np.float64]): data points + appended bias column of 1s
        ys (NDArray[np.float64]): labels
        learning_rate (float): gradient descent learning rate
        tolerance (float): convergence threshold, default is 1e-6
        plot (bool, optional): _description_. Defaults to False.

    Returns:
        Any: _description_
    """
    num_pts = xs.shape[0]
    dim = xs.shape[1]

    # init conditions
    wts = np.zeros(dim)
    sigmoid_all_wx_b = _sigmoid(np.dot(xs, wts))

    pointwise_costs1 = ys * np.log(sigmoid_all_wx_b)
    pointwise_costs2 = (1 - ys) * np.log(1 - sigmoid_all_wx_b)

    j_cost = -np.sum(pointwise_costs1 + pointwise_costs2) / num_pts
    ll_last = j_cost
    likelihoods = []
    iters = 0
    while True:
        iters = iters + 1

        # update weights with gradient descent,  rule: d_loss/dwt_j = (1/N)*SUM[sigmoid(x_i)-y_i)*x_j_i]
        wts = wts - learning_rate * np.dot(xs.T, sigmoid_all_wx_b - ys) / num_pts

        sigmoid_all_wx_b = _sigmoid(np.dot(xs, wts))

        pointwise_costs1 = ys * np.log(sigmoid_all_wx_b)
        pointwise_costs2 = (1 - ys) * np.log(1 - sigmoid_all_wx_b)

        j_cost = -np.sum(pointwise_costs1 + pointwise_costs2) / num_pts

        if iters % 50000 == 0:
            print(
                f"Iter: {iters} -- J_cost: {j_cost:.4f}, weights: {wts}",
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

        # return, if required.
        if np.abs(j_cost - ll_last) < tolerance:
            print(f"logistic regression took {iters} iterations.")
            return wts, likelihoods

        ll_last = j_cost


def normal_cdf_at_x(x: float | int, mean: float, variance: float) -> float:
    """Normal distr. integral, cdf at x | mean, variance

    Args:
        mean (float): sample mean
        variance (float): sample variance

    Returns:
        float: cdf at x | mean, variance
    """
    cdf_x = (1 + math.erf((x - mean) / np.sqrt(2 * variance))) / 2
    return cdf_x


def main() -> None:
    """_summary_"""

    # generate data
    num_points = 200
    xrange = [5, 80]
    mid_range = (xrange[0] + xrange[1]) / 2
    mean = np.random.uniform(0.75 * mid_range, 1.25 * mid_range)
    variance = np.random.uniform(0.1 * (2 * mid_range), 0.30 * (2 * mid_range))
    xs = np.linspace(xrange[0], xrange[1], num_points)

    ## ground truth CDF(xs)
    prob_xs_equal_1 = list(map(normal_cdf_at_x, xs, repeat(mean), repeat(variance)))

    ## generate labels from noise + probabilities
    err = 0.10
    random_prob_error = err * np.random.normal(size=num_points)
    prob_xs_plus_err = np.array(prob_xs_equal_1) + random_prob_error
    prob_xs_plus_err[prob_xs_plus_err > 1] = 1.0
    prob_xs_plus_err[prob_xs_plus_err < 0] = 0.0

    labels = np.array(
        [
            np.random.choice(
                [0, 1], p=[1 - prob_xs_plus_err[idx], prob_xs_plus_err[idx]]
            )
            for idx in range(num_points)
        ]
    )

    xs = np.stack((xs, np.ones(num_points))).T
    weights, _ = logistic_regression(
        xs, labels, learning_rate=0.001, tolerance=1e-9, plot=False
    )

    est_prob_xs = _sigmoid(np.dot(xs, weights))

    one_over_p = weights[0]
    t_over_p = -weights[1]

    est_mean = t_over_p * (1 / one_over_p)
    est_var = (np.pi**2 * (1 / one_over_p) ** 2) / 3

    # plot results
    plt.figure(figsize=(8, 5))
    plt.scatter(
        xs[:, 0],
        labels,
        c=labels,
        cmap="Spectral",
        s=8,
        label=f"Synthetic labels, error: \u00B1{err:.2f}",
    )
    plt.plot(
        xs[:, 0],
        prob_xs_equal_1,
        label=f"Ground Truth, mean: {mean:.1f}, variance: {variance:.1f}",
    )
    plt.plot(
        xs[:, 0],
        est_prob_xs,
        label=f"Estimate, mean: {est_mean:.1f}, variance: {est_var:.1f}",
    )
    plt.legend(loc=4, bbox_to_anchor=(1, 0.1))
    plt.show()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time()-t0:.3f} seconds.")
