"""Betthauser, 2020: logistic regresssion"""

import math
import time
from itertools import repeat
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

# from scipy.special import erf

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


@njit
def d_erf_dx(x):
    return (2 / np.sqrt(np.pi)) * np.exp(-(x**2))


@njit
def _d_sigmoid_dz(z):
    sig = 1 / (1 + np.exp(-z))
    return sig * (1 - sig)


def logistic_regression(
    xs: np.ndarray[float],
    ys: np.ndarray[float],
    learning_rate: float = 0.01,
    tolerance: float = 1e-7,
    plot: bool = False,
) -> Any:
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



@njit
def normal_cdf_at_x(x: float, mean: float, sigma: float) -> float:
    """
    Normal distr. integral, cdf at x | mean, variance

    Args:
        x (float): value at which to evaluate the CDF
        mean (float): sample mean
        sigma (float): sample variance

    Returns:
        float: cdf at x | mean, variance

    """
    return (1 + math.erf((x - mean) / (sigma * np.sqrt(2)))) / 2


def main() -> None:
    """_summary_"""
    # generate data
    num_points = 200
    xrange = [5, 95]
    mid_range = (xrange[0] + xrange[1]) / 2
    rng = np.random.default_rng()
    mean = rng.uniform(0.7 * mid_range, 1.3 * mid_range)
    stddev = rng.uniform(4, 9)
    xs = np.linspace(xrange[0], xrange[1], num_points)

    ## ground truth CDF(xs)
    prob_xs_equal_1 = np.array(list(map(normal_cdf_at_x, xs, repeat(mean), repeat(stddev))))

    ## generate labels from noise + probabilities
    min_err = 0.02  # 000001
    random_prob_error = min_err * rng.uniform(size=num_points)

    prob_xs_plus_err = prob_xs_equal_1 + random_prob_error
    # prob_xs_plus_err = (1 - 2 * min_err) * prob_xs_plus_err / np.max(prob_xs_plus_err) + min_err
    prob_xs_plus_err = (prob_xs_equal_1 - prob_xs_equal_1.min()) / (prob_xs_equal_1.max() - prob_xs_equal_1.min())

    labels = np.array(
        [
            rng.choice(
                [0, 1], p=[1 - prob_xs_plus_err[idx], prob_xs_plus_err[idx]]
            )
            for idx in range(num_points)
        ]
    )

    xs = np.stack((xs, np.ones(num_points))).T

    weights, _ = logistic_regression(xs, labels, learning_rate=0.003, tolerance=5e-8, plot=False)

    est_prob_xs = _sigmoid(np.dot(xs, weights))

    one_over_p = weights[0]
    t_over_p = -weights[1]

    actual_mean = mean
    actual_std = stddev

    est_mean = t_over_p * (1 / one_over_p)
    est_std = np.sqrt((np.pi**2 * (1 / one_over_p) ** 2) / 3)

    # plot results
    plt.figure(figsize=(8, 5))
    plt.scatter(
        xs[:, 0],
        labels,
        c=labels,
        cmap="Spectral",
        s=8,
        label=f"Synthetic labels, min. error: \u00b1{min_err:.2f}",
    )
    plt.plot(
        xs[:, 0],
        prob_xs_equal_1,
        label=f"Ground Truth, mean: {actual_mean:.1f}, std dev.: {actual_std:.1f}",
    )
    plt.plot(
        xs[:, 0],
        est_prob_xs,
        label=f"Estimate, mean: {est_mean:.1f}, std dev.: {est_std:.1f}",
    )
    plt.legend(loc=4, bbox_to_anchor=(1, 0.1))
    plt.show()


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter()-tmain:.3f} seconds.")
