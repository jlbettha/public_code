import time
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import default_rng

rng = default_rng(42)


class RANSAC:
    def __init__(self, n=10, k=100, t=0.05, d=10, model=None, loss=None, metric=None):
        self.n = n  # `n`: Minimum number of data points to estimate parameters
        self.k = k  # `k`: Maximum iterations allowed
        self.t = t  # `t`: Threshold value to determine if points are fit well
        self.d = d  # `d`: Number of close data points required to assert model fits well
        self.model = model  # `model`: class implementing `fit` and `predict`
        self.loss = loss  # `loss`: function of `y_true` and `y_pred` that returns a vector
        self.metric = metric  # `metric`: function of `y_true` and `y_pred` and returns a float
        self.best_fit = None
        self.best_error = np.inf

    def fit(self, x, y):
        for _ in range(self.k):
            ids = rng.permutation(x.shape[0])

            maybe_inliers = ids[: self.n]
            maybe_model = copy(self.model).fit(x[maybe_inliers], y[maybe_inliers])

            thresholded = self.loss(y[ids][self.n :], maybe_model.predict(x[ids][self.n :])) < self.t

            inlier_ids = ids[self.n :][np.flatnonzero(thresholded).flatten()]

            if inlier_ids.size > self.d:
                inlier_points = np.hstack([maybe_inliers, inlier_ids])
                better_model = copy(self.model).fit(x[inlier_points], y[inlier_points])

                this_error = self.metric(y[inlier_points], better_model.predict(x[inlier_points]))

                if this_error < self.best_error:
                    self.best_error = this_error
                    self.best_fit = better_model

        return self

    def predict(self, x):
        return self.best_fit.predict(x)


# @njit
def square_error_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2


# @njit
def mean_square_error(y_true, y_pred):
    return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]


# @njit
def quick_inv(x: np.ndarray, y: np.ndarray):
    return np.linalg.inv(x.T @ x) @ x.T @ y


# @njit
def quick_predict(x: np.ndarray, params: np.ndarray):
    return x @ params


class LinearRegressor:
    def __init__(self):
        self.params = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.hstack([np.ones((x.shape[0], 1)), x])
        self.params = quick_inv(x, y)
        return self

    def predict(self, x: np.ndarray):
        x = np.hstack([np.ones((x.shape[0], 1)), x])
        return quick_predict(x, self.params)


def main():
    regressor = RANSAC(model=LinearRegressor(), loss=square_error_loss, metric=mean_square_error)

    df = pd.read_csv("ransac_data.csv")
    x = df["x"].to_numpy().reshape(-1, 1)
    y = df["y"].to_numpy().reshape(-1, 1)

    regressor.fit(x, y)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1, 1)
    ax.set_box_aspect(1)

    plt.scatter(x, y)

    line = np.linspace(-1, 1, num=100).reshape(-1, 1)
    regressor.predict(line)
    plt.plot(line, regressor.predict(line), c="peru")
    plt.show()


if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = time.time()
    # main()
    # t2 = time.time()
    print(f"First run: {t1 - t0:.4f} seconds")
    # print(f"Second run: {t2 - t1:.4f} seconds")
    # print(f"Speedup: {(0.0701) / (t2 - t1):.2f}x")
