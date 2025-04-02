"""Betthauser, 2018: k-medians compression.
>> Use when number of samples is large.
>> Use when compressed data must be subset/members of original data.
"""

import time
import numpy as np
from typing import Any
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def k_medians_compression(
    data: NDArray[np.float64], k: int, plot: bool = False
) -> tuple[Any]:
    """Betthauser, 2018: k-medians clustering/compression
        goal is to find k<<N data points to represent all N.
        Use when compressed data must be members of original data.

    Args:
        data (NDArray[np.float64]): unlabeled N x D, N samples of D-dim data
        k (int): number of clusters

    Returns:
        NDArray[np.uint8]: array of new labels
    """
    k_medians = data[
        np.random.permutation(data.shape[0])[:k], :
    ]  # init with k datapoints

    k_last = k_medians
    iters = 0
    while True:
        iters = iters + 1

        labels = np.array(
            [
                np.argmin(cdist(np.reshape(data[idx, :], (1, -1)), k_medians))
                for idx in range(data.shape[0])
            ]
        )

        k_medians = np.squeeze(
            [np.median(data[labels == lbl, :], axis=0) for lbl in range(k)]
        )

        if plot and iters == 1:
            plt.cla()
            plt.scatter(data[:, 0], data[:, 1], c=labels, s=2, cmap="gist_ncar")
            plt.title(f"Iteration {iters}")
            plt.pause(0.001)

        if np.sum((k_medians - k_last) ** 2) == 0 or iters > 150:
            plt.close()
            print(f"k-medians compression took {iters} iterations.")
            return labels, iters, k_medians

        k_last = k_medians


def main() -> None:
    """_summary_"""
    num_pts = 5000
    k = num_pts // 25
    num_dims = 2

    # x = np.linspace(0, 12, num_pts)
    # y = 10 * (0.5 * x) + 2 * np.random.randn(num_pts)
    # data = np.c_[x, y]

    data = np.random.uniform(10, size=(num_pts, num_dims))
    labels, iters, k_medians = k_medians_compression(data, k, plot=False)

    kcolors = np.unique(labels)

    plt.cla()
    plt.scatter(data[:, 0], data[:, 1], zorder=-1, c=labels, s=2, cmap="gist_ncar")
    plt.scatter(
        k_medians[:, 0],
        k_medians[:, 1],
        c="k",
        s=15,
        zorder=1,
        marker="o",
        label="Cluster medians",
        cmap="gist_ncar",
    )
    plt.title(f"k-medians clustering, converged in {iters} iterations.")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter()-t0:.3f} seconds.")
