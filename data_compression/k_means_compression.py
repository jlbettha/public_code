""" Betthauser, 2018: k-means compression
    >> Use when number of samples is large. 
    >> Use when compressed data do not need to be members of original data.
"""

import time
import numpy as np
from typing import Any
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def k_means_compression(
    data: NDArray[np.float64], k: int, plot: bool = False
) -> tuple[Any]:
    """Betthauser, 2018: k-means clustering/compression
        goal is to find k<<N data centers to represent all N

    Args:
        data (NDArray[np.float64]): unlabeled N x D, N samples of D-dim data
        k (int): number of clusters

    Returns:
        NDArray[np.uint8]: array of new labels
    """
    k_means = data[
        np.random.permutation(data.shape[0])[:k], :
    ]  # init with k datapoints

    k_last = k_means
    iters = 0
    while True:
        iters = iters + 1

        labels = np.array(
            [
                np.argmin(cdist(np.reshape(data[idx, :], (1, -1)), k_means))
                for idx in range(data.shape[0])
            ]
        )

        k_means = np.squeeze(
            [np.mean(data[labels == lbl, :], axis=0) for lbl in range(k)]
        )

        if plot and iters == 1:
            plt.cla()
            plt.scatter(data[:, 0], data[:, 1], c=labels, s=2, cmap="gist_ncar")
            plt.title(f"Iteration {iters}")
            plt.pause(0.001)

        if np.sum((k_means - k_last) ** 2) == 0 or iters > 150:
            plt.close()
            print(f"k-means compression took {iters} iterations.")
            return labels, iters, k_means

        k_last = k_means


def main() -> None:
    """_summary_"""
    num_pts = 5000
    k = num_pts // 25
    num_dims = 2

    # x = np.linspace(0, 12, num_pts)
    # y = 10 * (0.5 * x) + 2 * np.random.randn(num_pts)
    # data = np.c_[x, y]

    data = np.random.uniform(10, size=(num_pts, num_dims))
    labels, iters, k_means = k_means_compression(data, k, plot=False)

    kcolors = np.unique(labels)

    plt.cla()
    plt.scatter(data[:, 0], data[:, 1], zorder=-1, c=labels, s=2, cmap="gist_ncar")
    plt.scatter(
        k_means[:, 0],
        k_means[:, 1],
        c="k",
        s=15,
        zorder=1,
        marker="o",
        label="Cluster means",
        cmap="gist_ncar",
    )
    plt.title(f"k-means clustering, converged in {iters} iterations.")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time()-t0:.3f} seconds.")
