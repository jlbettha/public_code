"""Betthauser, 2018: k-means clustering"""

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


def k_means_clustering(data: np.ndarray[float], k: int, plot: bool = False) -> tuple[Any]:
    """
    Betthauser, 2018: k-means clustering

    Args:
        data (NDArray[np.float64]): unlabeled N x D, N samples of D-dim data
        k (int): number of clusters
        plot (bool): whether to plot the clustering process

    Returns:
        NDArray[np.uint8]: array of new labels

    """
    rng = np.random.default_rng()
    k_means = data[rng.permutation(data.shape[0])[:k], :]  # init with k datapoints

    k_last = k_means
    iters = 0
    while True:
        iters = iters + 1

        labels = np.array(
            [np.argmin(cdist(np.reshape(data[idx, :], (1, -1)), k_means)) for idx in range(data.shape[0])]
        )

        k_means = np.squeeze([np.mean(data[labels == lbl, :], axis=0) for lbl in range(k)])

        if plot:
            plt.cla()
            plt.scatter(data[:, 0], data[:, 1], c=labels)
            plt.pause(0.001)

        if np.sum((k_means - k_last) ** 2) == 0:
            print(f"k-means clustering took {iters} iterations.")
            return labels, iters, k_means

        k_last = k_means


def _generate_data(num_clusters: int, dim: int, size_clusters: int) -> np.ndarray[float]:
    """
    Generate synthetic data of gaussian clusters

    Args:
        num_clusters (int): true number of clusters to generate
        dim (int): n-dimensionality of data
        size_clusters (int): number of points in each cluster

    Returns:
        np.ndarray[float]: synthetic data of n-dim gaussian clusters

    """
    rng = np.random.default_rng()
    means = rng.uniform(0, 40, size=(num_clusters, dim))
    covs = np.zeros((num_clusters, dim, dim))

    for c in range(num_clusters):
        amat = rng.standard_normal((dim, dim))
        amat = amat / amat.max()
        covs[c, :, :] = amat @ amat.T

    rand_factor = rng.uniform(0, 3, size=num_clusters)

    data = [
        rng.multivariate_normal(means[c, :], covs[c, :, :] * rand_factor[c], size=(size_clusters))
        for c in range(num_clusters)
    ]

    data = np.vstack(data)
    rand_idx = rng.permutation(data.shape[0])
    return data[rand_idx, :]


def main() -> None:
    """_summary_"""
    k = 7
    num_actual_clusters = 7
    num_dims = 2
    num_pts_per_cluster = 400

    data = _generate_data(num_actual_clusters, num_dims, num_pts_per_cluster)
    labels, iters, k_means = k_means_clustering(data, k, plot=False)

    plt.scatter(data[:, 0], data[:, 1], c=labels, s=3)
    plt.scatter(k_means[:, 0], k_means[:, 1], c="r", s=20, marker="x", label="Cluster means")
    plt.title(f"k-means clustering, converged in {iters} iterations.")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter() - tmain:.3f} seconds.")
