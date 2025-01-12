""" Betthauser, 2018: k-means clustering
"""

import time
import numpy as np
from typing import Any
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def k_means_clustering(
    data: NDArray[np.float64], k: int, plot: bool = False
) -> tuple[Any]:
    """Betthauser, 2018: k-means clustering

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

        if plot:
            plt.cla()
            plt.scatter(data[:, 0], data[:, 1], c=labels)
            plt.pause(0.001)

        if np.sum((k_means - k_last) ** 2) == 0:
            print(f"k-means clustering took {iters} iterations.")
            return labels, iters, k_means

        k_last = k_means


def _generate_data(
    num_clusters: int, dim: int, size_clusters: int
) -> NDArray[np.float64]:
    """generate synthetic data of gaussian clusters

    Args:
        num_clusters (int): true number of clusters to generate
        dim (int): n-dimensionality of data
        size_clusters (int): number of points in each cluster

    Returns:
        NDArray[np.float64]: synthetic data of n-dim gaussian clusters
    """
    means = np.random.uniform(0, 40, size=(num_clusters, dim))
    covs = np.zeros((num_clusters, dim, dim))

    for c in range(num_clusters):
        amat = np.random.randn(dim, dim)  # + np.eye(dim)
        amat = amat / amat.max()
        covs[c, :, :] = np.cov(amat)

    rand_factor = np.random.uniform(0, 5, size=num_clusters)

    data = [
        np.random.multivariate_normal(
            means[c, :], covs[c, :, :] * rand_factor[c], size=(size_clusters)
        )
        for c in range(num_clusters)
    ]

    data = np.vstack(data)
    rand_idx = np.random.permutation(data.shape[0])
    data = data[rand_idx, :]

    return data


def main() -> None:
    """_summary_"""
    k = 8
    num_actual_clusters = 8
    num_dims = 2
    num_pts_per_cluster = 500

    data = _generate_data(num_actual_clusters, num_dims, num_pts_per_cluster)
    labels, iters, k_means = k_means_clustering(data, k, plot=False)

    plt.scatter(data[:, 0], data[:, 1], c=labels, s=3)
    plt.scatter(
        k_means[:, 0], k_means[:, 1], c="r", s=20, marker="x", label="Cluster means"
    )
    plt.title(f"k-means clustering, converged in {iters} iterations.")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time()-t0:.3f} seconds.")
