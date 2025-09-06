"""
Betthauser, 2018: k-means compression
>> Use when number of samples is large.
>> Use when compressed data do not need to be members of original data.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


def k_means_compression(
    data: NDArray[np.float64],
    k: int,
    *,  # Forces all following parameters to be keyword-only
    plot: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.int32], int, NDArray[np.float64]]:
    """
    Betthauser, 2018: k-means clustering/compression
        goal is to find k<<N data centers to represent all N

    Args:
        data (NDArray[np.float64]): unlabeled N x D, N samples of D-dim data
        k (int): number of clusters
        plot (bool): whether to visualize clustering (keyword-only)
        rng (np.random.Generator, optional): random number generator instance

    Returns:
        tuple: (labels, iterations, centroids)
            - labels: cluster assignments for each data point
            - iterations: number of iterations until convergence
            - centroids: final cluster centers

    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialize centroids with k random datapoints
    init_indices = rng.permutation(data.shape[0])[:k]
    k_means = data[init_indices, :].copy()

    k_last = k_means.copy()
    max_iters: int = 150
    max_plot_dim: int = 2
    tolerance: float = 1e-6
    iters = 0
    while iters < max_iters:
        iters += 1

        # Vectorized distance computation - much faster than list comprehension
        distances = cdist(data, k_means)
        labels = np.argmin(distances, axis=1)

        # Update centroids - handle empty clusters
        new_k_means = np.zeros_like(k_means)
        for lbl in range(k):
            cluster_points = data[labels == lbl]
            if len(cluster_points) > 0:
                new_k_means[lbl] = np.mean(cluster_points, axis=0)
            else:
                # Keep previous centroid if no points assigned
                new_k_means[lbl] = k_means[lbl]

        # Visualization for 2D data only
        if plot and data.shape[1] == max_plot_dim and iters == 1:
            plt.cla()
            plt.scatter(data[:, 0], data[:, 1], c=labels, s=2, cmap="gist_ncar")
            plt.scatter(k_means[:, 0], k_means[:, 1], c="red", marker="x", s=50)
            plt.title(f"Iteration {iters}")
            plt.pause(0.001)

        # Check convergence with tolerance
        centroid_shift = np.sum((new_k_means - k_last) ** 2)
        if centroid_shift < tolerance:
            if plot:
                plt.close()
            print(f"k-means compression converged in {iters} iterations.")
            return labels.astype(np.int32), iters, new_k_means

        k_last = k_means.copy()
        k_means = new_k_means

    # Max iterations reached
    if plot:
        plt.close()
    print(f"k-means compression reached max iterations ({max_iters}).")
    return labels.astype(np.int32), iters, k_means


def main() -> None:
    """_summary_"""
    num_pts = 5000
    k = num_pts // 50
    num_dims = 2

    # x = np.linspace(0, 12, num_pts)
    # rng = np.random.default_rng()
    # data = rng.uniform(0, 10, size=(num_pts, num_dims))
    # labels, iters, k_means = k_means_compression(data, k, plot=False, rng=rng)
    rng = np.random.default_rng()
    data = rng.uniform(0, 10, size=(num_pts, num_dims))
    labels, iters, k_means = k_means_compression(data, k, plot=False)

    # kcolors = np.unique(labels)

    plt.cla()
    plt.scatter(data[:, 0], data[:, 1], zorder=-1, c=labels, s=2, cmap="gist_ncar")
    plt.scatter(
        k_means[:, 0],
        k_means[:, 1],
        c="k",
        s=15,
        zorder=1,
        marker="x",
        label="Cluster means",
    )
    plt.title(f"k-means clustering, converged in {iters} iterations.")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter() - t0:.3f} seconds.")
