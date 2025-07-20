"""
Betthauser, 2018: k-medians compression.
>> Use when number of samples is large.
>> Use when compressed data must be subset/members of original data.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


def k_medians_compression(
    data: NDArray[np.float64],
    k: int,
    *,
    plot: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.int32], int, NDArray[np.float64]]:
    """
    Betthauser, 2018: k-medians clustering/compression
        goal is to find k<<N data points to represent all N.
        Use when compressed data must be members of original data.

    Args:
        data (NDArray[np.float64]): unlabeled N x D, N samples of D-dim data
        k (int): number of clusters
        plot (bool): whether to visualize clustering (keyword-only)
        rng (np.random.Generator, optional): random number generator instance

    Returns:
        tuple: (labels, iterations, medians)
            - labels: cluster assignments for each data point
            - iterations: number of iterations until convergence
            - medians: final cluster medians (actual data points)

    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialize medians with k random datapoints
    init_indices = rng.permutation(data.shape[0])[:k]
    k_medians = data[init_indices, :].copy()

    k_last = k_medians.copy()
    iters = 0
    max_iters: int = 150
    max_plot_dim: int = 2
    tolerance: float = 1e-6

    while iters < max_iters:
        iters += 1

        # Compute distances and assign labels
        distances = cdist(data, k_medians)
        labels = np.argmin(distances, axis=1)

        # Update medians - find actual data points closest to cluster centers
        new_k_medians = np.zeros_like(k_medians)
        for lbl in range(k):
            cluster_points = data[labels == lbl]
            if len(cluster_points) > 0:
                # Find the median point (actual data point closest to centroid)
                cluster_center = np.mean(cluster_points, axis=0)
                distances_to_center = cdist(cluster_points, cluster_center.reshape(1, -1)).flatten()
                median_idx = np.argmin(distances_to_center)
                new_k_medians[lbl] = cluster_points[median_idx]
            else:
                # Keep previous median if no points assigned
                new_k_medians[lbl] = k_medians[lbl]

        # Visualization for 2D data
        if plot and data.shape[1] == max_plot_dim:
            plt.cla()
            plt.scatter(data[:, 0], data[:, 1], c=labels, s=2, cmap="gist_ncar")
            plt.scatter(k_medians[:, 0], k_medians[:, 1], c="red", marker="x", s=50)
            plt.title(f"K-medians Iteration {iters}")
            plt.pause(0.1)

        # Check convergence - medians should be stable
        median_shift = np.sum((new_k_medians - k_last) ** 2)
        if median_shift < tolerance:
            if plot:
                plt.close()
            print(f"k-medians compression converged in {iters} iterations.")
            return labels.astype(np.int32), iters, new_k_medians

        k_last = k_medians.copy()
        k_medians = new_k_medians

    # Max iterations reached
    if plot:
        plt.close()
    print(f"k-medians compression reached max iterations ({max_iters}).")
    return labels.astype(np.int32), iters, k_medians


def main() -> None:
    """_summary_"""
    num_pts = 5000
    k = num_pts // 25
    num_dims = 2

    # x = np.linspace(0, 12, num_pts)
    # y = 10 * (0.5 * x) + 2 * np.random.randn(num_pts)
    # data = np.c_[x, y]

    rng = np.random.default_rng()
    data = rng.uniform(0, 10, size=(num_pts, num_dims))
    labels, iters, k_medians = k_medians_compression(data, k, plot=False, rng=rng)

    # kcolors = np.unique(labels)

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
    )
    plt.title(f"k-medians clustering, converged in {iters} iterations.")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter() - t0:.3f} seconds.")
