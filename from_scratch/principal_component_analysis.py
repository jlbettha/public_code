# pylint: disable=C0103
"""
Betthauser, 2020:
PCA: principal component data projection
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from singular_value_decomposition import singular_value_decomposition


def pca_projection(a: np.ndarray, use_my_svd: bool = False) -> np.ndarray:
    """
    PCA principal component data projection of A

    Args:
        a (np.ndarray[float]): m x n matrix of m samples of n-dim data
        use_my_svd (bool, optional): Use my manual SVD implement. Defaults to False.

    Returns:
        np.ndarray[float]: principal component projection

    """
    a_c = a - np.mean(a, axis=0)

    if use_my_svd:
        u, sigmas, vt = singular_value_decomposition(a_c)
    else:
        u, sigmas, vt = np.linalg.svd(a_c, full_matrices=False)

    return u @ np.diag(sigmas)



def _generate_data(
    num_clusters: int, dim: int, size_clusters: int
) -> np.ndarray[float]:
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
    means = rng.uniform(0, 20, size=(num_clusters, dim))
    covs = np.zeros((num_clusters, dim, dim))

    for c in range(num_clusters):
        amat = rng.standard_normal((dim, dim))  # + np.eye(dim)
        amat = amat / amat.max()
        covs[c, :, :] = amat @ amat.T

    rand_factor = rng.uniform(0, 5, size=num_clusters)

    data = [
        rng.multivariate_normal(
            means[c, :], covs[c, :, :] * rand_factor[c], size=(size_clusters)
        )
        for c in range(num_clusters)
    ]

    return np.vstack(data)
    # rand_idx = rng.permutation(data.shape[0])
    # data = data[rand_idx, :]



def main() -> None:
    """PCA: principal component analysis"""
    ## init vars
    num_points = 600
    dim = 10
    num_classes = 3
    labels = np.vstack(
        [c * np.ones(num_points // num_classes) for c in range(num_classes)]
    )
    a = _generate_data(
        num_clusters=num_classes, dim=dim, size_clusters=num_points // num_classes
    )

    pcs = pca_projection(a)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(a[:, 0], a[:, 1], c=labels, s=3)
    plt.title(f"First 2 dimensions of {dim}-dim data")
    plt.subplot(1, 2, 2)
    plt.scatter(pcs[:, 0], pcs[:, 1], c=labels, s=3)
    plt.title("First 2 principal component projection")
    plt.show()


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter()-tmain:.3f} seconds.")
