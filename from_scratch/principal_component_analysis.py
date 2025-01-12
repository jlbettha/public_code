# pylint: disable=C0103
""" Betthauser, 2020: 
PCA: principal component data projection
"""

import time
import numpy as np
from numpy.typing import NDArray
from singular_value_decomposition import singular_value_decomposition
import matplotlib.pyplot as plt


def pca_projection(
    A: NDArray[np.float64], use_my_svd: bool = False
) -> NDArray[np.float64]:
    """PCA principal component data projection of A

    Args:
        A (NDArray[np.float64]): m x n matrix of m samples of n-dim data
        use_my_svd (bool, optional): Use my manual SVD implement. Defaults to False.

    Returns:
        NDArray[np.float64]: principal component projection
    """

    A_c = A - np.mean(A, axis=0)

    if use_my_svd:
        U, sigmas, VT = singular_value_decomposition(A_c)
    else:
        U, sigmas, VT = np.linalg.svd(A_c, full_matrices=False)

    PCs = U @ np.diag(sigmas)

    return PCs


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

    means = np.random.uniform(0, 20, size=(num_clusters, dim))
    covs = np.zeros((num_clusters, dim, dim))

    for c in range(num_clusters):
        amat = np.random.randn(dim, dim)  # + np.eye(dim)
        amat = amat / amat.max()
        covs[c, :, :] = amat @ amat.T

    rand_factor = np.random.uniform(0, 5, size=num_clusters)

    data = [
        np.random.multivariate_normal(
            means[c, :], covs[c, :, :] * rand_factor[c], size=(size_clusters)
        )
        for c in range(num_clusters)
    ]

    data = np.vstack(data)
    # rand_idx = np.random.permutation(data.shape[0])
    # data = data[rand_idx, :]

    return data


def main() -> None:
    """PCA: principal component analysis"""

    ## init vars
    num_points = 600
    dim = 10
    num_classes = 3
    labels = np.vstack(
        [c * np.ones(num_points // num_classes) for c in range(num_classes)]
    )
    A = _generate_data(
        num_clusters=num_classes, dim=dim, size_clusters=num_points // num_classes
    )

    PCs = pca_projection(A)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(A[:, 0], A[:, 1], c=labels, s=3)
    plt.title(f"First 2 dimensions of {dim}-dim data")
    plt.subplot(1, 2, 2)
    plt.scatter(PCs[:, 0], PCs[:, 1], c=labels, s=3)
    plt.title(f"First 2 principal component projection")
    plt.show()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time()-t0:.3f} seconds.")
