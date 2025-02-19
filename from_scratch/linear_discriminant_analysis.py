""" Betthauser, 2017: linear discriminant analysis
"""

import time
from typing import Any
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from singular_value_decomposition import singular_value_decomposition


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
        covs[c, :, :] = amat @ amat.T

    rand_factor = np.random.uniform(0, 5, size=num_clusters)

    data = [
        np.random.multivariate_normal(
            means[c, :], covs[c, :, :] * rand_factor[c], size=(size_clusters)
        )
        for c in range(num_clusters)
    ]

    labels = [[c] * size_clusters for c in range(num_clusters)]

    data = np.vstack(data)
    labels = np.hstack(labels)
    # rand_idx = np.random.permutation(data.shape[0])
    # data = data[rand_idx, :]

    return data, labels


def _get_class_means_covs(
    X: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Args:
        X (NDArray[np.float64]): data, shape: n x m
        y (NDArray[np.float64]): labels, shape: n x 1


    Returns:
        NDArray[np.float64]: stats for k classes
        mean (NDArray[np.float64]): means of k classes, shape: k x m
        covs (NDArray[np.float64]): covariance matrices of k classes, shape: k x m x m
    """
    unique_cls = np.unique(y)
    means = np.vstack([np.mean(X[y == cl, :], axis=0) for cl in unique_cls])
    covs = np.stack([np.cov(X[y == cl, :].T) for cl in unique_cls])
    return means, covs


def _get_between_class_scatter(
    means: NDArray[np.float64], total_mean: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Args:
        means (NDArray[np.float64]): class means k x m
        total_mean (NDArray[np.float64]): total mean m x 1

    Returns:
        NDArray[np.float64]: SB between class scatter matrix
    """
    num_c = means.shape[0]
    cls_mean_diffs = np.vstack([means[c, :] - total_mean for c in range(num_c)])
    sbs = np.stack([np.dot(cls_mean_diffs.T, cls_mean_diffs) for c in range(num_c)])
    sb_mat = np.sum(sbs, axis=0) / num_c
    return sb_mat


def main() -> None:
    """_summary_"""
    num_classes = 8
    num_dims = 10
    num_pts_per_class = 300

    X, y = _generate_data(num_classes, num_dims, num_pts_per_class)
    print(f"{X.shape=} {y.shape=}")

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    means, covs = _get_class_means_covs(X, y)
    print(f"{means.shape=} {covs.shape=}")

    total_mean = np.mean(X, axis=0)

    SW = np.sum(covs, axis=0) / num_classes
    SB = _get_between_class_scatter(means, total_mean)
    print(f"{SW.shape=} {SB.shape=}")

    ## manual implementation of SVD solver
    U, sigmas, VT = singular_value_decomposition(np.linalg.pinv(SW).dot(SB))
    # eigvals = sigmas
    W = U @ np.diag(sigmas)  # @ VT.T
    XW = X @ W
    meansXW, _ = _get_class_means_covs(XW, y)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=3)
    plt.scatter(
        means[:, 0], means[:, 1], c="r", s=20, marker="x", label="Cluster means"
    )
    plt.title("Raw data")

    plt.subplot(1, 3, 2)
    plt.scatter(XW[:, 0], XW[:, 1], c=y, s=3)
    plt.scatter(
        meansXW[:, 0], meansXW[:, 1], c="r", s=20, marker="x", label="Cluster means"
    )
    plt.title("LDA")

    # compare to PCA
    U, sigmas, VT = np.linalg.svd(X, full_matrices=False)
    PCs = U @ np.diag(sigmas)
    meansPC, _ = _get_class_means_covs(PCs, y)
    plt.subplot(1, 3, 3)
    plt.scatter(PCs[:, 0], PCs[:, 1], c=y, s=3)
    plt.scatter(
        meansPC[:, 0], meansPC[:, 1], c="r", s=20, marker="x", label="Cluster means"
    )
    plt.title("PCA")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tmain = time.time()
    main()
    print(f"Program took {time.time()-tmain:.3f} seconds.")
