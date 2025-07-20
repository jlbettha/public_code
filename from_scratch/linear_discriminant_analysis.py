"""Betthauser, 2017: linear discriminant analysis"""

import time

import matplotlib.pyplot as plt
import numpy as np
from singular_value_decomposition import singular_value_decomposition
from sklearn.preprocessing import MinMaxScaler


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
        amat = rng.standard_normal((dim, dim))  # + np.eye(dim)
        amat = amat / amat.max()
        covs[c, :, :] = amat @ amat.T

    rand_factor = rng.uniform(0, 5, size=num_clusters)

    data = [
        rng.multivariate_normal(means[c, :], covs[c, :, :] * rand_factor[c], size=(size_clusters))
        for c in range(num_clusters)
    ]

    labels = [[c] * size_clusters for c in range(num_clusters)]

    data = np.vstack(data)
    labels = np.hstack(labels)
    # rand_idx = rng.permutation(data.shape[0])
    # data = data[rand_idx, :]

    return data, labels


def _get_class_means_covs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute class-wise means and covariances

    Args:
        x (np.ndarray): data, shape: n x m
        y (np.ndarray): labels, shape: n x 1


    Returns:
        np.ndarray: stats for k classes
        mean (np.ndarray): means of k classes, shape: k x m
        covs (np.ndarray): covariance matrices of k classes, shape: k x m x m

    """
    unique_cls = np.unique(y)
    means = np.vstack([np.mean(x[y == cl, :], axis=0) for cl in unique_cls])
    covs = np.stack([np.cov(x[y == cl, :].T) for cl in unique_cls])
    return means, covs


def _get_between_class_scatter(means: np.ndarray[float], total_mean: np.ndarray[float]) -> np.ndarray[float]:
    """
    _summary_

    Args:
        means (NDArray[np.float64]): class means k x m
        total_mean (NDArray[np.float64]): total mean m x 1

    Returns:
        NDArray[np.float64]: SB between class scatter matrix

    """
    num_c = means.shape[0]
    cls_mean_diffs = np.vstack([means[c, :] - total_mean for c in range(num_c)])
    sbs = np.stack([np.dot(cls_mean_diffs.T, cls_mean_diffs) for c in range(num_c)])
    return np.sum(sbs, axis=0) / num_c


def main() -> None:
    """_summary_"""
    num_classes = 8
    num_dims = 10
    num_pts_per_class = 300

    x, y = _generate_data(num_classes, num_dims, num_pts_per_class)
    print(f"{x.shape=} {y.shape=}")

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    means, covs = _get_class_means_covs(x, y)
    print(f"{means.shape=} {covs.shape=}")

    total_mean = np.mean(x, axis=0)

    sw = np.sum(covs, axis=0) / num_classes
    sb = _get_between_class_scatter(means, total_mean)
    print(f"{sw.shape=} {sb.shape=}")

    ## manual implementation of SVD solver
    u, sigmas, vt = singular_value_decomposition(np.linalg.pinv(sw).dot(sb))
    # eigvals = sigmas
    w = u @ np.diag(sigmas)  # @ VT.T
    xw = x @ w
    meansxw, _ = _get_class_means_covs(xw, y)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=3)
    plt.scatter(means[:, 0], means[:, 1], c="r", s=20, marker="x", label="Cluster means")
    plt.title("Raw data")

    plt.subplot(1, 3, 2)
    plt.scatter(xw[:, 0], xw[:, 1], c=y, s=3)
    plt.scatter(meansxw[:, 0], meansxw[:, 1], c="r", s=20, marker="x", label="Cluster means")
    plt.title("LDA")

    # compare to PCA
    u, sigmas, vt = np.linalg.svd(x, full_matrices=False)
    pcs = u @ np.diag(sigmas)
    meanspc, _ = _get_class_means_covs(pcs, y)
    plt.subplot(1, 3, 3)
    plt.scatter(pcs[:, 0], pcs[:, 1], c=y, s=3)
    plt.scatter(meanspc[:, 0], meanspc[:, 1], c="r", s=20, marker="x", label="Cluster means")
    plt.title("PCA")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter() - tmain:.3f} seconds.")
