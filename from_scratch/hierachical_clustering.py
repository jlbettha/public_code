# pylint: disable=C0103
""" Betthauser, 2022 - Hierarchical clustering from scratch
        Currently, "load_breast_cancer" dataset.
"""

import time
import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy.typing import NDArray
from modules.my_distance_metrics import (
    jensen_shannon_dist,
    euclidean_dist,
    minkowski_dist,
    cosine_similarity,
)


def _flatten_lists(list_of_lists):
    if isinstance(list_of_lists, int):
        return list_of_lists
    try:
        flatter_list = list(np.ravel(list_of_lists))
    except:
        flatter_list = []
        for i in list_of_lists:
            flatter_list.extend(_flatten_lists(i))
    return flatter_list


def hierarchical_clustering(
    X: NDArray[np.float64], num_groups: int = 5
) -> NDArray[np.float64]:
    """hierarchical_clustering
    Args:
        X (NDArray[np.float64]): data
        num_groups (int): number of final cluster groups/tree level

    Returns:
        _type_: data sorted by features
    """
    N, num_features = X.shape

    clusters = [[n] for n in range(num_features)]

    while len(clusters) > num_groups:
        c_dist_matrix = np.zeros((len(clusters), len(clusters)))
        for i, clus1 in enumerate(clusters):
            c1 = _flatten_lists(clus1)
            Xc1 = np.reshape(X[:, c1], (N, -1))
            sum_feats1 = np.mean(Xc1, axis=1)
            pmf1 = sum_feats1 / np.sum(sum_feats1)

            for j, clus2 in enumerate(clusters):
                if j <= i:
                    continue
                c2 = _flatten_lists(clus2)
                Xc2 = np.reshape(X[:, c2], (N, -1))
                sum_feats2 = np.mean(Xc2, axis=1)
                pmf2 = sum_feats2 / np.sum(sum_feats2)
                c_dist_matrix[i, j] = euclidean_dist(pmf1, pmf2)

        c_min = np.min(c_dist_matrix[c_dist_matrix > 0])

        imax, jmax = [n.item() for n in np.where(c_dist_matrix == c_min)]
        clusters[imax] = [clusters[imax], clusters[jmax]]
        del clusters[jmax]

    new_order = []
    for clus in clusters:
        c = _flatten_lists(clus)
        # print(c)
        new_order.extend(c)

    X_clustered = X[:, new_order]

    return X_clustered, clusters, new_order


def main() -> None:
    """hierarchical clustering"""

    X, y = load_iris(return_X_y=True)
    # X, y = load_breast_cancer(return_X_y=True)
    num_classes = len(np.unique(y))

    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)
    print(f"{X_train.shape=}, {y_train.shape=}, {X_test.shape=}, {y_test.shape=}")

    tiny_noise = 1e-11 * np.random.uniform(size=(X_train.shape[0], X_train.shape[1]))
    X_train = X_train + tiny_noise

    X_train_clustered, clusters, new_order = hierarchical_clustering(
        X_train.T, num_groups=1
    )

    clean_clusters = [_flatten_lists(c) for c in clusters]
    print(clean_clusters, flush=True)

    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(X_train_clustered, aspect="auto")
    plt.xlabel(f"{len(clean_clusters)} hierarchical cluster(s)")
    plt.subplot(2, 1, 2)
    plt.imshow(np.atleast_2d(y_train[new_order]), aspect="auto")
    plt.xlabel("Actual labels")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tmain = time.time()
    main()
    print(f"Program took {time.time()-tmain:.3f} seconds.")
