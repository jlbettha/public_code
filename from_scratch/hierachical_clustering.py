# pylint: disable=C0103
"""
Betthauser, 2022 - Hierarchical clustering from scratch
Currently, "load_breast_cancer" dataset.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from modules.my_distance_metrics import (
    euclidean_dist,
)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def _flatten_lists(list_of_lists):
    if isinstance(list_of_lists, int):
        return list_of_lists
    try:
        flatter_list = list(np.ravel(list_of_lists))
    except ValueError:
        flatter_list = []
        for i in list_of_lists:
            flatter_list.extend(_flatten_lists(i))
    return flatter_list


def hierarchical_clustering(
    x: np.ndarray[float], num_groups: int = 5
) -> np.ndarray[float]:
    """
    hierarchical_clustering
    Args:
        x (np.ndarray[float]): data
        num_groups (int): number of final cluster groups/tree level

    Returns:
        _type_: data sorted by features

    """
    n, num_features = x.shape

    clusters = [[n] for n in range(num_features)]

    while len(clusters) > num_groups:
        c_dist_matrix = np.zeros((len(clusters), len(clusters)))
        for i, clus1 in enumerate(clusters):
            c1 = _flatten_lists(clus1)
            xc1 = np.reshape(x[:, c1], (n, -1))
            sum_feats1 = np.mean(xc1, axis=1)
            pmf1 = sum_feats1 / np.sum(sum_feats1)

            for j, clus2 in enumerate(clusters):
                if j <= i:
                    continue
                c2 = _flatten_lists(clus2)
                xc2 = np.reshape(x[:, c2], (n, -1))
                sum_feats2 = np.mean(xc2, axis=1)
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

    x_clustered = x[:, new_order]

    return x_clustered, clusters, new_order


def main() -> None:
    """Hierarchical clustering"""
    x, y = load_iris(return_X_y=True)
    # x, y = load_breast_cancer(return_X_y=True)
    # num_classes = len(np.unique(y))

    # scaler = MinMaxScaler()
    # x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01)

    rng = np.random.default_rng()
    tiny_noise = 1e-11 * rng.uniform(size=(x_train.shape[0], x_train.shape[1]))
    x_train = x_train + tiny_noise

    x_train_clustered, clusters, new_order = hierarchical_clustering(
        x_train.T, num_groups=1
    )

    clean_clusters = [_flatten_lists(c) for c in clusters]
    print(clean_clusters, flush=True)

    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(x_train_clustered, aspect="auto")
    plt.xlabel(f"{len(clean_clusters)} hierarchical cluster(s)")
    plt.subplot(2, 1, 2)
    plt.imshow(np.atleast_2d(y_train[new_order]), aspect="auto")
    plt.xlabel("Actual labels")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter()-tmain:.3f} seconds.")
