# pylint: disable=C0103
""" Betthauser, 2022 - Hierarchical clustering from scratch
        Currently, "load_breast_cancer" dataset.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy.typing import NDArray
from modules.my_information_measures import jensen_shannon_dist


def hierarchical_clustering(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """hierarchical_clustering
    Args:
        X (NDArray[np.float64]): data

    Returns:
        _type_: data sorted by features
    """
    _, num_features = X.shape

    dist_matrix = np.zeros((num_features, num_features))

    ungrouped_features = [f for f in range(num_features)]
    clusters = []
    for i in ungrouped_features:
        for j in ungrouped_features:
            if j <= i:
                continue
            p = X[:, i] / np.sum(X[:, i])
            q = X[:, j] / np.sum(X[:, j])
            dist_matrix[i, j] = jensen_shannon_dist(p, q)
    # print(f"{dist_matrix[dist_matrix > 0].min()=}, {dist_matrix.max()=}")
    current_min = np.min(dist_matrix[dist_matrix > 0])

    min_idx = np.where(dist_matrix == current_min)
    for n in min_idx:
        val = n.item()
        dist_matrix[val, :] = np.zeros(dist_matrix.shape[0])
        dist_matrix[:, val] = np.zeros(dist_matrix.shape[0])

    clusters.append([n.item() for n in min_idx])

    for n in min_idx:
        ungrouped_features.remove(n.item())

    while len(ungrouped_features) > 1:
        # print(len(ungrouped_features), flush=True)
        centers = []
        c_dist_matrix = np.zeros((len(clusters), num_features))
        for i, c in enumerate(clusters):
            Xc = X[:, c]
            sum_feats = np.sum(Xc, axis=1)
            pmf = sum_feats / np.sum(sum_feats)
            centers.append(pmf)
            for j in ungrouped_features:
                p = X[:, j] / np.sum(X[:, j])
                c_dist_matrix[i, j] = jensen_shannon_dist(p, pmf)

        d_min = np.min(dist_matrix[dist_matrix > 0])
        c_min = np.min(c_dist_matrix[c_dist_matrix > 0])

        if c_min < d_min:
            clust, feat = [n.item() for n in np.where(c_dist_matrix == c_min)]
            clusters[clust].extend([feat])
            ungrouped_features.remove(feat)
            dist_matrix[feat, :] = np.zeros(dist_matrix.shape[0])
            dist_matrix[:, feat] = np.zeros(dist_matrix.shape[0])
        else:
            min_idx = np.where(dist_matrix == d_min)
            for n in min_idx:
                val = n.item()
                dist_matrix[val, :] = np.zeros(dist_matrix.shape[0])
                dist_matrix[:, val] = np.zeros(dist_matrix.shape[0])
            clusters.append([n.item() for n in min_idx])

            for n in min_idx:
                ungrouped_features.remove(n.item())

    print(clusters)

    new_order = []
    for c in clusters:
        new_order.extend(c)

    X_clustered = X[:, new_order]

    return X_clustered


def main() -> None:
    """hierarchical clustering"""

    # X, y = load_iris(return_X_y=True)
    X, y = load_breast_cancer(return_X_y=True)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)
    print(f"{X_train.shape=}, {y_train.shape=}, {X_test.shape=}, {y_test.shape=}")

    X_train_clustered = hierarchical_clustering(X_train)

    plt.figure(figsize=(6, 6))
    plt.imshow(X_train_clustered, aspect="auto")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time()-t0:.3f} seconds.")
