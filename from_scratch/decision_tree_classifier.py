# pylint: disable=C0103
"""Betthauser, 2022 - Decision tree from scratch
Currently, "load_breast_cancer" dataset.
"""

import time
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_iris
# from sklearn import tree as sktree
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numba import njit


@njit
def entropy(hist1: np.ndarray[float]) -> float:
    """Betthauser 2016 -- Calculate joint entropy of an N-d distribution

    Args:
        hist1 (np.ndarray[float]): an N-D histogram or PMF

    Returns:
        np.float64: entropy of the ditribution
    """

    hist1 = hist1 / np.sum(hist1)
    nz_probs = np.array([-p * np.log(p) for p in hist1 if p > 1e-12])
    entrp = np.sum(nz_probs)
    return entrp


@njit
def information_gain(
    y: np.ndarray[float], x_1feature: np.ndarray[float], threshold: float
) -> float:
    """Information gain = entropy(parent) - weighted avg of entropy(children)

    Args:
        y (np.ndarray[float]): _description_
        x_1feature (np.ndarray[float]): _description_
        threshold (float): _description_

    Returns:
        float: Information gain
    """
    hist_parent = np.bincount(y)
    entropy_parent = entropy(hist_parent)

    left_children = y[np.where(x_1feature <= threshold)]
    right_children = y[np.where(x_1feature > threshold)]

    if len(left_children) == 0 or len(right_children) == 0:
        return 0

    left_hist = np.bincount(left_children)
    right_hist = np.bincount(right_children)
    entropy_left = entropy(left_hist)
    entropy_right = entropy(right_hist)
    entropy_children = (
        len(left_children) * entropy_left + len(right_children) * entropy_right
    ) / len(y)
    return entropy_parent - entropy_children


@njit
def gini_index(x: np.ndarray[float], w: np.ndarray[float] = None) -> float:
    """_summary_

    Args:
        x (np.ndarray[float]): _description_
        w (np.ndarray[float], optional): _description_. Defaults to None.

    Returns:
        float: _description_
    """
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        cumw = np.cumsum(sorted_w, dtype=np.float64)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=np.float64)
        return np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (
            cumxw[-1] * cumw[-1]
        )

    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=np.float64)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def accuracy(predictions, y_true) -> float:
    """_summary_"""
    return np.sum(predictions == y_true) / len(y_true)


class MyNode:
    """_summary_"""

    def __init__(
        self, num_features=None, thr=None, left_child=None, right_child=None, value=None
    ):
        self.num_features = num_features
        self.thr = thr
        self.left_child = left_child
        self.right_child = right_child
        self.value = value

    def is_leaf(self):
        """_summary_"""
        return self.value is not None


class MyDecisionTree:
    """MyDecisionTree"""

    def __init__(
        self, min_split: int = 2, max_depth: int = 15, prune_threshold: float = 0.0
    ):
        self.min_split = min_split
        self.max_depth = max_depth
        self.prune_threshold = prune_threshold
        self.root = None

    def train(self, X: np.ndarray[float], y: np.ndarray[float]):
        """build out decision tree based on train data

        Args:
            X (np.ndarray[float]): train data
            y (np.ndarray[float]): train labels
        """
        self.root = self._grow_next_branch(X, y, depth=0)

    def _grow_next_branch(
        self, X: np.ndarray[float], y: np.ndarray[float], depth: int = 0
    ):
        """_summary_

        Args:
            X (np.ndarray[float]): _description_
            y (np.ndarray[float]): _description_
            depth (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        if len(X.shape) < 2:
            num_datapoints, num_features = 1, len(X)
            num_labels = 1
        else:
            num_datapoints, num_features = X.shape
            num_labels = len(np.unique(y))

        # stopping criteria
        if (
            depth >= self.max_depth
            or num_labels == 1
            or num_datapoints < self.min_split
        ):
            return MyNode(value=int(stats.mode(y).mode))

        # feature_idx = np.random.permutation(num_features)
        feature_idx = np.arange(0, num_features, 1)

        max_info_gain, best_feat, best_thresh = self._best_split(X, y, feature_idx)
        if max_info_gain < self.prune_threshold:
            return MyNode(value=int(stats.mode(y).mode))
        else:
            left_idx = np.where(X[:, best_feat] <= best_thresh)
            right_idx = np.where(X[:, best_feat] > best_thresh)

            left = self._grow_next_branch(
                np.squeeze(X[left_idx, :]), y[left_idx], depth + 1
            )
            right = self._grow_next_branch(
                np.squeeze(X[right_idx, :]), y[right_idx], depth + 1
            )
            return MyNode(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        max_info_gain = -1e20
        split_idx, split_thresh = None, None
        for idx in feat_idxs:
            x_1col = X[:, idx]
            thresholds = np.unique(x_1col)
            for thr in thresholds:
                info_gain = information_gain(y, x_1col, thr)

                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    split_idx = idx
                    split_thresh = thr

        return max_info_gain, split_idx, split_thresh

    def _move_to_next_branch(self, x, node):
        """_summary_

        Args:
            x (_type_): single datapoint to predict
            node (_type_): this Node instance

        Returns:
            _type_: _description_
        """
        if node.is_leaf():
            return node.value

        # if node.info_gain < self.prune_threshold:

        if x[node.num_features] <= node.thr:
            return self._move_to_next_branch(x, node.left_child)
        return self._move_to_next_branch(x, node.right_child)

    def tree_predict(self, X_test: np.ndarray[float]) -> np.ndarray[int]:
        """predictor for MyDecisionTree class

        Args:
            xtest (np.ndarray[float]): unseen test data

        Returns:
            list[int]: prediction list
        """
        return np.array([self._move_to_next_branch(x, self.root) for x in X_test])


def main() -> None:
    """decision tree classifier"""
    max_depth = 10
    min_split = 2
    data = load_breast_cancer()
    # X, y = iris(return_X_y=True)
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(f"{X_train.shape=}, {y_train.shape=}, {X_test.shape=}, {y_test.shape=}")

    tree = MyDecisionTree(max_depth=max_depth, min_split=min_split, prune_threshold=0.0)

    tree.train(X_train, y_train)

    y_pred = tree.tree_predict(X_test)
    acc_test = accuracy(y_test, y_pred)

    trimmed_tree = MyDecisionTree(
        max_depth=max_depth, min_split=min_split, prune_threshold=0.1
    )
    trimmed_tree.train(X_train, y_train)
    y_pred_trim = trimmed_tree.tree_predict(X_test)
    acc_test_trim = accuracy(y_test, y_pred_trim)

    print(f"Testing accuracy, my model = {100*acc_test:.3f}%")
    print(f"Testing accuracy, my model pruned = {100*acc_test_trim:.3f}%")

    # compare with sklearn model and plot
    # sk_learn_tree = sktree.DecisionTreeClassifier(
    #     criterion="entropy",
    #     splitter="best",
    #     max_depth=max_depth,
    #     min_samples_split=min_split,
    # )
    # sk_learn_tree.fit(X_train, y_train)

    # y_sk_pred = sk_learn_tree.predict(X_test)
    # acc_sktest = accuracy(y_test, y_sk_pred)
    # print(f"Testing accuracy, sklearn model = {100*acc_sktest:.3f}%")

    # plt.figure(figsize=(8, 8))

    # sktree.plot_tree(
    #     sk_learn_tree,
    #     feature_names=data["feature_names"],
    #     class_names=data["target_names"],
    #     filled=True,
    # )
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter()-tmain:.3f} seconds.")
