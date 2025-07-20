"""_summary_"""

import numpy as np
from my_distance_metrics import minmax_scaling
from numba import njit

EPS = 1e-12


@njit
def entropy(hist1: np.ndarray[float]) -> float:
    """
    Betthauser 2016 -- Calculate entropy of an 1-d distribution

    Args:
        hist1 (np.ndarray[float]): an N-D histogram or PMF

    Returns:
        float: entropy of the ditribution

    """
    hist1 = hist1 / np.sum(hist1)
    nz_probs = np.array([-p * np.log(p) for p in hist1 if p > EPS])
    return np.sum(nz_probs)


@njit
def information_gain(y: np.ndarray[float], x_1feature: np.ndarray[float], threshold: float) -> float:
    """
    Information gain = entropy(parent) - weighted avg of entropy(children)

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
    entropy_children = (len(left_children) * entropy_left + len(right_children) * entropy_right) / len(y)
    return entropy_parent - entropy_children


@njit
def gini_index(x: np.ndarray[float], w: np.ndarray[float] = None) -> float:
    """
    Betthauser - 2024 - Get gini index, a measure of inequality (e.g., 0.0 means equal ditribution)

    Args:
        x (np.ndarray[float]): _description_
        w (np.ndarray[float]], optional): sample weights. Defaults to None.

    Returns:
        float: gini index/coefficient

    """
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        cumw = np.cumsum(sorted_w, dtype=np.float64)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=np.float64)
        return np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (cumxw[-1] * cumw[-1])

    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=np.float64)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


# returns joint histogram of 2 image sections
@njit
def joint_histogram_2d(patch1: np.ndarray[float], patch2: np.ndarray[float], bins: float = 255.0) -> np.ndarray[float]:
    """
    Betthauser - 2018 - Computes joint histogram of 2 image sections/patches
    Args:
        img1 (np.ndarray[float]): image patch 1
        img2 (np.ndarray[float]): image patch 2
        bins (float): number of bins
    Returns:
        np.ndarray[float]: joint_histogram
    """
    patch1 = minmax_scaling(patch1, max_val=bins).astype(np.uint8)
    patch2 = minmax_scaling(patch2, max_val=bins).astype(np.uint8)

    joint_histogram = np.zeros(bins, bins)
    for i in range(patch1.shape[0]):
        for j in range(patch1.shape[1]):
            joint_histogram[patch2[i, j], patch1[i, j]] += 1
    return joint_histogram


@njit
def mutual_info(image1: np.ndarray[float], image2: np.ndarray[float]) -> float:
    """
    Betthauser - 2018 - compute mutual information between 2 images/patches
    Args:
        image1 (np.ndarray[float]): image/patch
        image2 (np.ndarray[float]): another image/patch for comparison

    Returns:
        float: mutual information between the two images/patches

    """
    joint_hist = joint_histogram_2d(image1, image2)
    joint_entropy = entropy(joint_hist)
    hist1 = np.sum(joint_hist, axis=0)
    hist2 = np.sum(joint_hist, axis=1)
    entropy1 = entropy(hist1)
    entropy2 = entropy(hist2)
    return entropy1 + entropy2 - joint_entropy


@njit
def kl_divergence(p: np.ndarray[float], q: np.ndarray[float]) -> float:
    """
    Betthauser - 2018 - compute KL divergence between two PMFs

    Args:
        p (np.ndarray[float]): PMF of distribution p
        q (np.ndarray[float]): PMF of distribution q

    Returns:
        float:  KL divergence KL(p||q)

    """
    epsilon = 1e-12
    p = np.abs(p) + epsilon
    q = np.abs(q) + epsilon
    return np.sum(p * (np.log(p) - np.log(q)))


@njit
def kl_div_bidirectional(p: np.ndarray[float], q: np.ndarray[float]) -> float:
    """
    Betthauser - 2018 - compute Jeffreys/2-way KL divergence between two PMFs

    Args:
        p (np.ndarray[float]): PMF of distribution p
        q (np.ndarray[float]): PMF of distribution q

    Returns:
        float: Jeffreys bi-directional KL divergence KL(p||q) + KL(q||p)

    """
    epsilon = 1e-12
    p = np.abs(p) + epsilon
    q = np.abs(q) + epsilon
    return np.sum(p * (np.log(p) - np.log(q)) + q * (np.log(q) - np.log(p)))


@njit
def jensen_shannon_divergence(p: np.ndarray[float], q: np.ndarray[float]) -> float:
    """
    Betthauser - 2024 - jensen-shannon divergence

    Args:
        p (np.ndarray[float]): PMF of distribution p
        q (np.ndarray[float]): PMF of distribution q

    Returns:
        float: JS_divergence(P || Q) = 0.5[ D_kl(P || M ) + D_kl( Q || M ) ]
                    where M = 0.5(P+Q)

    """
    epsilon = 1e-12
    p = np.abs(p) + epsilon
    q = np.abs(q) + epsilon
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


@njit
def jensen_shannon_dist(p: np.ndarray[float], q: np.ndarray[float]) -> float:
    """
    Betthauser - 2024 - jensen-shannon distance metric

    Args:
        p (np.ndarray[float]): PMF of distribution p
        q (np.ndarray[float]): PMF of distribution q

    Returns:
        float: JS_distance = np.sqrt( JS_divergence )

    """
    epsilon = 1e-12
    p = np.abs(p) + epsilon
    q = np.abs(q) + epsilon
    m = 0.5 * (p + q)
    js_divergence = 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))
    return np.sqrt(js_divergence)


def main() -> None:
    print("my_information_measures.py is a module")


if __name__ == "__main__":
    main()
