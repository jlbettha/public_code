""" module summary """

import time
import numpy as np
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance, wasserstein_distance_nd
import matplotlib.pyplot as plt


####  Point-to-point distance functions  ##########
def euclidean_dist(a: NDArray[np.float64], b: NDArray[np.float64]) -> np.float64:
    """Betthauser - 2018 - compute euclidean distance between 2 points

    Args:
        a (NDArray[np.float64]): N-dimensional point A
        b (NDArray[np.float64]): N-dimensional point B

    Returns:
        np.float64: euclidean distance between points A and B
    """
    c = np.sqrt(np.sum((b - a) ** 2))
    return c


def cosine_similarity(a: NDArray[np.float64], b: NDArray[np.float64]) -> np.float64:
    """Betthauser - 2018 - compute cosine similarity between 2 vectors

    Args:
        a (NDArray[np.float64]): N-dimensional vector/point A
        b (NDArray[np.float64]): N-dimensional vector/point B

    Returns:
        np.float64: euclidean distance between points A and B
    """
    magnitude_a = np.sqrt(np.sum(a**2))
    magnitude_b = np.sqrt(np.sum(b**2))
    csim = np.dot(a, b) / (magnitude_a * magnitude_b)
    return csim


def manhattan_dist(a: NDArray[np.float64], b: NDArray[np.float64]) -> np.float64:
    """Betthauser - 2018 - compute manhattan distance between 2 points

    Args:
        a (NDArray[np.float64]): N-dimensional point A
        b (NDArray[np.float64]): N-dimensional point B

    Returns:
        np.float64: manhattan distance between points A and B
    """
    mand = np.sum(np.abs(b - a))
    return mand


def minkowski_dist(
    a: NDArray[np.float64], b: NDArray[np.float64], p: float
) -> np.float64:
    """Betthauser - 2018 - compute p-th root minkowski distance between 2 vectors

    Args:
        a (NDArray[np.float64]): N-dimensional point A
        b (NDArray[np.float64]): N-dimensional point A=B
        p (float): pth root

    Returns:
        np.float64: p-th root minkowski distance between points A and B
    """
    mink = np.sum(np.abs(b - a) ** p) ** (1 / p)
    return mink


####  Point-to-distribution distance functions  ##########
def mahalinobis_dist(
    y: NDArray[np.float64], x: NDArray[np.float64], sqrt_calc: bool = True
) -> np.float64:
    """Betthauser - 2018 - compute  mahalinobis distance from point to point cluster

    Args:
        y (NDArray[np.float64]): N-dimensional point
        x (NDArray[np.float64]): N-dimensional target cluster of points

    Returns:
        float: mahalinobis distance of point y to distribution x (exponent of multivariate gaussian)
    """
    mu = np.mean(x, axis=0)
    dist = ((y - mu) @ np.linalg.pinv(np.cov(x.T))) @ (y - mu).T
    if not sqrt_calc:
        return dist
    return np.sqrt(dist)


def z_score(x: float, mu: float, sigma: float) -> float:
    """Betthauser - 2017 - compute z_score of a data point wrt a distribution
    Args:
        x (float): data point
        mu (float): mean
        sigma (float): stardard deviation

    Returns:
        float: z_score of x
    """
    return (x - mu) / sigma


####  Distribution-to-distribution distance functions  ##########
def pearson_correlation(p: NDArray[np.float64], q: NDArray[np.float64]) -> np.float64:
    """Betthauser - 2018 - pearson correlation between distributions

    Args:
        p (NDArray[np.float64]): array p
        q (NDArray[np.float64]): array q

    Returns:
        np.float64: pearson correlation between normalized p and q
    """
    return np.corrcoef(p / np.sum(p), q / np.sum(q))[0, 1]


def jensen_shannon_divergence(
    p: NDArray[np.float64], q: NDArray[np.float64]
) -> np.float64:
    """Betthauser - 2024 - jensen-shannon divergence

    Args:
        p (NDArray[np.float64]): PMF of distribution p
        q (NDArray[np.float64]): PMF of distribution q

    Returns:
        np.float64: JS_divergence(P || Q) = 0.5[ D_kl(P || M ) + D_kl( Q || M ) ]
                    where M = 0.5(P+Q)
    """
    epsilon = 1e-12
    p = np.abs(p) + epsilon
    q = np.abs(q) + epsilon
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def jensen_shannon_distance(
    p: NDArray[np.float64], q: NDArray[np.float64]
) -> np.float64:
    """Betthauser - 2024 - jensen-shannon distance metric

    Args:
        p (NDArray[np.float64]): PMF of distribution p
        q (NDArray[np.float64]): PMF of distribution q

    Returns:
        np.float64: JS_distance = np.sqrt( JS_divergence )
    """
    epsilon = 1e-12
    p = np.abs(p) + epsilon
    q = np.abs(q) + epsilon
    m = 0.5 * (p + q)
    js_divergence = 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))
    return np.sqrt(js_divergence)


def wasserstein_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> np.float64:
    """Wasserstein distance or Kantorovich–Rubinstein metric

        # From wikipedia.org: Intuitively, if each distribution is viewed as a unit amount of earth (soil) piled on
        # M, the metric is the minimum "cost" of turning one pile into the other, which is assumed to be the amount
        # of earth that needs to be moved times the mean distance it has to be moved.

    Args:
        p (NDArray[np.float64]): PMF of distribution p
        q (NDArray[np.float64]): PMF of distribution q

    Returns:
        np.float64: W_p(P,Q) = (1/N) * SUM_i [ ||X_i - Y_i|| ^^ p ] ^^ (1/p)
    """
    epsilon = 1e-12
    p = np.abs(p) + epsilon
    q = np.abs(q) + epsilon
    if len(p.shape) == 2:
        return wasserstein_distance_nd(p, q)
    elif len(p.shape) == 1:
        return wasserstein_distance(p, q)
    # TODO: manual
    return ValueError


def wasserstein_dist_gaussian1d(
    mu1: NDArray[np.float64],
    C1: NDArray[np.float64],
    mu2: NDArray[np.float64],
    C2: NDArray[np.float64],
) -> np.float64:
    # Wasserstein distance of 2 gaussians ~N(mu1, C1) and ~N(mu2, C2)
    # From wikipedia.org: Intuitively, if each distribution is viewed as a unit amount of earth (soil) piled on
    # M, the metric is the minimum "cost" of turning one pile into the other, which is assumed to be the amount
    # of earth that needs to be moved times the mean distance it has to be moved.

    # W(mu1, mu2) = sqrt( ||mu1-mu2||_2^2 + trace(C1 + C2-2*(C2^0.5) @ C1 @ C2^0.5) ^ 0.5 )

    # TODO
    return NotImplementedError


def kl_divergence(p: NDArray[np.float64], q: NDArray[np.float64]) -> np.float64:
    """Betthauser - 2018 - compute KL divergence between two PMFs

    Args:
        p (NDArray[np.float64]): PMF of distribution p
        q (NDArray[np.float64]): PMF of distribution q

    Returns:
        np.float64:  KL divergence KL(p||q)
    """
    epsilon = 1e-12
    p = np.abs(p) + epsilon
    q = np.abs(q) + epsilon
    return np.sum(p * (np.log(p) - np.log(q)))


def kl_div_bidirectional(p: NDArray[np.float64], q: NDArray[np.float64]) -> np.float64:
    """Betthauser - 2018 - compute Jeffreys/2-way KL divergence between two PMFs

    Args:
        p (NDArray[np.float64]): PMF of distribution p
        q (NDArray[np.float64]): PMF of distribution q

    Returns:
        np.float64: Jeffreys bi-directional KL divergence KL(p||q) + KL(q||p)
    """
    epsilon = 1e-12
    p = np.abs(p) + epsilon
    q = np.abs(q) + epsilon
    jeffreys = np.sum(p * (np.log(p) - np.log(q)) + q * (np.log(q) - np.log(p)))
    return jeffreys


def kl_div_gaussian1d(mu1: float, v1: float, mu2: float, v2: float) -> float:
    """Betthauser - 2018 - compute KL divergence between two normal distributions

    Args:
        mu1 (float): _description_
        v1 (float): _description_
        mu2 (float): _description_
        v2 (float): _description_

    Returns:
        float: _description_
    """
    return np.log(v2 / v1) + ((v1 + (mu1 - mu2) ** 2) / (2 * v2)) - 0.5


def kl_div_gaussian1d_bidirectional(
    mu1: float, v1: float, mu2: float, v2: float
) -> float:
    """Betthauser - 2018 - compute KL divergence between two normal distributions

    Args:
        mu1 (float): _description_
        v1 (float): _description_
        mu2 (float): _description_
        v2 (float): _description_

    Returns:
        float: _description_
    """
    jeffreys = kl_div_gaussian1d(mu1=mu1, v1=v1, mu2=mu2, v2=v2) + kl_div_gaussian1d(
        mu1=mu2, v1=v2, mu2=mu1, v2=v1
    )
    return jeffreys


def bhattacharyya_dist(mu1: float, v1: float, mu2: float, v2: float) -> float:
    """Betthauser - 2021 - compute Bhattacharyya distance between two normal distributions
    Args:
        mu1 (float): mean of distribution 1
        v1 (float): variance of distribution 1
        mu2 (float): mean of distribution 2
        v2 (float): variance of distribution 2

    Returns:
        float: Bhattacharyya distance
    """
    part1 = 0.25 * np.log(0.25 * (v1 / v2 + v2 / v1 + 2))
    part2 = 0.25 * (((mu1 - mu2) ** 2) / (v1 + v2))
    return part1 + part2


def fisher_dist(mu1: float, v1: float, mu2: float, v2: float) -> float:
    """Betthauser - 2019 - compute Fisher distance between two normal distributions
        Note: Be aware that fisher_dist always = 0 if means are same,
        whereas bhat_dist only = 0 when means and vaiances are same.

    Args:
        mu1 (float): mean of distribution 1
        v1 (float): variance of distribution 1
        mu2 (float): mean of distribution 2
        v2 (float): variance of distribution 2

    Returns:
        float: Fisher distance

    """
    fish = ((mu1 - mu2) ** 2) / (v1 + v2)
    return fish


def entropy(hist1: NDArray[np.float64]) -> np.float64:
    """Betthauser 2016 -- Calculate joint entropy of an N-d distribution

    Args:
        hist1 (NDArray[np.float64]): an N-D histogram or PMF

    Returns:
        np.float64: entropy of the ditribution
    """
    hist1 = hist1 / np.sum(hist1)
    nz_probs = [-p * np.log(p) for p in hist1 if p > 1e-12]
    entrp = np.sum(nz_probs)
    return entrp


def minmax_scaling(
    data: NDArray[np.float64], max_val: float = 255
) -> NDArray[np.float64]:
    """Betthauser - 2018 - min-max scaling of data

    Args:
        data (NDArray[np.float64]): N-D data
        max_val (float, optional): max desired output value. Defaults to 255.

    Returns:
        NDArray[np.float64]: data min-max scaled in range [0, max_val]
    """
    return max_val * (data - data.min) / (data.max - data.min)


# returns joint histogram of 2 image sections
def joint_histogram_2d(
    patch1: NDArray[np.float64], patch2: NDArray[np.float64], bins: float = 255.0
) -> NDArray[np.float64]:
    """Computes joint histogram of 2 image sections/patches
    Args:
        img1 (NDArray[np.float64]): image patch 1
        img2 (NDArray[np.float64]): image patch 2
        bins (float): number of bins
    Returns:
        NDArray[np.float64]: joint_histogram
    """
    patch1 = minmax_scaling(patch1, max_val=bins).astype(np.uint8)
    patch2 = minmax_scaling(patch2, max_val=bins).astype(np.uint8)

    joint_histogram = np.zeros(bins, bins)
    for i in range(patch1.shape[0]):
        for j in range(patch1.shape[1]):
            joint_histogram[patch2[i, j], patch1[i, j]] += 1
    return joint_histogram


def mutual_info(image1: NDArray[np.float64], image2: NDArray[np.float64]) -> float:
    """Betthauser - 2018 - compute mutual information between 2 images/patches
    Args:
        image1 (NDArray[np.float64]): image/patch
        image2 (NDArray[np.float64]): another image/patch for comparison

    Returns:
        float: mutual information between the two images/patches
    """
    joint_hist = joint_histogram_2d(image1, image2)
    joint_entropy = entropy(joint_hist)
    hist1 = np.sum(joint_hist, axis=0)
    hist2 = np.sum(joint_hist, axis=1)
    entropy1 = entropy(hist1)
    entropy2 = entropy(hist2)
    mut_info = entropy1 + entropy2 - joint_entropy
    return mut_info


# TODO: Other Dist to Dist options for later
# ===========================================
# >> hellenger pmfs
# >> hellinger 2 normals
# >> renyi div pmfs - kl and shannon are renyi, a=1: special because it is only at a=1
#                     that the chain rule of conditional probability holds exactly.
# >> energy distance
# >> total variation, or statistical dist


def main() -> None:
    """_summary_"""
    point1 = np.array([3.25, 9.1, -2.7])
    point2 = np.array([-5.1, 0.95, 1.42])
    cluster1 = [5, 9, 3] + 1 * np.random.randn(1500, 3)
    cluster2 = [1, 1, 0] + 3 * np.random.randn(1500, 3)
    c1_1d = cluster1[:, 0]
    c2_1d = cluster2[:, 0]
    mu_a = np.mean(c1_1d)
    var_a = np.var(c1_1d)
    mu_b = np.mean(c2_1d)
    var_b = np.var(c2_1d)

    dists, _, _ = plt.hist([c1_1d, c2_1d], bins=62)
    # plt.show()

    distr_a = dists[0] / np.sum(dists[0])
    distr_b = dists[1] / np.sum(dists[1])

    print("\n  Point to point distances:")
    print(f"{cosine_similarity(point1, point2) = :.7f}")
    print(f"{manhattan_dist(point1, point2) = :.7f}")
    print(f"{euclidean_dist(point1, point2) = :.7f}")
    print(f"{minkowski_dist(point1, point2, 7) = :.7f}\n")

    print("\n  Point to cluster distances:")
    print(f"{mahalinobis_dist(point1, cluster1, sqrt_calc=False) = :.7f}")
    print(f"{mahalinobis_dist(point1, cluster2, sqrt_calc=True) = :.7f}\n")

    print("\n  Cluster to cluster distances:")
    print(f"{fisher_dist(mu_a, var_a, mu_b, var_b) = :.7f}")
    print(f"{bhattacharyya_dist(mu_a, var_a, mu_b, var_b) = :.3f}")
    print(f"{kl_divergence(distr_a, distr_b) = :.3f}")
    print(f"{kl_div_gaussian1d(mu_a, var_a, mu_b, var_b) = :.3f}\n")

    # plt.figure()
    # plt.scatter(cluster1[:, 0], cluster1[:, 1])
    # plt.scatter(cluster2[:, 0], cluster2[:, 1])
    # plt.show()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time() - t0:.3f} seconds")
