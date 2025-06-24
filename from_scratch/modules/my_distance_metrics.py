"""Betthauser - 2024 - Metrics for computing distances between points|points, points|distributions,
and distributions|distributions."""

import time
import numpy as np
from numba import njit


@njit
def sum_squared_error(x: np.ndarray[float], y: np.ndarray[float]) -> float:
    """Compute the sum of squared errors between two vectors x and y."""
    return np.sum((x - y) ** 2)


@njit
def compute_R2(x: np.ndarray[float], y: np.ndarray[float]) -> float:
    """Compute R2 between two vectors x and y."""
    ss_residual = sum_squared_error(x, y)
    ss_total = sum_squared_error(x, np.mean(x))
    return 1 - (ss_residual / ss_total)


####  Point-to-point distance functions  ##########
@njit
def euclidean_dist(a: np.ndarray[float], b: np.ndarray[float]) -> float:
    """Betthauser - 2018 - compute euclidean distance between 2 points

    Args:
        a (np.ndarray[float]): N-dimensional point A
        b (np.ndarray[float]): N-dimensional point B

    Returns:
        float: euclidean distance between points A and B
    """
    diff = b - a
    c = np.sqrt(np.sum(diff * diff))
    return c


@njit
def cosine_similarity(a: np.ndarray[float], b: np.ndarray[float]) -> float:
    """Betthauser - 2018 - compute cosine similarity between 2 vectors

    Args:
        a (np.ndarray[float]): N-dimensional vector/point A
        b (np.ndarray[float]): N-dimensional vector/point B

    Returns:
        float: euclidean distance between points A and B
    """
    magnitude_a = np.sqrt(np.sum(a * a))
    magnitude_b = np.sqrt(np.sum(b * b))
    csim = np.sum(a * b) / (magnitude_a * magnitude_b)
    return csim


@njit
def manhattan_dist(a: np.ndarray[float], b: np.ndarray[float]) -> float:
    """Betthauser - 2018 - compute manhattan distance between 2 points

    Args:
        a (np.ndarray[float]): N-dimensional point A
        b (np.ndarray[float]): N-dimensional point B

    Returns:
        float: manhattan distance between points A and B
    """
    mand = np.sum(np.abs(b - a))
    return mand


@njit
def minkowski_dist(a: np.ndarray[float], b: np.ndarray[float], p: float = 1.0) -> float:
    """Betthauser - 2018 - compute p-th root minkowski distance between 2 vectors

    Args:
        a (np.ndarray[float]): N-dimensional point A
        b (np.ndarray[float]): N-dimensional point A=B
        p (float, optional): pth root (default is 1, city block)

    Returns:
        float: p-th root minkowski distance between points A and B
        mink = np.sum(np.abs(b - a) ** p) ** (1 / p)
    """
    mink = np.sum(np.abs(b - a) ** p) ** (1 / p)
    return mink


####  Point-to-distribution distance functions  ##########
@njit
def mahalinobis_dist(y: np.ndarray[float], x: np.ndarray[float], sqrt_calc: bool = True) -> float:
    """Betthauser - 2018 - compute  mahalinobis distance from point to point cluster

    Args:
        y (np.ndarray[float]): N-dimensional point
        x (np.ndarray[float]): N-dimensional target cluster of points

    Returns:
        float: mahalinobis distance of point y to distribution x (exponent of multivariate gaussian)
    """
    # mu = np.mean(x, axis=0)
    mu = np.array([np.mean(x[:, i]) for i in range(x.shape[1])])  # jit-friendly version (axis=0 is a problem)
    diff_y_mu = y - mu
    dist = (diff_y_mu @ np.linalg.pinv(np.cov(x.T))) @ diff_y_mu.T
    if not sqrt_calc:
        return dist
    return np.sqrt(dist)


@njit
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
@njit
def pearson_correlation(p: np.ndarray[float], q: np.ndarray[float]) -> float:
    """Betthauser - 2018 - pearson correlation between distributions

    Args:
        p (np.ndarray[float]): array p
        q (np.ndarray[float]): array q

    Returns:
        float: pearson correlation between normalized p and q
    """
    return np.corrcoef(p / np.sum(p), q / np.sum(q))[0, 1]


@njit
def jensen_shannon_divergence(p: np.ndarray[float], q: np.ndarray[float]) -> float:
    """Betthauser - 2024 - jensen-shannon divergence

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
    """Betthauser - 2024 - jensen-shannon distance metric

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


@njit
def wasserstein_dist(p: np.ndarray[float], q: np.ndarray[float]) -> float:
    """Wasserstein distance or Kantorovich–Rubinstein metric

        # From wikipedia.org: Intuitively, if each distribution is viewed as a unit amount of earth (soil) piled on
        # M, the metric is the minimum "cost" of turning one pile into the other, which is assumed to be the amount
        # of earth that needs to be moved times the mean distance it has to be moved.

    Args:
        p (np.ndarray[float]): PMF of distribution p
        q (np.ndarray[float]): PMF of distribution q

    Returns:
        float: W_p(P,Q) = (1/N) * SUM_i [ ||X_i - Y_i|| ^^ p ] ^^ (1/p)
    """
    epsilon = 1e-12
    p = np.abs(p) + epsilon
    q = np.abs(q) + epsilon
    if len(p.shape) == 2:
        return None  # wasserstein_distance_nd(p, q)
    elif len(p.shape) == 1:
        return minkowski_dist(p, q) / len(p)
        # return wasserstein_distance(p, q)


@njit
def wasserstein_dist_gaussian1d(
    mu1: np.ndarray[float],
    cov1: np.ndarray[float],
    mu2: np.ndarray[float],
    cov2: np.ndarray[float],
) -> float:
    # Wasserstein distance of 2 gaussians ~N(mu1, C1) and ~N(mu2, C2)
    # From wikipedia.org: Intuitively, if each distribution is viewed as a unit amount of earth (soil) piled on
    # M, the metric is the minimum "cost" of turning one pile into the other, which is assumed to be the amount
    # of earth that needs to be moved times the mean distance it has to be moved.

    # W(mu1, mu2) = sqrt( ||mu1-mu2||_2^2 + trace(C1 + C2-2*(C2^0.5) @ C1 @ C2^0.5) ^ 0.5 )
    norm = np.sum((mu1 - mu2) ** 2)
    covterm = np.trace(cov1 + cov2 - 2 * ((cov2**0.5) @ cov1) @ (cov2**0.5)) ** 0.5
    return np.sqrt(norm + covterm)


@njit
def kl_divergence(p: np.ndarray[float], q: np.ndarray[float]) -> float:
    """Betthauser - 2018 - compute KL divergence between two PMFs

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
    """Betthauser - 2018 - compute Jeffreys/2-way KL divergence between two PMFs

    Args:
        p (np.ndarray[float]): PMF of distribution p
        q (np.ndarray[float]): PMF of distribution q

    Returns:
        float: Jeffreys bi-directional KL divergence KL(p||q) + KL(q||p)
    """
    epsilon = 1e-12
    p = np.abs(p) + epsilon
    q = np.abs(q) + epsilon
    log_p = np.log(p)
    log_q = np.log(q)
    jeffreys = np.sum(p * (log_p - log_q) + q * (log_q - log_p))
    return jeffreys


@njit
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
    diff_mu = mu1 - mu2
    return np.log(v2 / v1) + ((v1 + diff_mu * diff_mu) / (2 * v2)) - 0.5


@njit
def kl_div_gaussian1d_bidirectional(mu1: float, v1: float, mu2: float, v2: float) -> float:
    """Betthauser - 2018 - compute KL divergence between two normal distributions

    Args:
        mu1 (float): _description_
        v1 (float): _description_
        mu2 (float): _description_
        v2 (float): _description_

    Returns:
        float: _description_
    """
    jeffreys = kl_div_gaussian1d(mu1=mu1, v1=v1, mu2=mu2, v2=v2) + kl_div_gaussian1d(mu1=mu2, v1=v2, mu2=mu1, v2=v1)
    return jeffreys


@njit
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
    diff_mu = mu1 - mu2
    part2 = 0.25 * ((diff_mu * diff_mu) / (v1 + v2))
    return part1 + part2


@njit
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
    diff_mu = mu1 - mu2
    fish = (diff_mu * diff_mu) / (v1 + v2)
    return fish


@njit
def entropy(hist1: np.ndarray[float]) -> float:
    """Betthauser 2016 -- Calculate joint entropy of an N-d distribution

    Args:
        hist1 (np.ndarray[float]): an N-D histogram or PMF

    Returns:
        float: entropy of the ditribution
    """
    hist1 = hist1 / np.sum(hist1)
    nz_probs = np.array([-p * np.log(p) for p in hist1 if p > 1e-12])
    entrp = np.sum(nz_probs)
    return entrp


@njit
def minmax_scaling(data: np.ndarray[float], max_val: float = 255) -> np.ndarray[float]:
    """Betthauser - 2018 - min-max scaling of data

    Args:
        data (np.ndarray[float]): N-D data
        max_val (float, optional): max desired output value. Defaults to 255.

    Returns:
        np.ndarray[float]: data min-max scaled in range [0, max_val]
    """
    d_min = data.min()
    return max_val * (data - d_min) / (data.max() - d_min)


# returns joint histogram of 2 image sections
@njit
def joint_histogram_2d(patch1: np.ndarray[float], patch2: np.ndarray[float], bins: float = 255.0) -> np.ndarray[float]:
    """Computes joint histogram of 2 image sections/patches
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
def hist1d(
    vals: np.ndarray[float], bins: int, val_range: tuple[float, float], norm: bool = False
) -> np.ndarray[np.int64]:
    """JIT compute 1D histogram of values (fastest current method)

    Args:
        vals (np.ndarray[float]): _description_
        bins (int): _description_
        val_range (tuple[float, float]): _description_
        norm (bool, optional): _description_. Defaults to False.

    Returns:
        np.ndarray[np.int64]: _description_
    """
    hist = np.histogram(vals, bins, val_range)[0]
    if norm:
        hist = (hist / np.sum(hist)).astype(np.int64)
    return hist


@njit
def mutual_info(image1: np.ndarray[float], image2: np.ndarray[float]) -> float:
    """Betthauser - 2018 - compute mutual information between 2 images/patches
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

    range_min = np.min([c1_1d, c2_1d])
    range_max = np.max([c1_1d, c2_1d])

    distr_a = hist1d(vals=c1_1d, bins=64, val_range=(range_min, range_max), norm=True)
    distr_b = hist1d(vals=c2_1d, bins=64, val_range=(range_min, range_max), norm=True)

    print(" ** Point to point distances:")
    print(f"{cosine_similarity(point1, point2) = :.7f}")
    print(f"{manhattan_dist(point1, point2) = :.7f}")
    print(f"{euclidean_dist(point1, point2) = :.7f}")
    print(f"{minkowski_dist(point1, point2, 7) = :.7f}\n")

    print(" ** Point to distribution distances:")
    print(f"{mahalinobis_dist(point1, cluster1, sqrt_calc=False) = :.7f}")
    print(f"{mahalinobis_dist(point1, cluster1, sqrt_calc=True) = :.7f}\n")

    print(" **  Distribution to distribution distances:")
    print(f"{compute_R2(c1_1d, c2_1d) = :.7f}")
    print(f"{fisher_dist(mu_a, var_a, mu_b, var_b) = :.7f}")
    print(f"{bhattacharyya_dist(mu_a, var_a, mu_b, var_b) = :.3f}")
    print(f"{kl_div_bidirectional(distr_a, distr_b) = :.3f}")
    print(f"{kl_div_gaussian1d_bidirectional(mu_a, var_a, mu_b, var_b) = :.3f}")
    print(f"{wasserstein_dist(distr_a, distr_b) = :.3f}\n")
    # plt.figure()
    # plt.scatter(cluster1[:, 0], cluster1[:, 1])
    # plt.scatter(cluster2[:, 0], cluster2[:, 1])
    # plt.show()


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    tf1 = time.perf_counter() - t0
    print(f"Program took {tf1:.3f} seconds")

    t0 = time.perf_counter()
    main()
    tf2 = time.perf_counter() - t0
    print(f"1st run took {tf1:.3f} seconds")
    print(f"2nd run took {tf2:.3f} seconds")

    print(f"njit speed-up: {tf1/tf2:.3f}x")
