"""Betthauser, 2018: k-gaussian mixture clustering"""

import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

# from k_means_clustering import k_means_clustering


@njit
def _membership_prob(point: np.ndarray[float], c_mean: np.ndarray[float], c_cov: np.ndarray[float]) -> float:
    """
    Compute probability that point to belongs to this cluster (multivariate gaussian)

    Args:
        point (np.ndarray[float]): point to check
        c_mean (np.ndarray[float]): mean of cluster
        c_cov (np.ndarray[float]): covariance of cluster

    Returns:
        float: probability that point to belongs to this cluster

    """
    k = c_cov.shape[0]
    denom = np.sqrt(np.linalg.det(c_cov) * (2 * np.pi) ** k)
    mahal_dist = ((point - c_mean) @ np.linalg.pinv(c_cov)) @ (point - c_mean).T
    # prob = np.squeeze(np.exp(-0.5 * mahal_dist) / denom)
    prob = np.exp(-0.5 * mahal_dist[0][0]) / denom  # jit-friendly edit
    return prob + 1e-12


def _prob_dist_array(
    data_pt: np.ndarray[float],
    k_means: np.ndarray[float],
    k_covs: np.ndarray[float],
) -> np.ndarray[float]:
    """
    _summary_

    Args:
        data_pt (np.ndarray[float]): n-dimensional point to check
        k_means (np.ndarray[float]): means of k-clusters, shape: k x n
        k_covs (np.ndarray[float]): covariance matrices of k-clusters, shape: k x n x n

    Returns:
        np.ndarray[float]: array of probabilities that point belongs to k clusters

    """
    return np.array([_membership_prob(data_pt, mean, cov) for mean, cov in zip(k_means, k_covs, strict=False)])


def k_gaussian_mixture_clustering(  # noqa: PLR0915
    data: np.ndarray[float],
    k: int,
    tolerance: float = 1e-5,
    plot: bool = False,
    max_iters: int = 200,
) -> np.ndarray[int]:
    """
    Betthauser, 2018: k-gaussian mixture/EM clustering

    Args:
        data (NDArray[np.float64]): unlabeled N x D, N samples of D-dim data
        k (int): number of clusters
        tolerance (float): convergence threshold, default is 0.001
        plot (bool): if plots should be shown during iterations, default is False
        max_iters (int): maximum number of iterations, default is 200

    Returns:
        np.ndarray[int]: array of likeliest labels, num iters, cluster_means, cluster_covs, likelihoods

    """
    num_data_pts = data.shape[0]
    dim = data.shape[1]

    # init gmm
    rng = np.random.default_rng()
    while True:
        # _, _, cluster_means = k_means_clustering(data, k)  # init with k means
        rand_idx = rng.permutation(num_data_pts)
        cluster_means = data[rand_idx[:k], :]  # init with k random points in data

        cluster_covs = np.squeeze([np.eye(dim) for lbl in range(k)])  # init covs with identity

        try:
            soft_label_probs = np.array(
                [
                    _prob_dist_array(np.reshape(data[idx, :], (1, -1)), cluster_means, cluster_covs)
                    for idx in range(num_data_pts)
                ]
            )
        except RuntimeWarning:
            continue

        if np.any(np.isnan(soft_label_probs)):
            continue

        # soft_label_probs = np.squeeze(soft_label_probs)
        current_likeliest_labels = np.array([np.argmax(soft_label_probs[idx, :]) for idx in range(num_data_pts)])

        if len(np.unique(current_likeliest_labels)) == k:
            break

    cluster_weights_pi_k = np.ones(k) / k
    weighted_probs_nk = cluster_weights_pi_k * soft_label_probs
    sum_weighted_probs_n = np.sum(weighted_probs_nk, axis=1)

    log_likelihood = np.sum(np.log(sum_weighted_probs_n))  # / N

    best_llhd = log_likelihood
    best_means = cluster_means
    best_covs = cluster_covs
    best_labels = current_likeliest_labels

    likelihoods = []
    ll_last = log_likelihood

    iters = 0
    its = []
    while True:  # begin E-M steps
        iters = iters + 1

        # expectation ~~~~~~~~~~~~~~~~~
        gamma_nk = np.divide(weighted_probs_nk.T, sum_weighted_probs_n).T

        sum_gamma_k = np.sum(gamma_nk, axis=0)

        # maximization ~~~~~~~~~~~~~~~~
        cluster_weights_pi_k = sum_gamma_k / num_data_pts

        cluster_means = ((data.T @ gamma_nk) / sum_gamma_k).T

        cluster_covs = []
        for lbl in range(k):
            centered_data = data - cluster_means[lbl, :]
            ctc = (gamma_nk[:, lbl] * centered_data.T) @ centered_data
            cluster_covs.append(ctc / sum_gamma_k[lbl])

        cluster_covs = np.stack(cluster_covs)

        soft_label_probs = np.array(
            [
                _prob_dist_array(np.reshape(data[idx, :], (1, -1)), cluster_means, cluster_covs)
                for idx in range(num_data_pts)
            ]
        )

        weighted_probs_nk = cluster_weights_pi_k * soft_label_probs
        sum_weighted_probs_n = np.sum(weighted_probs_nk, axis=1)
        log_likelihood = np.sum(np.log(sum_weighted_probs_n)) / num_data_pts

        if iters % 10 == 0:
            print(f"Iter: {iters} -- Log-likelihood: {log_likelihood:.4f}", flush=True)

        likelihoods.append(log_likelihood)
        its.append(iters)

        current_likeliest_labels = np.array([np.argmax(soft_label_probs[idx, :]) for idx in range(num_data_pts)])

        if log_likelihood > best_llhd:
            best_llhd = log_likelihood
            best_means = cluster_means
            best_covs = cluster_covs
            best_labels = current_likeliest_labels

        if plot:
            plt.subplot(1, 2, 1)
            plt.cla()
            plt.scatter(data[:, 0], data[:, 1], c=current_likeliest_labels, s=3)
            plt.subplot(1, 2, 2)
            plt.cla()
            plt.plot(its, likelihoods)
            plt.ylabel("log-likelihood")
            plt.pause(0.001)

        if np.abs(log_likelihood - ll_last) < tolerance or iters > max_iters:
            plt.close()
            print(f"k-gaussian mixture clustering took {iters} iterations.")
            return (
                best_labels,
                best_means,
                best_covs,
                likelihoods,
            )

        ll_last = log_likelihood


def _generate_data(num_clusters: int, dim: int, size_clusters: int) -> np.ndarray[float]:
    """
    Generate synthetic data of gaussian clusters

    Args:
        num_clusters (int): true number of clusters to generate
        dim (int): n-dimensionality of data
        size_clusters (int): number of points in each cluster

    Returns:
        NDArray[np.float64]: synthetic data of n-dim gaussian clusters

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

    return np.vstack(data)
    # rand_idx = np.random.permutation(data.shape[0])
    # data = data[rand_idx, :]



def main() -> None:
    """_summary_"""
    k = 7
    num_actual_clusters = 7
    num_dims = 2
    num_pts_per_cluster = 500

    data = _generate_data(num_actual_clusters, num_dims, num_pts_per_cluster)

    labels, k_means, _, loglikes = k_gaussian_mixture_clustering(data, k, tolerance=1e-5, plot=False, max_iters=150)

    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=3)
    plt.scatter(k_means[:, 0], k_means[:, 1], c="r", s=20, marker="x", label="Cluster means")

    plt.title(f"k-gaussian mixture clustering, converged in {len(loglikes)} iterations.")

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, len(loglikes), 1), loglikes)
    plt.xlabel("iterations")
    plt.ylabel("log-likelihood")

    plt.show()


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter() - tmain:.3f} seconds.")
