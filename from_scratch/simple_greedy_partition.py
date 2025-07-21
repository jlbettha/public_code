"""Simple greedy partitioner of array s into k weighted sum groups"""

import time

import numpy as np
from numba import njit


@njit
def simple_greedy_partition(s: np.ndarray[float], k: int = 2, wts: np.ndarray[float] | None = None) -> list[list[float]]:
    """
    Partition array s into k approx equal sum groups

    Args:
        s (np.ndarray): array of numbers
        k (int, optional): number of partitions. Defaults to 2.
        wts (list[float] | None, optional): weights for each partition. Defaults to None.

    Raises:
        ValueError: len(s) < k

    Returns:
        list[list[float]]: list of k partition subsets

    """
    if wts is None:
        wts = np.array([1.0])

    wts = k * wts / np.sum(wts)
    print(wts)
    n = len(s)
    if n < k:
        msg = f"To partition into {k} groups, the number array must have at least {k} values."
        raise ValueError(msg)

    s[::-1].sort()
    groups: list[list[np.float64]] = [[s[i]] for i in range(k)]
    group_sums = s[:k]

    for i in np.arange(k, n):
        idx = np.argmin(group_sums)
        groups[idx].append(s[i])
        group_sums[idx] += s[i] * (wts[idx] ** -1)

    return groups


def main() -> None:
    num_partitions = 3
    wts = np.array([0.5, 1.0, 1.5])
    rng = np.random.default_rng()
    number_arr = 20.0 * rng.random(3000)
    print(f"{number_arr.shape=}, {np.sum(number_arr)=:.3f}")

    expected_group_sums = np.sum(number_arr) * wts / np.sum(wts)
    print(f"Expected groups: {expected_group_sums}")

    partitions = simple_greedy_partition(s=number_arr, k=num_partitions, wts=wts)

    actual_group_sums = [float(np.sum(group)) for group in partitions]

    print(f"Actual groups: {actual_group_sums}")  #:.3f}")
    print(f"RMSE: {'\u00b1'}{np.sqrt(np.mean((np.array(actual_group_sums) - expected_group_sums) ** 2)):.3f}")


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    treg = time.perf_counter() - tmain
    print(f"Program took {treg:.3f} seconds.")

    t0 = time.perf_counter()
    main()
    tjit = time.perf_counter() - t0
    print(f"Program took {tjit:.3f} seconds.")

    print(f"JIT speed-up: {treg / tjit:.3f}x")
