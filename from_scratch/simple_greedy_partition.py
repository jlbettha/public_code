"""Simple greedy partitioner of array s into k approx equal sum groups"""

import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


@njit
def simple_greedy_partition(s: np.ndarray[float], k: int = 2) -> list[list[float]]:
    """Partition array s into k approx equal sum groups

    Args:
        s (np.ndarray): array of numbers
        k (int, optional): number of partitions. Defaults to 2.

    Raises:
        ValueError: len(s) < k

    Returns:
        list[list[float]]: list of k partition subsets
    """
    n = len(s)
    if n < k:
        raise ValueError(
            f"To partition into {k} groups, the number array must have at least {k} values."
        )

    s[::-1].sort()
    groups: list[list[np.float64]] = [[s[i]] for i in range(k)]
    group_sums = s[:k]

    for i in np.arange(k, n):
        idx = np.argmin(group_sums)
        groups[idx].append(s[i])
        group_sums[idx] += s[i]

    return groups


def main() -> None:
    num_partitions = 3
    number_arr = 12.77 * np.random.rand(700)
    expected_group_sum = np.sum(number_arr) / num_partitions

    print(f"{number_arr.shape=}, {np.sum(number_arr)=:.3f}")

    partitions = simple_greedy_partition(s=number_arr, k=num_partitions)

    actual_group_sums = [float(np.sum(group)) for group in partitions]

    print(f"{expected_group_sum = :.3f}")
    print(f"{actual_group_sums = }")  #:.3f}")


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    treg = time.perf_counter() - tmain
    print(f"Program took {treg:.3f} seconds.")

    t0 = time.perf_counter()
    main()
    tjit = time.perf_counter() - t0
    print(f"Program took {tjit:.3f} seconds.")

    print(f"JIT speed-up: {treg/tjit:.3f}x")
