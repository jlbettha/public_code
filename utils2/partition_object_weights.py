"""Simple greedy partitioner of array s into k approx equal sum groups"""

# import time
import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt

np.random.seed(42)


def simple_partition(wts: np.ndarray[float], ids: np.ndarray[float], k: int = 2) -> list[list[float]]:
    """
    Partition array s into k approx equal sum groups

    Args:
        s (np.ndarray): array of numbers
        k (int, optional): number of partitions. Defaults to 2.

    Raises:
        ValueError: len(s) < k

    Returns:
        list[list[float]]: list of k partition subsets

    """
    n = len(wts)
    if n < k:
        raise ValueError(f"To partition into {k} groups, the number array must have at least {k} values.")

    wts, ids = (list(x) for x in zip(*sorted(zip(wts, ids, strict=False), key=lambda pair: pair[0], reverse=True), strict=False))

    groups: list[tuple[np.float64, str]] = [[tuple([wts[i], ids[i]])] for i in range(k)]
    # print(groups)
    group_sums = wts[:k]

    for i in np.arange(k, n):
        idx = np.argmin(group_sums)
        groups[idx].append([tuple([wts[i], ids[i]])])
        group_sums[idx] += wts[i]

    return groups


def main() -> None:
    case_type = "go"

    df = pd.read_excel(case_type + ".xlsx")
    ids = df["id"]
    wts = df["wt"]
    num_items = len(df["id"])
    wts = wts + 1e-8 * np.random.randn(num_items)
    num_unique = len(np.unique(wts))
    print(f"Fraction unique: {num_unique}/{num_items} = {num_unique / num_items:.3f}")
    assert num_unique == num_items
    # for k, v in item_dict.items():
    #     print(f"Item {k} has weight {v}")

    num_partitions = 2
    expected_group_sum = np.sum(wts) / num_partitions

    print(f"{wts.shape=}, {np.sum(wts)=:.3f}")

    partitions = simple_partition(wts=wts, ids=ids, k=num_partitions)

    actual_group_sums = []
    splits = []
    for group in partitions:
        grp_wt = 0
        print(f"group size: {len(group)}")
        ids = []
        for item in group:
            if isinstance(item, list):
                item = item[0]
            # print(type(item))
            wt, id = item
            grp_wt += wt
            # grp_wt = float(np.sum([item[0] for item in group]))
            ids.append(id)
        actual_group_sums.append(float(grp_wt))
        splits.append(ids)

    actual_group_sums = np.array(actual_group_sums)
    print(f"{expected_group_sum = :.5f}")
    print(f"{actual_group_sums = }")

    value = actual_group_sums * 3337.59  # 36.93  # 3337.59
    print(f"{value = }")

    with open(case_type + "_groups.txt", "w") as f:
        f.write(f"Group 1: {sorted(splits[0])}\n")
        f.write(f"Group 2: {sorted(splits[1])}\n")

    print(splits[0])
    print(splits[1])


if __name__ == "__main__":
    main()
