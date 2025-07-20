"""_summary_"""

import time

import numpy as np
from numba import njit


@njit
def rand_xy(rng):
    """_summary_"""
    x = rng.uniform(low=0.0, high=1.0)
    y = rng.uniform(low=0.0, high=1.0)
    return x**2 + y**2 <= 1


def main() -> None:
    """_summary_"""
    total_ct = 0
    lt_one_ct = 0
    num_iters = 10_000_000
    rng = np.random.default_rng()

    for i in range(num_iters):
        total_ct += 1
        if rand_xy(rng):
            lt_one_ct += 1
        if i % 100_000 == 0:
            pi_est = 4 * (lt_one_ct / total_ct)
            print(
                f"Iter {i} -- Pi estimate: {pi_est:.8f}, Squared error: {(np.pi - pi_est) ** 2:.8f}",
                flush=True,
            )
    pi_est = 4 * (lt_one_ct / total_ct)
    print(f"Final Pi estimate: {pi_est:.8f}, Squared error: {(np.pi - pi_est) ** 2:.8f}, Iters.: {num_iters:_}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter() - t0:.4f} seconds.")
