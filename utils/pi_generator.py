"""_summary_"""

import time
import numpy as np
from numba import njit
from numpy.random import uniform as npru


@njit
def rand_xy():
    """_summary_"""
    x = npru(low=0.0, high=1.0)
    y = npru(low=0.0, high=1.0)
    if x**2 + y**2 <= 1:
        return True
    return False


def main() -> None:
    """_summary_"""
    total_ct = 0
    lt_one_ct = 0
    num_iters = 10_000_000

    for i in range(num_iters):
        total_ct += 1
        if rand_xy():
            lt_one_ct += 1

        # if i % 1_000_000 == 0:
        #     pi_est = 4 * (lt_one_ct / total_ct)
        # print(
        #     f"Iter {i} -- Pi estimate: {pi_est:.8f}, Error: {np.pi-pi_est:.8f}",
        #     flush=True,
        # )
    pi_est = 4 * (lt_one_ct / total_ct)
    print(f"Final Pi estimate: {pi_est:.8f}, Error: {np.pi-pi_est:.8f}, Iters.: {num_iters:_}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter()-t0:.4f} seconds.")
