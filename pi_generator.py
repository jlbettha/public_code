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
    return x * x + y * y


def main() -> None:
    """_summary_"""
    total_ct = 0
    lt_one_ct = 0
    num_iters = 10_000_000

    for i in range(num_iters):
        total_ct += 1
        if rand_xy() <= 1:
            lt_one_ct += 1

        # if i % 1_000_000 == 0:
        #     pi_est = 4 * (lt_one_ct / total_ct)
        # print(
        #     f"Iter {i} -- Pi estimate: {pi_est:.8f}, Error: {np.pi-pi_est:.8f}",
        #     flush=True,
        # )
    pi_est = 4 * (lt_one_ct / total_ct)
    print(f"Final Pi estimate: {pi_est:.8f}, Error: {np.pi-pi_est:.8f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time()-t0:.4f} seconds.")
