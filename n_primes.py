""" Betthauser 2024 - prime generators from brute force to sieve of eratosthenes
"""

import time
import numpy as np
from numpy.typing import NDArray
from numba import njit, vectorize


@njit
def is_prime(n):
    if n == 2 or n == 3:
        return True
    if n < 2 or n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    r = int(n**0.5)

    # all primes > 3 are of the form 6n ± 1
    f = 5  # start with f=5 (which is prime)
    while f <= r:
        if n % f == 0:  # test f, f+2 for being prime
            return False
        if n % (f + 2) == 0:
            return False
        f += 6  # loop by 6
    return True


@njit
def n_primes_bf(n: int) -> NDArray[np.uint64]:
    """brute force method to list first n primes

    Args:
        n (int): desired number of primes

    Returns:
        list[int]: list of first n primes
    """
    if n <= 1:
        return np.array([1]).astype(np.uint64)

    list_primes = np.ones(n).astype(np.uint64)
    if n >= 2:
        list_primes[1] = 2
    if n == 2:
        return list_primes

    for idx in range(2, n):
        num = 1 + list_primes[idx - 1]
        p_not_found = True
        while p_not_found:
            p_not_found = False
            for factor in range(2, int(1 + (n + 1) / 2)):
                if num == factor:
                    continue
                if num % factor == 0:  # not prime
                    p_not_found = True
                    # print(f"{num} is divisible by {factor} and not prime", flush=True)
                    num += 1
                    break

        # print(f"prime number {idx+1} is {num}", flush=True)
        list_primes[idx] = num
    return list_primes


@njit
def n_primes_basic(n: int) -> NDArray[np.int64]:
    """basic method to list of first n primes

    Args:
        n (int): desired number of primes

    Returns:
        list[int]: list of first n primes
    """
    if n <= 1:
        return np.array([1]).astype(np.uint64)

    list_primes = np.ones(n).astype(np.uint64)
    if n >= 2:
        list_primes[1] = 2
    if n == 2:
        return list_primes

    for idx in range(2, n):
        num = 1 + list_primes[idx - 1]
        while True:
            if is_prime(num):
                break
            num += 1
        list_primes[idx] = num
    return list_primes


def n_primes_e_sieve(n: int) -> NDArray[np.int64]:
    """sieve of eratosthanes: fast method to list of first n primes

    Args:
        n (int): desired number of primes

    Returns:
        list[int]: list of first n primes
    """
    if n <= 1:
        return np.array([1]).astype(np.uint64)

    list_primes = np.ones(n).astype(np.uint64)
    if n >= 2:
        list_primes[1] = 2
    if n == 2:
        return list_primes
    # TODO
    return NotImplementedError


def n_primes_a_sieve(n: int) -> NDArray[np.int64]:
    """seive of atkins: fastest method to list of first n primes

    Args:
        n (int): desired number of primes

    Returns:
        list[int]: list of first n primes
    """
    if n <= 1:
        return np.array([1]).astype(np.uint64)

    list_primes = np.ones(n).astype(np.uint64)
    if n >= 2:
        list_primes[1] = 2
    if n == 2:
        return list_primes
    # TODO
    return NotImplementedError


def main() -> None:
    n = 30000
    _ = n_primes_bf(3)  # jit compile function on smaller run
    _ = n_primes_basic(3)

    t0 = time.time()
    _ = n_primes_bf(n)
    print(f"Brute force took {time.time()-t0:.3f} seconds.")

    t0 = time.time()
    _ = n_primes_basic(n)
    print(f"Basic method took {time.time()-t0:.3f} seconds.")


if __name__ == "__main__":

    main()
