"""Betthauser 2024 - prime generators from brute force to sieve of eratosthenes"""

import time

import numpy as np
from numba import njit


@njit
def is_prime(n):
    if n in {2, 3} or n < 9:  # noqa: PLR2004
        return True
    if n < 2 or n % 2 == 0 or n % 3 == 0:  # noqa: PLR2004
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
def n_primes_bf(n: int) -> np.ndarray[int]:
    """
    Brute force method to list first n primes

    Args:
        n (int): desired number of primes

    Returns:
        np.ndarray[int]: list of first n primes

    """
    if n <= 1:
        return np.array([1]).astype(np.uint64)

    list_primes = np.ones(n).astype(np.uint64)
    if n >= 2:  # noqa: PLR2004
        list_primes[1] = 2
    if n == 2:  # noqa: PLR2004
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
def n_primes_basic(n: int) -> np.ndarray[int]:
    """
    Basic method to list of first n primes

    Args:
        n (int): desired number of primes

    Returns:
        list[int]: list of first n primes

    """
    if n <= 1:
        return np.array([1]).astype(np.uint64)

    list_primes = np.ones(n).astype(np.uint64)
    if n >= 2:  # noqa: PLR2004
        list_primes[1] = 2
    if n == 2:  # noqa: PLR2004
        return list_primes

    for idx in range(2, n):
        num = 1 + list_primes[idx - 1]
        while True:
            if is_prime(num):
                break
            num += 1
        list_primes[idx] = num
    return list_primes


def n_primes_e_sieve(n: int) -> np.ndarray[int]:
    """
    Sieve of eratosthanes: fast method to list of first n primes

    Args:
        n (int): desired number of primes

    Returns:
        list[int]: list of first n primes

    """
    if n <= 1:
        return np.array([1]).astype(np.uint64)

    list_primes = np.ones(n).astype(np.uint64)
    if n >= 2:  # noqa: PLR2004
        list_primes[1] = 2
    if n == 2:  # noqa: PLR2004
        return list_primes
    # TODO: implement sieve of eratosthenes
    return NotImplementedError


def n_primes_a_sieve(n: int) -> np.ndarray[int]:
    """
    Seive of atkins: fastest method to list of first n primes

    Args:
        n (int): desired number of primes

    Returns:
        list[int]: list of first n primes

    """
    if n <= 1:
        return np.array([1]).astype(np.uint64)

    list_primes = np.ones(n).astype(np.uint64)
    if n >= 2:  # noqa: PLR2004
        list_primes[1] = 2
    if n == 2:  # noqa: PLR2004
        return list_primes
    # TODO: implement sieve of eratosthenes
    return NotImplementedError


def main() -> None:
    n = 30000
    _ = n_primes_bf(3)  # jit compile function on smaller run
    _ = n_primes_basic(3)

    t0 = time.perf_counter()
    _ = n_primes_bf(n)
    tfirst = time.perf_counter() - t0
    print(f"Brute force took {tfirst:.3f} seconds.")

    t0 = time.perf_counter()
    _ = n_primes_basic(n)
    tlast = time.perf_counter() - t0
    print(f"Basic method took {tlast:.3f} seconds.")
    print(f"jit speed-up: {tfirst / tlast:.3f}x")


if __name__ == "__main__":
    main()
