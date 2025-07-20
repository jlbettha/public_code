"""
Betthauser, 2018: orthonormal compression
>> Use when number of features (datapoint dimensions) is large.
"""

import time

import numpy as np
from numpy.typing import NDArray

ArrayTuple = tuple[NDArray[np.float64]]


def orthonormal_compression(
    data: NDArray[np.float64], factor: float = 0.2, rng: np.random.Generator = None
) -> ArrayTuple:
    """
    Betthauser, 2018: orthonormal_compression
        Projection reduces feature dimensionality

    Args:
        data (NDArray[np.float64]): unlabeled N x D, N samples of D-dim data
        factor (float): factor to reduce dimensionality
        rng (np.random.Generator, optional): random number generator for reproducibility
    Returns:
        Tuple[NDArray[np.uint8]]: projection matrix, transformed data

    """
    new_dim = int(data.shape[1] * factor)
    rng = np.random.default_rng()
    proj_matrix, _ = np.linalg.qr(rng.random((data.shape[1], new_dim)))
    proj_data = data @ proj_matrix
    return proj_matrix, proj_data


def main() -> None:
    """_summary_"""
    num_pts = 2000
    num_dims = 200

    rng = np.random.default_rng()
    data = rng.uniform(0, 10, size=(num_pts, num_dims))

    proj_matrix, proj_data = orthonormal_compression(data, factor=0.05, rng=rng)
    print(f"Projection matrix shape: {proj_matrix.shape}")
    print(f"Original data shape: {data.shape}")
    print(f"Projected data shape: {proj_data.shape}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter() - t0:.3f} seconds.")
