""" Betthauser, 2018: orthonormal compression 
    >> Use when number of features (datapoint dimensions) is large.
"""

import time
import numpy as np
from typing import Tuple
from numpy.typing import NDArray

ArrayTuple = Tuple[NDArray[np.float64]]


def orthonormal_compression(
    data: NDArray[np.float64], factor: float = 0.2
) -> ArrayTuple:
    """Betthauser, 2018: orthonormal_compression
        Projection reduces feature dimensionality

    Args:
        data (NDArray[np.float64]): unlabeled N x D, N samples of D-dim data
        factor (float): factor to reduce dimensionality
    Returns:
        Tuple[NDArray[np.uint8]]: projection matrix, transformed data
    """
    new_dim = int(data.shape[1] * factor)
    proj_matrix, _ = np.linalg.qr(np.random.rand(data.shape[1], new_dim))
    proj_data = data @ proj_matrix
    return proj_matrix, proj_data


def main() -> None:
    """_summary_"""
    num_pts = 2000
    num_dims = 200

    data = np.random.uniform(10, size=(num_pts, num_dims))

    proj_matrix, proj_data = orthonormal_compression(data, factor=0.05)
    print(f"Projection matrix shape: {proj_matrix.shape}")
    print(f"Original data shape: {data.shape}")
    print(f"Projected data shape: {proj_data.shape}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time()-t0:.3f} seconds.")
