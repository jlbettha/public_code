"""Betthauser, 2018: Discrete cosine transform compression
>> Use when number of features (datapoint dimensions) is large.
"""

import time
import numpy as np
import PIL.Image as Image
from typing import Tuple
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

ArrayTuple = Tuple[NDArray[np.float64]]


def dct_compression(data: NDArray[np.float64], factor: float = 0.2) -> ArrayTuple:
    """Betthauser, 2018: dct_compression
        Projection reduces feature dimensionality

    Args:
        data (NDArray[np.float64]): unlabeled N x D, N samples of D-dim data
        factor (float): factor to reduce dimensionality
    Returns:
        Tuple[NDArray[np.uint8]]: dct matrix, idct matrix, transformed data
    """
    new_dim = int(data.shape[1] * factor)
    dct_matrix = dct(np.eye(data.shape[1]), axis=0)
    idct_matrix = np.linalg.inv(dct_matrix)

    proj_data = data @ dct_matrix[:, :new_dim]
    return dct_matrix, idct_matrix, proj_data


def normalize_image(img: NDArray[np.float64]) -> NDArray[np.uint8]:
    """Min-max scale image to range [0,255] as uint8
    Args:
        img (NDArray[np.float64]): input image

    Returns:
        NDArray[np.uint8]: scaled image
    """
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)


def main() -> None:
    """_summary_"""
    num_pts = 2000
    num_dims = 200
    factor = 0.05
    data = np.random.uniform(10, size=(num_pts, num_dims))

    dct_matrix, idct_matrix, proj_data = dct_compression(data, factor=factor)
    print(f"DCT matrix shape: {dct_matrix.shape}")
    print(f"Original data shape: {data.shape}")
    print(f"Projected data shape: {proj_data.shape}")

    # img = Image.open("lena_noisy.png")
    # img = np.array(img)
    # dct_matrix, idct_matrix, proj_data = dct_compression(img, factor=factor)
    # dim = int(img.shape[0] * factor)
    # img_transformed = proj_data @ idct_matrix[:dim, :]

    # print(f"{img.shape=}, {img.min()=}, {img.max()=}")
    # print(f"{dct_matrix.shape=}, {dct_matrix.min()=}, {dct_matrix.max()=}")

    # plt.subplot(1, 2, 1)
    # plt.imshow(img, cmap="gray")
    # plt.axis("off")
    # plt.subplot(1, 2, 2)
    # plt.imshow(img_transformed, cmap="gray")
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter()-t0:.3f} seconds.")
