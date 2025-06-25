"""Betthauser, J. - 2018 - "Horn-Schunck Optical Flow Algorithm"""

import time
import matplotlib
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from numpy.typing import NDArray

matplotlib.rcParams["image.cmap"] = "gray"


def horn_schunck(
    img1: NDArray[np.float64], img2: NDArray[np.float64], alpha: float, iterations: int
) -> tuple[NDArray[np.float64]]:
    """_summary_

    Args:
        img1 (NDArray[np.float64]): _description_
        img2 (NDArray[np.float64]): _description_
        alpha (float): _description_
        iterations (int): _description_

    Returns:
        tuple[NDArray[np.float64]]: _description_
    """

    # set up initial velocities
    u_init = np.zeros([img1.shape[0], img1.shape[1]])
    v_init = np.zeros([img1.shape[0], img1.shape[1]])

    # Set initial value for the flow vectors
    U = u_init
    V = v_init

    # Estimate derivatives
    [fx, fy, ft] = spatiotemporal_image_derivatives(img1, img2)

    # Averaging kernel
    kernel = np.array([[0.5, 1, 0.5], [1, 1, 1], [0.5, 1, 0.5]])
    kernel = kernel / np.sum(kernel)

    # print(fx[100, 100], fy[100, 100], ft[100, 100])

    # Iteration to reduce error
    for _ in range(iterations):
        # Compute local averages of the flow vectors
        u_avg = ndi.convolve(U, kernel)
        v_avg = ndi.convolve(V, kernel)

        # common part of update step
        der = (fx * u_avg + fy * v_avg + ft) / (alpha**2 + fx**2 + fy**2)

        # iterative step
        U = u_avg - fx * der
        V = v_avg - fy * der

    return -U, -V


def spatiotemporal_image_derivatives(
    img1: NDArray[np.float64],
    img2: NDArray[np.float64],
) -> tuple[NDArray[np.float64]]:
    """_summary_

    Args:
        img1 (NDArray[np.float64]): _description_
        img2 (NDArray[np.float64]): _description_

    Returns:
        tuple[NDArray[np.float64]]: _description_
    """

    # build kernels for calculating derivatives
    kerneldX = np.array([[-1, 1], [-1, 1]]) * 0.25  # kernel for computing d/dx
    kerneldY = np.array([[-1, -1], [1, 1]]) * 0.25  # kernel for computing d/dy
    kerneldT = np.ones((2, 2)) * 0.25

    fx = ndi.convolve(img1, kerneldX) + ndi.convolve(img2, kerneldX)
    fy = ndi.convolve(img1, kerneldY) + ndi.convolve(img2, kerneldY)

    # ft = im2 - im1
    ft = ndi.convolve(img1, kerneldT) + ndi.convolve(img2, -kerneldT)

    return fx, fy, ft


def image_norm(img: NDArray[np.float64]) -> NDArray[np.float64]:
    """_summary_

    Args:
        img (NDArray[np.float64]): image

    Returns:
        NDArray[np.float64]: min-max normalized image in [0,1]
    """
    return (img - np.min(img)) / (np.max(img) - np.min(img))


#### MAIN #####
def main() -> None:
    """_summary_"""
    dicom_file = "./data/42211"
    factor = 4
    filter_size = 2 / factor
    arrow_spacing = 16 // factor
    scale = 12 // factor
    dim = 512 // factor

    dicom_data = pydicom.dcmread(dicom_file)
    print(dicom_data.pixel_array.shape)
    zoom_factor = dim / dicom_data.pixel_array.shape[1]  ## Scale image to 512x512
    N = dicom_data.pixel_array.shape[0]

    vol = np.zeros((N, dim, dim))
    for i in range(N):
        vol[i, :, :] = ndi.zoom(dicom_data.pixel_array[i, :, :], zoom=zoom_factor)

    vol = image_norm(vol)
    for _ in range(5):

        for i in range(35, vol.shape[0] - 10, 1):
            Iold = vol[i, :, :]
            Inew = vol[i + 1, :, :]

            Iold = ndi.gaussian_filter(Iold, filter_size)
            Inew = ndi.gaussian_filter(Inew, filter_size)

            U, V = horn_schunck(Iold, Inew, alpha=1, iterations=20)

            # plt.subplot(1, 3, 1)
            plt.cla()
            plt.imshow(Iold, cmap="gray")
            Y, X = np.mgrid[0:dim:arrow_spacing, 0:dim:arrow_spacing]
            U_small = U[Y, X]
            V_small = V[Y, X]
            plt.quiver(X, Y, U_small, V_small, color="r", scale=scale)

            # plt.subplot(1, 3, 2)
            # plt.cla()
            # plt.imshow(U, cmap="gray")
            # plt.subplot(1, 3, 3)
            # plt.cla()
            # plt.imshow(V, cmap="gray")
            plt.pause(0.001)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter() - t0:.3f} seconds.")
