""" _summary_
"""

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import keyboard

import cv2
from scipy import ndimage as ndi
from numpy.typing import NDArray


## Implement Lucas Kanade
def lucas_kanade(
    im1: NDArray[np.float64], im2: NDArray[np.float64], win: int = 7
) -> tuple[NDArray[np.float64]]:
    Ix = np.zeros(im1.shape)
    Iy = np.zeros(im1.shape)
    It = np.zeros(im1.shape)

    Ix[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2
    Iy[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2
    It[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]

    params = np.zeros(im1.shape + (5,))
    params[:, :, 0] = cv2.GaussianBlur(Ix * Ix, (5, 5), 3)
    params[:, :, 1] = cv2.GaussianBlur(Iy * Iy, (5, 5), 3)
    params[:, :, 2] = cv2.GaussianBlur(Ix * Iy, (5, 5), 3)
    params[:, :, 3] = cv2.GaussianBlur(Ix * It, (5, 5), 3)
    params[:, :, 4] = cv2.GaussianBlur(Iy * It, (5, 5), 3)

    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    win_params = (
        cum_params[2 * win + 1 :, 2 * win + 1 :]
        - cum_params[2 * win + 1 :, : -1 - 2 * win]
        - cum_params[: -1 - 2 * win, 2 * win + 1 :]
        + cum_params[: -1 - 2 * win, : -1 - 2 * win]
    )

    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)

    Ixx = win_params[:, :, 0]
    Iyy = win_params[:, :, 1]
    Ixy = win_params[:, :, 2]
    Ixt = -win_params[:, :, 3]
    Iyt = -win_params[:, :, 4]

    M_det = Ixx * Iyy - Ixy**2
    temp_u = Iyy * (-Ixt) + (-Ixy) * (-Iyt)
    temp_v = (-Ixy) * (-Ixt) + Ixx * (-Iyt)
    op_flow_x = np.where(M_det != 0, temp_u / M_det, 0)
    op_flow_y = np.where(M_det != 0, temp_v / M_det, 0)

    u[win + 1 : -1 - win, win + 1 : -1 - win] = op_flow_x[:-1, :-1]
    v[win + 1 : -1 - win, win + 1 : -1 - win] = op_flow_y[:-1, :-1]

    return -u, -v


def image_norm(img: NDArray[np.float64]) -> NDArray[np.float64]:
    """_summary_

    Args:
        img (NDArray[np.float64]): image

    Returns:
        NDArray[np.float64]: min-max normalized image in [0,1]
    """
    return (img - np.min(img)) / (np.max(img) - np.min(img))


if __name__ == "__main__":

    dicom_file = "./data/42211"
    factor = 2
    filter_size = 1 / factor
    arrow_spacing = 16 // factor
    scale = 40
    dim = 512 // factor

    dicom_data = pydicom.dcmread(dicom_file)
    zoom_factor = dim / dicom_data.pixel_array.shape[1]
    N = dicom_data.pixel_array.shape[0]

    vol = np.zeros((N, dim, dim))
    for i in range(N):
        vol[i, :, :] = ndi.zoom(dicom_data.pixel_array[i, :, :], zoom=zoom_factor)

    vol = image_norm(vol)
    for _ in range(5):

        for i in np.arange(35, vol.shape[0] - 10, 1):
            # if keyboard.is_pressed("q"):
            #     break

            Iold = vol[i, :, :]
            Inew = vol[i + 1, :, :]

            Iold = ndi.gaussian_filter(Iold, filter_size)
            Inew = ndi.gaussian_filter(Inew, filter_size)

            U, V = lucas_kanade(Iold, Inew, win=21)

            # plt.subplot(1, 3, 1)
            plt.cla()
            plt.imshow(Iold, cmap="gray")
            Y, X = np.mgrid[0:dim:arrow_spacing, 0:dim:arrow_spacing]
            U_small = U[Y, X]
            V_small = V[Y, X]
            plt.quiver(X, Y, U_small, V_small, color="r", scale=scale)
            plt.pause(0.001)

            # plt.subplot(1, 3, 2)
            # plt.cla()
            # plt.imshow(U)

            # plt.subplot(1, 3, 3)
            # plt.cla()
            # plt.imshow(V)
            # plt.tight_layout()
            # plt.pause(0.001)
