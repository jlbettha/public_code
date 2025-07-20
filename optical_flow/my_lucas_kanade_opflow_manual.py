"""_summary_"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from numpy.typing import NDArray
from scipy import ndimage as ndi


## Implement Lucas Kanade
def lucas_kanade(im1: NDArray[np.float64], im2: NDArray[np.float64], win: int = 7) -> tuple[NDArray[np.float64]]:
    """
    _summary_

    Args:
        im1 (NDArray[np.float64]): _description_
        im2 (NDArray[np.float64]): _description_
        win (int, optional): _description_. Defaults to 7.

    Returns:
        tuple[NDArray[np.float64]]: _description_

    """
    ix = np.zeros(im1.shape)
    iy = np.zeros(im1.shape)
    it = np.zeros(im1.shape)

    ix[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2
    iy[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2
    it[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]

    params = np.zeros((*im1.shape, 5))
    params[:, :, 0] = cv2.GaussianBlur(ix * ix, (5, 5), 3)
    params[:, :, 1] = cv2.GaussianBlur(iy * iy, (5, 5), 3)
    params[:, :, 2] = cv2.GaussianBlur(ix * iy, (5, 5), 3)
    params[:, :, 3] = cv2.GaussianBlur(ix * it, (5, 5), 3)
    params[:, :, 4] = cv2.GaussianBlur(iy * it, (5, 5), 3)

    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    win_params = (
        cum_params[2 * win + 1 :, 2 * win + 1 :]
        - cum_params[2 * win + 1 :, : -1 - 2 * win]
        - cum_params[: -1 - 2 * win, 2 * win + 1 :]
        + cum_params[: -1 - 2 * win, : -1 - 2 * win]
    )

    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)

    ixx = win_params[:, :, 0]
    iyy = win_params[:, :, 1]
    ixy = win_params[:, :, 2]
    ixt = -win_params[:, :, 3]
    iyt = -win_params[:, :, 4]

    m_det = ixx * iyy - ixy**2
    temp_u = iyy * (-ixt) + (-ixy) * (-iyt)
    temp_v = (-ixy) * (-ixt) + ixx * (-iyt)
    op_flow_x = np.where(m_det != 0, temp_u / m_det, 0)
    op_flow_y = np.where(m_det != 0, temp_v / m_det, 0)

    u[win + 1 : -1 - win, win + 1 : -1 - win] = op_flow_x[:-1, :-1]
    v[win + 1 : -1 - win, win + 1 : -1 - win] = op_flow_y[:-1, :-1]

    return -u, -v


def image_norm(img: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    _summary_

    Args:
        img (NDArray[np.float64]): image

    Returns:
        NDArray[np.float64]: min-max normalized image in [0,1]

    """
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def main() -> None:
    dicom_file = "./data/42211"
    factor = 2
    filter_size = 1 / factor
    arrow_spacing = 16 // factor
    scale = 40
    dim = 512 // factor

    dicom_data = pydicom.dcmread(dicom_file)
    zoom_factor = dim / dicom_data.pixel_array.shape[1]
    n = dicom_data.pixel_array.shape[0]

    vol = np.zeros((n, dim, dim))
    for i in range(n):
        vol[i, :, :] = ndi.zoom(dicom_data.pixel_array[i, :, :], zoom=zoom_factor)

    timestep = 0
    vol = image_norm(vol)
    for _ in range(15):
        for i in np.arange(35, vol.shape[0] - 10, 1):
            # if keyboard.is_pressed("q"):
            #     break
            timestep = timestep + 1
            img_old = vol[i, :, :]
            img_new = vol[i + 1, :, :]

            img_old = ndi.gaussian_filter(img_old, filter_size)
            img_new = ndi.gaussian_filter(img_new, filter_size)

            u, v = lucas_kanade(img_old, img_new, win=21)

            # plt.subplot(1, 3, 1)
            plt.cla()
            plt.imshow(img_old, cmap="gray")
            y, x = np.mgrid[0:dim:arrow_spacing, 0:dim:arrow_spacing]
            u_small = u[y, x]
            v_small = v[y, x]
            plt.quiver(x, y, u_small, v_small, color="r", scale=scale)
            plt.axis("off")
            plt.pause(0.001)
            # plt.savefig(f"./plots/myplot_{timestep:06d}.png")

            # plt.subplot(1, 3, 2)
            # plt.cla()
            # plt.imshow(U)

            # plt.subplot(1, 3, 3)
            # plt.cla()
            # plt.imshow(V)
            # plt.tight_layout()
            # plt.pause(0.001)


if __name__ == "__main__":
    main()
