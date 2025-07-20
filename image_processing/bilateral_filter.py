"""
Betthauser, 2018 -
This function implements 2-D bilateral filtering using
the method outlined in:

C. Tomasi and R. Manduchi. Bilateral Filtering for
Gray and Color Images. In Proceedings of the IEEE
International Conference on Computer Vision, 1998.

w is the half-size of the bilateral filter window. The standard
deviations of the bilateral filter are given by SIGMA,
where the spatial-domain standard deviation is given by
sigma_d and the intensity-domain standard deviation is
given by sigma_r.
"""

import time

import cv2

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    _summary_

    Args:
        img (NDArray[np.float64]): _description_

    Returns:
        NDArray[np.float64]: _description_

    """
    return (img - img.min()) / (img.max() - img.min())


def bilateral_filter(img: np.ndarray, sigma_d: float = 3, sigma_r: float = 0.1, w: int = 5) -> np.ndarray:
    """
    _summary_

    Args:
        img (NDArray[np.float64]): _description_
        sigma_d (float, optional): _description_. Defaults to 3.
        sigma_r (float, optional): _description_. Defaults to 0.1.
        w (int, optional): _description_. Defaults to 5.

    Returns:
        NDArray[np.float64]: _description_

    """
    if len(img.shape) == 2:  # noqa: PLR2004
        return bilateral_filter_gray(img, sigma_d, sigma_r, w)
    if len(img.shape) == 3:  # noqa: PLR2004
        return bilateral_filter_color(img, sigma_d, sigma_r, w)
    msg = "Image must be grayscale or color (2D or 3D array)."
    raise ValueError(msg)


def bilateral_filter_gray(img: np.ndarray, sigma_d: float = 3, sigma_r: float = 0.1, w: int = 5) -> np.ndarray:
    """
    _summary_

    Args:
        img (NDArray[np.float64]): image (2D array)
        sigma_d (float, optional): _description_. Defaults to 3.
        sigma_r (float, optional): _description_. Defaults to 0.1.
        w (int, optional): _description_. Defaults to 5.

    Returns:
        NDArray[np.float64]: _description_

    """
    img = normalize_image(img)

    x, y = np.meshgrid(np.arange(-w, w + 1, 1), np.arange(-w, w + 1, 1))
    g = np.exp(-(x**2 + y**2) / (2 * sigma_d**2))

    dim = img.shape
    filt_image = np.zeros(dim)
    for i in range(img.shape[0]):
        if i % 32 == 0:
            print(f"Percent complete: {100 * i / img.shape[0]:.3f}", flush=True)
        for j in range(img.shape[1]):
            # get local region
            i_min = np.max([i - w, 1])
            i_max = np.min([i + w, dim[0]])
            j_min = np.max([j - w, 1])
            j_max = np.min([j + w, dim[1]])
            i_img = img[i_min:i_max, j_min:j_max]

            # gaussian weights
            h = np.exp(-((i_img - img[i, j]) ** 2) / (2 * sigma_r**2))

            # filter response
            f = (
                h
                * g[
                    i_min - i + w + 1 : i_max - i + w + 1,
                    j_min - j + w + 1 : j_max - j + w + 1,
                ]
            )
            filt_image[i, j] = np.sum(f * i_img) / np.sum(f)

    return filt_image


def bilateral_filter_color(img: np.ndarray, sigma_d: float = 3, sigma_r: float = 0.1, w: int = 5) -> np.ndarray:
    """
    _summary_

    Args:
        img (NDArray[np.float64]): _description_
        sigma_d (float, optional): _description_. Defaults to 3.
        sigma_r (float, optional): _description_. Defaults to 0.1.
        w (int, optional): _description_. Defaults to 5.

    Returns:
        DArray[np.float64]: _description_

    """
    # RGB image to CIELab color space
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    x, y = np.meshgrid(np.arange(-w, w + 1, 1), np.arange(-w, w + 1, 1))
    g = np.exp(-(x**2 + y**2) / (2 * sigma_d**2))

    # Rescale range variance
    sigma_r = 100 * sigma_r

    dim = img.shape
    filt_image = np.zeros(dim)
    for i in range(img.shape[0]):
        if i % 32 == 0:
            print(f"Percent complete: {100 * i / img.shape[0]:.3f}", flush=True)
        for j in range(img.shape[1]):
            # get local region
            i_min = np.max([i - w, 1])
            i_max = np.min([i + w, dim[0]])
            j_min = np.max([j - w, 1])
            j_max = np.min([j + w, dim[1]])
            i_img = img[i_min:i_max, j_min:j_max, :]

            # gaussian weights
            dl = i_img[:, :, 0] - img[i, j, 0]
            da = i_img[:, :, 1] - img[i, j, 1]
            db = i_img[:, :, 2] - img[i, j, 2]
            h = np.exp(-(dl**2 + da**2 + db**2) / (2 * sigma_r**2))

            # filter response
            f = (
                h
                * g[
                    i_min - i + w + 1 : i_max - i + w + 1,
                    j_min - j + w + 1 : j_max - j + w + 1,
                ]
            )
            norm_f = np.sum(f)
            filt_image[i, j, 0] = np.sum(f * i_img[:, :, 0]) / norm_f
            filt_image[i, j, 1] = np.sum(f * i_img[:, :, 1]) / norm_f
            filt_image[i, j, 2] = np.sum(f * i_img[:, :, 2]) / norm_f

    filt_image = np.floor(filt_image + 0.5).astype(np.uint8)

    # Convert filtered image back to RGB color space.
    return cv2.cvtColor(filt_image, cv2.COLOR_LAB2RGB)


def main() -> None:
    """_summary_"""
    #  img = Image.open("lena_noisy.png")
    img = Image.open("./data/31734105968_14c1b75765_o.jpg")
    img = np.array(img)
    print(img.shape)

    scale_factor = 0.8  # percent of original size
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

    filt_image = bilateral_filter(img)

    plt.subplot(1, 2, 1)
    if len(img.shape) == 2:  # noqa: PLR2004
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.axis("off")
    plt.title("Input image")
    plt.subplot(1, 2, 2)
    if len(img.shape) == 2:  # noqa: PLR2004
        plt.imshow(filt_image, cmap="gray")
    else:
        plt.imshow(filt_image)
    plt.axis("off")
    plt.title("Bilateral filtered image")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter() - t0:.3f} seconds.")
