"""
Created on Wed Mar 20 00:33:40 2024

@author: jlbetthauser
"""

import time
import numpy as np
from scipy import ndimage as ndi
import math
import cv2
from scipy.signal import convolve2d
from skimage.filters import threshold_otsu, sobel
from skimage.color import rgb2gray
from skimage.util import img_as_float
from scipy.ndimage import gaussian_laplace
from numpy.typing import NDArray
from numba import njit

# from skimage.restoration import estimate_sigma
# from skimage.measure import blur_effect
# from statsmodels.stats.outliers_influence import variance_inflation_factor


try:
    from numpy import AxisError
except ImportError:
    from numpy.exceptions import AxisError


def blurriness2(image, h_size=11, channel_axis=None, reduce_func=np.mean) -> float:
    """metric that indicates the strength of blur in an image (0 for no blur, 1 for maximal blur).
            [1] Frederique Crete, et al. "The blur effect: perception and estimation with a new
            no-reference perceptual blur metric" Proc. SPIE 6492 (2007)
            https://hal.archives-ouvertes.fr/hal-00232709:DOI:'10.1117/12.702790'

    Args:
        image (_type_): image
        h_size (int, optional): Size of the re-blurring filter. Defaults to 11.
        channel_axis (_type_, optional): if None, the image is assumed to be grayscale (single-channel).
                                        Otherwise, this parameter indicates which axis of the array
                                        corresponds to color channels.. Defaults to None.
        reduce_func (_type_, optional): Function used to calculate the aggregation of blur metrics along all
                                        axes. If set to None, the entire list is returned, where the i-th
                                        element is the blur metric along the i-th axis.. Defaults to np.mean.

    Returns:
        float: Blur metric in [0,1]: by default, the maximum (JLB changed to mean) of blur metrics along all axes.
    """

    if channel_axis is not None:
        image = np.moveaxis(image, channel_axis, -1)
        image = rgb2gray(image)

    n_axes = image.ndim
    image = img_as_float(image)
    shape = image.shape
    B = []

    slices = tuple([slice(2, s - 1) for s in shape])
    for ax in range(n_axes):
        filt_im = ndi.uniform_filter1d(image, h_size, axis=ax)
        im_sharp = np.abs(sobel(image, axis=ax))
        im_blur = np.abs(sobel(filt_im, axis=ax))
        T = np.maximum(0, im_sharp - im_blur)
        M1 = np.sum(im_sharp[slices])
        M2 = np.sum(T[slices])
        B.append(np.abs(M1 - M2) / M1)

    return B if reduce_func is None else reduce_func(B)


def otsu_threshold(img, nbins=0.1):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thr = threshold_otsu(blur)
    # _,thr = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thr


@njit
def bhattacharyya_dist(mu1, v1, mu2, v2):
    part1 = 0.25 * np.log(0.25 * (v1 / v2 + v2 / v1 + 2))
    part2 = 0.25 * (((mu1 - mu2) ** 2) / (v1 + v2))
    return part1 + part2


def otsu_interclass_distance(img: NDArray[np.float64]) -> float:
    threshold = otsu_threshold(img)
    if threshold >= 255 or threshold <= 0:
        return 0.0

    mu1, v1 = np.nanmean(img[img >= threshold]), np.nanvar(img[img >= threshold])
    mu2, v2 = np.nanmean(img[img < threshold]), np.nanvar(img[img < threshold])
    if v1 <= 0 or v2 <= 0 or np.isnan(v1) or np.isnan(v2):
        return 0.0
    if mu1 <= 0 or mu2 <= 0 or np.isnan(mu1) or np.isnan(mu2):
        return 0.0
    if np.isinf(v1) or np.isinf(v2) or np.isinf(mu1) or np.isinf(mu2):
        return 0.0
    return bhattacharyya_dist(mu1, v1, mu2, v2)


## sigma
@njit
def estimate_variance(img):
    return np.nanvar(img)


# blur
def estimate_noise(img: NDArray[np.float64]) -> float:
    """_summary_

    Args:
        img (NDArray[np.float64]): image

    Returns:
        float: noise estimate
    """
    h, w = img.shape
    mfilter = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]

    sum_t = np.sum(np.absolute(convolve2d(img, mfilter)))
    sigma = sum_t * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    return sigma


## snr
@njit
def signal_to_noise(img: NDArray[np.float64]) -> float:
    """A rough analogue to signal-to-noise ratio of the input data.
        Returns the snr of img, here defined as the mean
        divided by the standard deviation.

    Args:
        img (NDArray[np.float64]): image

    Returns:
        float: snr
    """
    anz = img  # img[img > 1]
    n = anz.shape[0]
    if n <= 1:
        return 0.0
    m = np.nanmean(anz)
    sdv = np.sqrt(np.nanvar(anz))
    return m / sdv


def laplacian_edge_strength(img: NDArray[np.float64]) -> float:
    # lap = cv2.convertScaleAbs(cv2.Laplacian(img, 5))
    lap = np.abs(gaussian_laplace(img, sigma=3))
    return np.mean(lap[lap > 0])


def jlb_iqa(img):
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.float64)
    snr = signal_to_noise(img)
    est_noise = estimate_noise(img)
    blur2 = blurriness2(img)
    lap_edge_str = laplacian_edge_strength(img)
    est_var = estimate_variance(img)
    otsu = otsu_interclass_distance(img)
    return (snr, est_noise, blur2, lap_edge_str, est_var, otsu)


def main() -> None:
    img = cv2.imread("einstein.jpg")
    img = rgb2gray(img)
    print(jlb_iqa(img))


if __name__ == "__main__":
    tmain = time.time()
    main()
    tfirst = time.time() - tmain
    print(f"Program took {tfirst:.3f} seconds.")

    tmain = time.time()
    main()
    tlast = time.time() - tmain
    print(f"Program took {tlast:.3f} seconds.")

    print(f"njit speed-up: {tfirst/tlast:.3f}x")
