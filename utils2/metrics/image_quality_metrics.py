"""
Created on Wed Mar 20 00:33:40 2024.

@author: jlbetthauser
"""

import os
import time

import cv2
import numpy as np

# from distance_metrics import bhattacharyya_dist, minmax_scaling
from numba import njit
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_laplace
from scipy.signal import convolve2d
from skimage.color import rgb2gray  # noqa: F401, RUF100
from skimage.filters import sobel, threshold_otsu

# from skimage.restoration import estimate_sigma
from skimage.measure import blur_effect

# from statsmodels.stats.outliers_influence import variance_inflation_factor


def _segment_center_of_mass(label_mask: np.ndarray) -> np.ndarray:
    """
    Calculate the center of mass of a label mask.

    Args:
        label_mask (np.ndarray): The label mask to calculate the center of mass from.

    Returns:
        tuple[int, int, int]: The coordinates of the center of mass.

    """
    if not isinstance(label_mask, np.ndarray):
        msg = "label_mask must be a numpy array"
        raise TypeError(msg)

    if label_mask.ndim not in {2, 3}:
        msg = "label_mask must be a 2D or 3D array"
        raise ValueError(msg)

    if np.sum(label_mask) == 0:
        return (0.5 * np.array(label_mask.shape)).astype(int)

    return np.array(np.where(label_mask)).mean(axis=1).astype(int)


@njit
def _normalize_ndarray(arr: np.ndarray, max_val: float = 1.0) -> np.ndarray:
    """Normalize a nD numpy array to the range [0, max_val]."""
    min_val = np.min(arr)
    return max_val * (arr - min_val) / (np.max(arr) - min_val)


@njit
def _bhattacharyya_dist(mu1: float, v1: float, mu2: float, v2: float) -> float:
    """
    Betthauser - 2021 - compute Bhattacharyya distance between two normal distributions.

    Args:
        mu1 (float): mean of distribution 1
        v1 (float): variance of distribution 1
        mu2 (float): mean of distribution 2
        v2 (float): variance of distribution 2

    Returns:
        float: Bhattacharyya distance

    """
    part1 = 0.25 * np.log(0.25 * (v1 / v2 + v2 / v1 + 2))
    diff_mu = mu1 - mu2
    return 0.25 * ((diff_mu * diff_mu) / (v1 + v2)) + part1


def blurriness1(image: np.ndarray[float], h_size: int = 11) -> float:
    return blur_effect(image, h_size=h_size)


def blurriness2(
    image: np.ndarray[float],
    h_size: int = 11,
) -> float:
    """
    Metric that indicates the strength of blur in an image (0 for no blur, 1 for maximal blur).
            [1] Frederique Crete, et al. "The blur effect: perception and estimation with a new
            no-reference perceptual blur metric" Proc. SPIE 6492 (2007)
            https://hal.archives-ouvertes.fr/hal-00232709:DOI:'10.1117/12.702790'.

    Args:
        image (np.ndarray[float]): image
        h_size (int, optional): Size of the re-blurring filter. Defaults to 11.

    Returns:
        float: Blur metric in [0,1]: by default, the maximum (JLB changed to mean) of blur metrics along all axes.

    """
    b = np.zeros(2)
    slices = tuple([slice(2, s - 1) for s in image.shape])
    for ax in range(image.ndim):
        filt_im = ndi.uniform_filter1d(image, h_size, axis=ax)
        im_sharp = np.abs(sobel(image, axis=ax))
        im_blur = np.abs(sobel(filt_im, axis=ax))
        t = np.maximum(0, im_sharp - im_blur)
        m1 = np.sum(im_sharp[slices])
        m2 = np.sum(t[slices])
        b[ax] = np.abs(m1 - m2) / m1

    return np.max(b)


def otsu_threshold(img: np.ndarray[float]) -> float:
    """
    Calculate otsu's threshold.

    Args:
        img (np.ndarray[float]): input image

    Returns:
        float: otsu's threshold

    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    return threshold_otsu(blur)


def otsu_interclass_distance(img: np.ndarray[float]) -> float:
    """
    _summary_.

    Args:
        img (np.ndarray[float]): input image

    Returns:
        float: bhattacharya distance between image data above otsu threshold and below threshold

    """
    threshold = otsu_threshold(img)
    img = img.ravel()
    img = img[np.nonzero(img)]
    img_hi = img[np.where(img >= threshold)]
    img_lo = img[np.where(img < threshold)]
    mu1, v1 = np.nanmean(img_hi), np.nanvar(img_hi)
    mu2, v2 = np.nanmean(img_lo), np.nanvar(img_lo)
    if v1 <= 0 or v2 <= 0:
        return 0.0
    if mu1 <= 0 or mu2 <= 0:
        return 0.0
    return _bhattacharyya_dist(mu1, v1, mu2, v2)


@njit
def estimate_variance(img: np.ndarray[float]) -> float:
    """
    _summary_.

    Args:
        img (np.ndarray[float]): _description_

    Returns:
        float: _description_

    """
    img = img.ravel()
    img = img[np.nonzero(img)]
    return np.var(img)


def estimate_noise(img: np.ndarray[float]) -> float:
    """
    _summary_.

    Args:
        img (np.ndarray[float]): image

    Returns:
        float: noise estimate

    """
    h, w = img.shape
    mfilter = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])

    sum_t = np.sum(np.absolute(convolve2d(img, mfilter)))
    return sum_t * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))


@njit
def signal_to_noise(img: np.ndarray[float]) -> float:
    """
    A rough analogue to signal-to-noise ratio of the input data.
        Returns the snr of img, here defined as the mean
        divided by the standard deviation.

    Args:
        img (np.ndarray[float]): image

    Returns:
        float: snr

    """
    img = img.ravel()
    img = img[np.nonzero(img)]
    m = np.nanmean(img)
    sdv = np.sqrt(np.nanvar(img))
    if sdv == 0.0:
        return 0.0
    return m / sdv


def laplacian_edge_strength(img: np.ndarray[float]) -> float:
    """
    _summary_.

    Args:
        img (np.ndarray[float]): _description_

    Returns:
        float: _description_

    """
    # lap = cv2.convertScaleAbs(cv2.Laplacian(img, 5))
    lap = np.abs(gaussian_laplace(img, sigma=3))
    return np.nanmean(lap[np.nonzero(lap)])


def get_iqa_metrics(image: np.ndarray[float]) -> tuple[float]:
    """
    Compute all 6 no-reference measures.

    Args:
        image (np.ndarray[float]): gray-scale (2D) image

    Returns:
        tuple[float]: tuple of no-reference measures

    """
    img = _normalize_ndarray(image, max_val=1.0)
    snr = signal_to_noise(img)
    est_var = estimate_variance(img)
    otsu = otsu_interclass_distance(img)

    if img.ndim == 2:  # noqa: PLR2004
        est_noise = estimate_noise(img)
        blur2 = blurriness2(img)
        lap_edge_str = laplacian_edge_strength(img)

    else:
        xc, yc, zc = _segment_center_of_mass(img)

        est_noise = (estimate_noise(img[xc, :, :]) + estimate_noise(img[:, yc, :]) + estimate_noise(img[:, :, zc])) / 3
        blur2 = (blurriness2(img[xc, :, :]) + blurriness2(img[:, yc, :]) + blurriness2(img[:, :, zc])) / 3
        lap_edge_str = (
            laplacian_edge_strength(img[xc, :, :])
            + laplacian_edge_strength(img[:, yc, :])
            + laplacian_edge_strength(img[:, :, zc])
        ) / 3

    return {
        "snr": snr,
        "noise": est_noise,
        "blur": blur2,
        "edge_strength": lap_edge_str,
        "variance": est_var,
        "otsu": otsu,
    }


def main() -> None:
    pwd = os.path.dirname(os.path.abspath(__file__))
    print(pwd)

    img = cv2.imread(os.path.join(pwd, "einstein.jpg"))
    img = rgb2gray(img)
    img = _normalize_ndarray(img, max_val=1.0)

    metrics = get_iqa_metrics(img)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    tfirst = time.perf_counter() - tmain
    print(f"Program took {tfirst:.3f} seconds.")

    tmain = time.perf_counter()
    main()
    tlast = time.perf_counter() - tmain
    print(f"Program took {tlast:.3f} seconds.")

    print(f"jit speed-up: {tfirst / tlast:.3f}x")
